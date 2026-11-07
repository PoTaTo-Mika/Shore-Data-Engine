import argparse
import logging
import os
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

WATCH_DIR = Path(os.environ.get("SHORE_DATA_DIR", "data")).resolve()
AUDIO_EXTS = frozenset(
    {".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".wav", ".flac", ".aac", ".wma", ".webm"}
)
OPUS_BITRATE = os.environ.get("SHORE_OPUS_BITRATE", "128k")
STABLE_SECONDS = float(os.environ.get("SHORE_STABLE_SECONDS", "5"))
TRANSCODE_WORKERS = int(os.environ.get("SHORE_TRANSCODE_WORKERS", "4"))
DELETE_RAW = os.environ.get("SHORE_DELETE_RAW", "0") == "1"
SCAN_INTERVAL = float(os.environ.get("SHORE_SCAN_INTERVAL", "10"))
TAR_MAX_SIZE_GB = float(os.environ.get("SHORE_TAR_MAX_SIZE_GB", "1"))
TAR_MAX_SIZE = int(TAR_MAX_SIZE_GB * 1024 ** 3)
STALE_SECONDS = float(os.environ.get("SHORE_STALE_SECONDS", "3600"))

Observation = tuple[int, float]


def opus_path(source: Path) -> Path:
    return source.with_suffix(".opus")


def marker_path(source: Path) -> Path:
    return source.with_name(f".{source.name}.done")


def transcode_to_opus(source: Path, destination: Path) -> None:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-c:a",
        "libopus",
        "-b:a",
        OPUS_BITRATE,
        "-vbr",
        "on",
        "-f",
        "ogg",
        str(destination),
    ]
    subprocess.run(command, capture_output=True, check=True)


def transcode_one(source: Path) -> Path:
    """Publish one opus file; keep the raw file so crawlers still skip it."""
    target = opus_path(source)
    marker = marker_path(source)
    if not target.exists():
        part = target.with_name(target.name + ".part")
        try:
            transcode_to_opus(source, part)
            if part.stat().st_size == 0:
                raise RuntimeError(f"Empty opus output for {source}")
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
    if DELETE_RAW:
        source.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
    else:
        marker.write_text("done", encoding="utf-8")
    logger.info("Transcoded %s -> %s", source.name, target.name)
    return target


def is_stable(
    path: Path, observations: dict[Path, Observation], now: float
) -> bool:
    """A download is done once its mtime is old and two observations agree."""
    stat = path.stat()
    current = (stat.st_size, stat.st_mtime)
    previous = observations.get(path)
    observations[path] = current
    if now - stat.st_mtime < STABLE_SECONDS:
        return False
    return previous is None or previous == current


def transcode_pending(
    watch_dir: Path, observations: dict[Path, Observation] | None = None
) -> int:
    """Transcode settled raw downloads in parallel; returns the number attempted."""
    seen = observations if observations is not None else {}
    now = time.time()
    candidates: list[Path] = []
    current: set[Path] = set()
    for path in sorted(watch_dir.rglob("*")):
        relative = path.relative_to(watch_dir)
        if any(part.startswith(("_", ".")) for part in relative.parts):
            continue
        if not path.is_file() or path.name.endswith(".part"):
            continue
        if path.suffix.lower() == ".opus" or path.suffix.lower() not in AUDIO_EXTS:
            continue
        current.add(path)
        if marker_path(path).exists():
            continue
        if not is_stable(path, seen, now):
            continue
        candidates.append(path)
    for stale in set(seen) - current:
        del seen[stale]
    if not candidates:
        return 0

    with ThreadPoolExecutor(max_workers=TRANSCODE_WORKERS) as pool:
        futures = {pool.submit(transcode_one, path): path for path in candidates}
        for future in as_completed(futures):
            try:
                future.result()
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                logger.warning("Cannot transcode %s yet: %s", futures[future], exc)
    return len(candidates)


def scan_buckets(watch_dir: Path) -> dict[str, list[Path]]:
    """Group finished opus files by their directory relative to the watch root."""
    buckets: dict[str, list[Path]] = {}
    if not watch_dir.exists():
        return buckets
    for path in sorted(watch_dir.rglob("*")):
        relative = path.relative_to(watch_dir)
        if any(part.startswith(("_", ".")) for part in relative.parts):
            continue
        if not path.is_file() or path.suffix.lower() != ".opus":
            continue
        if path.name.endswith(".part"):
            continue
        bucket = "/".join(relative.parts[:-1])
        buckets.setdefault(bucket, []).append(path)
    for files in buckets.values():
        files.sort(key=lambda p: p.stat().st_mtime)
    return buckets


def split_into_batches(
    files: list[Path], max_bytes: int, now: float, stale_secs: float
) -> tuple[list[list[Path]], list[Path]]:
    batches: list[list[Path]] = []
    batch: list[Path] = []
    batch_bytes = 0
    for path in files:
        size = path.stat().st_size
        if batch and batch_bytes + size > max_bytes:
            batches.append(batch)
            batch = []
            batch_bytes = 0
        batch.append(path)
        batch_bytes += size
        if batch_bytes >= max_bytes:
            batches.append(batch)
            batch = []
            batch_bytes = 0
    rest: list[Path] = []
    if batch:
        oldest = min(path.stat().st_mtime for path in batch)
        if now - oldest >= stale_secs:
            batches.append(batch)
        else:
            rest = batch
    return batches, rest


def pack_batch(watch_dir: Path, bucket: str, batch: list[Path]) -> Path:
    flat = bucket.replace("/", "_") or "root"
    seq = len(list(watch_dir.glob(f"{flat}_*.tar")))
    tar_name = f"{flat}_{time.strftime('%Y%m%d_%H%M%S')}_{seq:03d}.tar"
    tar_path = watch_dir / tar_name
    tmp_path = watch_dir / f"{tar_name}.tmp"
    try:
        with tarfile.open(tmp_path, "w") as tar:
            for path in batch:
                tar.add(path, arcname=path.relative_to(watch_dir).as_posix())
        tmp_path.replace(tar_path)
    except (OSError, tarfile.TarError):
        tmp_path.unlink(missing_ok=True)
        raise
    logger.info("Packed %s (%d files)", tar_path.name, len(batch))
    return tar_path


def pack_bucket(
    watch_dir: Path,
    bucket: str,
    files: list[Path],
    now: float,
    stale_secs: float,
) -> list[Path]:
    batches, rest = split_into_batches(files, TAR_MAX_SIZE, now, stale_secs)
    packed = []
    for batch in batches:
        packed.append(pack_batch(watch_dir, bucket, batch))
        for path in batch:
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Cannot remove %s: %s", path, exc)
    if rest:
        oldest = min(path.stat().st_mtime for path in rest)
        logger.info(
            "Bucket %s holds %d files below threshold; forced pack in %.0f minutes",
            bucket,
            len(rest),
            max(0.0, stale_secs - (now - oldest)) / 60,
        )
    return packed


def pack_once(
    watch_dir: Path = WATCH_DIR,
    observations: dict[Path, Observation] | None = None,
) -> list[Path]:
    """Transcode first, then measure the real opus bytes before splitting batches."""
    watch_dir.mkdir(parents=True, exist_ok=True)
    transcode_pending(watch_dir, observations)
    now = time.time()
    packed: list[Path] = []
    for bucket, files in scan_buckets(watch_dir).items():
        packed.extend(pack_bucket(watch_dir, bucket, files, now, STALE_SECONDS))
    return packed


def start_monitor(watch_dir: Path = WATCH_DIR) -> None:
    watch_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Monitoring %s (%.1f GiB/tar, forced pack after %.0fs)",
        watch_dir,
        TAR_MAX_SIZE_GB,
        STALE_SECONDS,
    )
    observations: dict[Path, Observation] = {}
    while True:
        try:
            pack_once(watch_dir, observations)
        except Exception:
            logger.exception("Monitor iteration failed; retrying")
        time.sleep(SCAN_INTERVAL)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcode crawler downloads and pack buckets into tars."
    )
    parser.add_argument("--once", action="store_true", help="Run one pass and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.once:
        pack_once()
    else:
        start_monitor()


if __name__ == "__main__":
    main()
