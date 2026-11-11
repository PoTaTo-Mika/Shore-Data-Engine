"""Pack fully downloaded audio into fixed-size opus tars, deleting packed sources."""

import argparse
import logging
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIO_EXTS = frozenset(
    {".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".wav", ".flac", ".aac", ".wma", ".webm"}
)
DEFAULT_BITRATE = "128k"
DEFAULT_TAR_SIZE_GB = 1.0
DEFAULT_WORKERS = 4


def transcode(source: Path, bitrate: str) -> Path:
    """Return the opus for `source`, encoding it next to the source if missing."""
    if source.suffix.lower() == ".opus":
        return source
    target = source.with_suffix(".opus")
    if target.exists():
        return target  # Left by an interrupted run; it was published atomically.
    part = target.with_name(f".{target.name}.part")
    try:
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source), "-vn", "-c:a", "libopus", "-b:a", bitrate,
                "-vbr", "on", "-f", "ogg", str(part),
            ],
            capture_output=True,
            check=True,
        )  # fmt: skip
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
    return target


def pack(
    root: Path, tar_size: int, bitrate: str, workers: int
) -> tuple[list[Path], list[Path]]:
    """Pack audio under `root` per directory; returns (tars, sources that failed).

    Each tar holds at most `tar_size` opus bytes unless a single file is larger. Sources
    and their opus are deleted only after the tar holding them is in place.
    """
    sources = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTS
        and not any(p.startswith((".", "_")) for p in path.relative_to(root).parts)
    )
    origins: dict[Path, set[Path]] = {}
    failed: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(transcode, source, bitrate) for source in sources]
        for source, future in zip(sources, futures, strict=True):
            try:
                opus = future.result()
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="replace").strip()
                logger.warning("Cannot transcode %s: %s", source, stderr)
                failed.append(source)
                continue
            origins.setdefault(opus, {opus}).add(source)

    buckets: dict[Path, list[Path]] = {}
    for opus in sorted(origins):
        buckets.setdefault(opus.parent, []).append(opus)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tars: list[Path] = []
    for directory, members in buckets.items():
        batches: list[list[Path]] = [[]]
        batch_bytes = 0
        for opus in members:
            size = opus.stat().st_size
            if batches[-1] and batch_bytes + size > tar_size:
                batches.append([])
                batch_bytes = 0
            batches[-1].append(opus)
            batch_bytes += size

        prefix = "_".join(directory.relative_to(root).parts) or "root"
        for index, batch in enumerate(batches):
            tar_path = root / f"{prefix}_{stamp}_{index:03d}.tar"
            if tar_path.exists():
                raise FileExistsError(f"Refusing to overwrite {tar_path}")
            part = tar_path.with_name(f".{tar_path.name}.part")
            try:
                with tarfile.open(part, "w") as tar:
                    for opus in batch:
                        tar.add(opus, arcname=opus.relative_to(root).as_posix())
                part.replace(tar_path)
            finally:
                part.unlink(missing_ok=True)
            for opus in batch:
                for path in origins[opus]:
                    path.unlink()
            logger.info("Packed %s (%d files)", tar_path.name, len(batch))
            tars.append(tar_path)
    return tars, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack fully downloaded audio under ROOT into opus tars. "
        "Run only after all downloads finish; packed sources are deleted."
    )
    parser.add_argument("root", type=Path, nargs="?", default=Path("data"))
    parser.add_argument("--tar-size-gb", type=float, default=DEFAULT_TAR_SIZE_GB)
    parser.add_argument("--bitrate", default=DEFAULT_BITRATE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    tars, failed = pack(
        args.root.resolve(),
        int(args.tar_size_gb * 1024**3),
        args.bitrate,
        args.workers,
    )
    logger.info("Packed %d tars under %s", len(tars), args.root)
    if failed:
        raise SystemExit(f"{len(failed)} sources failed to transcode and were kept")


if __name__ == "__main__":
    main()
