import os
import time
import tarfile
from pathlib import Path

# ================= 配置区域 =================
WATCH_DIR = "data"             # 监控根目录
TAR_MAX_SIZE_GB = 1            # 每个 tar 最大体积 (GiB)
TAR_MAX_SIZE = int(TAR_MAX_SIZE_GB * 1024 ** 3)
SCAN_INTERVAL = 10             # 扫描间隔（秒）
STALE_SECONDS = 3600           # 文件滞留超过此时间强制打包 (1 小时)
# ===========================================

AUDIO_EXTS = {".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".wav", ".flac"}


def scan_buckets(watch_dir: Path) -> dict[str, list[Path]]:
    """扫描子目录，每个子目录是一个桶，返回桶名→音频文件列表（按 mtime 升序）."""
    buckets: dict[str, list[Path]] = {}
    if not watch_dir.exists():
        return buckets
    for sub in watch_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        files = [f for f in sub.iterdir()
                 if f.is_file() and f.suffix.lower() in AUDIO_EXTS
                 and not f.name.endswith(".part")]
        if files:
            files.sort(key=lambda f: f.stat().st_mtime)
            buckets[sub.name] = files
    return buckets


def pack_batch(watch_dir: Path, bucket_name: str, batch: list[Path], seq: int) -> Path | None:
    """将一批文件打包成 tar，成功返回 path."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    tar_name = f"{bucket_name}_{ts}_{seq:03d}.tar"
    tar_path = watch_dir / tar_name
    tmp_path = watch_dir / f"{tar_name}.tmp"

    try:
        size_mb = sum(f.stat().st_size for f in batch) / 1024 ** 2
        print(f"  [+] 打包: {tar_name} ({len(batch)} 文件, {size_mb:.0f} MiB)")
        with tarfile.open(tmp_path, "w") as tar:
            for f in batch:
                tar.add(str(f), arcname=f"{bucket_name}/{f.name}")
        os.rename(tmp_path, tar_path)
        print(f"  [√] 完成: {tar_path}")
        return tar_path
    except Exception as e:
        print(f"  [!] 打包失败: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None


def split_into_batches(files: list[Path], max_bytes: int,
                       now: float, stale_secs: float) -> tuple[list[list[Path]], list[Path]]:
    """按体积切分文件列表。返回 (待打包批次, 剩余不足1批的文件)."""
    batches: list[list[Path]] = []
    rest: list[Path] = []
    batch: list[Path] = []
    batch_bytes = 0

    for f in files:
        # 检查是否有文件已过期，触发强制打包
        fsize = f.stat().st_size

        # 当前批次塞不下时，输出批次并重置
        if batch and batch_bytes + fsize > max_bytes:
            batches.append(batch)
            batch = []
            batch_bytes = 0

        batch.append(f)
        batch_bytes += fsize

        if batch_bytes >= max_bytes:
            # 除最后一个文件外，检查是否需要将最后文件放回（保持 <max_bytes 原则）
            batches.append(batch)
            batch = []
            batch_bytes = 0

    # 剩余批次：判断是否要等
    if batch:
        oldest_mtime = min(f.stat().st_mtime for f in batch)
        if now - oldest_mtime >= stale_secs:
            batches.append(batch)  # 强制打包
        else:
            rest = batch

    return batches, rest


def start_monitor():
    watch_dir = Path(WATCH_DIR).resolve()
    print(f"[*] 开始监控: {watch_dir} "
          f"(打包阈值: {TAR_MAX_SIZE_GB} GiB/tar, "
          f"超时强制: {STALE_SECONDS}s)")

    seq_counter: dict[str, int] = {}

    while True:
        try:
            buckets = scan_buckets(watch_dir)
            now = time.time()

            for bucket_name, files in buckets.items():
                if not files:
                    continue
                seq = seq_counter.get(bucket_name, 0)

                batches, rest = split_into_batches(files, TAR_MAX_SIZE, now, STALE_SECONDS)

                for batch in batches:
                    tar = pack_batch(watch_dir, bucket_name, batch, seq)
                    if tar:
                        cleanup_files(batch)
                    seq += 1

                if rest:
                    oldest = min(f.stat().st_mtime for f in rest)
                    wait = max(0, STALE_SECONDS - (now - oldest))
                    print(f"  [ ] 桶 [{bucket_name}] 剩余 {len(rest)} 文件不足 {TAR_MAX_SIZE_GB} GiB, "
                          f"还需等待约 {wait/60:.0f} 分钟强制打包")

                seq_counter[bucket_name] = seq

        except Exception as e:
            print(f"[X] 监控循环出错: {e}")

        time.sleep(SCAN_INTERVAL)


def cleanup_files(files: list[Path]) -> int:
    count = 0
    for f in files:
        try:
            os.remove(f)
            count += 1
        except OSError as e:
            print(f"  [!] 删除失败 {f}: {e}")
    return count


if __name__ == "__main__":
    os.makedirs(WATCH_DIR, exist_ok=True)
    start_monitor()
