import os
import time
import boto3
from botocore.config import Config
from pathlib import Path
import concurrent.futures

# ================= 配置区域 =================
# 自己看底下内容配置，重要信息已脱敏

WATCH_DIR = "data"                # 监控 .tar 的目录
SCAN_INTERVAL = 10                # 扫描间隔（秒）
MAX_WORKERS = 4                   # 并发上传数
# ===========================================


def build_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET,
        config=Config(
            connect_timeout=60,
            read_timeout=60,
            retries={"max_attempts": 5},
        ),
    )


def upload_tar(s3, tar_path: Path) -> bool:
    """上传单个 tar 到 S3，成功返回 True."""
    # 从文件名提取语言桶: en_20260729_165520_000.tar → en
    lang = tar_path.name.split("_")[0]
    s3_key = f"{S3_PREFIX}{lang}/{tar_path.name}"
    size_mb = tar_path.stat().st_size / 1024 ** 2
    try:
        print(f"  [↑] 上传: {tar_path.name} ({size_mb:.0f} MiB) -> s3://{S3_BUCKET}/{s3_key}")
        s3.upload_file(str(tar_path), S3_BUCKET, s3_key, ExtraArgs={"ACL": "private"})
        return True
    except Exception as e:
        print(f"  [!] 上传失败 {tar_path.name}: {e}")
        return False


def scan_tars(watch_dir: Path) -> list[Path]:
    """扫描目录下所有 .tar 文件（跳过 .tmp）."""
    if not watch_dir.exists():
        return []
    return sorted(
        f for f in watch_dir.iterdir()
        if f.is_file() and f.suffix == ".tar"
    )


def start_monitor():
    watch_dir = Path(WATCH_DIR).resolve()
    print(f"[*] 开始监控 .tar: {watch_dir} -> s3://{S3_BUCKET}/{S3_PREFIX}")

    s3 = build_s3_client()
    processing: set[str] = set()  # 正在处理的文件名
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
    future_to_name: dict[concurrent.futures.Future, str] = {}

    while True:
        try:
            # 清理已完成的任务
            done = [f for f in future_to_name if f.done()]
            for f in done:
                name = future_to_name.pop(f)
                processing.discard(name)
                try:
                    if f.result():
                        tar_path = watch_dir / name
                        if tar_path.exists():
                            tar_path.unlink()
                            print(f"  [√] 已上传并删除: {name}")
                except Exception as e:
                    print(f"  [!] 任务异常 {name}: {e}")

            # 扫描新 tar
            for tar_path in scan_tars(watch_dir):
                if tar_path.name in processing:
                    continue
                processing.add(tar_path.name)
                future = executor.submit(upload_tar, s3, tar_path)
                future_to_name[future] = tar_path.name

        except Exception as e:
            print(f"[X] 监控循环出错: {e}")

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    os.makedirs(WATCH_DIR, exist_ok=True)
    start_monitor()
