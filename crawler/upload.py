"""Upload packed tars to object storage and remove the local copies."""

import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

if TYPE_CHECKING:
    from botocore.client import BaseClient

logger = logging.getLogger(__name__)

WATCH_DIR = Path(os.environ.get("SHORE_DATA_DIR", "data")).resolve()
SCAN_INTERVAL = float(os.environ.get("SHORE_SCAN_INTERVAL", "10"))
MAX_WORKERS = int(os.environ.get("SHORE_UPLOAD_WORKERS", "4"))

S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_KEY = os.environ.get("S3_KEY")
S3_SECRET = os.environ.get("S3_SECRET")
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "")


def build_s3_client() -> "BaseClient":
    for name, value in (
        ("S3_ENDPOINT", S3_ENDPOINT),
        ("S3_KEY", S3_KEY),
        ("S3_SECRET", S3_SECRET),
        ("S3_BUCKET", S3_BUCKET),
    ):
        if not value:
            raise ValueError(f"Missing environment variable: {name}")
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


def upload_tar(s3: "BaseClient", tar_path: Path) -> bool:
    lang = tar_path.name.split("_")[0]
    key = f"{S3_PREFIX}{lang}/{tar_path.name}"
    logger.info("Uploading %s to s3://%s/%s", tar_path.name, S3_BUCKET, key)
    try:
        s3.upload_file(str(tar_path), S3_BUCKET, key, ExtraArgs={"ACL": "private"})
        return True
    except (BotoCoreError, ClientError, OSError):
        logger.exception("Upload failed: %s", tar_path.name)
        return False


def scan_tars(watch_dir: Path) -> list[Path]:
    if not watch_dir.exists():
        return []
    return sorted(f for f in watch_dir.iterdir() if f.is_file() and f.suffix == ".tar")


def start_monitor() -> None:
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Watching %s for tars", WATCH_DIR)
    s3 = build_s3_client()
    processing: set[str] = set()
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    future_to_name: dict[Future[bool], str] = {}
    while True:
        try:
            for future in [f for f in future_to_name if f.done()]:
                name = future_to_name.pop(future)
                processing.discard(name)
                try:
                    if future.result():
                        tar_path = WATCH_DIR / name
                        if tar_path.exists():
                            tar_path.unlink()
                            logger.info("Uploaded and removed %s", name)
                except Exception:
                    logger.exception("Upload task failed: %s", name)

            for tar_path in scan_tars(WATCH_DIR):
                if tar_path.name in processing:
                    continue
                processing.add(tar_path.name)
                future = executor.submit(upload_tar, s3, tar_path)
                future_to_name[future] = tar_path.name
        except Exception:
            logger.exception("Monitor iteration failed; retrying")
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_monitor()
