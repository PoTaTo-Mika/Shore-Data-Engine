"""Append-only jsonl cursors and the single-run lock for the tar pipeline."""

import fcntl
import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

logger = logging.getLogger(__name__)

CURSOR_NAME = "_shore_cursors.jsonl"
LOCK_NAME = ".shore.lock"


@dataclass(kw_only=True)
class StageCursor:
    tar_name: str
    stage: str
    member: int
    written: int


def iter_records(path: Path) -> Iterator[dict[str, object]]:
    """Stream valid jsonl lines; truncate a torn tail so appends stay well-formed."""
    if not path.exists():
        return
    with path.open("r+b") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                return
            if not line.endswith(b"\n"):
                logger.warning("Truncating torn tail in %s", path.name)
                handle.truncate(start)
                return
            try:
                record = json.loads(line)
            except ValueError:
                logger.warning("Skipping broken line in %s", path.name)
                continue
            yield cast(dict[str, object], record)


def append_record(handle: TextIO, record: Mapping[str, object]) -> None:
    """Write one durable line; a pause must never leave progress only in memory."""
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def load_cursors(path: Path) -> dict[tuple[str, str], StageCursor]:
    """Replay the append-only log; a done event clears the cursor."""
    current: dict[tuple[str, str], StageCursor] = {}
    for record in iter_records(path):
        tar_name = cast(str, record["tar_name"])
        stage = cast(str, record["stage"])
        key = (tar_name, stage)
        if record.get("done"):
            current.pop(key, None)
            continue
        current[key] = StageCursor(
            tar_name=tar_name,
            stage=stage,
            member=cast(int, record["member"]),
            written=cast(int, record["written"]),
        )
    return current


@contextmanager
def pipeline_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_NAME
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another pipeline holds the lock at {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
