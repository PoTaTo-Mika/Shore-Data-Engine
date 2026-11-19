"""Append-only jsonl records, stage cursors, and the single-run lock."""

import fcntl
import json
import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO, cast

logger = logging.getLogger(__name__)

CURSOR_NAME = "_shore_cursors.jsonl"
LOCK_NAME = ".shore.lock"
TAIL_CHUNK_BYTES = 1 << 16


@dataclass(kw_only=True)
class StageCursor:
    tar_name: str
    stage: str
    member: int
    written: int


def iter_records(path: Path) -> Iterator[dict[str, object]]:
    """Stream complete jsonl lines read-only; a torn tail may be a concurrent append."""
    if not path.exists():
        return
    with path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                return
            try:
                record = json.loads(line)
            except ValueError:
                logger.warning("Skipping broken line in %s", path.name)
                continue
            yield cast(dict[str, object], record)


def truncate_torn_tail(path: Path) -> None:
    """Drop a partial last line left by a hard kill; call only under `pipeline_lock`."""
    if not path.exists():
        return
    with path.open("r+b") as handle:
        end = handle.seek(0, os.SEEK_END)
        position = end
        keep = 0
        while position > 0:
            step = min(TAIL_CHUNK_BYTES, position)
            position -= step
            handle.seek(position)
            index = handle.read(step).rfind(b"\n")
            if index >= 0:
                keep = position + index + 1
                break
        if keep < end:
            logger.warning("Truncating torn tail in %s", path.name)
            handle.truncate(keep)


def append_record(handle: TextIO, record: Mapping[str, object]) -> None:
    """Write one durable line; a pause must never leave progress only in memory."""
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


@dataclass(kw_only=True)
class Annotation:
    tar_name: str
    audio_id: str
    asr_text: str


def load_completed(path: Path) -> set[tuple[str, str]]:
    """Stream (tar name, audio id) pairs already written, for resuming a run."""
    return {
        (cast(str, record["tar_name"]), cast(str, record["audio_id"]))
        for record in iter_records(path)
    }


def append_annotation(handle: TextIO, annotation: Annotation) -> None:
    append_record(handle, asdict(annotation))


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
