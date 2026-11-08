"""Streaming jsonl annotations for audio members inside tars."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Annotation:
    tar_name: str
    audio_id: str
    asr_text: str


def load_completed(path: Path) -> set[tuple[str, str]]:
    """Read (tar name, audio id) pairs already written; quarantine a torn tail."""
    if not path.exists():
        return set()
    completed: set[tuple[str, str]] = set()
    valid: list[bytes] = []
    broken: list[bytes] = []
    for line in path.read_bytes().splitlines(keepends=True):
        if broken:
            broken.append(line)
            continue
        if not line.strip():
            valid.append(line)
            continue
        try:
            record = json.loads(line)
            key = (record["tar_name"], record["audio_id"])
        except (ValueError, KeyError, TypeError):
            broken.append(line)
            continue
        valid.append(line)
        completed.add(key)
    if broken:
        path.write_bytes(b"".join(valid))
        with Path(str(path) + ".bad").open("ab") as handle:
            handle.write(b"".join(broken))
        logger.warning("Quarantined %d broken lines from %s", len(broken), path.name)
    return completed


def append_annotation(handle: TextIO, annotation: Annotation) -> None:
    handle.write(json.dumps(asdict(annotation), ensure_ascii=False) + "\n")
    handle.flush()
