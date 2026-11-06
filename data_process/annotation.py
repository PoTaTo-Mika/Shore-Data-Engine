"""Streaming jsonl annotations for audio members inside tars."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO


@dataclass(kw_only=True)
class Annotation:
    tar_name: str
    audio_id: str
    asr_text: str


def load_completed(path: Path) -> set[tuple[str, str]]:
    """Read (tar name, audio id) pairs already written, for resuming a run."""
    if not path.exists():
        return set()
    completed: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                completed.add((record["tar_name"], record["audio_id"]))
    return completed


def append_annotation(handle: TextIO, annotation: Annotation) -> None:
    handle.write(json.dumps(asdict(annotation), ensure_ascii=False) + "\n")
    handle.flush()
