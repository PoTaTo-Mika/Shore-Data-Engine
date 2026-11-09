"""Streaming jsonl annotations for audio members inside tars."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO, cast

from data_process.state import append_record, iter_records


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
