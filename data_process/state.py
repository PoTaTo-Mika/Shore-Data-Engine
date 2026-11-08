"""On-disk cursors and the single-run lock for the tar pipeline."""

import fcntl
import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_NAME = "_shore_state.json"
LOCK_NAME = ".shore.lock"


@dataclass(kw_only=True)
class StageCursor:
    member: int = 0
    partial_bytes: int = 0
    written: int = 0


@dataclass(kw_only=True)
class PipelineState:
    root: Path
    stage_cursors: dict[str, dict[str, StageCursor]] = field(default_factory=dict)
    asr_tars: list[str] = field(default_factory=list)
    asr_last: tuple[str, str] | None = None

    def cursor(self, tar_name: str, stage: str) -> StageCursor | None:
        return self.stage_cursors.get(tar_name, {}).get(stage)

    def set_cursor(self, tar_name: str, stage: str, cursor: StageCursor) -> None:
        self.stage_cursors.setdefault(tar_name, {})[stage] = cursor
        self.save()

    def clear_cursor(self, tar_name: str, stage: str) -> None:
        entries = self.stage_cursors.get(tar_name)
        if entries is None:
            return
        entries.pop(stage, None)
        if not entries:
            del self.stage_cursors[tar_name]
        self.save()

    def save(self) -> None:
        payload = {
            "version": 1,
            "stage_cursors": {
                tar: {
                    stage: {
                        "member": cursor.member,
                        "partial_bytes": cursor.partial_bytes,
                        "written": cursor.written,
                    }
                    for stage, cursor in entries.items()
                }
                for tar, entries in self.stage_cursors.items()
            },
            "asr_tars": self.asr_tars,
            "asr_last": list(self.asr_last) if self.asr_last else None,
        }
        path = self.root / STATE_NAME
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)


def load_state(root: Path) -> PipelineState:
    path = root / STATE_NAME
    if not path.exists():
        return PipelineState(root=root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cursors = {
            tar: {
                stage: StageCursor(**entry) for stage, entry in entries.items()
            }
            for tar, entries in payload["stage_cursors"].items()
        }
        last = payload.get("asr_last")
        return PipelineState(
            root=root,
            stage_cursors=cursors,
            asr_tars=list(payload.get("asr_tars", [])),
            asr_last=(last[0], last[1]) if last else None,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Cannot read pipeline state %s: %s", path, exc)
        return PipelineState(root=root)


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
