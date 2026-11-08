"""Streaming read/write access to audio members inside tar datasets."""

import json
import logging
import tarfile
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import cast

from data_process.state import PipelineState, StageCursor

logger = logging.getLogger(__name__)

MANIFEST_NAME = "_shore.json"


@dataclass(kw_only=True)
class AudioRecord:
    tar_name: str
    audio_id: str
    data: bytes

    @property
    def extension(self) -> str:
        return Path(self.audio_id).suffix.lower()


Operator = Callable[[AudioRecord], Iterable[AudioRecord]]


def iter_tars(root: Path) -> Iterator[Path]:
    return iter(sorted(p for p in root.rglob("*.tar") if p.is_file()))


def completed_stages(tar_path: Path) -> set[str]:
    with tarfile.open(tar_path, "r") as tar:
        try:
            member = tar.getmember(MANIFEST_NAME)
        except KeyError:
            return set()
        handle = tar.extractfile(member)
        assert handle is not None
        manifest = cast(dict[str, object], json.load(handle))
    return set(cast(list[str], manifest["stages"]))


def iter_audio(tar_path: Path, extensions: set[str]) -> Iterator[AudioRecord]:
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile() or Path(member.name).suffix.lower() not in extensions:
                continue
            handle = tar.extractfile(member)
            assert handle is not None
            yield AudioRecord(
                tar_name=tar_path.name, audio_id=member.name, data=handle.read()
            )


def transform_tar(
    tar_path: Path,
    operator: Operator,
    extensions: set[str],
    stage: str,
    workers: int = 1,
    state: PipelineState | None = None,
) -> int:
    """Apply operator to audio members in parallel batches and replace the tar atomically.

    Output goes to a stage-specific partial tar. With ``state``, each committed member
    advances a cursor so a rerun resumes from that member instead of redoing the tar.
    The manifest inside the tar remains the completion mark.
    Returns the number of audio members written.
    """
    assert workers >= 1
    stages = completed_stages(tar_path)
    if stage in stages:
        logger.info("Stage %s already done for %s", stage, tar_path.name)
        if state is not None:
            state.clear_cursor(tar_path.name, stage)
        return 0

    partial = tar_path.with_name(f"{tar_path.name}.{stage}.partial")
    saved = state.cursor(tar_path.name, stage) if state is not None else None
    resume = saved is not None and partial.exists() and saved.member > 0
    if resume:
        assert saved is not None
        cursor = saved
        if partial.stat().st_size != cursor.partial_bytes:
            with partial.open("r+b") as handle:
                handle.truncate(cursor.partial_bytes)
        logger.info(
            "Resuming stage %s for %s at member %d",
            stage,
            tar_path.name,
            cursor.member,
        )
        with tarfile.open(partial, "r") as done:
            names = {
                member.name
                for member in done
                if member.isfile() and member.name != MANIFEST_NAME
            }
    else:
        partial.unlink(missing_ok=True)
        names = set()
        cursor = StageCursor()
        if state is not None:
            state.clear_cursor(tar_path.name, stage)

    count = cursor.written
    try:
        with tarfile.open(tar_path, "r") as src, ThreadPoolExecutor(
            max_workers=workers
        ) as pool:
            members = [member for member in src if member.name != MANIFEST_NAME]
            for offset in range(cursor.member, len(members), workers):
                batch = members[offset : offset + workers]
                jobs: list[
                    tuple[tarfile.TarInfo, AudioRecord | None, bytes | None]
                ] = []
                for member in batch:
                    if not member.isfile():
                        jobs.append((member, None, None))
                        continue
                    handle = src.extractfile(member)
                    assert handle is not None
                    data = handle.read()
                    if Path(member.name).suffix.lower() in extensions:
                        jobs.append(
                            (
                                member,
                                AudioRecord(
                                    tar_name=tar_path.name,
                                    audio_id=member.name,
                                    data=data,
                                ),
                                None,
                            )
                        )
                    else:
                        jobs.append((member, None, data))

                audio = [record for _, record, _ in jobs if record is not None]
                results = iter(pool.map(lambda record: list(operator(record)), audio))
                for member, record, data in jobs:
                    mode = "a" if partial.exists() else "w"
                    with tarfile.open(partial, mode) as dst:
                        if record is None:
                            dst.addfile(
                                member, BytesIO(data) if data is not None else None
                            )
                        else:
                            for output in next(results):
                                if output.audio_id in names:
                                    raise ValueError(
                                        f"Duplicate tar member: {output.audio_id}"
                                    )
                                names.add(output.audio_id)
                                info = tarfile.TarInfo(output.audio_id)
                                info.size = len(output.data)
                                info.mtime = member.mtime
                                info.mode = member.mode
                                dst.addfile(info, BytesIO(output.data))
                                count += 1
                    cursor.member += 1
                    cursor.partial_bytes = partial.stat().st_size
                    cursor.written = count
                    if state is not None:
                        state.set_cursor(tar_path.name, stage, cursor)

            with tarfile.open(partial, "r") as done:
                has_manifest = MANIFEST_NAME in done.getnames()
            if not has_manifest:
                stages.add(stage)
                payload = json.dumps({"stages": sorted(stages)}).encode("utf-8")
                with tarfile.open(partial, "a") as dst:
                    info = tarfile.TarInfo(MANIFEST_NAME)
                    info.size = len(payload)
                    dst.addfile(info, BytesIO(payload))

        partial.replace(tar_path)
        if state is not None:
            state.clear_cursor(tar_path.name, stage)
    except Exception:
        if state is None:
            partial.unlink(missing_ok=True)
        else:
            logger.warning(
                "Stage %s interrupted for %s; progress kept at member %d",
                stage,
                tar_path.name,
                cursor.member,
            )
        raise
    return count
