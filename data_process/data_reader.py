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

from data_process.state import append_record

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
    cursor_path: Path,
    workers: int = 1,
) -> int:
    """Stream members through operator and rewrite the tar locally in place.

    There is no partial tar: the destination is written to a transient file and
    replaces the source atomically, and it is removed on any interruption.
    Member cursors are appended to jsonl as soon as they commit.
    Returns the number of audio members written.
    """
    assert workers >= 1
    stages = completed_stages(tar_path)
    if stage in stages:
        logger.info("Stage %s already done for %s", stage, tar_path.name)
        return 0

    tmp_path = tar_path.with_name(f".{tar_path.name}.rewrite")
    names: set[str] = set()
    count = 0
    try:
        with cursor_path.open("a", encoding="utf-8") as cursors:
            with (
                tarfile.open(tar_path, "r") as src,
                tarfile.open(tmp_path, "w") as dst,
                ThreadPoolExecutor(max_workers=workers) as pool,
            ):
                members = [member for member in src if member.name != MANIFEST_NAME]
                for offset in range(0, len(members), workers):
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
                    results = iter(
                        pool.map(lambda record: list(operator(record)), audio)
                    )
                    for index, (member, record, data) in enumerate(jobs):
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
                        append_record(
                            cursors,
                            {
                                "tar_name": tar_path.name,
                                "stage": stage,
                                "member": offset + index + 1,
                                "written": count,
                            },
                        )

                stages.add(stage)
                payload = json.dumps({"stages": sorted(stages)}).encode("utf-8")
                info = tarfile.TarInfo(MANIFEST_NAME)
                info.size = len(payload)
                dst.addfile(info, BytesIO(payload))
            tmp_path.replace(tar_path)
            append_record(
                cursors,
                {"tar_name": tar_path.name, "stage": stage, "done": True},
            )
    finally:
        tmp_path.unlink(missing_ok=True)
    return count
