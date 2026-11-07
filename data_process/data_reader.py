"""Streaming read/write access to audio members inside tar datasets."""

import json
import logging
import tarfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import cast

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
    tar_path: Path, operator: Operator, extensions: set[str], stage: str
) -> int:
    """Stream every audio member through operator and replace the tar atomically.

    A completed stage is skipped on rerun; the manifest inside the tar records it.
    Returns the number of audio members written.
    """
    stages = completed_stages(tar_path)
    if stage in stages:
        logger.info("Stage %s already done for %s", stage, tar_path.name)
        return 0

    tmp_path = tar_path.with_name(tar_path.name + ".tmp")
    names: set[str] = set()
    count = 0
    try:
        with tarfile.open(tar_path, "r") as src, tarfile.open(tmp_path, "w") as dst:
            for member in src:
                if member.name == MANIFEST_NAME:
                    continue
                handle = src.extractfile(member) if member.isfile() else None
                if not member.isfile():
                    dst.addfile(member)
                    continue
                assert handle is not None
                if Path(member.name).suffix.lower() not in extensions:
                    dst.addfile(member, handle)
                    continue

                record = AudioRecord(
                    tar_name=tar_path.name, audio_id=member.name, data=handle.read()
                )
                for output in operator(record):
                    if output.audio_id in names:
                        raise ValueError(f"Duplicate tar member: {output.audio_id}")
                    names.add(output.audio_id)
                    info = tarfile.TarInfo(output.audio_id)
                    info.size = len(output.data)
                    info.mtime = member.mtime
                    info.mode = member.mode
                    dst.addfile(info, BytesIO(output.data))
                    count += 1

            stages.add(stage)
            payload = json.dumps({"stages": sorted(stages)}).encode("utf-8")
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size = len(payload)
            dst.addfile(info, BytesIO(payload))

        tmp_path.replace(tar_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return count
