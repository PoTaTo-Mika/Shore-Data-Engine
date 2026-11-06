"""Tests for streaming tar I/O and jsonl annotations."""

import io
import json
import tarfile
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from data_process.annotation import Annotation, append_annotation, load_completed
from data_process.data_reader import (
    AudioRecord,
    completed_stages,
    iter_audio,
    iter_tars,
    transform_tar,
)


def make_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def read_tar(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r") as tar:
        return {
            member.name: tar.extractfile(member).read()
            for member in tar
            if member.isfile()
        }


class TarReaderTests(unittest.TestCase):
    def test_iter_audio_streams_only_audio_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tar_path = root / "en_000.tar"
            make_tar(
                tar_path,
                {
                    "album/a.opus": b"audio",
                    "album/a.txt": b"text",
                    "notes.json": b"{}",
                },
            )
            records = list(iter_audio(tar_path, {".opus"}))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].tar_name, "en_000.tar")
            self.assertEqual(records[0].audio_id, "album/a.opus")
            self.assertEqual(records[0].data, b"audio")
            self.assertEqual(records[0].extension, ".opus")
            self.assertEqual(list(iter_tars(root)), [tar_path])

    def test_transform_tar_fans_out_and_marks_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tar_path = Path(tmp) / "en_000.tar"
            make_tar(tar_path, {"album/a.opus": b"audio", "album/a.txt": b"text"})

            def segment(record: AudioRecord) -> list[AudioRecord]:
                return [
                    replace(
                        record,
                        audio_id=f"album/a_{i:03d}.opus",
                        data=record.data + bytes([i]),
                    )
                    for i in range(2)
                ]

            self.assertEqual(
                transform_tar(tar_path, segment, {".opus"}, "stage2"), 2
            )
            content = read_tar(tar_path)
            self.assertEqual(content["album/a_000.opus"], b"audio\x00")
            self.assertEqual(content["album/a_001.opus"], b"audio\x01")
            self.assertEqual(content["album/a.txt"], b"text")
            self.assertEqual(completed_stages(tar_path), {"stage2"})
            self.assertEqual(
                json.loads(content["_shore.json"]), {"stages": ["stage2"]}
            )
            self.assertFalse(Path(str(tar_path) + ".tmp").exists())

            self.assertEqual(
                transform_tar(tar_path, segment, {".opus"}, "stage2"), 0
            )
            self.assertEqual(set(read_tar(tar_path)), set(content))

    def test_duplicate_member_is_rejected_without_touching_tar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tar_path = Path(tmp) / "en_000.tar"
            original = {"album/a.opus": b"audio"}
            make_tar(tar_path, original)

            def duplicate(record: AudioRecord) -> list[AudioRecord]:
                return [record, record]

            with self.assertRaises(ValueError):
                transform_tar(tar_path, duplicate, {".opus"}, "stage1")
            self.assertEqual(read_tar(tar_path), original)
            self.assertFalse(Path(str(tar_path) + ".tmp").exists())


class AnnotationTests(unittest.TestCase):
    def test_jsonl_schema_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.jsonl"
            self.assertEqual(load_completed(path), set())
            with path.open("a", encoding="utf-8") as handle:
                append_annotation(
                    handle,
                    Annotation(tar_name="en.tar", audio_id="a.opus", asr_text="你好"),
                )
                append_annotation(
                    handle,
                    Annotation(tar_name="en.tar", audio_id="b.opus", asr_text="world"),
                )
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(
                set(json.loads(lines[0])), {"tar_name", "audio_id", "asr_text"}
            )
            self.assertEqual(json.loads(lines[0])["asr_text"], "你好")
            self.assertEqual(
                load_completed(path),
                {("en.tar", "a.opus"), ("en.tar", "b.opus")},
            )
