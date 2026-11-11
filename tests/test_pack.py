"""Tests for packing downloaded audio into opus tars."""

import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crawler import pack


def fake_transcode(source: Path, bitrate: str) -> Path:
    if source.suffix == ".opus":
        return source
    if source.read_bytes() == b"broken":
        raise subprocess.CalledProcessError(1, "ffmpeg", stderr=b"invalid data")
    target = source.with_suffix(".opus")
    target.write_bytes(b"OPUS:" + source.read_bytes())
    return target


def tar_members(path: Path) -> dict[str, bytes]:
    with tarfile.open(path) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar}  # type: ignore[union-attr]


class PackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        patcher = patch.object(pack, "transcode", side_effect=fake_transcode)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def write(self, relative: str, data: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_packs_per_directory_and_deletes_sources(self) -> None:
        self.write("zh/album/a.mp3", b"aaa")
        self.write("zh/album/b.opus", b"bb")
        self.write("en/c.m4a", b"c")
        self.write("en/notes.txt", b"keep")
        tars, failed = pack.pack(self.root, 1 << 20, "128k", 2)

        self.assertEqual(failed, [])
        self.assertEqual([t.name.split("_")[0] for t in tars], ["en", "zh"])
        self.assertEqual(tar_members(tars[0]), {"en/c.opus": b"OPUS:c"})
        self.assertEqual(
            tar_members(tars[1]),
            {"zh/album/a.opus": b"OPUS:aaa", "zh/album/b.opus": b"bb"},
        )
        remaining = sorted(
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*")
            if p.is_file()
        )
        self.assertEqual(remaining, ["en/notes.txt", *[t.name for t in tars]])

    def test_splits_at_tar_size(self) -> None:
        for name in ("a", "b", "c"):
            self.write(f"zh/{name}.opus", b"0123456789")
        tars, _ = pack.pack(self.root, 20, "128k", 1)
        self.assertEqual(
            [list(tar_members(t)) for t in tars],
            [["zh/a.opus", "zh/b.opus"], ["zh/c.opus"]],
        )

    def test_failed_source_is_kept_and_reported(self) -> None:
        bad = self.write("zh/bad.mp3", b"broken")
        self.write("zh/good.mp3", b"ok")
        with self.assertLogs(pack.logger, level="WARNING") as logs:
            tars, failed = pack.pack(self.root, 1 << 20, "128k", 1)
        self.assertEqual(failed, [bad])
        self.assertTrue(bad.exists())
        self.assertIn("invalid data", logs.output[0])
        self.assertEqual(list(tar_members(tars[0])), ["zh/good.opus"])

    def test_hidden_and_staging_paths_are_ignored(self) -> None:
        self.write(".cache/a.mp3", b"a")
        self.write("_tmp/b.mp3", b"b")
        self.write("zh/.c.opus.part", b"c")
        tars, failed = pack.pack(self.root, 1 << 20, "128k", 1)
        self.assertEqual((tars, failed), ([], []))

    def test_tar_failure_keeps_sources(self) -> None:
        source = self.write("zh/a.mp3", b"a")
        with (
            patch.object(pack.tarfile, "open", side_effect=OSError("disk full")),
            self.assertRaises(OSError),
        ):
            pack.pack(self.root, 1 << 20, "128k", 1)
        self.assertTrue(source.exists())
        self.assertEqual(
            list(self.root.glob("*.tar")) + list(self.root.glob(".*.part")), []
        )

    def test_resume_after_interrupted_delete(self) -> None:
        self.write("zh/a.mp3", b"a")
        self.write("zh/a.opus", b"OPUS:a")
        tars, _ = pack.pack(self.root, 1 << 20, "128k", 1)
        self.assertEqual(tar_members(tars[0]), {"zh/a.opus": b"OPUS:a"})
        self.assertFalse((self.root / "zh" / "a.mp3").exists())


class TranscodeTests(unittest.TestCase):
    def test_real_ffmpeg_encodes_opus_and_publishes_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "tone.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=d=0.5",
                    str(source),
                ],
                check=True,
            )
            target = pack.transcode(source, "64k")
            self.assertEqual(target, source.with_suffix(".opus"))
            self.assertEqual(target.read_bytes()[:4], b"OggS")
            self.assertEqual(
                sorted(p.name for p in Path(tmp).iterdir()), ["tone.opus", "tone.wav"]
            )
            with self.assertRaises(subprocess.CalledProcessError):
                pack.transcode(Path(tmp) / "missing.mp3", "64k")
            self.assertEqual(
                sorted(p.name for p in Path(tmp).iterdir()), ["tone.opus", "tone.wav"]
            )
