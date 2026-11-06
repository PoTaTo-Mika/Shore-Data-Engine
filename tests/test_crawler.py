"""Tests for pack-time transcoding and fixed-size tar packing."""

import os
import subprocess
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from crawler import pack


def fake_transcode(source: Path, destination: Path) -> None:
    destination.write_bytes(b"OPUS:" + source.read_bytes())


class TranscodeTests(unittest.TestCase):
    def test_raw_becomes_opus_and_original_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            pack, "STABLE_SECONDS", 0
        ), patch.object(
            pack, "transcode_to_opus", side_effect=fake_transcode
        ) as transcode:
            root = Path(tmp)
            raw = root / "en" / "album" / "clip.mp3"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"raw")
            self.assertEqual(pack.transcode_pending(root), 1)
            self.assertEqual((root / "en" / "album" / "clip.opus").read_bytes(), b"OPUS:raw")
            self.assertTrue(raw.exists())
            self.assertTrue(pack.marker_path(raw).exists())
            self.assertEqual(pack.transcode_pending(root), 0)
            transcode.assert_called_once()

    def test_waits_for_two_unchanged_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(pack, "STABLE_SECONDS", 0):
            raw = Path(tmp) / "clip.wav"
            raw.write_bytes(b"a")
            old = 1_000_000.0
            os.utime(raw, (old, old))
            observations: dict[Path, pack.Observation] = {}
            self.assertTrue(pack.is_stable(raw, observations, time.time()))
            raw.write_bytes(b"longer content")
            os.utime(raw, (old, old))
            self.assertFalse(pack.is_stable(raw, observations, time.time()))

    def test_failed_transcode_keeps_raw_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(pack, "STABLE_SECONDS", 0):
            root = Path(tmp)
            raw = root / "clip.mp3"
            raw.write_bytes(b"raw")
            with patch.object(
                pack,
                "transcode_to_opus",
                side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
            ):
                self.assertEqual(pack.transcode_pending(root), 1)
            self.assertTrue(raw.exists())
            self.assertFalse(pack.marker_path(raw).exists())
            self.assertFalse((root / "clip.opus").exists())
            with patch.object(pack, "transcode_to_opus", side_effect=fake_transcode):
                self.assertEqual(pack.transcode_pending(root), 1)
            self.assertEqual((root / "clip.opus").read_bytes(), b"OPUS:raw")

    def test_delete_raw_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            pack, "STABLE_SECONDS", 0
        ), patch.object(pack, "DELETE_RAW", True), patch.object(
            pack, "transcode_to_opus", side_effect=fake_transcode
        ):
            root = Path(tmp)
            raw = root / "clip.mp3"
            raw.write_bytes(b"raw")
            pack.transcode_pending(root)
            self.assertFalse(raw.exists())
            self.assertFalse(pack.marker_path(raw).exists())
            self.assertEqual((root / "clip.opus").read_bytes(), b"OPUS:raw")


class PackTests(unittest.TestCase):
    def test_batch_size_is_measured_on_transcoded_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            pack, "STABLE_SECONDS", 0
        ), patch.object(pack, "TAR_MAX_SIZE", 8), patch.object(
            pack, "transcode_to_opus", side_effect=fake_transcode
        ):
            root = Path(tmp)
            bucket = root / "en" / "album"
            bucket.mkdir(parents=True)
            raw = bucket / "a.mp3"
            raw.write_bytes(b"raw")
            tars = pack.pack_once(root)
            self.assertEqual(len(tars), 1)
            with tarfile.open(tars[0]) as tar:
                members = tar.getmembers()
            self.assertEqual([m.name for m in members], ["en/album/a.opus"])
            self.assertEqual([m.size for m in members], [8])
            self.assertFalse((bucket / "a.opus").exists())
            self.assertTrue(raw.exists())

    def test_pack_once_splits_bucket_at_fixed_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bucket = root / "en" / "album"
            bucket.mkdir(parents=True)
            for name in ("a.opus", "b.opus", "c.opus"):
                (bucket / name).write_bytes(b"0123456789")
            with patch.object(pack, "TAR_MAX_SIZE", 10):
                tars = pack.pack_once(root)
            self.assertEqual(len(tars), 3)
            self.assertEqual(list(bucket.iterdir()), [])
            with tarfile.open(tars[0]) as tar:
                self.assertEqual(
                    [member.name for member in tar.getmembers()],
                    ["en/album/a.opus"],
                )

    def test_stale_files_are_forced_into_a_tar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bucket = root / "en"
            bucket.mkdir()
            clip = bucket / "clip.opus"
            clip.write_bytes(b"0123456789")
            os.utime(clip, (0, 0))
            with patch.object(pack, "TAR_MAX_SIZE", 1000), patch.object(
                pack, "STALE_SECONDS", 1
            ):
                tars = pack.pack_once(root)
            self.assertEqual(len(tars), 1)
            self.assertFalse(clip.exists())
