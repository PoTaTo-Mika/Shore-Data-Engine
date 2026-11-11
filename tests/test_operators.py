"""Tests for the local audio operators and their shared codec."""

import unittest
from io import BytesIO

import librosa
import numpy as np
import soundfile as sf

from data_process.codec import (
    SAMPLE_RATE,
    decode_audio,
    encode_opus,
    replace_audio,
    slice_record,
)
from data_process.data_reader import AudioRecord
from data_process.vad.rms import Slicer, get_rms
from data_process.vad.silero import plan_segments
from data_process.vad.ten import find_speech


def tone(seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def wav_record(waveform: np.ndarray, audio_id: str = "album/clip.wav") -> AudioRecord:
    buffer = BytesIO()
    sf.write(buffer, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return AudioRecord(tar_name="en.tar", audio_id=audio_id, data=buffer.getvalue())


class CodecTests(unittest.TestCase):
    def test_decode_resamples_and_mixes_channels(self) -> None:
        stereo = np.stack([tone(1.0), tone(1.0)], axis=1)
        decoded = decode_audio(wav_record(stereo), 16000, 1)
        self.assertEqual(decoded.shape[1], 1)
        self.assertAlmostEqual(decoded.shape[0] / 16000, 1.0, places=2)

    def test_replace_audio_keeps_wav_and_converts_others(self) -> None:
        waveform = tone(0.5)
        wav = replace_audio(wav_record(waveform), waveform, SAMPLE_RATE)
        self.assertEqual(wav.audio_id, "album/clip.wav")
        self.assertEqual(wav.data[:4], b"RIFF")
        other = replace_audio(
            wav_record(waveform, "album/clip.m4a"), waveform, SAMPLE_RATE
        )
        self.assertEqual(other.audio_id, "album/clip.opus")
        self.assertEqual(other.data[:4], b"OggS")

    def test_slice_record_names_segments_in_order(self) -> None:
        waveform = tone(2.0)
        records = slice_record(
            wav_record(waveform),
            waveform,
            SAMPLE_RATE,
            [(0, SAMPLE_RATE), (SAMPLE_RATE, 2 * SAMPLE_RATE)],
        )
        self.assertEqual(
            [r.audio_id for r in records],
            ["album/clip_000.opus", "album/clip_001.opus"],
        )
        self.assertTrue(all(r.data[:4] == b"OggS" for r in records))
        self.assertEqual(
            slice_record(wav_record(waveform), waveform, SAMPLE_RATE, []), []
        )

    def test_encode_opus_accepts_stereo(self) -> None:
        self.assertEqual(
            encode_opus(np.stack([tone(0.2)] * 2, axis=1), SAMPLE_RATE)[:4], b"OggS"
        )


class RmsSlicerTests(unittest.TestCase):
    def test_get_rms_matches_librosa(self) -> None:
        y = np.random.default_rng(0).standard_normal(10_000).astype(np.float32)
        expected = librosa.feature.rms(
            y=y, frame_length=2048, hop_length=512, center=True, pad_mode="constant"
        )[0]
        np.testing.assert_allclose(get_rms(y, 2048, 512), expected, rtol=1e-5)

    def test_splits_at_long_silence(self) -> None:
        waveform = np.concatenate(
            [tone(3.0), np.zeros(SAMPLE_RATE, np.float32), tone(3.0)]
        )
        slicer = Slicer(
            sr=SAMPLE_RATE,
            threshold=-30,
            min_length=2000,
            min_interval=300,
            hop_size=10,
            max_sil_kept=100,
        )
        ranges = slicer.slice(waveform)
        self.assertEqual(len(ranges), 2)
        self.assertLess(ranges[0][1], ranges[1][0])
        self.assertEqual(ranges[-1][1], len(waveform))

    def test_short_audio_is_kept_whole(self) -> None:
        slicer = Slicer(sr=SAMPLE_RATE)
        self.assertEqual(slicer.slice(tone(1.0)), [(0, SAMPLE_RATE)])


class SileroPlanTests(unittest.TestCase):
    def test_merges_splits_and_drops(self) -> None:
        plan = plan_segments(
            [(0.0, 4.0), (5.0, 9.0), (20.0, 21.0), (30.0, 75.0)],
            target_s=15.0,
            max_s=20.0,
            min_s=1.5,
            strict_gap_s=0.2,
            loose_gap_s=1.5,
        )
        self.assertEqual(plan[0], (0.0, 9.0))
        self.assertEqual(plan[1:], [(30.0, 45.0), (45.0, 60.0), (60.0, 75.0)])

    def test_long_span_only_bridges_strict_gaps(self) -> None:
        plan = plan_segments(
            [(0.0, 16.0), (17.0, 18.0), (18.1, 19.0)],
            target_s=15.0,
            max_s=20.0,
            min_s=0.5,
            strict_gap_s=0.2,
            loose_gap_s=1.5,
        )
        self.assertEqual(plan, [(0.0, 16.0), (17.0, 19.0)])


class TenSpanTests(unittest.TestCase):
    def test_groups_voiced_frames_with_hangover_and_padding(self) -> None:
        voiced = (
            [False] * 3
            + [True] * 5
            + [False]
            + [True] * 2
            + [False] * 4
            + [True]
            + [False] * 3
        )
        spans = find_speech(voiced, min_speech=3, min_silence=2, pad=1)
        self.assertEqual(spans, [(2, 12)])

    def test_open_span_runs_to_the_end(self) -> None:
        self.assertEqual(
            find_speech([False, True, True, True], min_speech=2, min_silence=2, pad=0),
            [(1, 4)],
        )
