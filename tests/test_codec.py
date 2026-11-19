"""Tests for the codec's decode-once / encode-last chain helpers."""

import unittest
from io import BytesIO

import numpy as np
import soundfile as sf

from data_process.codec import (
    SAMPLE_RATE,
    base_waveform,
    carry_pcm,
    decode_audio,
    ensure_encoded,
    replace_audio,
    slice_record,
)
from data_process.data_reader import AudioRecord


def wav_record(waveform: np.ndarray, audio_id: str = "album/clip.wav") -> AudioRecord:
    buffer = BytesIO()
    sf.write(buffer, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return AudioRecord(tar_name="en.tar", audio_id=audio_id, data=buffer.getvalue())


class ChainCacheTests(unittest.TestCase):
    def test_base_waveform_reuses_cache_without_decoding(self) -> None:
        pcm = np.zeros((1024, 2), dtype=np.float32)
        record = AudioRecord(tar_name="en.tar", audio_id="a.wav", data=b"not audio")
        record.pcm = pcm
        record.pcm_rate = SAMPLE_RATE
        self.assertIs(base_waveform(record), pcm)

    def test_base_waveform_decodes_stereo_when_no_cache(self) -> None:
        tone = np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)
        decoded = base_waveform(wav_record(tone.astype(np.float32)))
        self.assertEqual(decoded.shape[1], 2)
        self.assertAlmostEqual(decoded.shape[0] / SAMPLE_RATE, 1.0, places=2)

    def test_carry_pcm_defers_encoding(self) -> None:
        pcm = np.zeros((32, 2), dtype=np.float32)
        carried = carry_pcm(wav_record(np.zeros(32, np.float32)), pcm)
        self.assertEqual(carried.data, b"")
        self.assertIs(carried.pcm, pcm)
        self.assertEqual(carried.pcm_rate, SAMPLE_RATE)

    def test_ensure_encoded_encodes_terminal_pcm_only(self) -> None:
        tone = np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)
        stereo = np.stack([tone, tone], axis=1).astype(np.float32)
        wav = ensure_encoded(carry_pcm(wav_record(tone.astype(np.float32)), stereo))
        self.assertEqual(wav.audio_id, "album/clip.wav")
        self.assertEqual(wav.data[:4], b"RIFF")
        self.assertIsNone(wav.pcm)
        opus = ensure_encoded(
            carry_pcm(wav_record(tone.astype(np.float32), "album/clip.mp3"), stereo)
        )
        self.assertEqual(opus.audio_id, "album/clip.opus")
        self.assertEqual(opus.data[:4], b"OggS")

    def test_ensure_encoded_passes_through_encoded_members(self) -> None:
        record = wav_record(np.zeros(16, np.float32))
        self.assertIs(ensure_encoded(record), record)

    def test_slice_record_clears_the_pcm_cache(self) -> None:
        tone = np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)
        record = carry_pcm(
            wav_record(tone.astype(np.float32)),
            np.stack([tone, tone], axis=1).astype(np.float32),
        )
        clips = slice_record(record, tone.astype(np.float32), SAMPLE_RATE, [(0, 1024)])
        self.assertEqual(len(clips), 1)
        self.assertIsNone(clips[0].pcm)
        self.assertEqual(clips[0].data[:4], b"OggS")

    def test_replace_audio_clears_the_pcm_cache(self) -> None:
        tone = np.sin(2 * np.pi * 220 * np.arange(1024) / SAMPLE_RATE).astype(
            np.float32
        )
        record = carry_pcm(wav_record(tone), np.stack([tone, tone], axis=1))
        encoded = replace_audio(record, tone, SAMPLE_RATE)
        self.assertIsNone(encoded.pcm)
        self.assertEqual(encoded.data[:4], b"RIFF")

    def test_cached_stereo_round_trips_through_decode(self) -> None:
        tone = np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)
        stereo = decode_audio(wav_record(tone.astype(np.float32)), SAMPLE_RATE, 2)
        record = AudioRecord(tar_name="en.tar", audio_id="a.wav", data=b"")
        record.pcm = stereo
        record.pcm_rate = SAMPLE_RATE
        mono = base_waveform(record).mean(axis=1)
        self.assertEqual(mono.shape[0], stereo.shape[0])


if __name__ == "__main__":
    unittest.main()
