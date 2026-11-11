"""ffmpeg-backed decoding and encoding shared by the local audio operators."""

import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf

from data_process.data_reader import AudioRecord

SAMPLE_RATE = 44100
OPUS_BITRATE = "128k"
ENCODE_WORKERS = 4


def run_ffmpeg(arguments: list[str], data: bytes | None = None) -> bytes:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *arguments,
    ]
    return subprocess.run(command, input=data, capture_output=True, check=True).stdout


def decode_audio(record: AudioRecord, sample_rate: int, channels: int) -> np.ndarray:
    """Decode a member to float32 PCM shaped (samples, channels)."""
    # Containers such as m4a may need seeking, so ffmpeg reads a file instead of a pipe.
    with tempfile.NamedTemporaryFile(suffix=record.extension) as source:
        source.write(record.data)
        source.flush()
        pcm = run_ffmpeg(
            [
                "-i",
                source.name,
                "-ac",
                str(channels),
                "-ar",
                str(sample_rate),
                "-f",
                "f32le",
                "pipe:1",
            ]
        )
    return np.frombuffer(pcm, dtype=np.float32).reshape(-1, channels)


def encode_opus(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Encode float PCM shaped (samples,) or (samples, channels) as ogg/opus."""
    channels = 1 if waveform.ndim == 1 else waveform.shape[1]
    return run_ffmpeg(
        [
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-b:a",
            OPUS_BITRATE,
            "-f",
            "ogg",
            "pipe:1",
        ],
        np.ascontiguousarray(waveform, dtype=np.float32).tobytes(),
    )


def replace_audio(
    record: AudioRecord, waveform: np.ndarray, sample_rate: int
) -> AudioRecord:
    """Re-encode a member; wav stays 16-bit wav, every other format becomes opus."""
    if record.extension == ".wav":
        buffer = BytesIO()
        sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
        return replace(record, data=buffer.getvalue())
    audio_id = str(Path(record.audio_id).with_suffix(".opus"))
    return replace(record, audio_id=audio_id, data=encode_opus(waveform, sample_rate))


def slice_record(
    record: AudioRecord,
    waveform: np.ndarray,
    sample_rate: int,
    segments: list[tuple[int, int]],
) -> list[AudioRecord]:
    """Encode each (start, end) sample range as a `<stem>_<index>.opus` member."""
    stem = str(Path(record.audio_id).with_suffix(""))
    with ThreadPoolExecutor(max_workers=ENCODE_WORKERS) as pool:
        encoded = pool.map(
            lambda segment: encode_opus(waveform[segment[0] : segment[1]], sample_rate),
            segments,
        )
        return [
            replace(record, audio_id=f"{stem}_{index:03d}.opus", data=data)
            for index, data in enumerate(encoded)
        ]
