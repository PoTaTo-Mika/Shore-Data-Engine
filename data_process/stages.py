"""Byte-level audio operators applied to tar members by the engine stages."""

import importlib
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np

from data_process.data_reader import AudioRecord, Operator

OPUS_BITRATE = "128k"
DEFAULT_UVR_MODEL = "htdemucs"
DEFAULT_SLICER_THRESHOLD = -30.0
DEFAULT_MIN_LENGTH_MS = 10000
SLICER_SAMPLE_RATE = 44100
SLICER_MIN_INTERVAL_MS = 500
SLICER_HOP_SIZE_MS = 10
SLICER_MAX_SIL_KEPT_MS = 500
SLICER_ENCODE_WORKERS = 4


def run_ffmpeg(data: bytes, arguments: list[str]) -> bytes:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *arguments,
    ]
    result = subprocess.run(command, input=data, capture_output=True, check=True)
    return result.stdout


def build_uvr(params: dict[str, object]) -> Operator:
    """Stage 1: replace each member with its isolated vocals (Demucs)."""
    # Heavy optional deps are loaded only when this stage is configured.
    module = importlib.import_module("data_process.uvr.run_demucs")
    torch = importlib.import_module("torch")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    separator = module.demucs.api.Separator(
        model=cast(str, params.get("model", DEFAULT_UVR_MODEL)), device=device
    )
    # One GPU model is shared by all workers; CPU-side ffmpeg runs concurrently.
    # CUDA streams are per-thread, so inference must finish before the lock is released.
    gpu_lock = threading.Lock()
    use_cuda = torch.cuda.is_available()

    def separate(record: AudioRecord) -> list[AudioRecord]:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / Path(record.audio_id).name
            source.write_bytes(record.data)
            waveform, samplerate, format_type = module.load_audio(str(source))
            with gpu_lock:
                _, separated = separator.separate_tensor(waveform, samplerate)
                vocals = separated["vocals"].detach().cpu()
                if use_cuda:
                    torch.cuda.synchronize()
            output = Path(module.get_output_path(str(source), format_type))
            module.save_audio(vocals, samplerate, str(output), format_type)
            audio_id = str(Path(record.audio_id).with_suffix(output.suffix))
            return [replace(record, audio_id=audio_id, data=output.read_bytes())]

    return separate


def build_slicer(params: dict[str, object]) -> Operator:
    """Stage 2: split each member into VAD segments, one tar member per segment."""
    # Heavy optional deps are loaded only when this stage is configured.
    sf = importlib.import_module("soundfile")
    slicer_cls = importlib.import_module("data_process.slicer.rms_vad").Slicer
    threshold = cast(float, params.get("threshold", DEFAULT_SLICER_THRESHOLD))
    min_length = cast(int, params.get("min_length", DEFAULT_MIN_LENGTH_MS))

    def slice_audio(record: AudioRecord) -> list[AudioRecord]:
        pcm = run_ffmpeg(
            record.data,
            [
                "-i", "pipe:0", "-ac", "1", "-ar", str(SLICER_SAMPLE_RATE),
                "-f", "wav", "pipe:1",
            ],
        )
        waveform, samplerate = sf.read(BytesIO(pcm), dtype="float32")
        slicer = slicer_cls(
            sr=samplerate,
            threshold=threshold,
            min_length=min_length,
            min_interval=SLICER_MIN_INTERVAL_MS,
            hop_size=SLICER_HOP_SIZE_MS,
            max_sil_kept=SLICER_MAX_SIL_KEPT_MS,
        )
        chunks = slicer.slice(np.asarray(waveform))
        if not chunks:
            return []
        stem = str(Path(record.audio_id).with_suffix(""))

        def encode_chunk(chunk: np.ndarray, index: int) -> AudioRecord:
            buffer = BytesIO()
            sf.write(buffer, chunk, samplerate, format="WAV", subtype="PCM_16")
            opus = run_ffmpeg(
                buffer.getvalue(),
                [
                    "-i", "pipe:0", "-c:a", "libopus", "-b:a", OPUS_BITRATE,
                    "-f", "ogg", "pipe:1",
                ],
            )
            return replace(
                record, audio_id=f"{stem}_{index:03d}.opus", data=opus
            )

        with ThreadPoolExecutor(
            max_workers=min(SLICER_ENCODE_WORKERS, len(chunks))
        ) as pool:
            return list(pool.map(encode_chunk, chunks, range(len(chunks))))

    return slice_audio
