"""MDX-Net (ONNX) vocal isolation operator."""

import threading
from pathlib import Path
from typing import Literal, cast, get_args

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download

from data_process.codec import SAMPLE_RATE, decode_audio, replace_audio
from data_process.data_reader import AudioRecord, Operator

Stem = Literal["vocals", "instrumental"]

HOP_LENGTH = 1024
DEFAULT_N_FFT = 7680
DEFAULT_DIM_F = 3072
DEFAULT_DIM_T = 256
DEFAULT_BATCH_SIZE = 4


def build_operator(params: dict[str, object]) -> Operator:
    """Replace each member with vocals separated by an MDX-Net spectrogram model."""
    model_path = Path(cast(str, params["model_path"]))
    if not model_path.exists():
        hf_hub_download(
            repo_id=cast(str, params["repo_id"]),
            filename=model_path.name,
            local_dir=model_path.parent,
        )
    stem = cast(Stem, params.get("stem", "vocals"))
    if stem not in get_args(Stem):
        raise ValueError(f"Unknown MDX-Net stem: {stem}")
    n_fft = cast(int, params.get("n_fft", DEFAULT_N_FFT))
    dim_f = cast(int, params.get("dim_f", DEFAULT_DIM_F))
    dim_t = cast(int, params.get("dim_t", DEFAULT_DIM_T))
    compensate = cast(float, params["compensate"])
    batch_size = cast(int, params.get("batch_size", DEFAULT_BATCH_SIZE))

    available = ort.get_available_providers()
    providers = [
        p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available
    ]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name

    n_bins = n_fft // 2 + 1
    chunk_size = HOP_LENGTH * (dim_t - 1)
    trim = n_fft // 2
    step = chunk_size - 2 * trim
    window = torch.hann_window(n_fft)
    # One session serves all stage workers; only inference holds the lock.
    lock = threading.Lock()

    def separate(record: AudioRecord) -> list[AudioRecord]:
        mix = decode_audio(record, SAMPLE_RATE, 2).T
        length = mix.shape[1]
        pad = step - length % step
        padded = np.pad(mix, ((0, 0), (trim, pad + trim)))
        chunks = np.stack(
            [padded[:, i : i + chunk_size] for i in range(0, length + pad, step)]
        )

        # Spectrogram channels are (left real, left imag, right real, right imag).
        stft = torch.stft(
            torch.from_numpy(chunks).reshape(-1, chunk_size),
            n_fft,
            HOP_LENGTH,
            window=window,
            return_complex=True,
        )
        spec = (
            torch.view_as_real(stft)
            .permute(0, 3, 1, 2)
            .reshape(-1, 4, n_bins, dim_t)[:, :, :dim_f]
            .numpy()
        )
        with lock:
            pred = np.concatenate(
                [
                    session.run(None, {input_name: spec[i : i + batch_size]})[0]
                    for i in range(0, len(spec), batch_size)
                ]
            )

        pred = np.pad(pred, ((0, 0), (0, 0), (0, n_bins - dim_f), (0, 0)))
        bins = torch.from_numpy(pred).reshape(-1, 2, n_bins, dim_t).permute(0, 2, 3, 1)
        waves = torch.istft(
            torch.view_as_complex(bins.contiguous()), n_fft, HOP_LENGTH, window=window
        ).reshape(-1, 2, chunk_size)
        joined = waves[:, :, trim:-trim].permute(1, 0, 2).reshape(2, -1)[:, :length]
        source = joined.numpy() * compensate
        vocals = source if stem == "vocals" else mix - source
        return [replace_audio(record, vocals.T, SAMPLE_RATE)]

    return separate
