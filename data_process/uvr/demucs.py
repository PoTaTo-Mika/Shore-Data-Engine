"""Demucs vocal isolation operator."""

import threading
from typing import cast

import demucs.api
import torch

from data_process.codec import decode_audio, replace_audio
from data_process.data_reader import AudioRecord, Operator

DEFAULT_MODEL = "htdemucs"


def build_operator(params: dict[str, object]) -> Operator:
    """Replace each member with its isolated vocals."""
    separator = demucs.api.Separator(
        model=cast(str, params.get("model", DEFAULT_MODEL)),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    # One model serves all stage workers; only inference holds the lock.
    lock = threading.Lock()

    def separate(record: AudioRecord) -> list[AudioRecord]:
        waveform = decode_audio(record, separator.samplerate, separator.audio_channels)
        mix = torch.from_numpy(waveform.T.copy())
        with lock:
            _, stems = separator.separate_tensor(mix)
            vocals = stems["vocals"].cpu().numpy().T
        return [replace_audio(record, vocals, separator.samplerate)]

    return separate
