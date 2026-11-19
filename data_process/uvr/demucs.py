"""Demucs vocal isolation operator."""

import threading
from typing import cast

import demucs.api
import torch

from data_process.codec import SAMPLE_RATE, base_waveform, carry_pcm
from data_process.data_reader import AudioRecord, Operator

DEFAULT_MODEL = "htdemucs"


def build_operator(params: dict[str, object]) -> Operator:
    """Isolate vocals, passing the PCM on so a following slicer skips a re-decode."""
    separator = demucs.api.Separator(
        model=cast(str, params.get("model", DEFAULT_MODEL)),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    assert separator.samplerate == SAMPLE_RATE, "demucs model must run at SAMPLE_RATE"
    # One model serves all stage workers; only inference holds the lock.
    lock = threading.Lock()

    def separate(record: AudioRecord) -> list[AudioRecord]:
        mix = torch.from_numpy(base_waveform(record).T.copy())
        with lock:
            _, stems = separator.separate_tensor(mix)
            vocals = stems["vocals"].cpu().numpy().T
        return [carry_pcm(record, vocals)]

    return separate
