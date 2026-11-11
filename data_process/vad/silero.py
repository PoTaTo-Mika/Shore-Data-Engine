"""Silero VAD slicer for noisy speech."""

import math
import threading
from typing import cast

import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from data_process.codec import SAMPLE_RATE, decode_audio, slice_record
from data_process.data_reader import AudioRecord, Operator

VAD_SAMPLE_RATE = 16000
DEFAULT_THRESHOLD = 0.6
DEFAULT_MIN_SILENCE_MS = 500
DEFAULT_TARGET_S = 15.0
DEFAULT_MAX_S = 20.0
DEFAULT_MIN_S = 1.5
DEFAULT_STRICT_GAP_S = 0.2
DEFAULT_LOOSE_GAP_S = 1.5


def plan_segments(
    speech: list[tuple[float, float]],
    *,
    target_s: float,
    max_s: float,
    min_s: float,
    strict_gap_s: float,
    loose_gap_s: float,
) -> list[tuple[float, float]]:
    """Merge spans toward `target_s`, split ones over `max_s`, drop ones under `min_s`.

    Gaps under `loose_gap_s` are bridged while a span is shorter than `target_s`, and
    only gaps under `strict_gap_s` once it is longer.
    """
    merged: list[tuple[float, float]] = []
    for start, end in speech:
        if merged:
            last_start, last_end = merged[-1]
            gap = start - last_end
            max_gap = loose_gap_s if last_end - last_start < target_s else strict_gap_s
            if end - last_start <= max_s and gap < max_gap:
                merged[-1] = (last_start, end)
                continue
        merged.append((start, end))

    planned: list[tuple[float, float]] = []
    for start, end in merged:
        pieces = math.ceil((end - start) / target_s) if end - start > max_s else 1
        width = (end - start) / pieces
        planned.extend(
            (start + i * width, start + (i + 1) * width) for i in range(pieces)
        )
    return [(start, end) for start, end in planned if end - start >= min_s]


def build_operator(params: dict[str, object]) -> Operator:
    """Split each member at Silero VAD speech boundaries, one opus member per clip."""
    model = load_silero_vad()
    threshold = cast(float, params.get("threshold", DEFAULT_THRESHOLD))
    min_silence_ms = cast(int, params.get("min_silence_ms", DEFAULT_MIN_SILENCE_MS))
    plan = {
        "target_s": cast(float, params.get("target_s", DEFAULT_TARGET_S)),
        "max_s": cast(float, params.get("max_s", DEFAULT_MAX_S)),
        "min_s": cast(float, params.get("min_s", DEFAULT_MIN_S)),
        "strict_gap_s": cast(float, params.get("strict_gap_s", DEFAULT_STRICT_GAP_S)),
        "loose_gap_s": cast(float, params.get("loose_gap_s", DEFAULT_LOOSE_GAP_S)),
    }
    # The model keeps recurrent state per call, so workers must not run it concurrently.
    lock = threading.Lock()

    def slice_audio(record: AudioRecord) -> list[AudioRecord]:
        vad_input = torch.from_numpy(
            decode_audio(record, VAD_SAMPLE_RATE, 1)[:, 0].copy()
        )
        with lock:
            timestamps = get_speech_timestamps(
                vad_input,
                model,
                threshold=threshold,
                sampling_rate=VAD_SAMPLE_RATE,
                min_silence_duration_ms=min_silence_ms,
                return_seconds=True,
            )
        speech = [(float(t["start"]), float(t["end"])) for t in timestamps]
        waveform = decode_audio(record, SAMPLE_RATE, 1)[:, 0]
        segments = [
            (round(start * SAMPLE_RATE), min(len(waveform), round(end * SAMPLE_RATE)))
            for start, end in plan_segments(speech, **plan)
        ]
        return slice_record(record, waveform, SAMPLE_RATE, segments)

    return slice_audio
