"""TEN VAD slicer for short-utterance speech."""

import math
from typing import cast

import numpy as np
from ten_vad import TenVad

from data_process.codec import SAMPLE_RATE, decode_audio, slice_record
from data_process.data_reader import AudioRecord, Operator

VAD_SAMPLE_RATE = 16000
HOP_SIZE = 256
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SPEECH_MS = 300
DEFAULT_MIN_SILENCE_MS = 200
DEFAULT_PAD_MS = 50


def find_speech(
    voiced: list[bool], *, min_speech: int, min_silence: int, pad: int
) -> list[tuple[int, int]]:
    """Group voiced frames into padded [start, end) frame spans.

    A span closes after `min_silence` unvoiced frames and is kept if it has at least
    `min_speech` frames.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    silence = 0
    for i, flag in enumerate(voiced):
        if flag:
            start = i if start is None else start
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= min_silence:
                spans.append((start, i - silence + 1))
                start = None
                silence = 0
    if start is not None:
        spans.append((start, len(voiced)))
    return [
        (max(0, begin - pad), min(len(voiced), end + pad))
        for begin, end in spans
        if end - begin >= min_speech
    ]


def build_operator(params: dict[str, object]) -> Operator:
    """Split each member at TEN VAD speech boundaries, one opus member per clip."""
    threshold = cast(float, params.get("threshold", DEFAULT_THRESHOLD))
    frame_ms = HOP_SIZE * 1000 / VAD_SAMPLE_RATE
    min_speech = math.ceil(
        cast(int, params.get("min_speech_ms", DEFAULT_MIN_SPEECH_MS)) / frame_ms
    )
    min_silence = math.ceil(
        cast(int, params.get("min_silence_ms", DEFAULT_MIN_SILENCE_MS)) / frame_ms
    )
    pad = round(cast(int, params.get("pad_ms", DEFAULT_PAD_MS)) / frame_ms)

    def slice_audio(record: AudioRecord) -> list[AudioRecord]:
        vad_input = decode_audio(record, VAD_SAMPLE_RATE, 1)[:, 0]
        pcm = (np.clip(vad_input, -1.0, 1.0) * 32767).astype(np.int16)
        frames = pcm[: len(pcm) // HOP_SIZE * HOP_SIZE].reshape(-1, HOP_SIZE)
        # The detector is stateful, so each member gets a fresh instance.
        vad = TenVad(hop_size=HOP_SIZE, threshold=threshold)
        voiced = [vad.process(frame)[1] == 1 for frame in frames]
        spans = find_speech(
            voiced, min_speech=min_speech, min_silence=min_silence, pad=pad
        )

        waveform = decode_audio(record, SAMPLE_RATE, 1)[:, 0]
        samples_per_frame = HOP_SIZE * SAMPLE_RATE / VAD_SAMPLE_RATE
        segments = [
            (
                round(begin * samples_per_frame),
                min(len(waveform), round(end * samples_per_frame)),
            )
            for begin, end in spans
        ]
        return slice_record(record, waveform, SAMPLE_RATE, segments)

    return slice_audio
