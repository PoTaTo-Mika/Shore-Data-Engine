"""RMS-threshold slicer for clean speech."""

from typing import cast

import numpy as np

from data_process.codec import SAMPLE_RATE, decode_audio, slice_record
from data_process.data_reader import AudioRecord, Operator

DEFAULT_THRESHOLD_DB = -30.0
DEFAULT_MIN_LENGTH_MS = 10000
DEFAULT_MIN_INTERVAL_MS = 500
DEFAULT_HOP_SIZE_MS = 10
DEFAULT_MAX_SIL_KEPT_MS = 500


def get_rms(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Centered frame-wise RMS, matching `librosa.feature.rms` with constant padding."""
    y = np.pad(y, frame_length // 2)
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length]
    return np.sqrt(np.mean(np.abs(frames) ** 2, axis=-1))


class Slicer:
    """Cut mono audio at long enough silences, keeping clips above `min_length`."""

    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ) -> None:
        if not min_length >= min_interval >= hop_size:
            raise ValueError("Require min_length >= min_interval >= hop_size")
        if not max_sil_kept >= hop_size:
            raise ValueError("Require max_sil_kept >= hop_size")
        interval_samples = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(interval_samples), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(interval_samples / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def slice(self, waveform: np.ndarray) -> list[tuple[int, int]]:
        """Return (start, end) sample ranges of the kept clips."""
        total_samples = waveform.shape[0]
        if (total_samples + self.hop_size - 1) // self.hop_size <= self.min_length:
            return [(0, total_samples)]
        rms_list = get_rms(waveform, self.win_size, self.hop_size)
        sil_tags: list[tuple[int, int]] = []
        silence_start: int | None = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Cut at the quietest frames, keeping at most max_sil_kept of silence.
            if i - silence_start <= self.max_sil_kept:
                pos = int(rms_list[silence_start : i + 1].argmin()) + silence_start
                sil_tags.append((0, pos) if silence_start == 0 else (pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = int(
                    rms_list[
                        i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                    ].argmin()
                )
                pos += i - self.max_sil_kept
                pos_l = (
                    int(
                        rms_list[
                            silence_start : silence_start + self.max_sil_kept + 1
                        ].argmin()
                    )
                    + silence_start
                )
                pos_r = (
                    int(rms_list[i - self.max_sil_kept : i + 1].argmin())
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = (
                    int(
                        rms_list[
                            silence_start : silence_start + self.max_sil_kept + 1
                        ].argmin()
                    )
                    + silence_start
                )
                pos_r = (
                    int(rms_list[i - self.max_sil_kept : i + 1].argmin())
                    + i
                    - self.max_sil_kept
                )
                sil_tags.append((0, pos_r) if silence_start == 0 else (pos_l, pos_r))
                clip_start = pos_r
            silence_start = None

        total_frames = rms_list.shape[0]
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = (
                int(rms_list[silence_start : silence_end + 1].argmin()) + silence_start
            )
            sil_tags.append((pos, total_frames + 1))
        if not sil_tags:
            return [(0, total_samples)]

        frames = []
        if sil_tags[0][0] > 0:
            frames.append((0, sil_tags[0][0]))
        frames.extend(
            (sil_tags[i][1], sil_tags[i + 1][0]) for i in range(len(sil_tags) - 1)
        )
        if sil_tags[-1][1] < total_frames:
            frames.append((sil_tags[-1][1], total_frames))
        return [
            (begin * self.hop_size, min(total_samples, end * self.hop_size))
            for begin, end in frames
        ]


def build_operator(params: dict[str, object]) -> Operator:
    """Split each member at RMS silences, one opus member per clip."""
    slicer = Slicer(
        sr=SAMPLE_RATE,
        threshold=cast(float, params.get("threshold", DEFAULT_THRESHOLD_DB)),
        min_length=cast(int, params.get("min_length", DEFAULT_MIN_LENGTH_MS)),
        min_interval=cast(int, params.get("min_interval", DEFAULT_MIN_INTERVAL_MS)),
        hop_size=cast(int, params.get("hop_size", DEFAULT_HOP_SIZE_MS)),
        max_sil_kept=cast(int, params.get("max_sil_kept", DEFAULT_MAX_SIL_KEPT_MS)),
    )

    def slice_audio(record: AudioRecord) -> list[AudioRecord]:
        waveform = decode_audio(record, SAMPLE_RATE, 1)[:, 0]
        return slice_record(record, waveform, SAMPLE_RATE, slicer.slice(waveform))

    return slice_audio
