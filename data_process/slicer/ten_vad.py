import os
import math
import librosa
import soundfile
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from ten_vad import TenVad 

def process_audio(audio_path, output_dir):
    # 与 lets_slice.py 一致：在子进程内完成切分与写盘
    hop_size = 256
    threshold = 0.5
    min_speech_ms = 300
    min_silence_ms = 200
    pad_ms = 50

    # get audio name
    audio_name = os.path.basename(audio_path).split('.')[0]

    # 读取并重采样到 44100Hz，单声道，float32 [-1, 1]
    waveform, sr = librosa.load(audio_path, sr=44100)

    # 转为 int16 PCM 供 VAD 使用
    pcm = np.asarray(np.clip(waveform, -1.0, 1.0) * 32767.0, dtype=np.int16)

    # 对齐整帧
    total_frames = len(pcm) // hop_size
    if total_frames == 0:
        os.makedirs(output_dir, exist_ok=True)
        return
    pcm = pcm[: total_frames * hop_size]
    waveform = waveform[: total_frames * hop_size]
    frames = pcm.reshape(total_frames, hop_size)

    # VAD
    vad = TenVad(hop_size=hop_size, threshold=threshold)
    frame_ms = hop_size * 1000.0 / sr
    min_silence_f = max(1, int(math.ceil(min_silence_ms / frame_ms)))
    min_speech_f = max(1, int(math.ceil(min_speech_ms / frame_ms)))
    pad_f = int(round(pad_ms / frame_ms))

    voiced_flags = []
    for f in frames:
        prob, flag = vad.process(np.ascontiguousarray(f))
        voiced_flags.append(bool(flag != 0 or prob > threshold))

    # 段落提取（带静音挂起）
    segments = []
    in_seg = False
    seg_start = 0
    sil_cnt = 0

    for i, v in enumerate(voiced_flags):
        if v:
            if not in_seg:
                in_seg = True
                seg_start = i
            sil_cnt = 0
        else:
            if in_seg:
                sil_cnt += 1
                if sil_cnt >= min_silence_f:
                    seg_end = i - sil_cnt  # 最后一个有声帧的索引
                    # 过滤太短的段
                    if seg_end - seg_start + 1 >= min_speech_f:
                        s = max(0, seg_start - pad_f) * hop_size
                        e = min(total_frames, seg_end + 1 + pad_f) * hop_size
                        segments.append((s, e))
                    in_seg = False
                    sil_cnt = 0

    # 收尾：最后仍在说话
    if in_seg:
        seg_end = total_frames - 1
        if seg_end - seg_start + 1 >= min_speech_f:
            s = max(0, seg_start - pad_f) * hop_size
            e = min(total_frames, seg_end + 1 + pad_f) * hop_size
            segments.append((s, e))

    # 写出切片（结构与 lets_slice.py 一致）
    os.makedirs(output_dir, exist_ok=True)
    for i, (s, e) in enumerate(segments):
        chunk = waveform[s:e]
        soundfile.write(os.path.join(output_dir, f'{audio_name}_{i}.wav'), chunk, sr)


def _process_single(args):
    audio_path, output_dir = args
    process_audio(audio_path, output_dir)


def process_folder(folder_path, max_workers=None):
    # 收集所有音频文件，递归处理子目录
    audio_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))

    print(f"找到 {len(audio_files)} 个音频文件需要处理")

    # 为每个音频文件计算输出目录并并发处理
    tasks = []
    for audio_file in audio_files:
        # 获取相对于根目录的路径，确定输出目录
        rel_path = os.path.relpath(audio_file, folder_path)
        audio_dir = os.path.dirname(rel_path)

        if audio_dir:  # 如果在子目录中
            output_dir = os.path.join(folder_path, audio_dir, 'sliced')
        else:  # 如果直接在根目录中
            output_dir = os.path.join(folder_path, 'sliced')

        tasks.append((audio_file, output_dir))

    if max_workers is None:
        try:
            max_workers = os.cpu_count() or 1
        except Exception:
            max_workers = 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(_process_single, tasks), total=len(tasks), desc="切片处理音频文件"))


if __name__ == '__main__':
    folder_path = 'data'
    process_folder(folder_path)