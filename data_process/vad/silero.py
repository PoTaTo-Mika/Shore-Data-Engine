import os
import torch
import soundfile as sf
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings
import subprocess
import shutil
import math

warnings.filterwarnings("ignore")

# ================= 全局变量 =================
vad_model = None
vad_utils = None
# ===========================================

def merge_segments(segments, min_duration=3.0, target_duration=15.0, max_duration=60.0, strict_gap=0.5, loose_gap=2.0):
    """
    第一步：尽可能合并短片段。
    """
    if not segments:
        return []

    merged = []
    current_seg = {'start': segments[0]['start'], 'end': segments[0]['end']}

    for i in range(1, len(segments)):
        next_seg = segments[i]
        
        current_dur = current_seg['end'] - current_seg['start']
        gap = next_seg['start'] - current_seg['end']
        next_dur = next_seg['end'] - next_seg['start']
        
        potential_dur = current_dur + gap + next_dur
        
        should_merge = False

        # 如果合并后超过上限，绝对不合并
        if potential_dur > max_duration:
            should_merge = False
        
        # 还没够长，尽量合并
        elif current_dur < target_duration:
            if gap < loose_gap:
                should_merge = True
        
        # 已经够长但没超限，只有间隙很小才合并
        else:
            if gap < strict_gap:
                should_merge = True

        if should_merge:
            current_seg['end'] = next_seg['end']
        else:
            merged.append(current_seg)
            current_seg = {'start': next_seg['start'], 'end': next_seg['end']}

    merged.append(current_seg)
    return merged

def split_long_segment(segment, target_duration=15.0, max_duration=20.0, overlap=0.0):
    """
    [新增] 第二步：强制切分超长片段。
    如果一个片段经过合并逻辑后依然超过 max_duration (比如 VAD 识别出一整段 10分钟的音频)，
    则强制将其均匀切分为 target_duration 长度的小段。
    """
    start = segment['start']
    end = segment['end']
    duration = end - start
    
    # 如果在允许范围内，直接返回
    if duration <= max_duration:
        return [segment]
    
    # 开始强制切分
    # 例如 600s 的音频，切成 ~15s 一段
    chunks = []
    curr_start = start
    
    while curr_start < end:
        curr_end = min(curr_start + target_duration, end)
        
        # 如果剩下的不足 3秒，且前面已经有片段，尝试延长前一个片段（避免末尾出现碎渣）
        if (end - curr_end) < 3.0 and chunks:
             chunks[-1]['end'] = end
             break
        
        chunks.append({'start': curr_start, 'end': curr_end})
        
        # 移动指针 (减去重叠量，确保连贯性，TTS通常不需要重叠，设为0)
        curr_start = curr_end - overlap
        
        # 如果已经到了终点
        if curr_start >= end:
            break
            
    return chunks

def init_worker(local_repo_path):
    global vad_model, vad_utils
    torch.set_num_threads(1)
    try:
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir=local_repo_path, 
            model='silero_vad', 
            source='local',
            trust_repo=True
        )
    except Exception as e:
        print(f"进程初始化失败: {e}")
        # 这里不抛出异常，允许部分进程失败重试或跳过
        pass

def ffmpeg_slice(input_path, output_path, start_time, end_time):
    duration = end_time - start_time
    cmd = [
        'ffmpeg', 
        '-nostdin', '-hide_banner', '-loglevel', 'error',
        '-ss', f"{start_time:.3f}",
        '-t', f"{duration:.3f}",
        '-i', input_path,
        '-c', 'copy', # 注意：copy模式在某些MP3上可能导致时间戳不准，如果还出问题改成 -c:a pcm_s16le
        '-avoid_negative_ts', 'make_zero',
        '-y',
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def process_audio(audio_path, output_dir):
    global vad_model, vad_utils
    
    if vad_model is None or vad_utils is None:
        return

    (get_speech_timestamps, _, read_audio, _, _) = vad_utils
    
    try:
        wav_vad = read_audio(audio_path, sampling_rate=16000)
        
        # 1. 获取基础时间戳
        # 建议提高 threshold 到 0.6 或 0.7 以减少背景噪音被识别为说话
        speech_timestamps = get_speech_timestamps(
            wav_vad, 
            vad_model, 
            sampling_rate=16000,
            return_seconds=True,
            threshold=0.6,          # <--- 修改建议：稍微提高阈值
            min_silence_duration_ms=500 
        )
        
        # 2. 合并碎片
        merged_timestamps = merge_segments(
            speech_timestamps, 
            min_duration=3.0,
            target_duration=15.0,
            max_duration=60.0, 
            strict_gap=0.2,
            loose_gap=1.5         
        )
        
        # 3. [关键修改] 检查并强制切分超长片段
        final_timestamps = []
        for seg in merged_timestamps:
            # 这里的 max_duration 设为 20s，意味着如果有漏网之鱼超过20s，强制切开
            # 这样就能彻底杜绝 10分钟的音频出现
            splitted = split_long_segment(seg, target_duration=15.0, max_duration=20.0)
            final_timestamps.extend(splitted)

        if not final_timestamps:
            return

        os.makedirs(output_dir, exist_ok=True)
        
        file_name = os.path.basename(audio_path)
        name_part, ext_part = os.path.splitext(file_name)
        ext_part = ext_part.lower()
        
        is_lossless_format = ext_part in ['.wav', '.flac']
        
        data = None
        sr = None
        
        if is_lossless_format:
            data, sr = sf.read(audio_path)
        
        saved_count = 0
        
        for i, seg in enumerate(final_timestamps):
            start_sec = seg['start']
            end_sec = seg['end']
            
            if (end_sec - start_sec) < 1.5:
                continue

            save_name = f"{name_part}_{str(i).zfill(3)}{ext_part}" # 增加位数以防切片过多
            save_path = os.path.join(output_dir, save_name)
            
            if is_lossless_format:
                start_sample = int(start_sec * sr)
                end_sample = int(end_sec * sr)
                start_sample = max(0, start_sample)
                end_sample = min(len(data), end_sample)
                
                chunk = data[start_sample:end_sample]
                sf.write(save_path, chunk, sr)
                saved_count += 1
            else:
                success = ffmpeg_slice(audio_path, save_path, start_sec, end_sec)
                if success:
                    saved_count += 1
        
        if saved_count > 0:
            try:
                os.remove(audio_path)
            except Exception:
                pass
            
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

def _process_single_wrapper(args):
    audio_path, output_dir = args
    process_audio(audio_path, output_dir)

def process_folder_recursive(root_folder, max_workers=None):
    tasks = []
    print("正在扫描音频文件...")
    
    valid_extensions = ('.wav', '.mp3', '.flac', '.opus', '.ogg', '.m4a', '.aac')
    
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(valid_extensions):
                full_path = os.path.join(root, file)
                if 'sliced' in root.split(os.sep): 
                    continue
                output_dir = os.path.join(root, 'sliced')
                tasks.append((full_path, output_dir))
    
    print(f"找到 {len(tasks)} 个音频文件。")
    if not tasks:
        return

    # 检查本地缓存
    hub_dir = torch.hub.get_dir()
    local_repo_path = os.path.join(hub_dir, 'snakers4_silero-vad_master')
    
    # 简单的本地模型检查逻辑
    if not os.path.exists(local_repo_path):
        print("未在默认路径找到 Silero VAD，尝试在线加载一次以缓存...")
        try:
            torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
            # 再次寻找
            candidates = [d for d in os.listdir(hub_dir) if 'silero' in d]
            if candidates:
                local_repo_path = os.path.join(hub_dir, candidates[0])
        except Exception as e:
            print(f"模型下载/加载失败: {e}")
            return
    
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 2)
    
    print(f"开始处理，使用 {max_workers} 个进程...")
    
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(local_repo_path,)) as executor:
        list(tqdm(executor.map(_process_single_wrapper, tasks), total=len(tasks), unit="file"))

if __name__ == '__main__':
    DATA_DIR = 'data' 
    process_folder_recursive(DATA_DIR)