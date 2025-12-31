import os
import torch
import soundfile as sf
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings
import subprocess
import shutil

warnings.filterwarnings("ignore")

# ================= 全局变量 =================
vad_model = None
vad_utils = None
# ===========================================

def merge_segments(segments, min_duration=3.0, target_duration=15.0, max_duration=60.0, strict_gap=0.5, loose_gap=2.0):
    """
    重写后的合并逻辑，专为 TTS 数据集优化。
    
    参数:
    - min_duration: 最小保留时长，小于这个长度的孤立片段会被丢弃（除非它被合并）。
    - target_duration: 期望的理想时长 (约 15s)。
    - max_duration: 硬性最大时长 (约 60s)，防止 OOM。
    - strict_gap: 当片段已经够长(>target)时，只有静音小于此值才继续合并 (倾向于切分)。
    - loose_gap: 当片段还太短(<target)时，只要静音小于此值就允许合并 (倾向于凑长)。
    """
    if not segments:
        return []

    merged = []
    # 初始化当前片段
    current_seg = {'start': segments[0]['start'], 'end': segments[0]['end']}

    for i in range(1, len(segments)):
        next_seg = segments[i]
        
        # 计算当前时长和间隙
        current_dur = current_seg['end'] - current_seg['start']
        gap = next_seg['start'] - current_seg['end']
        next_dur = next_seg['end'] - next_seg['start']
        
        # 预测合并后的总时长
        potential_dur = current_dur + gap + next_dur
        
        should_merge = False

        # --- 核心判断逻辑 ---
        
        # 1. 如果合并后超过硬性上限 (60s)，绝对不合并 -> 强制切分
        if potential_dur > max_duration:
            should_merge = False
        
        # 2. 如果当前还未达到目标长度 (15s)，且静音在允许范围内 (loose_gap)，尝试合并
        elif current_dur < target_duration:
            if gap < loose_gap:
                should_merge = True
        
        # 3. 如果已经超过目标长度 (15s)，但还在最大长度 (60s) 内
        #    此时变得“挑剔”，只有静音非常短 (strict_gap) 才合并（也就是连贯语气）
        #    否则就在这里切开，保持在 15s 左右
        else:
            if gap < strict_gap:
                should_merge = True

        if should_merge:
            current_seg['end'] = next_seg['end']
        else:
            merged.append(current_seg)
            current_seg = {'start': next_seg['start'], 'end': next_seg['end']}

    merged.append(current_seg)
    
    # 最后过滤掉极其短的碎片 (可选)
    final_merged = [s for s in merged if (s['end'] - s['start']) >= 1.0]
    
    return final_merged

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
        raise e

def ffmpeg_slice(input_path, output_path, start_time, end_time):
    # 计算持续时间
    duration = end_time - start_time
    cmd = [
        'ffmpeg', 
        '-nostdin', '-hide_banner', '-loglevel', 'error',
        '-ss', f"{start_time:.3f}",
        '-t', f"{duration:.3f}",
        '-i', input_path,
        '-c', 'copy',
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
        # 1. 准备 VAD 音频
        wav_vad = read_audio(audio_path, sampling_rate=16000)
        
        # 2. 获取时间戳 (VAD 本身只负责找有没有人说话)
        speech_timestamps = get_speech_timestamps(
            wav_vad, 
            vad_model, 
            sampling_rate=16000,
            return_seconds=True,
            threshold=0.5,
            min_silence_duration_ms=500 
        )
        
        # 3. 合并 (使用新的逻辑和参数)
        merged_timestamps = merge_segments(
            speech_timestamps, 
            min_duration=3.0,     # 最小允许 3秒
            target_duration=15.0, # 理想目标 15秒
            max_duration=60.0,    # 最大限制 60秒
            strict_gap=0.2,       # 超过15s后，如果停顿超过0.3s就切分
            loose_gap=1.5         
        )
        
        if not merged_timestamps:
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
        
        for i, seg in enumerate(merged_timestamps):
            start_sec = seg['start']
            end_sec = seg['end']
            
            # 双重保险：过滤太短的片段（小于1.5秒通常无法用于TTS）
            if (end_sec - start_sec) < 1.5:
                continue

            save_name = f"{name_part}_{str(i).zfill(3)}{ext_part}"
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

    if shutil.which("ffmpeg") is None:
        print("\n[注意] 未检测到 FFmpeg！非 Wav/Flac 文件可能无法正确处理。\n")
    
    print("正在检查并下载模型到本地缓存...")
    try:
        torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
    except:
        pass 
    
    hub_dir = torch.hub.get_dir()
    local_repo_path = os.path.join(hub_dir, 'snakers4_silero-vad_master')
    
    if not os.path.exists(local_repo_path):
        candidates = [d for d in os.listdir(hub_dir) if 'silero' in d]
        if candidates:
            local_repo_path = os.path.join(hub_dir, candidates[0])
        else:
            # 如果本地实在没有，这里可能会报错，建议第一次运行联网
            pass
    
    print(f"使用模型路径: {local_repo_path}")

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 2) # 保留一点 CPU
    
    print(f"开始处理，使用 {max_workers} 个进程...")
    
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(local_repo_path,)) as executor:
        list(tqdm(executor.map(_process_single_wrapper, tasks), total=len(tasks), unit="file"))

if __name__ == '__main__':
    # 将这里修改为你的数据目录
    DATA_DIR = r'data' 
    process_folder_recursive(DATA_DIR)