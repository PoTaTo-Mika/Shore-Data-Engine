import os
import torch
import soundfile as sf
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings
import subprocess  # 新增：用于调用 ffmpeg
import shutil

warnings.filterwarnings("ignore")

# ================= 全局变量 =================
vad_model = None
vad_utils = None
# ===========================================

def merge_segments(segments, min_duration=5.0, target_duration=10.0, max_gap=1.5):
    """
    保持原有的合并逻辑不变
    """
    if not segments:
        return []

    merged = []
    current_seg = {'start': segments[0]['start'], 'end': segments[0]['end']}

    for i in range(1, len(segments)):
        next_seg = segments[i]
        current_dur = current_seg['end'] - current_seg['start']
        gap = next_seg['start'] - current_seg['end']
        
        should_merge = False
        if current_dur < min_duration:
            if gap < 2.0: 
                should_merge = True
        elif (current_dur + (next_seg['end'] - next_seg['start'])) < target_duration and gap < max_gap:
            should_merge = True

        if should_merge:
            current_seg['end'] = next_seg['end']
        else:
            merged.append(current_seg)
            current_seg = {'start': next_seg['start'], 'end': next_seg['end']}

    merged.append(current_seg)
    return merged

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
    """
    使用 FFmpeg 进行无损流式切分 (-c copy)
    """
    duration = end_time - start_time
    # -ss: 开始时间, -t: 持续时间, -c copy: 直接复制流不重编码, -y: 覆盖输出
    # -avoid_negative_ts make_zero: 修正时间戳，防止播放器识别错误
    cmd = [
        'ffmpeg', 
        '-nostdin', '-hide_banner', '-loglevel', 'error', # 静默模式
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
        # print(f"FFmpeg 切分失败: {input_path}")
        return False

def process_audio(audio_path, output_dir):
    global vad_model, vad_utils
    
    if vad_model is None or vad_utils is None:
        return

    (get_speech_timestamps, _, read_audio, _, _) = vad_utils
    
    try:
        # 1. 准备 VAD 音频
        # silero 的 read_audio 支持大多数 ffmpeg 能读的格式 (opus, ogg, mp3 等)
        # 它会在内存中解码为 PCM Tensor，无需中间文件
        wav_vad = read_audio(audio_path, sampling_rate=16000)
        
        # 2. 获取时间戳
        speech_timestamps = get_speech_timestamps(
            wav_vad, 
            vad_model, 
            sampling_rate=16000,
            return_seconds=True,
            threshold=0.5,
            min_silence_duration_ms=500 
        )
        
        # 3. 合并
        merged_timestamps = merge_segments(
            speech_timestamps, 
            min_duration=5.0, 
            target_duration=12.0,
            max_gap=0.8
        )
        
        if not merged_timestamps:
            return

        os.makedirs(output_dir, exist_ok=True)
        
        # 获取文件名和扩展名
        file_name = os.path.basename(audio_path)
        name_part, ext_part = os.path.splitext(file_name)
        ext_part = ext_part.lower()
        
        # 判断是否为无损格式 (可以直接由 Python 库处理) 或 有损格式 (建议 FFmpeg 处理)
        # WAV 和 FLAC 是无损的，可以直接读写 numpy 数组
        is_lossless_format = ext_part in ['.wav', '.flac']
        
        data = None
        sr = None
        
        # 只有在需要 Python 切分时才读取全量音频
        if is_lossless_format:
            data, sr = sf.read(audio_path)
        
        saved_count = 0
        
        for i, seg in enumerate(merged_timestamps):
            start_sec = seg['start']
            end_sec = seg['end']
            
            # 过滤太短的片段
            if (end_sec - start_sec) < 1.0:
                continue

            save_name = f"{name_part}_{str(i).zfill(3)}{ext_part}"
            save_path = os.path.join(output_dir, save_name)
            
            if is_lossless_format:
                # ====== 策略 A: 内存切分 (适用于 Wav/Flac) ======
                start_sample = int(start_sec * sr)
                end_sample = int(end_sec * sr)
                start_sample = max(0, start_sample)
                end_sample = min(len(data), end_sample)
                
                chunk = data[start_sample:end_sample]
                sf.write(save_path, chunk, sr)
                saved_count += 1
            else:
                # ====== 策略 B: FFmpeg 无损流拷贝 (适用于 Opus/Mp3/Ogg) ======
                # 直接调用 FFmpeg 对原文件进行剪切
                success = ffmpeg_slice(audio_path, save_path, start_sec, end_sec)
                if success:
                    saved_count += 1
        
        # 6. 删除原始文件
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
    
    # 扩展支持的格式列表
    valid_extensions = ('.wav', '.mp3', '.flac', '.opus', '.ogg', '.m4a', '.aac')
    
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(valid_extensions):
                full_path = os.path.join(root, file)
                output_dir = os.path.join(root, 'sliced')
                tasks.append((full_path, output_dir))
    
    print(f"找到 {len(tasks)} 个音频文件。")
    if not tasks:
        return

    # 检查 FFmpeg 是否存在
    if shutil.which("ffmpeg") is None:
        print("\n[错误] 未检测到 FFmpeg！处理非 Wav/Flac 格式需要 FFmpeg。")
        print("请安装 FFmpeg 并添加到环境变量，或者只处理 Wav 文件。\n")
        # 这里可以选择 return 退出，或者继续尝试
    
    # ================= 准备本地模型路径 (保持不变) =================
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
            raise FileNotFoundError(f"未找到 Silero 模型缓存。")
    
    print(f"使用本地模型路径: {local_repo_path}")
    # ==========================================================

    if max_workers is None:
        max_workers = max(1, 16)
    
    print(f"开始处理，使用 {max_workers} 个进程...")
    
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(local_repo_path,)) as executor:
        list(tqdm(executor.map(_process_single_wrapper, tasks), total=len(tasks), unit="file"))

if __name__ == '__main__':
    DATA_DIR = 'data'
    process_folder_recursive(DATA_DIR)