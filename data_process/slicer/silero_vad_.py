import os
import torch
import soundfile as sf
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ================= 全局变量 =================
# 这些变量将在每个子进程中被初始化一次
vad_model = None
vad_utils = None
# ===========================================

def merge_segments(segments, min_duration=5.0, target_duration=10.0, max_gap=1.5):
    """
    (保持原有的合并逻辑不变)
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
    """
    子进程初始化函数：每个进程只运行一次。
    在这里加载模型，避免重复 I/O 和 网络请求。
    """
    global vad_model, vad_utils
    
    # 限制进程内的线程数，防止 CPU 争抢
    torch.set_num_threads(1)
    
    # 使用 source='local' 彻底断绝网络请求
    # 注意：这里加载的是本地缓存的路径
    try:
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir=local_repo_path, 
            model='silero_vad', 
            source='local',  # 关键修改：强制本地加载
            trust_repo=True
        )
    except Exception as e:
        print(f"进程初始化失败: {e}")
        raise e

def process_audio(audio_path, output_dir):
    """
    处理单个音频文件
    不再需要传递 model 和 utils，直接使用全局变量
    """
    # 引用全局变量
    global vad_model, vad_utils
    
    if vad_model is None or vad_utils is None:
        return # 初始化失败的情况

    (get_speech_timestamps, _, read_audio, _, _) = vad_utils
    
    try:
        # 1. 准备 VAD 音频
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
            # print(f"警告: {audio_path} 未检测到语音，跳过") # 减少日志输出
            return

        # 4. 读取原音频切分
        data, sr = sf.read(audio_path)
        
        audio_name = os.path.basename(audio_path).rsplit('.', 1)[0]
        os.makedirs(output_dir, exist_ok=True)
        
        saved_count = 0
        for i, seg in enumerate(merged_timestamps):
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            
            start_sample = max(0, start_sample)
            end_sample = min(len(data), end_sample)
            
            if (end_sample - start_sample) / sr < 1.0:
                continue
                
            chunk = data[start_sample:end_sample]
            save_name = f"{audio_name}_{str(i).zfill(3)}.wav"
            save_path = os.path.join(output_dir, save_name)
            
            sf.write(save_path, chunk, sr)
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
    """
    包装器现在非常简单，只负责解包参数调用逻辑
    """
    audio_path, output_dir = args
    process_audio(audio_path, output_dir)

def process_folder_recursive(root_folder, max_workers=None):
    tasks = []
    print("正在扫描音频文件...")
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                full_path = os.path.join(root, file)
                output_dir = os.path.join(root, 'sliced')
                tasks.append((full_path, output_dir))
    
    print(f"找到 {len(tasks)} 个音频文件。")
    if not tasks:
        return

    # ================= 关键步骤：准备本地模型路径 =================
    print("正在检查并下载模型到本地缓存...")
    # 在主进程先下载一次，确保缓存存在
    try:
        torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
    except:
        pass # 可能已经有了，忽略网络错误
    
    # 获取 torch hub 的本地缓存路径
    # 通常是 ~/.cache/torch/hub/snakers4_silero-vad_master
    hub_dir = torch.hub.get_dir()
    local_repo_path = os.path.join(hub_dir, 'snakers4_silero-vad_master')
    
    if not os.path.exists(local_repo_path):
        # 尝试另一种常见的命名（取决于版本）
        local_repo_path = os.path.join(hub_dir, 'snakers4_silero-vad_v4.0') # 示例，具体看下载下来的文件夹名
        if not os.path.exists(local_repo_path):
             # 最后的兜底：列出目录找含有 silero 的文件夹
            candidates = [d for d in os.listdir(hub_dir) if 'silero' in d]
            if candidates:
                local_repo_path = os.path.join(hub_dir, candidates[0])
            else:
                raise FileNotFoundError(f"未找到 Silero 模型缓存，请确保至少成功联网运行过一次 torch.hub.load。路径检查: {hub_dir}")
    
    print(f"使用本地模型路径: {local_repo_path}")
    # ==========================================================

    if max_workers is None:
        max_workers = max(1, 16)
    
    print(f"开始处理，使用 {max_workers} 个进程...")
    
    # 使用 initializer 初始化每个进程的模型
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(local_repo_path,)) as executor:
        list(tqdm(executor.map(_process_single_wrapper, tasks), total=len(tasks), unit="file"))

if __name__ == '__main__':
    DATA_DIR = 'data'
    process_folder_recursive(DATA_DIR)