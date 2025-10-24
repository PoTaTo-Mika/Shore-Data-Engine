import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from funasr import AutoModel
from tqdm import tqdm
import torch
import multiprocessing as mp
import json

# 支持的音频格式扩展名
SUPPORTED_FORMATS = {
    '.opus', '.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', 
    '.wma', '.aiff', '.pcm', '.amr', '.3gp'
}

def find_audio_files(directory):
    """递归查找所有支持的音频文件"""
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            _, ext = os.path.splitext(file.lower())
            if ext in SUPPORTED_FORMATS:
                audio_files.append(os.path.join(root, file))
    return audio_files

def get_timestamp_path(audio_path):
    """获取时间戳文件路径"""
    return f"{audio_path}.timestamp"

def load_timestamps(timestamp_path):
    """加载时间戳文件"""
    try:
        with open(timestamp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['timestamps']
    except Exception as e:
        print(f"加载时间戳文件失败 {timestamp_path}: {e}")
        return None

def save_timestamps(timestamp_path, timestamps):
    """保存时间戳到文件"""
    try:
        with open(timestamp_path, 'w', encoding='utf-8') as f:
            json.dump({'timestamps': timestamps}, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存时间戳文件失败 {timestamp_path}: {e}")
        return False

def check_files_status(audio_files):
    """检查哪些文件需要VAD处理，哪些已经有时间戳"""
    need_vad = []
    has_timestamp = []
    
    for audio_path in audio_files:
        timestamp_path = get_timestamp_path(audio_path)
        if os.path.exists(timestamp_path):
            has_timestamp.append(audio_path)
        else:
            need_vad.append(audio_path)
    
    return need_vad, has_timestamp

def process_vad_single(args_tuple):
    """VAD阶段：单个音频文件的VAD处理"""
    audio_path, device = args_tuple
    
    try:
        # 在子进程内部加载模型到指定GPU
        vad_model = AutoModel(model='./checkpoints/fsmn-vad', 
                              device=device,
                              disable_update=True)
        
        # VAD推理，获取时间戳
        results = vad_model.generate(input=audio_path)
        
        if not results or not results[0].get('value'):
            return f"Skipped (no segments): {audio_path}", None
        
        timestamps = results[0]['value']
        
        # 保存时间戳
        timestamp_path = get_timestamp_path(audio_path)
        if save_timestamps(timestamp_path, timestamps):
            return f"VAD Success: {audio_path}", timestamps
        else:
            return f"VAD Error (save failed): {audio_path}", None

    except Exception as e:
        error_message = f"VAD Error: {audio_path}: {e}"
        print(error_message)
        return error_message, None

def run_vad_stage(audio_files, num_workers, num_gpus):
    """运行VAD阶段"""
    print(f"\n{'='*60}")
    print(f"VAD处理 - 共 {len(audio_files)} 个文件需要处理")
    print(f"{'='*60}")
    
    # 准备任务列表，轮询分配GPU
    tasks = []
    for i, audio_file in enumerate(audio_files):
        gpu_id = i % num_gpus
        device = f"cuda:{gpu_id}"
        tasks.append((audio_file, device))
    
    print(f"使用 {num_workers} 个 worker 和 {num_gpus} 个 GPU 进行VAD处理...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_vad_single, tasks), 
                           total=len(tasks), 
                           desc="VAD进度"))
    
    # 统计结果
    success = sum(1 for res, _ in results if "VAD Success" in res)
    errors = [res for res, _ in results if "Error" in res]
    
    print(f"\nVAD处理完成: 成功 {success}/{len(audio_files)}")
    if errors:
        print(f"失败 {len(errors)} 个，部分失败日志:")
        for err in errors[:5]:
            print(f"  {err}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="使用 FSMN-VAD 生成音频时间戳")
    parser.add_argument('--input_dir', type=str, default='./data', help="输入音频根目录")
    parser.add_argument('--vad_workers', type=int, default=8, help="VAD阶段并发进程数")
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help="使用的GPU数量")
    
    args = parser.parse_args()

    if not torch.cuda.is_available() or args.num_gpus == 0:
        print("错误：未检测到CUDA设备或num_gpus为0。请在有GPU的环境下运行。")
        return
        
    if args.num_gpus > torch.cuda.device_count():
        print(f"警告：请求使用 {args.num_gpus} GPU, 但只有 {torch.cuda.device_count()} 可用。")
        args.num_gpus = torch.cuda.device_count()

    print(f"支持的格式: {', '.join(sorted(SUPPORTED_FORMATS))}")
    print(f"正在从 '{args.input_dir}' 扫描音频文件...")
    audio_files = find_audio_files(args.input_dir)
    
    if not audio_files:
        print("未找到任何支持的音频文件。")
        return
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 检查文件状态
    need_vad, has_timestamp = check_files_status(audio_files)
    
    print(f"\n文件状态:")
    print(f"  需要VAD处理: {len(need_vad)} 个")
    print(f"  已有时间戳: {len(has_timestamp)} 个")
    
    # VAD处理
    if len(need_vad) > 0:
        run_vad_stage(need_vad, args.vad_workers, args.num_gpus)
    else:
        print("\n所有文件已有时间戳，无需处理")
    
    print(f"\n{'='*60}")
    print("VAD处理完成！")
    print(f"{'='*60}")

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
        print("多进程启动方式已设置为 'spawn'，以确保CUDA安全。")
    except RuntimeError:
        pass
    
    main()