import os
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor
from funasr import AutoModel
from tqdm import tqdm
import torch
import multiprocess as mp

# 支持的音频格式扩展名
SUPPORTED_FORMATS = {
    '.opus', '.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', 
    '.wma', '.aiff', '.pcm', '.amr', '.3gp'
}

# 音频格式对应的FFmpeg编码器
CODEC_MAP = {
    '.opus': 'libopus',
    '.wav': 'pcm_s16le', 
    '.mp3': 'libmp3lame',
    '.m4a': 'aac',
    '.flac': 'flac',
    '.aac': 'aac',
    '.ogg': 'libvorbis',
    '.wma': 'wmav2',
    '.aiff': 'pcm_s16le',
    '.pcm': 'pcm_s16le',
    '.amr': 'libopencore_amrnb',
    '.3gp': 'aac'
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

def get_output_extension(input_path, output_format=None):
    """根据输入文件和输出格式确定输出文件扩展名"""
    if output_format:
        return f'.{output_format.lstrip(".")}'
    
    input_ext = os.path.splitext(input_path.lower())[1]
    # 默认保持原格式，除了某些特殊情况
    if input_ext in ['.pcm', '.aiff']:
        return '.wav'  # 将PCM和AIFF转换为更通用的WAV
    return input_ext

def get_codec_params(input_path, output_path):
    """根据输入输出格式确定FFmpeg编码参数"""
    output_ext = os.path.splitext(output_path.lower())[1]
    
    if output_ext in CODEC_MAP:
        codec = CODEC_MAP[output_ext]
        # 对于无损格式使用复制，有损格式使用指定编码器
        if output_ext in ['.wav', '.flac', '.aiff']:
            return ['-c:a', 'copy']  # 无损格式直接复制
        else:
            return ['-c:a', codec]
    
    # 默认使用libopus
    return ['-c:a', 'libopus']

def process_single_audio(args_tuple):
    audio_path, input_base_dir, output_base_dir, device, output_format = args_tuple
    
    try:
        # 1. 在子进程内部加载模型到指定GPU
        vad_model = AutoModel(model='./checkpoints/fsmn-vad', 
                              device=device,
                              disable_update=True)
        
        # 2. VAD推理，获取时间戳
        results = vad_model.generate(input=audio_path)
        
        if not results or not results[0].get('value'):
            return f"Skipped (no segments): {audio_path}"
        
        timestamps = results[0]['value']
        
        # 3. 计算并创建输出目录，保持原始目录结构
        relative_path = os.path.relpath(os.path.dirname(audio_path), input_base_dir)
        current_output_dir = os.path.join(output_base_dir, relative_path)
        os.makedirs(current_output_dir, exist_ok=True)
        
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        output_ext = get_output_extension(audio_path, output_format)
        
        # 4. 遍历时间戳，使用ffmpeg进行切割
        for i, (start_ms, end_ms) in enumerate(timestamps):
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            
            output_filename = os.path.join(current_output_dir, f"{base_filename}_{i:04d}{output_ext}")
            
            # 获取编码参数
            codec_params = get_codec_params(audio_path, output_filename)
            
            # 构建FFmpeg命令
            command = [
                'ffmpeg',
                '-i', audio_path,
                '-ss', str(start_sec),
                '-to', str(end_sec),
                *codec_params,
                '-vn',
                '-nostdin',
                '-y',
                '-loglevel', 'error',
                output_filename
            ]
            
            subprocess.run(command, check=True)
            
        return f"Success: {audio_path}"

    except Exception as e:
        error_message = f"Error processing {audio_path}: {e}"
        print(error_message)
        return error_message


def main():
    parser = argparse.ArgumentParser(description="使用 FSMN-VAD 和 FFmpeg 对大规模音频进行并行切分")
    parser.add_argument('--input_dir', type=str, default='./data', help="输入音频根目录")
    parser.add_argument('--output_dir', type=str, default='./data', help="输出根目录")
    parser.add_argument('--num_workers', type=int, default=64, help="并发进程数")
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help="使用的GPU数量")
    parser.add_argument('--output_format', type=str, default=None, 
                       help="输出格式 (如: wav, mp3, opus等)，默认保持原格式")
    
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
        
    print(f"找到 {len(audio_files)} 个文件。准备分配任务...")
    
    # 准备任务列表，轮询分配GPU
    tasks = []
    for i, audio_file in enumerate(audio_files):
        gpu_id = i % args.num_gpus
        device = f"cuda:{gpu_id}"
        tasks.append((audio_file, args.input_dir, args.output_dir, device, args.output_format))
    
    print(f"使用 {args.num_workers} 个 worker 和 {args.num_gpus} 个 GPU 开始并行处理...")
    if args.output_format:
        print(f"输出格式设置为: {args.output_format}")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(executor.map(process_single_audio, tasks), total=len(tasks), desc="VAD和切分进度"))

    print("\n处理完成！")
    errors = [res for res in results if res.startswith("Error")]
    if errors:
        print(f"\n有 {len(errors)} 个文件处理失败，部分失败日志如下:")
        for err in errors[:10]:
            print(err)

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
        print("多进程启动方式已设置为 'spawn'，以确保CUDA安全。")
    except RuntimeError:
        pass
    
    main()