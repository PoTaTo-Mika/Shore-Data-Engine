import torch
import demucs.api
import tqdm
import os
import logging
import multiprocessing
from multiprocessing import Process
from typing import List, Optional, Tuple
import subprocess
from io import BytesIO
import soundfile as sf
from demucs.apply import BagOfModels

# 设置多进程启动方法为 spawn（必须在其他导入之前）
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

# 配置logging
def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(processName)s PID:%(process)d] - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/demucs.log', encoding='utf-8')
        ]
    )

def compile_separator_model(separator: demucs.api.Separator, gpu_index: Optional[int] = None):
    """对Separator内部的模型应用torch.compile进行优化"""
    log_prefix = f"[GPU {gpu_index}] " if gpu_index is not None else ""
    logging.info(f"{log_prefix}Applying torch.compile to the model...")
    
    loaded_model = separator.model
    
    try:
        if isinstance(loaded_model, BagOfModels):
            logging.info(f"{log_prefix}Compiling BagOfModels with {len(loaded_model.models)} sub-models.")
            for i, sub_model in enumerate(loaded_model.models):
                loaded_model.models[i] = torch.compile(sub_model)
        else:
            separator.model = torch.compile(loaded_model)
        logging.info(f"{log_prefix}Model compiled successfully.")
    except Exception as e:
        logging.warning(f"{log_prefix}Failed to compile model: {e}")
        
    return separator

def audio_to_wav_memory(audio_path: str) -> BytesIO:
    """将任意音频格式转换为WAV并存储在内存中"""
    try:
        cmd = ['ffmpeg', '-i', audio_path, '-acodec', 'pcm_f32le', '-ar', '44100', '-ac', '2', '-f', 'wav', 'pipe:1']
        result = subprocess.run(cmd, capture_output=True, check=True)
        return BytesIO(result.stdout)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg conversion failed: {e.stderr.decode()}")

def wav_memory_to_opus(wav_buffer: BytesIO, output_path: str):
    """将内存中的WAV转换为Opus格式"""
    try:
        cmd = ['ffmpeg', '-i', 'pipe:0', '-acodec', 'libopus', '-b:a', '128k', 
               '-vbr', 'on', '-compression_level', '10', '-y', output_path]
        wav_buffer.seek(0)
        subprocess.run(cmd, input=wav_buffer.read(), capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg encoding to Opus failed: {e.stderr.decode()}")

def mark_as_finished(file_path: str):
    """标记文件为已处理"""
    finished_path = os.path.splitext(file_path)[0] + '.finish'
    with open(finished_path, 'w') as f:
        f.write('processed')

def is_finished(file_path: str) -> bool:
    """检查文件是否已处理"""
    finished_path = os.path.splitext(file_path)[0] + '.finish'
    return os.path.exists(finished_path)

def load_audio(audio_path: str) -> Tuple[torch.Tensor, int, str]:
    """
    加载音频文件
    返回: (audio_tensor, samplerate, format_type)
    format_type: 'wav' | 'opus' | 'other'
    """
    _, ext = os.path.splitext(audio_path)
    ext = ext.lower()
    
    # 如果是原生WAV，直接读取
    if ext == '.wav':
        audio_data, samplerate = sf.read(audio_path)
        audio_tensor = torch.from_numpy(audio_data.T).float()
        return audio_tensor, samplerate, 'wav'
    
    # 如果是Opus
    if ext == '.opus':
        wav_buffer = audio_to_wav_memory(audio_path)
        wav_buffer.seek(0)
        audio_data, samplerate = sf.read(wav_buffer)
        wav_buffer.close()
        audio_tensor = torch.from_numpy(audio_data.T).float()
        return audio_tensor, samplerate, 'opus'
    
    # 其他格式
    wav_buffer = audio_to_wav_memory(audio_path)
    wav_buffer.seek(0)
    audio_data, samplerate = sf.read(wav_buffer)
    wav_buffer.close()
    audio_tensor = torch.from_numpy(audio_data.T).float()
    return audio_tensor, samplerate, 'other'

def get_output_path(input_path: str, format_type: str) -> str:
    """
    根据输入格式确定输出路径
    - wav -> wav (保持原格式)
    - opus -> opus (保持原格式)
    - other -> opus (转换为opus)
    """
    base_name = os.path.splitext(input_path)[0]
    
    if format_type == 'wav':
        return input_path  # 保持原WAV路径
    elif format_type == 'opus':
        return input_path  # 保持原Opus路径
    else:
        return base_name + '.opus'  # 其他格式转为Opus

def save_audio(vocals_tensor: torch.Tensor, samplerate: int, output_path: str, format_type: str):
    """
    保存音频文件
    - wav: 直接保存为WAV（覆盖原文件）
    - opus: 内存转码保存为Opus（覆盖原文件）
    - other: 内存转码保存为Opus
    """
    vocals_np = vocals_tensor.cpu().numpy().T
    
    # 原生WAV：直接保存为WAV（覆盖）
    if format_type == 'wav':
        sf.write(output_path, vocals_np, samplerate, format='WAV', subtype='PCM_16')
        return
    
    # Opus或其他格式：内存转码为Opus
    wav_buffer = BytesIO()
    sf.write(wav_buffer, vocals_np, samplerate, format='WAV', subtype='PCM_16')
    wav_memory_to_opus(wav_buffer, output_path)
    wav_buffer.close()

def process_audio(audio_path: str, separator: demucs.api.Separator):
    """处理单个音频文件"""
    if is_finished(audio_path):
        logging.info(f"File already processed, skipping: {audio_path}")
        return
    
    logging.info(f"Processing: {audio_path}")
    
    try:
        # 加载音频
        audio_tensor, samplerate, format_type = load_audio(audio_path)
        
        # 分离人声
        _, separated = separator.separate_tensor(audio_tensor, samplerate)
        vocals_tensor = separated['vocals']
        
        # 确定输出路径
        output_path = get_output_path(audio_path, format_type)
        
        # 如果是其他格式转为Opus，需要删除原文件
        should_delete_original = (format_type == 'other' and output_path != audio_path)
        
        # 保存结果（直接覆盖）
        save_audio(vocals_tensor, samplerate, output_path, format_type)
        
        # 删除原文件（如果需要）
        if should_delete_original and os.path.exists(audio_path):
            os.remove(audio_path)
            logging.info(f"Deleted original file: {audio_path}")
        
        # 标记完成（使用输出路径）
        mark_as_finished(output_path)
        logging.info(f"Successfully processed: {audio_path} -> {output_path}")
            
    except Exception:
        logging.exception(f"Failed to process: {audio_path}")

def list_audio_files(folder_path: str) -> List[str]:
    """列出所有待处理的音频文件"""
    audio_extensions = {'.opus', '.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
    audio_files: List[str] = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in audio_extensions:
                full_path = os.path.join(root, file)
                if not is_finished(full_path):
                    audio_files.append(full_path)
    return audio_files

def split_evenly(items: List[str], num_parts: int) -> List[List[str]]:
    """将列表均匀分割成多个部分"""
    if num_parts <= 0:
        return [items]
    if not items:
        return [[] for _ in range(num_parts)]
    
    base, extra = divmod(len(items), num_parts)
    result: List[List[str]] = []
    start = 0
    
    for i in range(num_parts):
        end = start + base + (1 if i < extra else 0)
        result.append(items[start:end])
        start = end
    
    return result

def worker_run_on_gpu(gpu_index: int, files: List[str]):
    """GPU工作进程"""
    setup_logging()
    torch.set_float32_matmul_precision('high')
    device_str = f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu'
    
    try:
        if device_str.startswith('cuda:'):
            torch.cuda.set_device(gpu_index)
    except Exception as e:
        logging.warning(f"[GPU {gpu_index}] Failed to set CUDA device: {e}")
        device_str = 'cpu'

    logging.info(f"[GPU {gpu_index}] Worker starting. Files: {len(files)}, Device: {device_str}")

    try:
        separator = demucs.api.Separator(model='htdemucs', device=device_str)
        separator = compile_separator_model(separator, gpu_index=gpu_index)
    except Exception as e:
        logging.error(f"[GPU {gpu_index}] Failed to initialize Separator: {e}")
        return

    logging.info(f"[GPU {gpu_index}] Processing files (first file may be slower due to JIT)...")
    for idx, audio_file in enumerate(files, 1):
        try:
            logging.info(f"[GPU {gpu_index}] ({idx}/{len(files)}) => {audio_file}")
            process_audio(audio_file, separator)
        except Exception:
            logging.exception(f"[GPU {gpu_index}] Error processing: {audio_file}")

def process_folder(folder_path: str):
    """处理文件夹中的所有音频文件"""
    audio_files = list_audio_files(folder_path)
    logging.info(f"Found {len(audio_files)} new audio files to process.")

    if not audio_files:
        logging.info("No new files to process.")
        return

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # 单GPU或CPU模式
    if num_gpus <= 1:
        device_str = 'cuda:0' if num_gpus == 1 else 'cpu'
        logging.info(f"Single process mode on device: {device_str}")
        
        if device_str.startswith('cuda:'):
            try:
                torch.cuda.set_device(0)
            except Exception:
                device_str = 'cpu'
        
        separator = demucs.api.Separator(model='htdemucs', device=device_str)
        separator = compile_separator_model(separator)
            
        logging.info("Processing files (first file may be slower due to JIT)...")
        for audio_file in tqdm.tqdm(audio_files, desc="Processing"):
            process_audio(audio_file, separator)
        return

    # 多GPU并行模式
    logging.info(f"Multi-process mode with {num_gpus} GPUs.")
    chunks = split_evenly(audio_files, num_gpus)
    processes: List[Process] = []
    
    for gpu_index in range(num_gpus):
        p = Process(target=worker_run_on_gpu, args=(gpu_index, chunks[gpu_index]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    logging.info("All processes finished.")

if __name__ == '__main__':
    setup_logging()
    
    # 检查FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("FFmpeg not found. Please install it and add to PATH.")
        exit(1)
    
    folder = './data'  # 修改为你的目标文件夹
    process_folder(folder)