import torch
import demucs.api
import tqdm
import os
import logging
import multiprocessing
from multiprocessing import Process
from typing import List, Optional
import subprocess
from io import BytesIO
import soundfile as sf
import numpy as np
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
    """
    对Separator内部的模型应用torch.compile进行优化。
    这是一个原地操作。
    """
    log_prefix = f"[GPU {gpu_index}] " if gpu_index is not None else ""
    logging.info(f"{log_prefix}Applying torch.compile to the model. This may take a moment...")
    
    loaded_model = separator.model
    
    try:
        # 检查是否为模型集成包
        if isinstance(loaded_model, BagOfModels):
            logging.info(f"{log_prefix}Model is a BagOfModels with {len(loaded_model.models)} sub-models.")
            for i, sub_model in enumerate(loaded_model.models):
                logging.info(f"{log_prefix}  - Compiling sub-model {i}...")
                loaded_model.models[i] = torch.compile(sub_model)
            logging.info(f"{log_prefix}All sub-models compiled successfully.")
        else:
            logging.info(f"{log_prefix}Model is a single instance. Compiling...")
            separator.model = torch.compile(loaded_model)
            logging.info(f"{log_prefix}Model compiled successfully.")
            
    except Exception as e:
        logging.warning(f"{log_prefix}Failed to compile the model. Will proceed in normal (eager) mode. Error: {e}")
        
    return separator

def opus_to_wav_memory(opus_path: str) -> BytesIO:
    try:
        cmd = ['ffmpeg', '-i', opus_path, '-acodec', 'pcm_f32le', '-ar', '44100', '-ac', '2', '-f', 'wav', 'pipe:1']
        result = subprocess.run(cmd, capture_output=True, check=True)
        return BytesIO(result.stdout)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg opus->wav failed: {e.stderr.decode()}")

def wav_memory_to_opus(wav_buffer: BytesIO, opus_path: str):
    try:
        cmd = ['ffmpeg', '-i', 'pipe:0', '-acodec', 'libopus', '-b:a', '96k', '-vbr', 'on', '-compression_level', '10', '-y', opus_path]
        wav_buffer.seek(0)
        subprocess.run(cmd, input=wav_buffer.read(), capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg wav->opus failed: {e.stderr.decode()}")

def mark_as_finished(file_path: str):
    base_name = os.path.splitext(file_path)[0]
    finished_path = base_name + '.finish'
    with open(finished_path, 'w') as f:
        f.write('processed')
    logging.info(f"Marked as finished: {finished_path}")

def is_finished(file_path: str) -> bool:
    base_name = os.path.splitext(file_path)[0]
    finished_path = base_name + '.finish'
    return os.path.exists(finished_path)

def process_audio(audio_path, separator):
    if is_finished(audio_path):
        logging.info(f"File already processed, skipping: {audio_path}")
        return
    
    logging.info(f"Processing: {audio_path}")
    wav_input_buffer, wav_output_buffer = None, None
    try:
        wav_input_buffer = opus_to_wav_memory(audio_path)
        wav_input_buffer.seek(0)
        audio_data, samplerate = sf.read(wav_input_buffer)
        
        audio_tensor = torch.from_numpy(audio_data.T).float()
        
        _, separated = separator.separate_tensor(audio_tensor, samplerate)
        vocals_tensor = separated['vocals']
        
        vocals_np = vocals_tensor.cpu().numpy().T
        
        wav_output_buffer = BytesIO()
        sf.write(wav_output_buffer, vocals_np, samplerate, format='WAV', subtype='FLOAT')
        
        wav_memory_to_opus(wav_output_buffer, audio_path)
        mark_as_finished(audio_path)
        logging.info(f"Successfully processed: {audio_path}")
            
    except Exception:
        logging.exception(f"Failed to process: {audio_path}")
    finally:
        if wav_input_buffer: wav_input_buffer.close()
        if wav_output_buffer: wav_output_buffer.close()

def _list_audio_files(folder_path: str) -> List[str]:
    audio_files: List[str] = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.opus'):
                full_path = os.path.join(root, file)
                if not is_finished(full_path):
                    audio_files.append(full_path)
    return audio_files

def _split_evenly(items: List[str], num_parts: int) -> List[List[str]]:
    if num_parts <= 0: return [items]
    length = len(items)
    if length == 0: return [[] for _ in range(num_parts)]
    base, extra = divmod(length, num_parts)
    result: List[List[str]] = []
    start = 0
    for i in range(num_parts):
        end = start + base + (1 if i < extra else 0)
        result.append(items[start:end])
        start = end
    return result

def _worker_run_on_gpu(gpu_index: int, files: List[str]):
    setup_logging()
    torch.set_float32_matmul_precision('high')
    device_str = f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu'
    
    try:
        if device_str.startswith('cuda:'): torch.cuda.set_device(gpu_index)
    except Exception as e:
        logging.warning(f"[GPU {gpu_index}] Failed to set CUDA device, using CPU: {e}")
        device_str = 'cpu'

    logging.info(f"[GPU {gpu_index}] Worker starting. Files to process: {len(files)}. Device: {device_str}")

    try:
        
        separator = demucs.api.Separator(model='htdemucs', device=device_str)
        
        separator = compile_separator_model(separator, gpu_index=gpu_index)

    except Exception as e:
        logging.error(f"[GPU {gpu_index}] Failed to initialize Separator: {e}")
        return

    # 第一个文件会触发JIT编译，可能较慢
    logging.info(f"[GPU {gpu_index}] Starting to process files. The first file may take longer due to JIT compilation.")
    for idx, audio_file in enumerate(files, 1):
        try:
            logging.info(f"[GPU {gpu_index}] ({idx}/{len(files)}) ==> {audio_file}")
            process_audio(audio_file, separator)
        except Exception:
            logging.exception(f"[GPU {gpu_index}] Critical error processing: {audio_file}")

def process_folder(folder_path: str):
    audio_files = _list_audio_files(folder_path)
    logging.info(f"Found {len(audio_files)} new audio files to process.")

    if not audio_files:
        logging.info("No new files to process.")
        return

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_gpus <= 1:
        device_str = 'cuda:0' if num_gpus == 1 else 'cpu'
        logging.info(f"Starting in single process mode on device: {device_str}")
        
        if device_str.startswith('cuda:'):
            try: torch.cuda.set_device(0)
            except Exception: device_str = 'cpu'
        
        # 步骤 1: 初始化 Separator
        separator = demucs.api.Separator(model='htdemucs', device=device_str)
        
        # 步骤 2: 编译模型
        separator = compile_separator_model(separator)
            
        logging.info("Starting to process files. The first file may take longer due to JIT compilation.")
        for audio_file in tqdm.tqdm(audio_files, desc="Processing audio files"):
            process_audio(audio_file, separator)
        return

    # 多GPU并行
    logging.info(f"Starting in multi-process mode with {num_gpus} GPUs.")
    chunks = _split_evenly(audio_files, num_gpus)
    processes: List[Process] = []
    for gpu_index in range(num_gpus):
        p = Process(target=_worker_run_on_gpu, args=(gpu_index, chunks[gpu_index]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    logging.info("All processes finished.")

if __name__ == '__main__':
    setup_logging()
    
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("FFmpeg not found. Please ensure it is installed and in your system's PATH.")
        exit(1)
    
    folder = '/hdd_common/gm_data' # 修改为你的目标文件夹
    process_folder(folder)