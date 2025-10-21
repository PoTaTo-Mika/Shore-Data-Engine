import torch
import demucs.api
import tqdm
import os
import logging
import multiprocessing
from multiprocessing import Process
from typing import List
import subprocess
import shutil
from io import BytesIO
import soundfile as sf
import numpy as np

# 设置多进程启动方法为 spawn（必须在其他导入之前）
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

# 配置logging
def setup_logging():
    # 确保logs目录存在
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(processName)s PID:%(process)d] - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler('logs/demucs.log', encoding='utf-8')  # 输出到文件
        ]
    )

def opus_to_wav_memory(opus_path: str) -> BytesIO:
    """将opus文件转换为内存中的wav数据"""
    try:
        cmd = [
            'ffmpeg', '-i', opus_path,
            '-acodec', 'pcm_f32le',
            '-ar', '44100',
            '-ac', '2',
            '-f', 'wav',
            'pipe:1'
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        return BytesIO(result.stdout)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg转码失败: {e.stderr.decode()}")

def wav_memory_to_opus(wav_buffer: BytesIO, opus_path: str):
    """将内存中的wav数据转码为opus文件"""
    try:
        cmd = [
            'ffmpeg', '-i', 'pipe:0',
            '-acodec', 'libopus',
            '-b:a', '96k',
            '-vbr', 'on',
            '-compression_level', '10',
            '-y',
            opus_path
        ]
        wav_buffer.seek(0)
        subprocess.run(cmd, input=wav_buffer.read(), capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg转码失败: {e.stderr.decode()}")

def mark_as_finished(file_path: str):
    """标记文件为已处理完成"""
    base_name = os.path.splitext(file_path)[0]
    finished_path = base_name + '.finish'
    with open(finished_path, 'w') as f:
        f.write('processed')
    logging.info(f"已标记文件为完成: {finished_path}")

def is_finished(file_path: str) -> bool:
    """检查文件是否已经处理完成"""
    base_name = os.path.splitext(file_path)[0]
    finished_path = base_name + '.finish'
    return os.path.exists(finished_path)

def process_audio(audio_path, separator):
    """处理音频文件，全程在内存中进行"""
    
    # 检查是否已经处理过
    if is_finished(audio_path):
        logging.info(f"文件已处理过，跳过: {audio_path}")
        return
    
    logging.info(f"开始处理: {audio_path}")
    
    wav_input_buffer = None
    wav_output_buffer = None
    
    try:
        # 步骤1: opus -> wav (内存)
        wav_input_buffer = opus_to_wav_memory(audio_path)
        
        # 步骤2: 使用demucs处理wav数据
        wav_input_buffer.seek(0)
        audio_data, samplerate = sf.read(wav_input_buffer)
        
        # 转换为torch tensor格式 (channels, samples)
        if audio_data.ndim == 1:
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).float()
        else:
            audio_tensor = torch.from_numpy(audio_data.T).float()
        
        # 使用separator处理 - 返回 (origin, separated_dict)
        origin, separated = separator.separate_tensor(audio_tensor, samplerate)
        
        # 提取人声轨道
        vocals_tensor = separated['vocals']
        
        # 步骤3: 将分离后的音频保存到内存buffer
        vocals_tensor = vocals_tensor.cpu()
        vocals_np = vocals_tensor.numpy()
        
        # 确保是 (samples, channels) 格式
        if vocals_np.ndim == 2 and vocals_np.shape[0] < vocals_np.shape[1]:
            vocals_np = vocals_np.T
        elif vocals_np.ndim == 1:
            vocals_np = vocals_np[:, np.newaxis]
        
        # 创建输出buffer
        wav_output_buffer = BytesIO()
        sf.write(wav_output_buffer, vocals_np, samplerate, format='WAV', subtype='FLOAT')
        
        # 步骤4: wav -> opus (覆盖原文件)
        wav_memory_to_opus(wav_output_buffer, audio_path)
        
        # 步骤5: 标记文件为完成
        mark_as_finished(audio_path)
        logging.info(f"成功处理: {audio_path}")
            
    except Exception as e:
        logging.exception(f"处理失败: {audio_path}, 错误: {e}")
    finally:
        # 清理内存buffer
        if wav_input_buffer:
            wav_input_buffer.close()
        if wav_output_buffer:
            wav_output_buffer.close()

def _list_audio_files(folder_path: str) -> List[str]:
    audio_files: List[str] = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.opus') and not file.endswith('.finish'):
                full_path = os.path.join(root, file)
                # 检查是否已经处理过
                if not is_finished(full_path):
                    audio_files.append(full_path)
    return audio_files

def _split_evenly(items: List[str], num_parts: int) -> List[List[str]]:
    if num_parts <= 0:
        return [items]
    length = len(items)
    if length == 0:
        return [[] for _ in range(num_parts)]
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
    
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device_str = f'cuda:{gpu_index}' if use_cuda else 'cpu'
    
    try:
        if use_cuda:
            torch.cuda.set_device(gpu_index)
    except Exception as e:
        logging.warning(f"[GPU {gpu_index}] 设置CUDA设备失败，使用CPU: {e}")
        device_str = 'cpu'

    logging.info(f"[GPU {gpu_index}] 启动，待处理文件数: {len(files)}, 设备: {device_str}")

    try:
        separator = demucs.api.Separator(model='htdemucs', device=device_str)
    except TypeError:
        if device_str.startswith('cuda:'):
            try:
                torch.cuda.set_device(int(device_str.split(':')[1]))
            except Exception:
                pass
        separator = demucs.api.Separator(model='htdemucs')

    for idx, audio_file in enumerate(files, 1):
        try:
            logging.info(f"[GPU {gpu_index}] ({idx}/{len(files)}) 处理: {audio_file}")
            process_audio(audio_file, separator)
        except Exception as e:
            logging.exception(f"[GPU {gpu_index}] 处理失败: {audio_file}")

def process_folder(folder_path: str):
    audio_files = _list_audio_files(folder_path)
    logging.info(f"Found {len(audio_files)} audio files to process")

    if not audio_files:
        logging.info("没有发现可处理的音频文件")
        return

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_gpus <= 1:
        device_str = 'cuda:0' if num_gpus == 1 else 'cpu'
        logging.info(f"单进程模式启动，设备: {device_str}")
        
        try:
            if device_str.startswith('cuda:'):
                torch.cuda.set_device(0)
        except Exception:
            device_str = 'cpu'
            
        try:
            separator = demucs.api.Separator(model='htdemucs', device=device_str)
        except TypeError:
            separator = demucs.api.Separator(model='htdemucs')
            
        for audio_file in tqdm.tqdm(audio_files, desc="Processing audio files"):
            process_audio(audio_file, separator)
        return

    # 多GPU并行
    chunks = _split_evenly(audio_files, num_gpus)
    processes: List[Process] = []
    for gpu_index in range(num_gpus):
        p = Process(target=_worker_run_on_gpu, args=(gpu_index, chunks[gpu_index]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    setup_logging()
    
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("未找到ffmpeg，请确保ffmpeg已安装并在PATH中")
        exit(1)
    
    folder = '/hdd_common/gm_data'
    process_folder(folder)