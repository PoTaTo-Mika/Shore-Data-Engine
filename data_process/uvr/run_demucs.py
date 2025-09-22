import torch
import demucs.api
import tqdm
import os
import logging
from multiprocessing import Process
from typing import List

# 配置logging
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

def process_audio(audio_path, separator):
    origin, separated = separator.separate_audio_file(audio_path)
    logging.info(f"Processing {audio_path}")
    if 'vocals' in separated:
        demucs.api.save_audio(
            wav=separated['vocals'],
            path=audio_path,
            samplerate=separator.samplerate,
            bits_per_sample=32
        )
    logging.info(f"Successfully processed {audio_path}")

def _list_audio_files(folder_path: str) -> List[str]:
    audio_files: List[str] = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
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
    # 选择设备
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device_str = f'cuda:{gpu_index}' if use_cuda else 'cpu'
    try:
        if use_cuda:
            torch.cuda.set_device(gpu_index)
    except Exception as e:
        logging.warning(f"[GPU {gpu_index}] 设置CUDA设备失败，将尝试CPU: {e}")
        device_str = 'cpu'

    logging.info(f"[GPU {gpu_index}] 启动，待处理文件数: {len(files)}, 设备: {device_str}")

    # 加载分离器（尽量绑定到指定设备）
    separator = None
    try:
        separator = demucs.api.Separator(model='htdemucs_ft', device=device_str)  # type: ignore[arg-type]
    except TypeError:
        # 兼容旧版本API不接受device参数的情况
        if device_str.startswith('cuda:'):
            try:
                torch.cuda.set_device(int(device_str.split(':')[1]))
            except Exception:
                pass
        separator = demucs.api.Separator(model='htdemucs_ft')

    for idx, audio_file in enumerate(files, 1):
        try:
            logging.info(f"[GPU {gpu_index}] ({idx}/{len(files)}) 处理: {audio_file}")
            process_audio(audio_file, separator)
        except Exception as e:
            logging.exception(f"[GPU {gpu_index}] 处理失败: {audio_file}, 错误: {e}")

def process_folder(folder_path: str):
    # 收集所有音频文件
    audio_files = _list_audio_files(folder_path)
    logging.info(f"Found {len(audio_files)} audio files to process")

    if not audio_files:
        logging.info("没有发现可处理的音频文件，退出。")
        return

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_gpus <= 1:
        # 回退到单进程（单GPU或CPU）
        device_str = 'cuda:0' if num_gpus == 1 else 'cpu'
        logging.info(f"单进程模式启动，设备: {device_str}")
        try:
            if device_str.startswith('cuda:'):
                torch.cuda.set_device(0)
        except Exception:
            device_str = 'cpu'
        try:
            separator = demucs.api.Separator(model='htdemucs_ft', device=device_str)  # type: ignore[arg-type]
        except TypeError:
            separator = demucs.api.Separator(model='htdemucs_ft')
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
        
# Linux 下直接执行
folder = 'data'
process_folder(folder)


