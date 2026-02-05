import os
import json
import logging
import torch
import sys
from tqdm import tqdm
from pathlib import Path

from qwen_asr import Qwen3ASRModel

# 配置logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/qwen_asr_transcription.log', encoding='utf-8')
    ]
)

def process_into_list(folder, model):
    """
    遍历文件夹中的音频文件并使用 Qwen3ASRModel 进行转写，结果保存至 transcription.json。
    """
    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.webm', '.opus']
    folder_path = Path(folder)
    json_output_path = folder_path / "transcription.json"
    
    # 1. 加载现有数据，支持断点续传
    if json_output_path.exists():
        try:
            with open(json_output_path, "r", encoding='utf-8') as f:
                transcription_pairs = json.load(f)
            logging.info(f"Loaded existing transcription data with {len(transcription_pairs)} entries from {folder}")
        except Exception as e:
            logging.error(f"Error loading {json_output_path}: {e}")
            transcription_pairs = {}
    else:
        transcription_pairs = {}

    # 2. 收集待处理的音频文件
    all_audio_files = []
    for audio_file in folder_path.rglob('*'):
        if audio_file.is_file() and audio_file.suffix.lower() in audio_extensions:
            all_audio_files.append(audio_file)
    
    # 排除已处理的文件
    audio_files_to_process = []
    for f in all_audio_files:
        abs_path = str(f.absolute())
        if abs_path not in transcription_pairs:
            audio_files_to_process.append(f)
    
    if not audio_files_to_process:
        logging.info(f"No new files to process in {folder}.")
        return

    logging.info(f"Processing {len(audio_files_to_process)} files in {folder}")

    # 3. 分批调用模型进行推理
    # 使用模型配置的 batch_size，防止内存溢出
    batch_size = getattr(model, 'max_inference_batch_size', 32)
    if batch_size <= 0: batch_size = 32
    
    for i in tqdm(range(0, len(audio_files_to_process), batch_size), desc=f"ASR: {folder_path.name}"):
        batch = audio_files_to_process[i:i+batch_size]
        audio_paths = [str(f.absolute()) for f in batch]
        
        try:
            # 调用封装好的 transcribe 方法
            results = model.transcribe(
                audio=audio_paths,
                language=None,  # 设为 None 以开启自动语言检测
            )
            
            # 更新结果
            for path, res in zip(audio_paths, results):
                transcription_pairs[path] = res.text
            
            # 实时保存，防止崩溃丢失进度
            with open(json_output_path, "w", encoding='utf-8') as f:
                json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logging.error(f"Error processing batch starting at {batch[0].name}: {e}")

def _iter_sliced_dirs(root_dir: str):
    """
    寻找 root_dir 下所有名为 'sliced' 的目录
    """
    root_path = Path(root_dir)
    return [p for p in root_path.rglob('sliced') if p.is_dir()]

def _process_all_sliced(root_dir: str, model):
    """
    核心批量处理逻辑
    """
    sliced_dirs = _iter_sliced_dirs(root_dir)
    logging.info(f"Found {len(sliced_dirs)} 'sliced' directories in {root_dir}")
    
    for sliced_dir in sliced_dirs:
        logging.info(f"Starting processing directory: {sliced_dir}")
        process_into_list(str(sliced_dir), model)

if __name__ == "__main__":
    # 配置数据根目录
    DATA_ROOT = "data"
    
    # 初始化 Qwen3-ASR 模型
    # 参考原始 MVP 配置
    logging.info("Initializing Qwen3-ASR model...")

    if sys.platform == "linux":
        model = Qwen3ASRModel.LLM(
            model="./checkpoints/Qwen3-ASR-1.7B",
            tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            max_inference_batch_size=128 # vLLM 吞吐量大，可以设置得大一些
        )
    else:
        model = Qwen3ASRModel.from_pretrained(
            "./checkpoints/Qwen3-ASR-1.7B",
            dtype=torch.bfloat16,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            max_inference_batch_size=32,
            max_new_tokens=256,
        )
    
    logging.info("Starting batch processing...")
    _process_all_sliced(DATA_ROOT, model)
    logging.info("All tasks completed successfully!")

