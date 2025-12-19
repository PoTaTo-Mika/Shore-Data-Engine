import os
import json
import logging
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from typing import List, Dict
import math
from datetime import datetime
from funasr import AutoModel
import torch

# 配置日志
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/funasr_transcription.log', encoding='utf-8')
    ]
)

# ================= 配置区域 =================
DEFAULT_CONFIG = {
    "num_gpus": 8,
    "num_mega_batches": 1,
    # 注意：这里的 batch_size 仅表示 GPU 进程一次从队列里取多少个文件来减少通信开销
    # 实际送入模型时会被强制设为 1
    "batch_size": 16,  
    "data_root": "data",
    "model_dir": "./checkpoints/Fun-ASR-Nano-2512",
    "remote_code_path": "./Fun-ASR-main/model.py",
    "lang": "zh",      # zh, en, ja, auto
    "itn": True        # 是否进行逆文本标准化
}
# ===========================================

class GPUWorker:
    """GPU工作进程"""
    
    def __init__(self, gpu_id: int, config: Dict):
        self.gpu_id = gpu_id
        self.config = config
        self.model = None
        
    def initialize_models(self):
        """初始化模型"""
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        
        logging.info(f"GPU {self.gpu_id}: Loading Fun-ASR model...")
        
        try:
            self.model = AutoModel(
                model=self.config['model_dir'],
                trust_remote_code=True,
                remote_code=self.config['remote_code_path'],
                device="cuda:0", 
                disable_update=True
            )
            logging.info(f"GPU {self.gpu_id}: Model loaded successfully.")
        except Exception as e:
            logging.error(f"GPU {self.gpu_id}: Failed to load model: {e}")
            raise e
    
    def process_batch(self, batch_paths: List[str]) -> Dict[str, str]:
        """处理音频文件批次"""
        results = {}
        
        # --- 核心修改 ---
        # 即使 batch_paths 包含多个文件，我们也必须一个一个送入模型
        # 因为 model.py 抛出了 NotImplementedError("batch decoding is not implemented")
        for audio_path in batch_paths:
            try:
                # 强制 batch_size=1
                # input 必须是一个列表，包含一个路径
                res_list = self.model.generate(
                    input=[audio_path], 
                    cache={}, 
                    batch_size=1,  # ！！！强制为 1 ！！！
                    language=self.config['lang'],
                    itn=self.config['itn']
                )
                
                # 提取结果
                if res_list and len(res_list) > 0:
                    text = self._extract_text(res_list[0])
                    results[audio_path] = text
                else:
                    results[audio_path] = ""
                    
            except Exception as e:
                logging.error(f"GPU {self.gpu_id}: Error processing {os.path.basename(audio_path)}: {e}")
                results[audio_path] = ""
        
        return results
    
    @staticmethod
    def _extract_text(res) -> str:
        """从 Fun-ASR 输出提取文本"""
        if isinstance(res, dict):
            return str(res.get('text', '')).strip()
        elif isinstance(res, str):
            return res.strip()
        return ""


def worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, config: Dict):
    """工作进程函数"""
    try:
        worker = GPUWorker(gpu_id, config)
        worker.initialize_models()
        
        while True:
            task = task_queue.get()
            if task is None:
                break
            
            batch_id, batch_paths = task
            # 执行推理
            results = worker.process_batch(batch_paths)
            result_queue.put((gpu_id, batch_id, results))
            
    except Exception as e:
        logging.error(f"GPU {gpu_id}: Worker fatal error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put((gpu_id, -1, {}))


def split_into_chunks(items: List, num_chunks: int) -> List[List]:
    if num_chunks <= 0: return [items]
    chunk_size = math.ceil(len(items) / num_chunks)
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def process_folder_parallel(folder: str, config: Dict):
    """并行处理文件夹音频文件"""
    num_gpus = config['num_gpus']
    num_mega_batches = config['num_mega_batches']
    
    # 允许的音频后缀
    audio_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.webm', '.opus'}
    folder_path = Path(folder)
    json_output_path = folder_path / "transcription.json"
    temp_results_dir = folder_path / "temp_results"
    temp_results_dir.mkdir(exist_ok=True)
    
    # 加载已有结果
    transcription_pairs = {}
    if json_output_path.exists():
        try:
            with open(json_output_path, "r", encoding='utf-8') as f:
                transcription_pairs = json.load(f)
            logging.info(f"Loaded {len(transcription_pairs)} existing entries.")
        except json.JSONDecodeError:
            pass

    # 收集文件
    audio_files = [p for p in folder_path.rglob('*') 
                   if p.is_file() and p.suffix.lower() in audio_extensions]
    
    pending_files = [str(p.absolute()) for p in audio_files 
                     if str(p.absolute()) not in transcription_pairs]
    
    logging.info(f"Folder: {folder} | Total: {len(audio_files)} | Pending: {len(pending_files)}")
    
    if not pending_files:
        return
    
    mega_batches = split_into_chunks(pending_files, num_mega_batches)
    
    for mega_batch_idx, mega_batch in enumerate(mega_batches):
        if not mega_batch: continue
        
        logging.info(f"Processing batch block {mega_batch_idx + 1}/{len(mega_batches)} "
                    f"({len(mega_batch)} files)")
        
        task_queue, result_queue = mp.Queue(), mp.Queue()
        
        # 这里的 task_chunk_size 决定了队列里一个任务包包含多少个文件
        # 虽然这里可能是 16，但 Worker 内部会 1 个 1 个跑，不会报错
        task_chunk_size = config['batch_size'] 
        mini_tasks = [mega_batch[i:i + task_chunk_size] for i in range(0, len(mega_batch), task_chunk_size)]
        
        for batch_id, batch in enumerate(mini_tasks):
            task_queue.put((batch_id, batch))
        
        for _ in range(num_gpus):
            task_queue.put(None)
        
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(target=worker_process, args=(gpu_id, task_queue, result_queue, config))
            p.start()
            processes.append(p)
        
        completed_tasks = 0
        with tqdm(total=len(mini_tasks), desc=f"Progress") as pbar:
            while completed_tasks < len(mini_tasks):
                gpu_id, batch_id, results = result_queue.get()
                
                if batch_id == -1:
                    completed_tasks += 1
                    pbar.update(1)
                    continue
                
                # 保存临时结果
                temp_file = temp_results_dir / f"gpu_{gpu_id}_mb_{mega_batch_idx}_b_{batch_id}.json"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                transcription_pairs.update(results)
                completed_tasks += 1
                pbar.update(1)
        
        for p in processes:
            p.join()
        
        # 保存Checkpoint
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)
    
    _finalize_processing(transcription_pairs, json_output_path, temp_results_dir)


def _finalize_processing(transcription_pairs: Dict, json_output_path: Path, temp_results_dir: Path):
    for temp_file in temp_results_dir.glob("*.json"):
        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                transcription_pairs.update(json.load(f))
        except Exception:
            pass
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)
    
    try:
        import shutil
        shutil.rmtree(temp_results_dir)
    except Exception:
        pass


def process_sliced_parallel(root_dir: str, config: Dict):
    root_path = Path(root_dir)
    sliced_dirs = [p for p in root_path.rglob('sliced') if p.is_dir()]
    
    logging.info(f"Found {len(sliced_dirs)} 'sliced' directories")
    
    for idx, sliced_dir in enumerate(sliced_dirs, 1):
        logging.info(f"Processing {idx}/{len(sliced_dirs)}: {sliced_dir}")
        process_folder_parallel(str(sliced_dir), config)


def load_config_from_file(config_path="configs/funasr_config.json"):
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config.update(json.load(f))
        except Exception:
            pass
    return config

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    config = load_config_from_file()
    
    # 打印配置
    print(json.dumps(config, indent=2))
    
    start_time = datetime.now()
    process_sliced_parallel(root_dir=config["data_root"], config=config)
    
    logging.info(f"Total time: {datetime.now() - start_time}")
    
    try:
        from tools.calculate_time import calculate_time as cct
        cct(config["data_root"])
    except ImportError:
        pass