from funasr import AutoModel
import os
import json
import logging
from tqdm import tqdm
from pathlib import Path
import torch
import multiprocessing as mp
from typing import List, Dict
import math
from datetime import datetime

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


class GPUWorker:
    """GPU工作进程"""
    
    def __init__(self, gpu_id: int, checkpoint_dir: str = "./checkpoints", lang: str = 'zh'):
        self.gpu_id = gpu_id
        self.checkpoint_dir = checkpoint_dir
        self.asr_model = None
        self.punc_model = None
        self.lang = lang
        
    def initialize_models(self):
        """初始化模型"""
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.cuda.set_device(0)
        
        logging.info(f"GPU {self.gpu_id}: Loading models...")
        
        # 根据语言
        if self.lang == "chuan":
            asr_model_name = "paraformer-chuan"
        else: 
            asr_model_name = "paraformer-zh"

        # 初始化ASR模型
        self.asr_model = AutoModel(
            model=f"{self.checkpoint_dir}/{asr_model_name}",
            device="cuda:0"
        )

        # 初始化标点模型
        self.punc_model = AutoModel(
            model=f"{self.checkpoint_dir}/ct-punc",
            device="cuda:0"
        )

        self.asr_model.model = torch.compile(self.asr_model.model)
        self.punc_model.model = torch.compile(self.punc_model.model)
        
        logging.info(f"GPU {self.gpu_id}: Models loaded")
    
    def process_batch(self, batch_paths: List[str], batch_size: int = 16) -> Dict[str, str]:
        """处理音频文件批次"""
        results = {}
        
        for start in range(0, len(batch_paths), batch_size):
            sub_batch = batch_paths[start:start + batch_size]
            
            try:
                # ASR批量推理
                asr_results = self.asr_model.generate(
                    input=sub_batch,
                    batch_size=len(sub_batch),
                    disable_pbar=True
                )
                
                if not isinstance(asr_results, list):
                    asr_results = [asr_results]
                
                # 标点修复
                for path, res in zip(sub_batch, asr_results):
                    asr_text = self._extract_text(res)
                    final_text = self._apply_punctuation(asr_text) if asr_text else ""
                    results[path] = final_text
                    
            except Exception as e:
                logging.error(f"GPU {self.gpu_id}: Batch error: {e}")
                for path in sub_batch:
                    if path not in results:
                        results[path] = ""
        
        return results
    
    def _apply_punctuation(self, text: str) -> str:
        """应用标点修复"""
        try:
            punc_res = self.punc_model.generate(input=text)
            return self._extract_text(punc_res) or text
        except Exception as e:
            logging.warning(f"GPU {self.gpu_id}: Punctuation failed: {e}")
            return text
    
    @staticmethod
    def _extract_text(res) -> str:
        """从模型输出提取文本"""
        if isinstance(res, str):
            return res.strip()
        elif isinstance(res, dict) and 'text' in res:
            return str(res['text']).strip()
        elif isinstance(res, list) and res:
            first_item = res[0]
            if isinstance(first_item, dict) and 'text' in first_item:
                return str(first_item['text']).strip()
            elif isinstance(first_item, str):
                return first_item.strip()
        return ""


def worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, 
                   checkpoint_dir: str, batch_size: int, lang: str):
    """工作进程函数"""
    try:
        worker = GPUWorker(gpu_id, checkpoint_dir, lang)
        worker.initialize_models()
        
        while True:
            task = task_queue.get()
            if task is None:  # 结束信号
                break
            
            batch_id, batch_paths = task
            logging.info(f"GPU {gpu_id}: Processing batch {batch_id} ({len(batch_paths)} files)")
            
            results = worker.process_batch(batch_paths, batch_size)
            result_queue.put((gpu_id, batch_id, results))
            
    except Exception as e:
        logging.error(f"GPU {gpu_id}: Worker error: {e}")
        result_queue.put((gpu_id, -1, {}))


def split_into_chunks(items: List, num_chunks: int) -> List[List]:
    """分割列表为指定数量的块"""
    chunk_size = math.ceil(len(items) / num_chunks)
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def process_folder_parallel(folder: str, num_gpus: int = 8, num_mega_batches: int = 10,
                           batch_size: int = 8, checkpoint_dir: str = "./checkpoints", lang: str = 'zh'):
    """并行处理文件夹音频文件"""
    
    audio_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.webm', '.opus'}
    folder_path = Path(folder)
    json_output_path = folder_path / "transcription.json"
    temp_results_dir = folder_path / "temp_results"
    temp_results_dir.mkdir(exist_ok=True)
    
    # 加载已有结果
    transcription_pairs = {}
    if json_output_path.exists():
        with open(json_output_path, "r", encoding='utf-8') as f:
            transcription_pairs = json.load(f)
        logging.info(f"Loaded {len(transcription_pairs)} existing entries")
    
    # 收集待处理文件
    audio_files = [p for p in folder_path.rglob('*') 
                   if p.is_file() and p.suffix.lower() in audio_extensions]
    pending_files = [str(p.absolute()) for p in audio_files 
                     if str(p.absolute()) not in transcription_pairs]
    
    logging.info(f"Found {len(audio_files)} audio files, {len(pending_files)} pending")
    
    if not pending_files:
        logging.info("No files to process")
        return
    
    # 处理每个大批次
    mega_batches = split_into_chunks(pending_files, num_mega_batches)
    
    for mega_batch_idx, mega_batch in enumerate(mega_batches):
        logging.info(f"Processing mega batch {mega_batch_idx + 1}/{len(mega_batches)} "
                    f"({len(mega_batch)} files)")
        
        # 创建并行处理环境
        task_queue, result_queue = mp.Queue(), mp.Queue()
        mini_batches = split_into_chunks(mega_batch, num_gpus * 10)
        
        # 填充任务队列
        for batch_id, batch in enumerate(mini_batches):
            task_queue.put((batch_id, batch))
        for _ in range(num_gpus):
            task_queue.put(None)
        
        # 启动工作进程
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=worker_process,
                args=(gpu_id, task_queue, result_queue, checkpoint_dir, batch_size, lang)
            )
            p.start()
            processes.append(p)
        
        # 收集结果
        completed_batches = 0
        with tqdm(total=len(mini_batches), desc=f"Mega batch {mega_batch_idx + 1}") as pbar:
            while completed_batches < len(mini_batches):
                gpu_id, batch_id, results = result_queue.get()
                
                if batch_id == -1:  # 错误信号
                    logging.error(f"GPU {gpu_id} error")
                    continue
                
                # 保存临时结果
                temp_file = temp_results_dir / f"gpu_{gpu_id}_batch_{batch_id}.json"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                transcription_pairs.update(results)
                completed_batches += 1
                pbar.update(1)
        
        # 清理进程
        for p in processes:
            p.join()
        
        # 保存进度
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)
    
    # 最终合并和清理
    _finalize_processing(transcription_pairs, json_output_path, temp_results_dir)


def _finalize_processing(transcription_pairs: Dict, json_output_path: Path, temp_results_dir: Path):
    """最终处理步骤"""
    # 合并临时结果
    for temp_file in temp_results_dir.glob("*.json"):
        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                transcription_pairs.update(json.load(f))
        except Exception as e:
            logging.error(f"Failed to merge {temp_file}: {e}")
    
    # 保存最终结果
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)
    
    # 清理临时文件
    try:
        import shutil
        shutil.rmtree(temp_results_dir)
        logging.info("Temporary files cleaned")
    except Exception as e:
        logging.warning(f"Cleanup failed: {e}")
    
    logging.info(f"Transcription completed! Total: {len(transcription_pairs)} files")


def process_sliced_parallel(root_dir: str = "data", num_gpus: int = 8, 
                                num_mega_batches: int = 10, batch_size: int = 16, lang: str = "zh"):
    """并行处理所有sliced目录"""
    root_path = Path(root_dir)
    sliced_dirs = [p for p in root_path.rglob('sliced') if p.is_dir()]
    
    logging.info(f"Found {len(sliced_dirs)} 'sliced' directories")
    
    for idx, sliced_dir in enumerate(sliced_dirs, 1):
        logging.info(f"Processing {idx}/{len(sliced_dirs)}: {sliced_dir}")
        
        process_folder_parallel(
            str(sliced_dir),
            num_gpus=num_gpus,
            num_mega_batches=num_mega_batches,
            batch_size=batch_size,
            lang=lang
        )


def load_config(config_path="configs/funasr_config.json"):
    """从JSON文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using default values")
        return {
            "num_gpus": 8,
            "num_mega_batches": 10,
            "batch_size": 16,
            "data_root": "data"
        }
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing config file: {e}")
        raise

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # 从JSON文件加载配置
    config = load_config()
    
    # 使用配置参数
    NUM_GPUS = config["num_gpus"]
    NUM_MEGA_BATCHES = config["num_mega_batches"]
    BATCH_SIZE = config["batch_size"]
    DATA_ROOT = config["data_root"]
    LANG = config["lang"]
    
    # 开始处理
    start_time = datetime.now()
    logging.info(f"Starting parallel processing with {NUM_GPUS} GPUs")
    
    process_sliced_parallel(
        root_dir=DATA_ROOT,
        num_gpus=NUM_GPUS,
        num_mega_batches=NUM_MEGA_BATCHES,
        batch_size=BATCH_SIZE,
        lang=LANG
    )
    
    duration = datetime.now() - start_time
    logging.info(f"Total processing time: {duration}")