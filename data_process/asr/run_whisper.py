import os
import json
import logging
import platform
import gc
import math
import multiprocessing
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import torch
from faster_whisper import WhisperModel

# ================= 配置与初始化 =================
# 必须设置为 spawn，否则 VLLM 和 CTranslate2 会冲突
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# 限制每个进程内的 CPU 线程数，防止多显卡进程争抢 CPU 导致死锁或变慢
os.environ["OMP_NUM_THREADS"] = "1"

# 设置根目录
ROOT_DIR = Path(__file__).parent.parent.parent
try:
    os.chdir(ROOT_DIR)
except:
    pass

# 配置 logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Process:%(processName)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/whisper_transcription.log', encoding='utf-8')
    ]
)

################# 辅助函数 #################

def clear_gpu_memory():
    """强制清理 GPU 显存"""
    logging.info("Cleaning GPU memory & Garbage Collection...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def get_audio_duration(audio_path):
    """获取音频文件时长（秒），优先读取元数据"""
    try:
        import librosa
        # 仅读取头部元数据，速度快
        duration = librosa.get_duration(path=str(audio_path))
        return duration
    except Exception:
        try:
            # 失败兜底
            import librosa
            y, sr = librosa.load(str(audio_path), sr=None)
            return len(y) / sr
        except Exception as e:
            logging.error(f"Error getting duration for {audio_path}: {e}")
            return None

def filter_audio_files_by_duration(audio_files, max_duration=30, min_duration=0, max_workers=None):
    """多线程并发检查文件时长"""
    def process_file(audio_file):
        duration = get_audio_duration(audio_file)
        if duration is not None:
            if min_duration <= duration and (max_duration is None or duration <= max_duration):
                return audio_file
        return None
    
    filtered_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        futures = {executor.submit(process_file, f): f for f in audio_files}
        # 获取结果
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                filtered_files.append(res)
    
    return filtered_files

################# Stage 2: 基础转写 (多进程 Worker) #################

def process_single_folder_basic(folder, model, device_id, duration_filter=None):
    """
    单个文件夹的处理逻辑 (运行在子进程中)
    """
    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.webm', '.opus']
    folder_path = Path(folder)
    json_output_path = folder_path / "transcription.json"
    
    # 1. 加载或初始化 JSON
    if json_output_path.exists():
        try:
            with open(json_output_path, "r", encoding='utf-8') as f:
                transcription_pairs = json.load(f)
        except:
            transcription_pairs = {}
    else:
        transcription_pairs = {}

    # 2. 扫描文件
    all_files = [p for p in folder_path.rglob('*') if p.is_file() and p.suffix.lower() in audio_extensions]
    
    # 3. 筛选未处理的文件
    files_to_process = [f for f in all_files if str(f.absolute()) not in transcription_pairs]
    
    # 4. 如果有时长过滤需求（比如只处理 >30s 的）
    # 注意：为了效率，建议先不做时长检查，因为文件可能很多。
    # 这里根据 logic: 如果之前 VLLM 跑过了，理论上 json 里已经有了。
    # 这里主要处理剩下的。为了保险，我们可以在循环里做简单的 check。
    
    if not files_to_process:
        return

    logging.info(f"[GPU-{device_id}] Processing {folder_path.name}: {len(files_to_process)} new files")

    for audio_file in files_to_process:
        try:
            # 再次检查时长（可选，视性能需求而定）
            # 如果 duration_filter 存在，我们在这里做检查，避免读取大量无效文件
            if duration_filter:
                dur = get_audio_duration(audio_file)
                min_d, max_d = duration_filter
                if dur is None: continue
                if dur < min_d: continue
                if max_d is not None and dur > max_d: continue

            audio_path = str(audio_file.absolute())
            
            # === Faster-Whisper 推理 ===
            # 抗幻觉参数集
            segments, info = model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True, # 关键：过滤静音
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=False, # 关键：防止死循环
                repetition_penalty=1.2, # 关键：防止复读机
                temperature=[0.0, 0.2] 
            )

            text_segments = [segment.text for segment in segments]
            transcription_text = "".join(text_segments).strip()

            transcription_pairs[audio_path] = transcription_text

            # 实时保存（防止崩溃丢失）
            with open(json_output_path, "w", encoding='utf-8') as f:
                json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logging.error(f"[GPU-{device_id}] Error in {audio_file.name}: {e}")

def worker_basic_process(sliced_dirs, gpu_id):
    """
    Stage 2 的工作进程入口
    """
    device_str = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    logging.info(f"Worker launched on {device_str} for {len(sliced_dirs)} folders.")
    
    try:
        # 在进程内加载模型，指定 device_index
        # 使用 float16 提升速度
        compute_type = "float16"
        
        model = WhisperModel(
            "large-v3", 
            device="cuda" if torch.cuda.is_available() else "cpu",
            device_index=gpu_id, 
            compute_type=compute_type,
            download_root='./checkpoints/whisper-large-v3'
        )
        
        # 遍历分配给该进程的文件夹列表
        for idx, folder in enumerate(sliced_dirs):
            process_single_folder_basic(folder, model, gpu_id, duration_filter=(30, None))
            
            if (idx + 1) % 5 == 0:
                gc.collect() # 定期清理 Python 垃圾

    except Exception as e:
        logging.error(f"Worker {gpu_id} failed: {e}")
    finally:
        logging.info(f"Worker {gpu_id} finished.")

################# Stage 1: VLLM 加速推理 #################

def process_stage_1_vllm(all_sliced_dirs):
    """
    使用 VLLM 处理短音频。
    VLLM 本身使用 Tensor Parallelism (占用所有卡运行一个实例)，
    所以这里不需要多进程，直接单进程跑即可。
    """
    try:
        from vllm import LLM, SamplingParams
        import librosa
        from vllm.distributed.parallel_state import destroy_model_parallel
    except ImportError:
        logging.error("VLLM not installed.")
        return

    logging.info("Initializing VLLM for Stage 1 (Short Audio)...")
    
    # 初始化 VLLM (占用所有 GPU)
    try:
        llm = LLM(
            model="openai/whisper-large-v3",
            tensor_parallel_size=torch.cuda.device_count(), # 占满所有卡
            max_model_len=448,
            max_num_seqs=256,
            gpu_memory_utilization=0.9,
            download_dir="./checkpoints"
        )
    except Exception as e:
        logging.error(f"VLLM Init Failed: {e}")
        return

    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=200)
    
    # 收集所有任务
    # 为了提高 VLLM 效率，我们可以跨文件夹收集所有短文件做成一个巨大的 Batch
    # 但为了保证断点续传的 JSON 写入安全，还是按文件夹粒度处理比较稳妥，或者按 batch 写入
    
    # 这里简化逻辑：遍历所有文件夹，把没做的短音频挑出来
    for folder in tqdm(all_sliced_dirs, desc="VLLM Processing Directories"):
        folder_path = Path(folder)
        json_path = folder_path / "transcription.json"
        
        # 读取现有进度
        if json_path.exists():
            with open(json_path, "r", encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
            
        # 找文件
        files = [p for p in folder_path.rglob('*') if p.suffix.lower() in ['.mp3','.wav','.m4a','.opus'] and p.is_file()]
        targets = [f for f in files if str(f.absolute()) not in data]
        
        # 过滤短音频 (<=30s)
        # 注意：这里是单线程过滤，如果文件巨多可能会慢，但通常 sliced 文件夹内文件不会特别多
        batch_prompts = []
        batch_paths = []
        
        for f in targets:
            try:
                # 快速检查时长
                if get_audio_duration(f) <= 30.0: # 稍微放宽一点点
                    y, sr = librosa.load(str(f), sr=16000)
                    batch_prompts.append({
                        "prompt": "<|startoftranscript|>",
                        "multi_modal_data": {"audio": (y, sr)},
                    })
                    batch_paths.append(str(f.absolute()))
            except:
                pass
        
        if not batch_prompts:
            continue
            
        # 推理
        try:
            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            
            # 更新结果
            for i, out in enumerate(outputs):
                data[batch_paths[i]] = out.outputs[0].text.strip()
            
            # 写入
            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logging.error(f"VLLM Error in {folder.name}: {e}")

    # === 清理 VLLM ===
    logging.info("Stage 1 Complete. Destroying VLLM instance...")
    destroy_model_parallel()
    del llm
    clear_gpu_memory()

################# 主流程 #################

def main():
    data_root = "data"
    all_sliced_dirs = [p for p in Path(data_root).rglob('sliced') if p.is_dir()]
    
    if not all_sliced_dirs:
        logging.warning("No 'sliced' directories found!")
        return

    # ================= Stage 1: VLLM (Short Audio) =================
    # 只有 Linux 且有卡才跑 VLLM
    run_vllm = platform.system() == "Linux" and torch.cuda.is_available()
    
    if run_vllm:
        print("\n" + "="*60)
        print("STAGE 1: VLLM Processing (Using ALL GPUs as one unit)")
        print("="*60)
        process_stage_1_vllm(all_sliced_dirs)
    else:
        logging.info("Skipping VLLM (Not Linux or No GPU).")

    # ================= Stage 2: Faster-Whisper (Long/Remaining) =================
    # 使用多进程，每张卡跑一个 Worker
    print("\n" + "="*60)
    print("STAGE 2: Faster-Whisper (Multi-GPU Processing)")
    print("="*60)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        num_workers = 1
        logging.warning("No GPU detected. Using 1 CPU worker.")
    else:
        num_workers = num_gpus
        logging.info(f"Spawning {num_workers} workers for {num_workers} GPUs.")

    # 任务分配算法：把文件夹列表切分成 num_workers 份
    chunks = [[] for _ in range(num_workers)]
    for i, d in enumerate(all_sliced_dirs):
        chunks[i % num_workers].append(d)

    processes = []
    for i in range(num_workers):
        if not chunks[i]: continue
        
        # 启动进程
        # i 对应 gpu_id
        p = multiprocessing.Process(
            target=worker_basic_process,
            args=(chunks[i], i),
            name=f"Worker-GPU-{i}"
        )
        p.start()
        processes.append(p)

    # 等待所有进程结束
    for p in processes:
        p.join()

    print("\n" + "="*60)
    print("ALL PROCESSING DONE.")
    print("="*60)

    try:
        from tools.calculate_time import calculate_time as cct
        cct(data_root)
    except:
        pass

if __name__ == "__main__":
    main()