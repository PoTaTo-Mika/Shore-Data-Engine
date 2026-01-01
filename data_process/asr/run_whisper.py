import os
import json
import logging
import platform
import threading
import torch
import gc
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
from queue import Queue
try:
    from faster_whisper import WhisperModel
except ImportError:
    logging.warning("faster_whisper not found. Please install via 'pip install faster-whisper'")

# 一定要加！！！！
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# 某些环境下Faster Whisper配合多线程需要设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置根目录
ROOT_DIR = Path(__file__).parent.parent.parent
os.chdir(ROOT_DIR)

# 配置logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/whisper_transcription.log', encoding='utf-8')
    ]
)

#################辅助函数#################

def get_audio_duration(audio_path):
    """获取音频文件时长（秒）"""
    try:
        import librosa
        # 仅加载头部元数据，加快速度
        duration = librosa.get_duration(path=str(audio_path))
        return duration
    except Exception as e:
        # fallback
        try:
            import librosa
            audio_data, sample_rate = librosa.load(str(audio_path), sr=None, duration=None)
            duration = len(audio_data) / sample_rate
            return duration
        except Exception as e2:
            logging.error(f"Error getting duration for {audio_path}: {e2}")
            return None

def filter_audio_files_by_duration(audio_files, max_duration=30, min_duration=0, max_workers=None):
    # 保持原有逻辑不变
    def process_file(audio_file):
        duration = get_audio_duration(audio_file)
        if duration is not None:
            if min_duration <= duration and (max_duration is None or duration <= max_duration):
                return audio_file, duration, True
        return audio_file, duration, False
    
    filtered_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, file): file for file in audio_files}
        for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                          total=len(audio_files), 
                          desc="Filtering audio by duration", 
                          unit="file"):
            try:
                audio_file, duration, included = future.result()
                if included:
                    filtered_files.append(audio_file)
            except Exception as e:
                logging.warning(f"Error processing {future_to_file[future].name}: {e}")
    return filtered_files

################# Faster-Whisper 多卡转写 (替代原基础转写) #################

def process_into_list(folder, duration_filter=None):
    """
    使用 Faster-Whisper 进行多GPU并行转写，具备低幻觉参数配置。
    """
    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.webm', '.opus']
    folder_path = Path(folder)
    json_output_path = folder_path / "transcription.json"
    
    # 加载现有数据
    if json_output_path.exists():
        with open(json_output_path, "r", encoding='utf-8') as f:
            transcription_pairs = json.load(f)
        logging.info(f"Loaded existing transcription data with {len(transcription_pairs)} entries")
    else:
        transcription_pairs = {}

    # 1. 收集和过滤文件
    all_audio_files = []
    for audio_file in folder_path.rglob('*'):
        if audio_file.is_file() and audio_file.suffix.lower() in audio_extensions:
            all_audio_files.append(audio_file)
    
    # 排除已处理的文件
    audio_files_to_process = []
    for f in all_audio_files:
        if str(f.absolute()) not in transcription_pairs:
            audio_files_to_process.append(f)
        else:
            logging.debug(f"Skipping {f.name} (already done)")

    if not audio_files_to_process:
        logging.info("No new files to process.")
        return

    # 应用时长过滤
    if duration_filter is not None:
        min_dur, max_dur = duration_filter
        logging.info(f"Filtering {len(audio_files_to_process)} files by duration ({min_dur}-{max_dur})...")
        audio_files_to_process = filter_audio_files_by_duration(audio_files_to_process, max_duration=max_dur, min_duration=min_dur)
    
    logging.info(f"Total files to process with Faster-Whisper: {len(audio_files_to_process)}")

    # 2. 多GPU Worker 设置
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        logging.warning("No GPU detected, falling back to CPU (slow).")
        gpu_indices = [-1] # CPU
    else:
        gpu_indices = list(range(gpu_count))
        logging.info(f"Detected {gpu_count} GPUs. Launching workers...")

    # 线程安全的写锁
    write_lock = threading.Lock()
    file_queue = Queue()
    for f in audio_files_to_process:
        file_queue.put(f)

    # 进度条
    pbar = tqdm(total=len(audio_files_to_process), desc="Faster-Whisper Processing", unit="file")

    def worker_func(gpu_id):
        """每个Worker在一个独立的GPU上加载模型并处理队列"""
        device = "cuda" if gpu_id >= 0 else "cpu"
        device_index = gpu_id if gpu_id >= 0 else 0
        
        try:
            # 加载模型：large-v3-turbo
            # float16=True 可以在GPU上节省显存并加速
            logging.info(f"[GPU {gpu_id}] Loading Faster-Whisper model...")
            model = WhisperModel(
                "large-v3-turbo", 
                device=device, 
                device_index=device_index, 
                compute_type="float16",
                download_root='./checkpoints/whisper-large-v3-turbo'
            )
        except Exception as e:
            logging.error(f"[GPU {gpu_id}] Failed to load model: {e}")
            return

        while not file_queue.empty():
            try:
                audio_file = file_queue.get(block=False)
            except Exception:
                break # 队列空了

            audio_path = str(audio_file.absolute())
            
            try:
                # --- 低幻觉参数配置 ---
                # vad_filter=True: 过滤静音，防止对静音进行幻觉翻译
                # condition_on_previous_text=False: 防止之前的错误累积导致循环
                # temperature=0: 使用贪婪解码，最稳定
                segments, info = model.transcribe(
                    audio_path,
                    beam_size=5,
                    best_of=5,
                    temperature=0,
                    condition_on_previous_text=False, 
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500), # 激进一点的VAD
                    word_timestamps=False
                )
                
                # Faster-whisper返回的是generator，必须遍历才能执行
                text_segments = [segment.text for segment in segments]
                full_text = "".join(text_segments).strip()

                # 线程安全写入
                with write_lock:
                    transcription_pairs[audio_path] = full_text
                    # 每处理一个就保存（虽然频繁IO，但安全）
                    with open(json_output_path, "w", encoding='utf-8') as f:
                        json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)
                
                pbar.update(1)
                logging.debug(f"[GPU {gpu_id}] Processed {audio_file.name}")

            except Exception as e:
                logging.error(f"[GPU {gpu_id}] Error processing {audio_file.name}: {e}")
                pbar.update(1) # 即使失败也要更新进度条
                continue

    # 3. 启动线程池
    # 为什么用Thread而不是Process？
    # Faster-Whisper底层是CTranslate2 (C++)，它会释放GIL，因此多线程可以有效利用多卡。
    # 相比多进程，多线程共享内存开销小，启动快。
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_indices)) as executor:
        futures = [executor.submit(worker_func, gpu_id) for gpu_id in gpu_indices]
        concurrent.futures.wait(futures)

    pbar.close()
    logging.info(f"Faster-Whisper Transcription saved to {json_output_path}")

################# VLLM加速推理 #################

def process_into_list_vllm(folder, max_duration=30):
    try:
        import time
        from vllm import LLM, SamplingParams
        import librosa
    except ImportError as e:
        logging.error(f"Failed to import VLLM modules: {e}")
        return
    
    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.webm', '.opus']
    folder_path = Path(folder)
    json_output_path = folder_path / "transcription.json"
    
    if json_output_path.exists():
        with open(json_output_path, "r", encoding='utf-8') as f:
            transcription_pairs = json.load(f)
        logging.info(f"Loaded existing VLLM transcription data with {len(transcription_pairs)} entries")
    else:
        transcription_pairs = {}

    # 加载模型
    try:
        logging.info("Loading VLLM Whisper model...")
        llm = LLM(
            model="openai/whisper-large-v3",
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=448, # 限制上下文长度防止OOM
            max_num_seqs=200,  # 稍微调低以增加稳定性
            enforce_eager=False,
            download_dir="./checkpoints"
        )
    except Exception as e:
        logging.error(f"Failed to load VLLM model: {e}")
        return

    # --- Prompt 和 采样参数修正 ---
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=200,
        skip_special_tokens=True, # 尝试在解码层跳过
    )

    # 收集文件
    all_audio_files = []
    for audio_file in folder_path.rglob('*'):
        if audio_file.is_file() and audio_file.suffix.lower() in audio_extensions:
            all_audio_files.append(audio_file)
    
    # 过滤未处理且时长符合的文件
    files_to_process = []
    for f in all_audio_files:
        if str(f.absolute()) not in transcription_pairs:
            files_to_process.append(f)
    
    audio_files = filter_audio_files_by_duration(files_to_process, max_duration=max_duration)
    logging.info(f"VLLM Processing: {len(audio_files)} new files (<= {max_duration}s)")
    
    batch_size = 256
    
    for i in tqdm(range(0, len(audio_files), batch_size), desc="VLLM Batches", unit="batch"):
        batch_files = audio_files[i:i+batch_size]
        batch_prompts = []
        batch_file_paths = []
        
        for audio_file in batch_files:
            try:
                audio_path = str(audio_file.absolute())
                audio_data, sample_rate = librosa.load(str(audio_file), sr=16000)
                
                # --- Prompt 修正: 添加 <|notimestamps|> ---
                # VLLM Whisper通常使用特殊的Prompt格式来控制行为
                # 加入 <|notimestamps|> 可以有效抑制 <|0.00|> 的输出
                prompt_content = "<|startoftranscript|><|notimestamps|>"
                
                prompt = {
                    "prompt": prompt_content,
                    "multi_modal_data": {
                        "audio": (audio_data, sample_rate),
                    },
                }
                
                batch_prompts.append(prompt)
                batch_file_paths.append(audio_path)
            except Exception as e:
                logging.error(f"Error preparing {audio_file}: {e}")
        
        if not batch_prompts:
            continue
            
        try:
            start_time = time.time()
            outputs = llm.generate(batch_prompts, sampling_params)
            
            for idx, output in enumerate(outputs):
                if idx < len(batch_file_paths):
                    audio_path = batch_file_paths[idx]
                    # strip() 有助于去掉可能残留的空白
                    generated_text = output.outputs[0].text.strip()
                    transcription_pairs[audio_path] = generated_text
            
            with open(json_output_path, "w", encoding='utf-8') as f:
                json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logging.error(f"Error during VLLM inference: {e}")
            continue

    logging.info(f"VLLM finished. Total files: {len(transcription_pairs)}")
    
    # --- 显存清理 ---
    # VLLM 占用显存非常厉害，必须手动清理以供后续 Faster-Whisper 使用
    logging.info("Cleaning up VLLM resources...")
    from vllm.distributed.parallel_state import destroy_model_parallel
    try:
        destroy_model_parallel()
    except:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("GPU cache cleared.")

################# 主流程 #################

if __name__ == "__main__":
    data_root = "data"

    def _iter_sliced_dirs(root_dir: str):
        root_path = Path(root_dir)
        return [p for p in root_path.rglob('sliced') if p.is_dir()]

    def _process_all_sliced(root_dir: str, use_vllm: bool = False, duration_filter=None):
        sliced_dirs = _iter_sliced_dirs(root_dir)
        logging.info(f"Found {len(sliced_dirs)} 'sliced' directories")
        
        for sliced_dir in sliced_dirs:
            logging.info(f"Processing directory: {sliced_dir}")
            if use_vllm:
                process_into_list_vllm(str(sliced_dir), max_duration=30)
            else:
                process_into_list(str(sliced_dir), duration_filter=duration_filter)

    # 流程控制
    use_vllm = True
    
    if use_vllm and platform.system() != "Linux":
        print(f"警告：vLLM仅支持Linux平台，当前系统为 {platform.system()}")
        use_vllm = False

    if use_vllm:
        print("=" * 60)
        print("第一阶段：VLLM (Short Audio <= 30s) + <|notimestamps|> fix")
        print("=" * 60)
        _process_all_sliced(data_root, use_vllm=True)
        
        print("\n" + "=" * 60)
        print("第二阶段：Faster-Whisper Multi-GPU (Long Audio > 30s) + Low Hallucination")
        print("=" * 60)
        # 注意：这里我们过滤掉0-30秒的，只处理30秒以上的
        _process_all_sliced(data_root, use_vllm=False, duration_filter=(30, None))
    else:
        print("=" * 60)
        print("使用 Faster-Whisper 处理所有音频...")
        print("=" * 60)
        _process_all_sliced(data_root, use_vllm=False)
    
    print("\n" + "=" * 60)
    print("所有音频处理完成！")