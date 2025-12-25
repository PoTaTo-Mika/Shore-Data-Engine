import whisper as wsp
import os
import json
import logging
import platform
from tqdm import tqdm
from pathlib import Path
import torch

# 一定要加！！！！
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# 设置根目录
ROOT_DIR = Path(__file__).parent.parent.parent
os.chdir(ROOT_DIR)

# 配置logging
# 确保logs目录存在
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('logs/whisper_transcription.log', encoding='utf-8')  # 输出到文件
    ]
)

#################辅助函数#################

def get_audio_duration(audio_path):
    """获取音频文件时长（秒）"""
    try:
        import librosa
        audio_data, sample_rate = librosa.load(str(audio_path), sr=None, duration=None)
        duration = len(audio_data) / sample_rate
        return duration
    except Exception as e:
        logging.error(f"Error getting duration for {audio_path}: {e}")
        return None

import concurrent.futures

def filter_audio_files_by_duration(audio_files, max_duration=30, min_duration=0, max_workers=None):

    def process_file(audio_file):
        duration = get_audio_duration(audio_file)
        if duration is not None:
            if min_duration <= duration and (max_duration is None or duration <= max_duration):
                logging.debug(f"File {audio_file.name}: {duration:.2f}s - included")
                return audio_file, duration, True
            else:
                logging.debug(f"File {audio_file.name}: {duration:.2f}s - excluded")
        return audio_file, duration, False
    
    filtered_files = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_file, file): file for file in audio_files}
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                          total=len(audio_files), 
                          desc="Filtering audio by duration", 
                          unit="file"):
            try:
                audio_file, duration, included = future.result()
                if included:
                    filtered_files.append(audio_file)
            except Exception as e:
                file_name = future_to_file[future].name
                logging.warning(f"Error processing {file_name}: {e}")
    
    return filtered_files

#################基础转写#################

def process_audio(audio_path, model):
    result = model.transcribe(str(audio_path))  # 确保路径是字符串格式
    return result['text']

def process_into_list(folder, duration_filter=None):

    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.webm', '.opus']
    
    folder_path = Path(folder)
    json_output_path = folder_path / "transcription.json"
    
    # 如果JSON文件已存在，加载现有数据
    if json_output_path.exists():
        with open(json_output_path, "r", encoding='utf-8') as f:
            transcription_pairs = json.load(f)
        logging.info(f"Loaded existing transcription data with {len(transcription_pairs)} entries")
    else:
        transcription_pairs = {}

    model = wsp.load_model('large-v3-turbo', download_root='./checkpoints/whisper-large-v3-turbo')

    # 先收集所有音频文件
    audio_files = []
    for audio_file in folder_path.rglob('*'):
        if audio_file.is_file() and audio_file.suffix.lower() in audio_extensions:
            audio_files.append(audio_file)
    
    logging.info(f"Found {len(audio_files)} audio files in total")
    
    # 根据时长过滤音频文件
    if duration_filter is not None:
        min_dur, max_dur = duration_filter
        audio_files = filter_audio_files_by_duration(audio_files, max_duration=max_dur, min_duration=min_dur)
        logging.info(f"After duration filtering ({min_dur}s-{max_dur if max_dur else 'inf'}s): {len(audio_files)} files to process")
    
    # 使用tqdm显示进度
    for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
        try:
            # 获得绝对路径
            audio_path = str(audio_file.absolute())
            
            # 检查是否已经处理过这个文件
            if audio_path in transcription_pairs:
                logging.info(f"Skipping already processed file: {audio_file}")
                continue
                
            logging.info(f"Processing {audio_file}")
            # 这边用函数是为了方便兼容后面走tensorrt的格式
            transcription_text = process_audio(str(audio_file), model)

            transcription_pairs[audio_path] = transcription_text

            # 立即保存到JSON文件
            with open(json_output_path, "w", encoding='utf-8') as f:
                json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)

            logging.info(f"Finish Transcription and Saved: {transcription_text}")
        
        except Exception as e:
            logging.error(f"Overcome Error: {e} with {audio_file}")
            # 然后我们就不添加进去了

    logging.info(f"Transcription saved to {json_output_path}")
    logging.info(f"Total {len(transcription_pairs)} files processed")

#################VLLM加速推理#################

def process_into_list_vllm(folder, max_duration=30):

    try:
        # 在函数内部导入vllm相关模块，避免Windows环境问题
        import time
        from vllm import LLM, SamplingParams
        import librosa
        
        logging.info("VLLM modules imported successfully")
        
    except ImportError as e:
        logging.error(f"Failed to import VLLM modules: {e}")
        logging.error("Please install VLLM or use the basic whisper function instead")
        return
    
    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.webm', '.opus']
    
    folder_path = Path(folder)
    json_output_path = folder_path / "transcription.json"
    
    # 如果JSON文件已存在，加载现有数据
    if json_output_path.exists():
        with open(json_output_path, "r", encoding='utf-8') as f:
            transcription_pairs = json.load(f)
        logging.info(f"Loaded existing VLLM transcription data with {len(transcription_pairs)} entries")
    else:
        transcription_pairs = {}

    # 创建VLLM Whisper模型实例
    try:
        llm = LLM(
            model="openai/whisper-large-v3",
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=448,
            max_num_seqs=400,
            kv_cache_dtype="auto",
            download_dir="./checkpoints"
        )
        logging.info("VLLM Whisper model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load VLLM model: {e}")
        return

    # 创建采样参数
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=200,
    )

    # 先收集所有音频文件
    all_audio_files = []
    for audio_file in folder_path.rglob('*'):
        if audio_file.is_file() and audio_file.suffix.lower() in audio_extensions:
            all_audio_files.append(audio_file)
    
    logging.info(f"Found {len(all_audio_files)} audio files in total")
    
    # 只处理时长小于max_duration的音频文件
    audio_files = filter_audio_files_by_duration(all_audio_files, max_duration=max_duration)
    logging.info(f"Found {len(audio_files)} audio files <= {max_duration}s to process with VLLM")
    
    # 批量处理音频文件
    batch_size = 256  # 可以根据GPU内存调整批次大小
    
    for i in tqdm(range(0, len(audio_files), batch_size), desc="Processing batches", unit="batch"):
        batch_files = audio_files[i:i+batch_size]
        batch_prompts = []
        batch_file_paths = []
        
        # 准备批次数据
        for audio_file in batch_files:
            try:
                audio_path = str(audio_file.absolute())
                
                # 检查是否已经处理过这个文件
                if audio_path in transcription_pairs:
                    logging.info(f"Skipping already processed file: {audio_file}")
                    continue
                
                # 使用librosa读取音频文件
                audio_data, sample_rate = librosa.load(str(audio_file), sr=16000)
                
                # 准备VLLM格式的prompt
                prompt = {
                    "prompt": "<|startoftranscript|>",
                    "multi_modal_data": {
                        "audio": (audio_data, sample_rate),
                    },
                }
                
                batch_prompts.append(prompt)
                batch_file_paths.append(audio_path)
                
            except Exception as e:
                logging.error(f"Error preparing audio file {audio_file}: {e}")
                continue
        
        # 如果批次为空，跳过
        if not batch_prompts:
            continue
            
        try:
            # 使用VLLM进行批量推理
            start_time = time.time()
            outputs = llm.generate(batch_prompts, sampling_params)
            inference_time = time.time() - start_time
            
            logging.info(f"Batch inference completed in {inference_time:.2f}s for {len(batch_prompts)} files")
            
            # 处理输出结果 - 批量更新transcription_pairs
            batch_results = []
            for idx, output in enumerate(outputs):
                if idx < len(batch_file_paths):
                    audio_path = batch_file_paths[idx]
                    generated_text = output.outputs[0].text.strip()
                    
                    transcription_pairs[audio_path] = generated_text
                    batch_results.append((Path(audio_path).name, generated_text[:50]))
            
            # 批次保存到JSON文件 - 减少IO操作
            with open(json_output_path, "w", encoding='utf-8') as f:
                json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)
            
            # 批量日志输出
            for filename, preview in batch_results:
                logging.info(f"Transcribed: {filename} -> {preview}...")
            
            logging.info(f"Batch saved: {len(batch_results)} files written to {json_output_path.name}")
            
        except Exception as e:
            logging.error(f"Error during VLLM inference: {e}")
            continue

    # 计算总时长

    logging.info(f"VLLM transcription saved to {json_output_path}")
    logging.info(f"Total {len(transcription_pairs)} files processed")

if __name__ == "__main__":
    # 新增：遍历 data 下所有包含 sliced 的专辑目录，并在各自 sliced 目录生成 transcription.json
    data_root = "data"

    def _iter_sliced_dirs(root_dir: str):
        root_path = Path(root_dir)
        # 查找所有名为 sliced 的目录
        return [p for p in root_path.rglob('sliced') if p.is_dir()]

    def _process_all_sliced(root_dir: str, use_vllm: bool = False, duration_filter=None):

        sliced_dirs = _iter_sliced_dirs(root_dir)
        logging.info(f"Found {len(sliced_dirs)} 'sliced' directories under {root_dir}")
        for sliced_dir in sliced_dirs:
            logging.info(f"Processing sliced directory: {sliced_dir}")
            if use_vllm:
                # VLLM只处理30秒以内的音频
                process_into_list_vllm(str(sliced_dir), max_duration=30)
            else:
                # 基础Whisper根据duration_filter处理
                process_into_list(str(sliced_dir), duration_filter=duration_filter)

    # 两阶段处理流程
    use_vllm = True  # 第一阶段使用VLLM

    # 检查系统平台
    if use_vllm and platform.system() != "Linux":
        print(f"警告：vLLM仅支持Linux平台，当前系统为 {platform.system()}")
        print("自动切换到基础Whisper进行转录...")
        use_vllm = False

    if use_vllm:
        print("=" * 60)
        print("第一阶段：使用VLLM加速推理处理短音频（≤30秒）...")
        print("=" * 60)
        _process_all_sliced(data_root, use_vllm=True)
        
        print("\n" + "=" * 60)
        print("第二阶段：使用基础Whisper处理长音频（>30秒）...")
        print("=" * 60)
        # 第二阶段：处理30秒以上的音频
        _process_all_sliced(data_root, use_vllm=False, duration_filter=(30, None))
    else:
        print("=" * 60)
        print("使用基础Whisper进行转录（所有音频）...")
        print("=" * 60)
        _process_all_sliced(data_root, use_vllm=False)
    
    print("\n" + "=" * 60)
    print("所有音频处理完成！")
    print("=" * 60)

    from tools.calculate_time import calculate_time as cct
    cct(data_root)