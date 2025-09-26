# Kimi-Audio, Qwen-Audio, Mimo-Audio, Qwen-Omni...
import librosa
import numpy as np
import json
import os
import logging
import subprocess
from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import torch

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('logs/audio_language_model_caption.log', encoding='utf-8')  # 输出到文件
    ]
)


def Qwen3_Omni_Recognition(audio_path, model, processor, sampling_params, asr_text=None):
    # 往挂上的LLM发请求
    llm = model
    # 构建消息内容
    content = [
        {"type": "audio", "audio": audio_path}
    ]
    
    # 如果有ASR转写文本，添加到消息中
    if asr_text:
        content.append({"type": "text", "text": f"转写文本：{asr_text}"})
    
    content.append({"type": "text", "text": f"请结合音频和转写文本，描述音频内容，包括情绪，副语言，语种等多种语言学特征。只需要以一整段文本的形式回复即可，无需使用markdown逐个列举。"})
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    
    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        "mm_processor_kwargs": {
            "use_audio_in_video": True,
        },
    }

    if images is not None:
        inputs['multi_modal_data']['image'] = images
    if videos is not None:
        inputs['multi_modal_data']['video'] = videos
    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios

    outputs = llm.generate([inputs], sampling_params=sampling_params)
    
    # 安全地获取结果
    if hasattr(outputs[0], 'outputs') and len(outputs[0].outputs) > 0:
        if hasattr(outputs[0].outputs[0], 'text'):
            return outputs[0].outputs[0].text
        else:
            logging.error(f"outputs[0].outputs[0] has no 'text' attribute: {outputs[0].outputs[0]}")
            return str(outputs[0].outputs[0])
    else:
        logging.error(f"Unexpected output structure: {outputs[0]}")
        return str(outputs[0])

def process_list(folder, model, processor, sampling_params, asr_json_path=None):
    from pathlib import Path
    import json
    from tqdm import tqdm
    
    # 支持的音频文件扩展名
    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.webm', '.opus']
    
    folder_path = Path(folder)
    json_output_path = folder_path / "describe_results.json"
    
    # 加载ASR转写结果
    asr_data = {}
    if asr_json_path and Path(asr_json_path).exists():
        try:
            with open(asr_json_path, "r", encoding='utf-8') as f:
                asr_data = json.load(f)
            logging.info(f"Loaded ASR data with {len(asr_data)} entries from {asr_json_path}")
        except Exception as e:
            logging.error(f"Error loading ASR data from {asr_json_path}: {e}")
            logging.warning("Continuing without ASR data")
    elif asr_json_path:
        logging.warning(f"ASR file not found: {asr_json_path}, continuing without ASR data")
    else:
        logging.info("No ASR file specified, processing audio-only")
    
    # 如果JSON文件已存在，加载现有数据
    if json_output_path.exists():
        with open(json_output_path, "r", encoding='utf-8') as f:
            describe_results = json.load(f)
        logging.info(f"Loaded existing describe data with {len(describe_results)} entries")
    else:
        describe_results = {}
    
    # 收集所有音频文件
    audio_files = []
    for audio_file in folder_path.rglob('*'):
        if audio_file.is_file() and audio_file.suffix.lower() in audio_extensions:
            audio_files.append(audio_file)
    
    logging.info(f"Found {len(audio_files)} audio files to process")
    
    # 使用tqdm显示进度
    for audio_file in tqdm(audio_files, desc="Processing describe recognition", unit="file"):
        try:
            # 获得绝对路径
            audio_path = str(audio_file.absolute())
            
            # 检查是否已经处理过这个文件
            if audio_path in describe_results:
                logging.info(f"Skipping already processed file: {audio_file}")
                continue
                
            logging.info(f"Processing describe for: {audio_file}")
            
            # 获取对应的ASR转写文本
            asr_text = asr_data.get(audio_path)
            if asr_text:
                logging.info(f"Found ASR text for {audio_file}: {asr_text[:100]}...")
            else:
                logging.info(f"No ASR text found for {audio_file}, using audio-only analysis")
            
            # 进行情感识别
            describe = Qwen3_Omni_Recognition(audio_path, model, processor, sampling_params, asr_text)
            
            # 保存结果
            describe_results[audio_path] = describe
            
            # 立即保存到JSON文件
            with open(json_output_path, "w", encoding='utf-8') as f:
                json.dump(describe_results, f, ensure_ascii=False, indent=2)
            
            logging.info(f"describe recognized and saved: {audio_file} -> {describe}")
        
        except Exception as e:
            import traceback
            logging.error(f"Error processing {audio_file}: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            # 继续处理下一个文件
    
    logging.info(f"Describe recognition completed. Results saved to {json_output_path}")
    logging.info(f"Total {len(describe_results)} files processed")

if __name__ == "__main__":

    with open("configs/allm.json", "r") as f:
        config = json.load(f)
        model_name = config["model"]
        cur_path = 'checkpoints/' + model_name.split('/')[-1]
        if not os.path.exists(cur_path):
           subprocess.run(["huggingface-cli", "download", model_name, "--local-dir", cur_path])
        
        if 'Qwen' in model_name:
            model = LLM(
                model=cur_path, trust_remote_code=True, 
                gpu_memory_utilization=0.9,  # 降低GPU内存使用率
                tensor_parallel_size=torch.cuda.device_count(),
                limit_mm_per_prompt={'image': 0, 'video': 1, 'audio': 1} 
                if 'Captioner' not in model_name else {'audio': 1},
                max_num_seqs=4,  # 减少并发序列数
                max_model_len=4096,  # 减少最大模型长度
                seed=1145,
            )
            processor = Qwen3OmniMoeProcessor.from_pretrained(cur_path)

        sampling_params = SamplingParams(
        temperature=0.7,  # README建议使用0.7
        top_p=0.8,        # README建议使用0.8
        top_k=20,         # README建议使用20
        max_tokens=4096, # README建议使用16384
    )

    # 获取ASR文件路径（如果存在）
    asr_path = config.get("asr_text_path")
    process_list(config["data_path"], model, processor, sampling_params, asr_path)



    