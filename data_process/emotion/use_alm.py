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
    
    content.append({"type": "text", "text": f"请结合音频和转写文本分析这段内容中的情感，只输出情感类别：happy, amused, excited, satisfied, relaxed, sad, frustrated, guilty, angry, fearful, anxious, disgust, jealous, surprised, confused, curious, eager, adoring, interested, neutral."})
    # 整体来说，除了 happy,sad,angry,fearful,surprised,disgusted,neutral
    # 还有更加细分的 curious,depressed,excited,relaxed,anxious,jealous,frustrated,disappointed
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

def process_list(folder, model, processor, sampling_params, asr_json_path=None, batch_size=16):
    from pathlib import Path
    import json
    from tqdm import tqdm
    
    # 支持的音频文件扩展名
    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.webm', '.opus']
    
    folder_path = Path(folder)
    json_output_path = folder_path / "emotion_results.json"
    
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
            emotion_results = json.load(f)
        logging.info(f"Loaded existing emotion data with {len(emotion_results)} entries")
    else:
        emotion_results = {}
    
    # 收集所有音频文件
    audio_files = []
    for audio_file in folder_path.rglob('*'):
        if audio_file.is_file() and audio_file.suffix.lower() in audio_extensions:
            audio_files.append(audio_file)
    
    logging.info(f"Found {len(audio_files)} audio files to process")
    
    # 筛选出待处理的文件（跳过已处理的）
    pending_files = [f for f in audio_files if str(f.absolute()) not in emotion_results]
    logging.info(f"Pending files to process: {len(pending_files)}")
    
    if len(pending_files) == 0:
        logging.info("All files already processed!")
        return
    
    # 批量处理
    total_batches = (len(pending_files) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(0, len(pending_files), batch_size), 
                          desc="Processing emotion recognition batches", 
                          total=total_batches,
                          unit="batch"):
        batch_files = pending_files[batch_idx:batch_idx + batch_size]
        batch_inputs = []
        batch_paths = []
        batch_file_names = []
        
        # 准备批量输入
        for audio_file in batch_files:
            try:
                audio_path = str(audio_file.absolute())
                asr_text = asr_data.get(audio_path)
                
                # 构建消息内容
                content = [{"type": "audio", "audio": audio_path}]
                
                if asr_text:
                    content.append({"type": "text", "text": f"转写文本：{asr_text}"})
                
                content.append({
                    "type": "text", 
                    "text": "请结合音频和转写文本分析这段内容中的情感，只输出情感类别：happy, amused, excited, satisfied, relaxed, sad, frustrated, guilty, angry, fearful, anxious, disgust, jealous, surprised, confused, curious, eager, adoring, interested, neutral."
                })
                
                messages = [{"role": "user", "content": content}]
                
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
                
                batch_inputs.append(inputs)
                batch_paths.append(audio_path)
                batch_file_names.append(audio_file.name)
                
            except Exception as e:
                import traceback
                logging.error(f"Error preparing input for {audio_file}: {e}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        # 如果这个批次没有有效的输入，跳过
        if len(batch_inputs) == 0:
            logging.warning(f"Batch {batch_idx // batch_size + 1} has no valid inputs, skipping")
            continue
        
        # 批量推理
        try:
            logging.info(f"Processing batch {batch_idx // batch_size + 1}/{total_batches} with {len(batch_inputs)} files")
            outputs = model.generate(batch_inputs, sampling_params=sampling_params)
            
            # 处理批量结果
            for audio_path, file_name, output in zip(batch_paths, batch_file_names, outputs):
                try:
                    # 安全地获取结果
                    if hasattr(output, 'outputs') and len(output.outputs) > 0:
                        if hasattr(output.outputs[0], 'text'):
                            emotion = output.outputs[0].text
                        else:
                            emotion = str(output.outputs[0])
                    else:
                        emotion = str(output)
                    
                    emotion_results[audio_path] = emotion
                    logging.info(f"Emotion recognized: {file_name} -> {emotion}")
                    
                except Exception as e:
                    logging.error(f"Error extracting result for {file_name}: {e}")
                    emotion_results[audio_path] = "ERROR"
            
            # 每个批次处理完后立即保存
            with open(json_output_path, "w", encoding='utf-8') as f:
                json.dump(emotion_results, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Batch {batch_idx // batch_size + 1} completed and saved")
            
        except Exception as e:
            import traceback
            logging.error(f"Error during batch inference for batch {batch_idx // batch_size + 1}: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            
            # 批量推理失败时，尝试单个处理这个批次
            logging.info(f"Attempting to process batch {batch_idx // batch_size + 1} files individually...")
            for audio_path, file_name, single_input in zip(batch_paths, batch_file_names, batch_inputs):
                try:
                    single_output = model.generate([single_input], sampling_params=sampling_params)
                    
                    if hasattr(single_output[0], 'outputs') and len(single_output[0].outputs) > 0:
                        if hasattr(single_output[0].outputs[0], 'text'):
                            emotion = single_output[0].outputs[0].text
                        else:
                            emotion = str(single_output[0].outputs[0])
                    else:
                        emotion = str(single_output[0])
                    
                    emotion_results[audio_path] = emotion
                    logging.info(f"Individual processing succeeded: {file_name} -> {emotion}")
                    
                except Exception as single_e:
                    logging.error(f"Individual processing also failed for {file_name}: {single_e}")
                    emotion_results[audio_path] = "ERROR"
            
            # 保存单个处理的结果
            with open(json_output_path, "w", encoding='utf-8') as f:
                json.dump(emotion_results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Emotion recognition completed. Results saved to {json_output_path}")
    logging.info(f"Total {len(emotion_results)} files processed")
    logging.info(f"Successful: {sum(1 for v in emotion_results.values() if v != 'ERROR')}")
    logging.info(f"Failed: {sum(1 for v in emotion_results.values() if v == 'ERROR')}")

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
                max_num_seqs=32,  
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



    