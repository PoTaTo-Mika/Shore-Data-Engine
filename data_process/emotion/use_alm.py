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


def Qwen3_Omni_Recognition(audio_path,model):
    # 往挂上的LLM发请求
    llm = model
    messages = [
        {
            "role": "system",
            "content": [
                "You are a helpful assistant that can help analyze the emotion from audio and video clips.",
                "The user will give you several audio clips, please tell the emotion in the text format."
                "The emotion can be categorized into 7 categories: happy, sad, angry, fearful, surprised, disgusted, and neutral."
                "You should only output the emotion in the text format, no other text."
            ], 
        },
        {
            "role": "user",
            "content": [
                {"audio": audio_path},
                "Please tell me what emotion is shown in the audio clip."
            ],
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

    return outputs[0].outputs[0].text

def process_list(folder, model):
    audio_files = []
    for audio_file in folder.rglob('*'):
        audio_files.append(audio_file)
    logging.info(f"Found {len(audio_files)} audio files to process")
    
    for audio_file in audio_files:
        

if __name__ == "__main__":

    with open("configs/allm.json", "r") as f:
        config = json.load(f)
        model_name = config["model"]
        cur_path = 'checkpoints/' + model_name.split('/')[-1]
        if not os.path.exists(cur_path):
           subprocess.run(["huggingface-cli", "download", model_name, "--local-dir", cur_path])
        
        if 'Qwen' in model_name:
            model = LLM(
                model=cur_path, trust_remote_code=True, gpu_memory_utilization=0.95,
                tensor_parallel_size=torch.cuda.device_count(),
                limit_mm_per_prompt={'image': 0, 'video': 1, 'audio': 1} 
                if 'Captioner' not in model_name else {'audio': 1},
                max_num_seqs=8,
                max_model_len=32768,
                seed=1145,
            )
            processor = Qwen3OmniMoeProcessor.from_pretrained(cur_path)

        sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )



    