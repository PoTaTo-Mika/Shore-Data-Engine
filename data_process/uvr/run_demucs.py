import torch
import demucs.api
import tqdm
import os
import logging

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('demucs.log', encoding='utf-8')  # 输出到文件
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

def process_folder(folder_path):
    separator = demucs.api.Separator(model='htdemucs_ft') # 需要320MB的空间
    # 收集所有音频文件
    audio_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    logging.info(f"Found {len(audio_files)} audio files to process")
    
    # 使用tqdm显示进度条处理所有音频文件
    for audio_file in tqdm.tqdm(audio_files, desc="Processing audio files"):
        process_audio(audio_file, separator)
        
if __name__ == '__main__':
    folder = './data'
    process_folder(folder)


