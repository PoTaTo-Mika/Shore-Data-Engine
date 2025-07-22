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
    separator = demucs.api.Separator(model='htdemucs')
    for file in tqdm.tqdm(os.listdir(folder_path)):
        if file.endswith('.wav'):
            process_audio(os.path.join(folder_path, file), separator)
        
if __name__ == '__main__':
    folder = './data'
    process_folder(folder)


