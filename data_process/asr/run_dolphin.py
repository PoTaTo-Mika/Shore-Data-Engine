# 这个主要用于方言识别以及小语种识别
import dolphin
import os
import json
import logging
from tqdm import tqdm
from pathlib import Path

def process_audio(audio_path, model):
    waveform = dolphin.load_audio(audio_path)
    result = model(waveform)
    # result = model(waveform, lang_sym = 'zh', region_sym="CN") 
    # 这个是强制指定语言的
    return result.text

def process_into_list(folder):
    
    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.webm', '.opus']
    transcription_pairs = {}

    model = dolphin.load_model("base", "../../checkpoints/dolphin", "cuda")

    folder_path = Path(folder)
    
    # 先收集所有音频文件
    audio_files = []
    for audio_file in folder_path.rglob('*'):
        if audio_file.is_file() and audio_file.suffix.lower() in audio_extensions:
            audio_files.append(audio_file)
    
    logging.info(f"Found {len(audio_files)} audio files to process")
    
    # 使用tqdm显示进度
    for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
        try:
            logging.info(f"Processing {audio_file}")
            # 这边用函数是为了方便兼容后面走tensorrt的格式
            transcription_text = process_audio(audio_file, model)

            # 获得绝对路径
            audio_path = str(audio_file.absolute())
            transcription_pairs[audio_path] = transcription_text

            logging.info(f"Finish Transcription: {transcription_text}")
        
        except Exception as e:
            logging.error(f"Overcome Error: {e} with {audio_file}")
            # 然后我们就不添加进去了
    
    json_output_path = folder_path / "transcription.json"

    # 保存json
    with open(json_output_path, "w", encoding='utf-8') as f:
        json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)

    from tools.calculate_time import calculate_time

    folder_path = Path(folder)
    total_duration = calculate_time(folder_path)

    logging.info(f"Transcription saved to {json_output_path}")
    logging.info(f"Total {len(transcription_pairs)} files processed")
    logging.info(f"Total duration: {total_duration/3600:.2f} hours")

if __name__ == "__main__":
    folder = "qtfm"
    process_into_list(folder)