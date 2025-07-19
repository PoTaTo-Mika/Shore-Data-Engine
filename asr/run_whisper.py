import whisper as wsp
import os
import json
import logging
from tqdm import tqdm
from pathlib import Path

def process_audio(audio_path, model):
    result = model.transcribe(audio_path)
    return result['text']

def process_into_list(folder):

    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.webm', '.opus']
    transcription_pairs = {}

    model = wsp.load_model('../checkpoints/whisper-large-v3')

    folder_path = Path(folder)
    for audio_file in folder_path.rglob('*'):
        if audio_file.is_file() and audio_file.suffix.lower() in audio_extensions:
            try:
                logging.info(f"Processing {audio_file}")
                # 这边用函数是为了方便兼容后面走tensorrt的格式
                transcription_text = process_audio(audio_file, model)

                # 获得绝对路径
                audio_path = str(audio_file.absolute())
                transcription_pairs[audio_path] = transcription_text

                logging.info(f"Finish Transcription: {transcription_text}")
            
            except Exception as e:
                logging(f"Overcome Error: {e} with {audio_file}")
                # 然后我们就不添加进去了
    
    json_output_path = folder_path / "transcription.json"

    # 保存json
    with open(json_output_path, "w",encoding='utf-8') as f:
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