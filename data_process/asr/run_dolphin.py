# 这个主要用于方言识别以及小语种识别
import dolphin
import os
import json
import logging
from tqdm import tqdm
from pathlib import Path

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('dolphin_transcription.log', encoding='utf-8')  # 输出到文件
    ]
)

def process_audio(audio_path, model):
    waveform = dolphin.load_audio(audio_path)
    result = model(waveform)
    # result = model(waveform, lang_sym = 'zh', region_sym="CN") 
    # 这个是强制指定语言的
    return result.text

def process_into_list(folder):
    
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

    model = dolphin.load_model("base", "./checkpoints/dolphin", "cuda")

    # 先收集所有音频文件
    audio_files = []
    for audio_file in folder_path.rglob('*'):
        if audio_file.is_file() and audio_file.suffix.lower() in audio_extensions:
            audio_files.append(audio_file)
    
    logging.info(f"Found {len(audio_files)} audio files to process")
    
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
            transcription_text = process_audio(audio_file, model)
            transcription_pairs[audio_path] = transcription_text

            # 立即保存到JSON文件
            with open(json_output_path, "w", encoding='utf-8') as f:
                json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)

            logging.info(f"Finish Transcription and Saved: {transcription_text}")
        
        except Exception as e:
            logging.error(f"Overcome Error: {e} with {audio_file}")
            # 然后我们就不添加进去了

    from tools.calculate_time import calculate_time

    total_duration = calculate_time(folder_path)

    logging.info(f"Transcription saved to {json_output_path}")
    logging.info(f"Total {len(transcription_pairs)} files processed")
    logging.info(f"Total duration: {total_duration/3600:.2f} hours")

if __name__ == "__main__":
    logging.warning("Dolphin can only process audio less than 30 seconds")
    logging.warning("Please check the audio duration before processing")
    folder = "data/test"
    process_into_list(folder)