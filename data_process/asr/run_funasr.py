from funasr import AutoModel
import os
import json
import logging
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
import librosa
import tempfile

# 配置 logging，保持与 run_whisper.py 一致
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('logs/funasr_transcription.log', encoding='utf-8')  # 输出到文件
    ]
)

def process_audio(audio_path: str, model: AutoModel) -> str:
    temp_path = None
    try:
        # 预处理音频：检查采样率并转换为16kHz
        info = sf.info(audio_path)
        if info.samplerate != 16000:
            logging.debug(f"Converting {audio_path} from {info.samplerate}Hz to 16000Hz")
            audio_data, _ = librosa.load(audio_path, sr=16000)
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            sf.write(temp_path, audio_data, 16000)
            processed_path = temp_path
        else:
            processed_path = audio_path
        
        # 进行语音识别
        res = model.generate(input=processed_path)
        
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        # 提取文本内容
        if isinstance(res, str):
            return res.strip()
        elif isinstance(res, list) and len(res) > 0:
            first_item = res[0]
            if isinstance(first_item, dict) and 'text' in first_item:
                return str(first_item['text']).strip()
            elif isinstance(first_item, str):
                return first_item.strip()
        elif isinstance(res, dict) and 'text' in res:
            return str(res['text']).strip()
        
        return ""
        
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        logging.error(f"Failed to transcribe {audio_path}: {e}")
        raise


def process_folder(folder: str):
    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.webm', '.opus']

    folder_path = Path(folder)
    json_output_path = folder_path / "transcription.json"

    # 加载已有转写结果，支持增量转写
    if json_output_path.exists():
        with open(json_output_path, "r", encoding='utf-8') as f:
            transcription_pairs = json.load(f)
        logging.info(f"Loaded existing transcription data with {len(transcription_pairs)} entries")
    else:
        transcription_pairs = {}

    # 加载 FunASR 模型
    logging.info("Loading FunASR model ... (this may take a while for the first time)")
    model = AutoModel(
        model="paraformer-zh",   # 主 ASR 模型
        vad_model="fsmn-vad",   # 语音活动检测
        punc_model="ct-punc",   # 标点恢复
        vad_kwargs={"max_single_segment_time": 60000}
    )
    logging.info("Model loaded successfully")

    # 收集全部音频文件
    audio_files = [p for p in folder_path.rglob('*') if p.is_file() and p.suffix.lower() in audio_extensions]
    logging.info(f"Found {len(audio_files)} audio files to process")

    # 开始批量处理
    for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
        audio_path = str(audio_file.absolute())

        # 跳过已处理文件
        if audio_path in transcription_pairs:
            logging.debug(f"Skipping already processed file: {audio_file}")
            continue

        try:
            logging.info(f"Processing {audio_file}")
            transcription_text = process_audio(audio_path, model)
            transcription_pairs[audio_path] = transcription_text

            # 实时保存
            with open(json_output_path, "w", encoding='utf-8') as f:
                json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)

            logging.info(f"Finish transcription: {audio_file}")
        except Exception as e:
            logging.error(f"Skipping file {audio_file} due to error: {e}")
            continue

    # 统计总时长
    try:
        from tools.calculate_time import calculate_time
        total_duration = calculate_time(folder_path)
        logging.info(f"Total duration: {total_duration/3600:.2f} hours")
    except Exception as e:
        logging.warning(f"Failed to calculate total duration: {e}")

    logging.info(f"Transcription saved to {json_output_path}")
    logging.info(f"Total {len(transcription_pairs)} files processed")


if __name__ == "__main__":
    target_folder = "data/sliced"  # 默认路径，可根据需要修改
    process_folder(target_folder)
