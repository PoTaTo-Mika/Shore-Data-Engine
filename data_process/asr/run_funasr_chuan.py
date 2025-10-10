from funasr import AutoModel
import os
import json
import logging
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
import librosa
import tempfile

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('logs/sichuanese_transcription.log', encoding='utf-8')  # 输出到文件
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
    logging.info("Loading FunASR models ... (this may take a while for the first time)")

    asr_model = AutoModel(
        model="./checkpoints/paraformer-chuan",
    )

    punc_model = AutoModel(
        model="./checkpoints/ct-punc",
    )
    logging.info("ASR (paraformer-zh) and PUNC (ct-punc) models loaded successfully")

    audio_files = [p for p in folder_path.rglob('*') if p.is_file() and p.suffix.lower() in audio_extensions]
    logging.info(f"Found {len(audio_files)} audio files to process")

    pending_files = []
    for p in audio_files:
        abs_path = str(p.absolute())
        if abs_path not in transcription_pairs:
            pending_files.append(p)

    logging.info(f"Pending {len(pending_files)} files to process (skipped {len(audio_files) - len(pending_files)} already done)")