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
    logging.info("Loading FunASR models ... (this may take a while for the first time)")
    # 1) 仅加载 ASR（Paraformer），以便进行真正的批量推理
    asr_model = AutoModel(
        model="paraformer-zh",
    )
    # 2) 单独加载标点模型（串行文本修复，轻量）
    punc_model = AutoModel(
        model="ct-punc",
    )
    logging.info("ASR (paraformer-zh) and PUNC (ct-punc) models loaded successfully")

    # 收集全部音频文件
    audio_files = [p for p in folder_path.rglob('*') if p.is_file() and p.suffix.lower() in audio_extensions]
    logging.info(f"Found {len(audio_files)} audio files to process")

    # 生成待处理列表（过滤已处理）
    pending_files = []
    for p in audio_files:
        abs_path = str(p.absolute())
        if abs_path not in transcription_pairs:
            pending_files.append(p)

    logging.info(f"Pending {len(pending_files)} files to process (skipped {len(audio_files) - len(pending_files)} already done)")

    # 批量参数：环境变量
    # FUNASR_BATCH_SIZE：每批样本数；FUNASR_BATCH_SIZE_S：每批总时长（秒）。若都设置，优先使用时长分批。
    try:
        batch_size_env = os.environ.get("FUNASR_BATCH_SIZE", "16")
        batch_size = max(1, int(batch_size_env))
    except Exception:
        batch_size = 16

    batch_size_s_env = os.environ.get("FUNASR_BATCH_SIZE_S")
    batch_size_s = None
    try:
        if batch_size_s_env is not None:
            batch_size_s = max(1, int(batch_size_s_env))
    except Exception:
        batch_size_s = None

    # 批量识别（ASR），随后逐条标点（PUNC）
    for start in tqdm(range(0, len(pending_files), batch_size), desc="Processing batches", unit="batch"):
        batch_paths = [str(p.absolute()) for p in pending_files[start:start + batch_size]]
        if not batch_paths:
            continue

        try:
            logging.info(f"ASR batch: {start} - {start + len(batch_paths) - 1} (size={len(batch_paths)})")

            # ASR 批量推理
            gen_kwargs = {
                "input": batch_paths,
                "disable_pbar": True,
            }
            if batch_size_s is not None:
                gen_kwargs["batch_size_s"] = batch_size_s
            else:
                gen_kwargs["batch_size"] = len(batch_paths)

            asr_results = asr_model.generate(**gen_kwargs)

            # 统一成 list
            if not isinstance(asr_results, list):
                asr_results = [asr_results]

            # 逐条进行标点修复（串行）
            for path, res in zip(batch_paths, asr_results):
                # 提取 ASR 文本
                asr_text = ""
                try:
                    if isinstance(res, str):
                        asr_text = res.strip()
                    elif isinstance(res, dict) and 'text' in res:
                        asr_text = str(res['text']).strip()
                    elif isinstance(res, list) and len(res) > 0:
                        first_item = res[0]
                        if isinstance(first_item, dict) and 'text' in first_item:
                            asr_text = str(first_item['text']).strip()
                        elif isinstance(first_item, str):
                            asr_text = first_item.strip()
                except Exception:
                    asr_text = ""

                # 标点修复
                final_text = asr_text
                try:
                    if asr_text:
                        punc_res = punc_model.generate(input=asr_text)
                        # 解析标点输出
                        if isinstance(punc_res, list) and len(punc_res) > 0:
                            item0 = punc_res[0]
                            if isinstance(item0, dict) and 'text' in item0:
                                final_text = str(item0['text']).strip()
                            elif isinstance(item0, str):
                                final_text = item0.strip()
                        elif isinstance(punc_res, dict) and 'text' in punc_res:
                            final_text = str(punc_res['text']).strip()
                        elif isinstance(punc_res, str):
                            final_text = punc_res.strip()
                except Exception as e:
                    logging.warning(f"Punctuation failed for {Path(path).name}: {e}")
                    final_text = asr_text

                transcription_pairs[path] = final_text

            # 批次保存
            with open(json_output_path, "w", encoding='utf-8') as f:
                json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)

            logging.info(f"Batch saved: {len(batch_paths)} files")

        except Exception as e:
            logging.error(f"Skipping batch starting at index {start} due to error: {e}")
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
