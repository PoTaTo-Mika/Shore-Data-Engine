import os
from typing import Optional
import dashscope

def transcribe_with_qwen_asr(
    audio_file_path: str,
    api_key: Optional[str] = None,
    model: str = "qwen3-asr-flash",
    context_text: str = "",
    enable_lid: bool = True,
    enable_itn: bool = False,
    language: Optional[str] = None,
) -> str:
    audio_uri = audio_file_path if audio_file_path.startswith("file://") else f"file://{audio_file_path}"

    messages = [
        {
            "role": "system",
            "content": [
                {"text": context_text},
            ],
        },
        {
            "role": "user",
            "content": [
                {"audio": audio_uri},
            ],
        },
    ]

    asr_options = {
        "enable_lid": enable_lid,
        "enable_itn": enable_itn,
    }
    if language:
        asr_options["language"] = language

    # 优先使用传入的 api_key；否则尝试从本地配置读取；最后回退到环境变量
    resolved_api_key = api_key
    if not resolved_api_key:
        try:
            import json
            from pathlib import Path
            # 修改配置文件路径到项目根目录下的configs文件夹
            project_root = Path(__file__).resolve().parents[2]
            config_path = project_root / "configs" / "qwen_asr.json"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                resolved_api_key = cfg.get("api_key") or None
        except Exception:
            resolved_api_key = None

    response = dashscope.MultiModalConversation.call(
        api_key=resolved_api_key or os.getenv("DASHSCOPE_API_KEY"),
        model=model,
        messages=messages,
        result_format="message",
        asr_options=asr_options,
    )

    # 解析返回并处理 finish_reason
    try:
        choices = response["output"]["choices"]
        first_choice = choices[0]
        finish_reason = first_choice.get("finish_reason")
        if finish_reason == "stop":
            message = first_choice.get("message", {})
            for item in message.get("content", []) or []:
                if isinstance(item, dict) and "text" in item:
                    return item["text"]
            raise RuntimeError("ASR 已完成但未返回文本内容")
        else:
            print(f"finish_reason: {finish_reason}")
            raise RuntimeError(f"ASR 未完成，finish_reason={finish_reason}")
    except Exception:
        # 原样抛出，调用方可捕获并处理
        raise

def process_into_list(folder: str) -> None:
    import json
    import logging
    from pathlib import Path
    from tqdm import tqdm

    audio_extensions = [
        ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".webm", ".opus",
    ]

    folder_path = Path(folder)
    json_output_path = folder_path / "transcription.json"

    if json_output_path.exists():
        with open(json_output_path, "r", encoding="utf-8") as f:
            transcription_pairs = json.load(f)
        logging.info(f"Loaded existing transcription data with {len(transcription_pairs)} entries")
    else:
        transcription_pairs = {}

    audio_files = []
    for audio_file in folder_path.rglob("*"):
        if audio_file.is_file() and audio_file.suffix.lower() in audio_extensions:
            audio_files.append(audio_file)

    logging.info(f"Found {len(audio_files)} audio files to process")

    for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
        try:
            audio_path = str(audio_file.absolute())
            if audio_path in transcription_pairs:
                logging.info(f"Skipping already processed file: {audio_file}")
                continue

            logging.info(f"Processing {audio_file}")
            text = transcribe_with_qwen_asr(audio_path)
            transcription_pairs[audio_path] = text

            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)

            logging.info(f"Finish Transcription and Saved: {text}")
        except Exception as e:
            import logging
            logging.error(f"Overcome Error: {e} with {audio_file}")

    try:
        from tools.calculate_time import calculate_time
        total_duration = calculate_time(folder_path)
        logging.info(f"Total duration: {total_duration/3600:.2f} hours")
    except Exception:
        pass

    logging.info(f"Transcription saved to {json_output_path}")
    logging.info(f"Total {len(transcription_pairs)} files processed")

if __name__ == "__main__":
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"

    # 查找 data 下的所有子目录中的 sliced 目录
    sliced_dirs = []
    if data_root.exists():
        for subdir in data_root.iterdir():
            if subdir.is_dir():
                sliced = subdir / "sliced"
                if sliced.exists() and sliced.is_dir():
                    sliced_dirs.append(sliced)

    if not sliced_dirs:
        logging.warning("No sliced directories found under data/*/sliced")
    else:
        for sd in sliced_dirs:
            logging.info(f"Start processing: {sd}")
            process_into_list(str(sd))
            logging.info(f"Finished processing: {sd}")