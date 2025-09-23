from fish_audio_sdk import Session, TTSRequest, ReferenceAudio
import argparse
import json
from pathlib import Path


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_one(cfg: dict, text: str, out_path: str | None = None) -> Path:
    api_key = cfg.get("API_KEY", "").strip()
    if not api_key:
        raise ValueError("config[API_KEY] 不能为空")

    out_dir = Path(cfg.get("OUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 默认使用 config 中的 FORMAT 后缀
    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        fmt = str(cfg.get("FORMAT", "mp3")).lower()
        if fmt not in {"mp3", "wav", "pcm"}:
            fmt = "mp3"
        out_file = out_dir / f"output.{fmt}"

    reference_path = str(cfg.get("REFERENCE", "")).strip()
    ref_text = cfg.get("REF_TEXT", "")

    # 读取并校验与文档一致的配置项
    fmt = str(cfg.get("FORMAT", "mp3")).lower()
    if fmt not in {"mp3", "wav", "pcm"}:
        fmt = "mp3"

    try:
        chunk_length = int(cfg.get("CHUNK_LENGTH", 200) or 200)
    except (TypeError, ValueError):
        chunk_length = 200
    if not (100 <= chunk_length <= 300):
        chunk_length = 200

    try:
        mp3_bitrate = int(cfg.get("MP3_BITRATE", 128) or 128)
    except (TypeError, ValueError):
        mp3_bitrate = 128
    if mp3_bitrate not in {64, 128, 192}:
        mp3_bitrate = 128

    latency = str(cfg.get("LATENCY", "balanced"))
    if latency not in {"normal", "balanced"}:
        latency = "balanced"

    normalize = bool(cfg.get("NORMALIZE", True))

    sample_rate_val = cfg.get("SAMPLE_RATE", None)
    try:
        sample_rate = int(sample_rate_val) if sample_rate_val is not None else None
    except (TypeError, ValueError):
        sample_rate = None

    reference_id_val = cfg.get("REFERENCE_ID")
    reference_id = (
        str(reference_id_val).strip() if isinstance(reference_id_val, str) and reference_id_val.strip() else None
    )

    # 可选采样控制
    try:
        top_p = float(cfg.get("TOP_P", 0.7))
    except (TypeError, ValueError):
        top_p = 0.7
    try:
        temperature = float(cfg.get("TEMPERATURE", 0.7))
    except (TypeError, ValueError):
        temperature = 0.7

    if reference_path:
        rp = Path(reference_path)
        if rp.is_file():
            ref_bytes = rp.read_bytes()
            request = TTSRequest(
                text=text,
                format=fmt,
                chunk_length=chunk_length,
                mp3_bitrate=mp3_bitrate,
                latency=latency,
                normalize=normalize,
                sample_rate=sample_rate,
                reference_id=reference_id,
                top_p=top_p,
                temperature=temperature,
                references=[
                    ReferenceAudio(
                        audio=ref_bytes,
                        text=ref_text,
                    )
                ],
            )
        else:
            # 引用文件不存在则退回零样本
            request = TTSRequest(
                text=text,
                format=fmt,
                chunk_length=chunk_length,
                mp3_bitrate=mp3_bitrate,
                latency=latency,
                normalize=normalize,
                sample_rate=sample_rate,
                reference_id=reference_id,
                top_p=top_p,
                temperature=temperature,
            )
    else:
        request = TTSRequest(
            text=text,
            format=fmt,
            chunk_length=chunk_length,
            mp3_bitrate=mp3_bitrate,
            latency=latency,
            normalize=normalize,
            sample_rate=sample_rate,
            reference_id=reference_id,
            top_p=top_p,
            temperature=temperature,
        )

    session = Session(apikey=api_key)
    backend = str(cfg.get("BACKEND", "speech-1.5"))
    if backend not in {"speech-1.5", "speech-1.6", "agent-x0", "s1", "s1-mini"}:
        backend = "speech-1.5"
    with out_file.open("wb") as f:
        for chunk in session.tts(request, backend=backend):
            f.write(chunk)
    return out_file

def generate_json(cfg: dict, file_path: str):
    mapping_path = Path(file_path)
    with mapping_path.open("r", encoding="utf-8") as f:
        name_to_text = json.load(f)
    if not isinstance(name_to_text, dict):
        raise ValueError("--mapping JSON 必须是 {filename: text} 的字典")

    out_dir = Path(cfg.get("OUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)

    default_ext = str(cfg.get("FORMAT", "mp3")).lower() or "mp3"
    if default_ext not in {"mp3", "wav", "pcm"}:
        default_ext = "mp3"

    for file_name, text in name_to_text.items():
        if not isinstance(file_name, str) or not isinstance(text, str):
            raise ValueError("--mapping 中的 key 和 value 都必须是字符串")
        # 统一强制输出扩展名与 FORMAT 一致（即使传入 .txt 或其他后缀）
        rel = Path(file_name)
        rel = rel.with_suffix(f".{default_ext}")
        out_path = out_dir / rel
        generate_one(cfg, text, out_path=str(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parents[3] / "configs" / "fish_audio.json"),
        help="Path to config.json",
    )
    parser.add_argument(
        "--mapping",
        default="data/distill/mapping.json",
        help="Path to JSON mapping: {\"filename\": \"text\", ...}",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    mapping_path = args.mapping.strip()

    if mapping_path:
        generate_json(cfg, mapping_path)
    else:
        text = cfg.get("TEXT", "").strip()
        if not text:
            raise ValueError("未提供 --mapping，且 config[TEXT] 为空")
        generate_one(cfg, text)


if __name__ == "__main__":
    main()