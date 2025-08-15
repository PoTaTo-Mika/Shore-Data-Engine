import os
import logging
from pathlib import Path
from typing import List, Tuple, Union, Optional

import librosa
import soundfile as sf
from tqdm import tqdm

from funasr import AutoModel


# 日志配置（与 run_funasr 风格保持一致）
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/vad.log', encoding='utf-8')
    ]
)


AudioPath = Union[str, Path]


def _ensure_16k_wav(audio_path: AudioPath) -> Tuple[str, Optional[str]]:
    """
    若输入音频采样率非 16k，则转换到临时 16k wav 文件并返回其路径；
    返回 (processed_path, temp_path)。调用方需在使用后删除 temp_path。
    """
    info = sf.info(str(audio_path))
    if info.samplerate == 16000:
        return str(audio_path), None

    logging.debug(f"Converting to 16kHz: {audio_path} ({info.samplerate} -> 16000)")
    audio_data, _ = librosa.load(str(audio_path), sr=16000, mono=True)

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_path = tmp.name
    tmp.close()
    sf.write(tmp_path, audio_data, 16000)
    return tmp_path, tmp_path


def _parse_vad_result_item(item) -> List[Tuple[float, float]]:
    """
    解析 FunASR VAD 单条结果为秒级区间列表 [(start_s, end_s), ...]

    兼容多种可能的返回结构：
    - {"value": [[s_ms, e_ms], ...]}  (常见)
    - {"value": [[s_s, e_s], ...]}   (少见)
    - {"speech_segments": [{"start": s, "end": e}, ...]}
    - 直接返回 [[s, e], ...]
    单位判定：若数值 > 1000 视为毫秒，否则视为秒。
    """
    segments_raw = None

    if isinstance(item, dict):
        if 'value' in item:
            segments_raw = item['value']
        elif 'speech_segments' in item:
            segments_raw = item['speech_segments']
        elif 'segments' in item:
            segments_raw = item['segments']
    elif isinstance(item, list):
        segments_raw = item

    result: List[Tuple[float, float]] = []
    if not segments_raw:
        return result

    def _to_sec(a: float, b: float) -> Tuple[float, float]:
        # 粗略判断单位：大于 1000 认为是毫秒
        if max(a, b) > 1000.0:
            return a / 1000.0, b / 1000.0
        return float(a), float(b)

    for seg in segments_raw:
        if isinstance(seg, dict):
            s = seg.get('start')
            e = seg.get('end')
            if s is None or e is None:
                continue
            s_s, e_s = _to_sec(float(s), float(e))
            if e_s > s_s:
                result.append((s_s, e_s))
        elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
            s_s, e_s = _to_sec(float(seg[0]), float(seg[1]))
            if e_s > s_s:
                result.append((s_s, e_s))

    return result


def _slice_and_save(audio_path: AudioPath, segments_s: List[Tuple[float, float]], output_dir: Path) -> int:
    """将区间切分并保存为 wav，返回保存数量。"""
    if not segments_s:
        return 0

    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = Path(audio_path).stem
    count = 0
    for idx, (start_s, end_s) in enumerate(segments_s):
        beg = max(0, int(start_s * sr))
        end = min(len(y), int(end_s * sr))
        if end <= beg:
            continue
        out_path = output_dir / f"{base}_seg{idx + 1}.wav"
        sf.write(str(out_path), y[beg:end], sr)
        count += 1
    return count


def process_folder(folder: str, batch_size: int = 16, model_name: Optional[str] = None):
    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.webm', '.opus']

    root = Path(folder)
    files: List[Path] = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in audio_extensions]
    if not files:
        logging.info(f"No audio files found under: {folder}")
        return

    # 模型加载
    model_str = model_name or os.environ.get('FUNASR_VAD_MODEL', 'fsmn-vad')
    logging.info(f"Loading FunASR VAD model: {model_str}")
    vad_model = AutoModel(model=model_str)
    logging.info("VAD model loaded")

    # 分批处理
    for start in tqdm(range(0, len(files), batch_size), desc="VAD batches", unit="batch"):
        batch_paths = files[start:start + batch_size]
        if not batch_paths:
            continue

        # 确保 16k 输入（如需）
        processed_paths: List[str] = []
        temps: List[Optional[str]] = []
        try:
            for p in batch_paths:
                proc, tmp = _ensure_16k_wav(p)
                processed_paths.append(proc)
                temps.append(tmp)

            # 推理（若批量失败，回退到单条处理）
            try:
                res = vad_model.generate(input=processed_paths, disable_pbar=True, batch_size=len(processed_paths))
            except Exception as e:
                logging.warning(f"Batch VAD failed, fallback to single inference: {e}")
                res = []
                for proc in processed_paths:
                    item = vad_model.generate(input=proc, disable_pbar=True)
                    res.append(item if isinstance(item, dict) or isinstance(item, list) else item)

            # 统一为 list
            if not isinstance(res, list):
                res = [res]

            # 对齐长度（有的实现返回 [ {..} ]）
            if len(res) == 1 and len(processed_paths) > 1:
                res = res * len(processed_paths)

            # 逐条解析并保存
            for src_path, item in zip(batch_paths, res):
                segments_s = _parse_vad_result_item(item if not isinstance(item, list) else (item[0] if item else {}))
                out_dir = src_path.parent / 'sliced'
                saved = _slice_and_save(src_path, segments_s, out_dir)
                logging.info(f"{src_path.name}: saved {saved} segments")

        finally:
            # 清理临时文件
            for tmp in temps:
                if tmp and os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch VAD slicing using FunASR FSMN-VAD")
    parser.add_argument("folder", type=str, help="输入目录，递归遍历处理")
    parser.add_argument("--batch_size", type=int, default=None, help="每批处理的文件数，默认读取环境变量 FUNASR_BATCH_SIZE 或 16")
    parser.add_argument("--model", type=str, default=None, help="FunASR VAD 模型名，默认 'fsmn-vad'，亦可用环境变量 FUNASR_VAD_MODEL 覆盖")

    args = parser.parse_args()

    # 批大小：优先命令行，其次环境变量，最后默认 16
    if args.batch_size is not None:
        bs = max(1, int(args.batch_size))
    else:
        try:
            bs = max(1, int(os.environ.get("FUNASR_BATCH_SIZE", "16")))
        except Exception:
            bs = 16

    process_folder(args.folder, batch_size=bs, model_name=args.model)


