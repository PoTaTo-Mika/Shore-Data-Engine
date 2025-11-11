import json
import os
from pathlib import Path
from typing import Dict, Iterable
import shutil

# Vibe coding with GPT-5

def iter_sliced_dirs(data_root: Path) -> Iterable[Path]:
    for p in data_root.rglob('sliced'):
        if p.is_dir():
            yield p


def load_transcription(json_path: Path) -> Dict[str, str]:
    try:
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def extract_filename_from_key(key: str) -> str:
    # 按用户要求优先按 '/' 分割取最后一个
    if '/' in key:
        return key.rsplit('/', 1)[-1]
    # 兼容 Windows 绝对路径情况
    if '\\' in key:
        return key.rsplit('\\', 1)[-1]
    return key


def write_text_label_for_audio(audio_path: Path, text: str, preferred_filename: str | None = None) -> Path:
    audio_dir = audio_path.parent
    if preferred_filename:
        # 使用从 key 中抽取出来的文件名来确定标签文件名
        stem, _sep, _ext = preferred_filename.partition('.')
        label_name = f"{stem}.txt"
    else:
        label_name = f"{audio_path.stem}.txt"
    label_path = audio_dir / label_name
    label_path.write_text(text.strip() + "\n", encoding='utf-8')
    return label_path


def get_album_name(sliced_dir: Path) -> str:
    """从 sliced 目录路径中提取专辑名称"""
    # 假设路径结构为: data/{album_name}/sliced
    parent_dir = sliced_dir.parent
    return parent_dir.name if parent_dir != sliced_dir else "unknown"


def move_transcription_json(sliced_dir: Path, target_dir: Path) -> None:
    """移动并重命名 transcription.json 文件"""
    json_path = sliced_dir / 'transcription.json'
    if not json_path.exists():
        return
    
    album_name = get_album_name(sliced_dir)
    # 生成新的文件名，防止冲突
    new_filename = f"transcription_{album_name}.json"
    target_path = target_dir / new_filename
    
    # 如果目标文件已存在，添加数字后缀
    counter = 1
    original_target = target_path
    while target_path.exists():
        stem = original_target.stem
        target_path = original_target.parent / f"{stem}_{counter}.json"
        counter += 1
    
    # 创建目标目录（如果不存在）
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 移动文件
    shutil.move(str(json_path), str(target_path))
    print(f"[extract_label] 已移动 transcription.json -> {target_path}")


def process_one_sliced_dir(sliced_dir: Path) -> None:
    json_path = sliced_dir / 'transcription.json'
    pairs = load_transcription(json_path)
    if not pairs:
        return

    created = 0
    skipped = 0
    for abs_audio, transcript in pairs.items():
        if not isinstance(transcript, str):
            continue
        # 绝对路径到 Path
        audio_path = Path(abs_audio)
        # 如果文件不存在，仍按 key 的文件名写到当前 sliced 目录
        if not audio_path.exists():
            fallback_name = extract_filename_from_key(abs_audio)
            stem, _sep, _ext = fallback_name.partition('.')
            label_path = sliced_dir / f"{stem}.txt"
            try:
                label_path.write_text(transcript.strip() + "\n", encoding='utf-8')
                created += 1
            except Exception:
                skipped += 1
            continue

        # 正常情况：把标签写在音频同目录（通常就在 sliced 或其子目录）
        preferred_name = extract_filename_from_key(abs_audio)
        try:
            write_text_label_for_audio(audio_path, transcript, preferred_filename=preferred_name)
            created += 1
        except Exception:
            skipped += 1

    print(f"[extract_label] {sliced_dir}: created={created}, skipped={skipped}")


def main():
    data_root = Path('data')
    if not data_root.exists():
        print("[extract_label] 'data' 目录不存在，已跳过")
        return

    sliced_dirs = list(iter_sliced_dirs(data_root))
    if not sliced_dirs:
        print("[extract_label] 未找到任何 'sliced' 目录")
        return

    # 创建目标目录
    target_dir = data_root / 'transcription'
    
    # 先处理所有标签提取
    for sliced_dir in sliced_dirs:
        process_one_sliced_dir(sliced_dir)
    
    # 然后移动所有 transcription.json 文件
    print("\n[extract_label] 开始移动 transcription.json 文件...")
    for sliced_dir in sliced_dirs:
        move_transcription_json(sliced_dir, target_dir)


if __name__ == '__main__':
    main()