import os
import soundfile as sf
import librosa
from pathlib import Path
import time

supported_formats = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.opus']

def calculate_time(folder_path):
    
    # 使用pathlib的Path对象来遍历
    folder = Path(folder_path)
    
    # 检查文件夹是否存在
    if not folder.exists():
        print(f"文件夹 {folder_path} 不存在")
        return
    
    # 查找所有子目录中的sliced目录
    sliced_dirs = []
    for subdir in folder.iterdir():
        if subdir.is_dir():
            sliced_path = subdir / "sliced"
            if sliced_path.exists() and sliced_path.is_dir():
                sliced_dirs.append((subdir.name, sliced_path))
    
    if not sliced_dirs:
        print("未找到任何包含sliced目录的子目录")
        return
    
    print(f"\n{'='*50}")
    print("专辑切片总时长统计:")
    print(f"{'='*50}")
    
    total_all_duration = 0
    
    for album_name, sliced_dir in sliced_dirs:
        audio_files = []
        # 遍历sliced目录中的所有音频文件
        for file_path in sliced_dir.rglob('*'):
            if file_path.is_file():
                file_extension = file_path.suffix.lower()
                if file_extension in supported_formats:
                    audio_files.append(str(file_path))
        
        # 计算该sliced目录的总时长
        total_duration = 0
        for audio_file in audio_files:
            try:
                duration = librosa.get_duration(path=audio_file)
                total_duration += duration
            except Exception as e:
                print(f"无法读取文件 {audio_file}: {e}")
        
        total_all_duration += total_duration
        # 直接打印专辑和时长信息
        print(f"{album_name}: {total_duration/3600:.2f} 小时")
    
    print(f"{'='*50}")
    print(f"所有专辑切片总时长: {total_all_duration/3600:.2f} 小时")

if __name__ == "__main__":
    folder_path = "data"
    calculate_time(folder_path)