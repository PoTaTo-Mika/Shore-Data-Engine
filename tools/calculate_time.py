import os
import soundfile as sf
import librosa
from pathlib import Path
import time

supported_formats = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.opus']

def calculate_time(folder_path):
    
    audio_files = []
    
    # 使用pathlib的Path对象来遍历
    folder = Path(folder_path)
    
    # 检查文件夹是否存在
    if not folder.exists():
        print(f"文件夹 {folder_path} 不存在")
        return 0  # 返回0而不是audio_files列表
    
    # 遍历文件夹和所有子文件夹
    for file_path in folder.rglob('*'):
        # 检查是否为文件
        if file_path.is_file():
            # 获取文件扩展名
            file_extension = file_path.suffix.lower()
            
            # 检查是否为支持的音频格式
            if file_extension in supported_formats:
                audio_files.append(str(file_path))
    
    # 统计总长度
    total_duration = 0
    for audio_file in audio_files:
        try:
            # 使用librosa获取音频时长
            duration = librosa.get_duration(path=audio_file)
            total_duration += duration
        except Exception as e:
            print(f"无法读取文件 {audio_file}: {e}")
    
    print(f"找到 {len(audio_files)} 个音频文件")
    print(f"总时长: {total_duration:.2f} 秒 ({total_duration/3600:.2f} 小时)")
    
    return total_duration 

if __name__ == "__main__":
    folder_path = "qtfm"
    total_duration = calculate_time(folder_path)
    print(f"总时长: {total_duration:.2f} 秒 ({total_duration/3600:.2f} 小时)")