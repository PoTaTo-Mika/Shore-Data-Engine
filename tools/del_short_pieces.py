import librosa
import os
import json
import argparse
from pathlib import Path

def get_audio_duration(file_path):
    """获取音频文件时长（毫秒）"""
    try:
        duration = librosa.get_duration(path=file_path)
        return duration * 1000  # 转换为毫秒
    except Exception as e:
        print(f"无法读取音频文件 {file_path}: {e}")
        return None

def delete_short_audio_files(directory, min_duration_ms=2000):
    """删除指定目录下短于指定时长的音频文件"""
    directory = Path(directory)
    if not directory.exists():
        print(f"目录不存在: {directory}")
        return
    
    # 支持的音频格式
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
    
    deleted_count = 0
    total_count = 0
    
    # 递归遍历目录
    for audio_file in directory.rglob('*'):
        if audio_file.is_file() and audio_file.suffix.lower() in audio_extensions:
            total_count += 1
            print(f"检查文件: {audio_file}")
            
            duration_ms = get_audio_duration(audio_file)
            if duration_ms is not None:
                if duration_ms < min_duration_ms:
                    try:
                        audio_file.unlink()  # 删除文件
                        print(f"已删除短音频文件: {audio_file} (时长: {duration_ms:.1f}ms)")
                        deleted_count += 1
                    except Exception as e:
                        print(f"删除文件失败 {audio_file}: {e}")
                else:
                    print(f"保留文件: {audio_file} (时长: {duration_ms:.1f}ms)")
    
    print(f"\n处理完成!")
    print(f"总共检查了 {total_count} 个音频文件")
    print(f"删除了 {deleted_count} 个短音频文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="删除指定目录下短于指定时长的音频文件")
    parser.add_argument("directory", help="要检查的目录路径")
    parser.add_argument("--min-duration", type=int, default=2000, 
                       help="最短音频时长（毫秒），默认2000ms")
    
    args = parser.parse_args()
    
    print(f"开始检查目录: {args.directory}")
    print(f"最短时长阈值: {args.min_duration}ms")
    print("-" * 50)
    
    delete_short_audio_files(args.directory, args.min_duration)
