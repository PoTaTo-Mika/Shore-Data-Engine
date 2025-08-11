from data_process.slicer.slicer import Slicer
import librosa
import soundfile
import os
from tqdm import tqdm
# for those audio with not much noise

def process_audio(audio_path, slicer, output_dir):
    # get audio name
    audio_name = os.path.basename(audio_path).split('.')[0]
    waveform, sr = librosa.load(audio_path, sr=44100)
    chunks = slicer.slice(waveform)
    
    # Create sliced directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T  # Swap axes if the audio is stereo.
        soundfile.write(os.path.join(output_dir, f'{audio_name}_{i}.wav'), chunk, sr)  # Save sliced audio files with soundfile.

def process_folder(folder_path):
    slicer = Slicer(
        sr = 44100,
        threshold = -30,
        min_length = 10000, # 10s
        min_interval = 500, # 500ms
        hop_size = 10,
        max_sil_kept = 500,
    )
    
    # 收集所有音频文件，递归处理子目录
    audio_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print(f"找到 {len(audio_files)} 个音频文件需要处理")
    
    # 处理每个音频文件
    for audio_file in tqdm(audio_files, desc="切片处理音频文件"):
        # 获取相对于根目录的路径，确定输出目录
        rel_path = os.path.relpath(audio_file, folder_path)
        audio_dir = os.path.dirname(rel_path)
        
        if audio_dir:  # 如果在子目录中
            output_dir = os.path.join(folder_path, audio_dir, 'sliced')
        else:  # 如果直接在根目录中
            output_dir = os.path.join(folder_path, 'sliced')
            
        process_audio(audio_file, slicer, output_dir)

if __name__ == '__main__':
    folder_path = 'data'
    process_folder(folder_path)