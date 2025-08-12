from data_process.slicer.slicer import Slicer
import librosa
import soundfile
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
# for those audio with not much noise

def process_audio(audio_path, output_dir):
    # 在子进程内构造 slicer，避免对象跨进程序列化问题
    slicer = Slicer(
        sr=44100,
        threshold=-30,
        min_length=10000,  # 10s
        min_interval=500,  # 500ms
        hop_size=10,
        max_sil_kept=500,
    )

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


def _process_single(args):
    audio_path, output_dir = args
    process_audio(audio_path, output_dir)

def process_folder(folder_path, max_workers=None):
    # 收集所有音频文件，递归处理子目录
    audio_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print(f"找到 {len(audio_files)} 个音频文件需要处理")
    
    # 为每个音频文件计算输出目录并并发处理
    tasks = []
    for audio_file in audio_files:
        # 获取相对于根目录的路径，确定输出目录
        rel_path = os.path.relpath(audio_file, folder_path)
        audio_dir = os.path.dirname(rel_path)

        if audio_dir:  # 如果在子目录中
            output_dir = os.path.join(folder_path, audio_dir, 'sliced')
        else:  # 如果直接在根目录中
            output_dir = os.path.join(folder_path, 'sliced')

        tasks.append((audio_file, output_dir))

    if max_workers is None:
        try:
            max_workers = os.cpu_count() or 1
        except Exception:
            max_workers = 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(_process_single, tasks), total=len(tasks), desc="切片处理音频文件"))

if __name__ == '__main__':
    folder_path = 'data'
    process_folder(folder_path)