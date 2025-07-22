from slicer import Slicer
import librosa
import soundfile
import os
from tqdm import tqdm
# for those audio with not much noise

def process_audio(audio_path, slicer):
    # get audio name
    audio_name = audio_path.split('/')[-1].split('.')[0]
    waveform, sr = librosa.load(audio_path, sr=44100)
    chunks = slicer.slice(waveform)
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T  # Swap axes if the audio is stereo.
        soundfile.write(f'{audio_name}_{i}.wav', chunk, sr)  # Save sliced audio files with soundfile.

def process_folder(folder_path):
    slicer = Slicer(
        sr = 44100,
        threshold = -30,
        min_length = 10000, # 10s
        min_interval = 500, # 500ms
        hop_size = 10,
        max_sil_kept = 500,
    )
    for file in tqdm(os.listdir(folder_path)):
        if file.endswith('.wav'):
            process_audio(os.path.join(folder_path, file), slicer)

if __name__ == '__main__':
    folder_path = './data/wav'
    process_folder(folder_path)