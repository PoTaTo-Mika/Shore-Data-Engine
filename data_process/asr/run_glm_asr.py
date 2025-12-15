import os
import json
import torch
import torchaudio
import logging
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
)

# ================= 配置区域 =================
CHECKPOINT_DIR = "checkpoints/glm-asr-nano" 
DATA_ROOT = "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_FEAT_CFG = {
    "chunk_length": 30,
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "WhisperProcessor",
    "return_attention_mask": False,
    "sampling_rate": 16000,
}

# ================= 辅助函数 =================

def get_audio_token_length(seconds, merge_factor=2):
    """计算音频对应的Token长度 (核心逻辑)"""
    def get_T_after_cnn(L_in, dilation=1):
        for padding, kernel_size, stride in [(1,3,1), (1,3,2)]:
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

    mel_len = int(seconds * 100)
    audio_len_after_cnn = get_T_after_cnn(mel_len)
    audio_token_num = (audio_len_after_cnn - merge_factor) // merge_factor + 1
    # 限制最大长度
    return min(audio_token_num, 1500 // merge_factor)

# ================= 模型封装类 =================

class AudioLLMTranscriber:
    def __init__(self, checkpoint_dir, device):
        self.device = device
        print(f"正在加载模型: {checkpoint_dir} ...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
            self.feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)
            self.config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir,
                config=self.config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device)
            self.model.eval()
            self.merge_factor = getattr(self.config, "merge_factor", 2)
            print("模型加载完成。")
        except Exception as e:
            print(f"模型加载失败: {e}")
            exit(1)

    def transcribe_file(self, audio_path: str, max_new_tokens=128):
        # 1. 加载音频
        wav, sr = torchaudio.load(audio_path)
        wav = wav[:1, :] # 确保单声道
        if sr != self.feature_extractor.sampling_rate:
            wav = torchaudio.transforms.Resample(sr, self.feature_extractor.sampling_rate)(wav)

        # 2. 预处理 (目前逻辑只处理前30秒，适配inference.py逻辑)
        chunk_size = 30 * self.feature_extractor.sampling_rate
        chunk = wav[:, :chunk_size]
        
        mel = self.feature_extractor(
            chunk.numpy(),
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding="max_length",
        )["input_features"]

        seconds = chunk.shape[1] / self.feature_extractor.sampling_rate
        num_tokens = get_audio_token_length(seconds, self.merge_factor)

        # 3. 构建 Prompt (严格遵循 inference.py)
        tokens = []
        tokens += self.tokenizer.encode("<|user|>")
        tokens += self.tokenizer.encode("\n")
        tokens += self.tokenizer.encode("<|begin_of_audio|>")
        
        audio_offsets = [len(tokens)]
        tokens += [0] * num_tokens # 音频占位符
        
        tokens += self.tokenizer.encode("<|end_of_audio|>")
        audio_length = [num_tokens]
        
        tokens += self.tokenizer.encode("<|user|>")
        tokens += self.tokenizer.encode("\nPlease transcribe this audio into text")
        tokens += self.tokenizer.encode("<|assistant|>")
        tokens += self.tokenizer.encode("\n")

        # 4. 准备输入
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        attention_mask = torch.ones(1, len(tokens), dtype=torch.long).to(self.device)
        audios = mel.to(self.device).to(torch.bfloat16)

        model_inputs = {
            "inputs": input_ids,
            "attention_mask": attention_mask,
            "audios": audios,
            "audio_offsets": [audio_offsets],
            "audio_length": [audio_length],
        }

        # 5. 生成
        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        prompt_len = input_ids.size(1)
        transcript_ids = generated[0, prompt_len:].cpu().tolist()
        transcript = self.tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()
        
        return transcript

# ================= 业务逻辑 =================

def get_sliced_dirs(root_dir):
    """查找所有名为 sliced 的子目录"""
    root_path = Path(root_dir)
    return [p for p in root_path.rglob('sliced') if p.is_dir()]

def process_single_folder(folder_path: Path, transcriber: AudioLLMTranscriber):
    """处理单个文件夹：读取->转写->保存"""
    json_output_path = folder_path / "transcription.json"
    audio_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.webm', '.opus'}
    
    # 加载现有进度
    transcription_pairs = {}
    if json_output_path.exists():
        try:
            with open(json_output_path, "r", encoding='utf-8') as f:
                transcription_pairs = json.load(f)
            print(f"  - 已加载现有记录: {len(transcription_pairs)} 条")
        except Exception:
            print("  - JSON文件损坏，将重新创建")

    # 扫描音频文件
    audio_files = [f for f in folder_path.iterdir() if f.suffix.lower() in audio_extensions]
    
    # 过滤掉已经转写过的
    files_to_process = [f for f in audio_files if str(f.absolute()) not in transcription_pairs]
    
    if not files_to_process:
        print(f"  - 文件夹 {folder_path.name} 无需更新。")
        return

    print(f"  - 待处理文件数: {len(files_to_process)}")

    # 遍历处理
    for audio_file in tqdm(files_to_process, desc=f"Processing {folder_path.parent.name}", unit="file"):
        abs_path = str(audio_file.absolute())
        try:
            text = transcriber.transcribe_file(abs_path)
            transcription_pairs[abs_path] = text
            
            # 实时保存 (像 run_whisper.py 一样)
            with open(json_output_path, "w", encoding='utf-8') as f:
                json.dump(transcription_pairs, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"\n  [Error] 处理文件失败 {audio_file.name}: {e}")

def main():
    print("=" * 60)
    print("Audio-LLM 批量转写脚本 (MVP)")
    print("=" * 60)

    # 1. 检查模型路径
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"错误: 找不到模型目录 '{CHECKPOINT_DIR}'。")
        print("请修改脚本顶部的 CHECKPOINT_DIR 变量。")
        return

    # 2. 初始化模型 (只加载一次)
    transcriber = AudioLLMTranscriber(CHECKPOINT_DIR, DEVICE)

    # 3. 扫描 data 目录下的 sliced 文件夹
    sliced_dirs = get_sliced_dirs(DATA_ROOT)
    
    if not sliced_dirs:
        print(f"在 '{DATA_ROOT}' 下未找到任何 'sliced' 目录。")
        return
        
    print(f"共发现 {len(sliced_dirs)} 个 sliced 目录待处理。")

    # 4. 循环处理每个目录
    for folder in sliced_dirs:
        print(f"\n正在处理目录: {folder}")
        process_single_folder(folder, transcriber)

    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()