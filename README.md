## Shore-Data-Engine

面向语音数据收集与预处理的一体化流水线，目标是为 TTS/ASR 等任务快速构建高质量音频-文本数据集。

包含模块：
- 爬虫：蜻蜓 FM、喜马拉雅 FM、VoiceWiki 语音及文本采集
- 数据切分：基于 RMS 阈值的静音切分
- ASR：Whisper、FunASR、Dolphin、Qwen-ASR 多模型转写
- UVR：Demucs 声部分离（提取人声）
- 文本润色：调用大模型对转写结果进行自动润色
- 数据蒸馏/TTS：Fish-Audio、火山引擎 Seed-TTS 文本转语音

### 目录结构

```text
Shore-Data-Engine/
  crawler/            # 爬虫脚本（qtfm, xmly, voicewiki）
  data/               # 数据目录（建议将原始/切分/转写等放置在此）
  data_process/
    asr/              # ASR 转写脚本（whisper, funasr, dolphin, qwen_asr）
    slicer/           # 静音切分脚本
    uvr/              # Demucs 声部分离
    distill/          # 蒸馏
  logs/               # 日志输出目录
  tools/              # 辅助工具（统计时长、导出标签、LLM润色）
  requirements.txt
  README.md
```

### 环境要求
- Python 3.9+（推荐 3.10）
- 建议使用 GPU（NVIDIA CUDA）以加速 ASR/UVR
- 对于 `torch`/`torchaudio`，请根据显卡和 CUDA 版本从官方指引安装对应版本
- Windows 与 Linux 均可运行；`vLLM` 仅在 Linux 上支持
- 若使用 `crawler/xmly/crawler.py`，需要本机安装 Node.js 以运行 `decode.js`
- 若使用 Qwen-ASR，需安装 `dashscope` 并在配置或环境变量中提供 API Key

### 安装
1) 创建虚拟环境（推荐）

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows PowerShell
# 或
source .venv/bin/activate  # Linux / macOS
```

2) 安装依赖

```bash
pip install -U pip
pip install -r requirements.txt

# 根据你的 CUDA 环境，单独安装合适版本的 torch/torchaudio（如果上一步未自动装好）
# 例如：参照 PyTorch 官网指令安装
```

3) 可选：准备检查点目录
- Whisper 默认会下载到 `./checkpoints/whisper-large-v3-turbo`
- Dolphin 模型保存到 `./checkpoints/dolphin`
- Demucs 会自动下载到缓存或本地路径

### 快速开始
以下示例演示常见的“切分 -> 转写 -> 统计 -> 可选润色/分离”流程。

1) 将原始音频放入 `data/` 下（支持 `.wav/.mp3/.m4a/.ogg/.flac/.aac/.wma/.webm/.opus`）

2) 静音切分

```bash
python data_process/slicer/lets_slice.py
```

输出切分片段到每个数据子目录内的 `sliced/` 子目录（如 `data/<专辑>/sliced/`；若音频直接位于 `data/` 根目录，则输出到 `data/sliced/`）。

3) 选择一种 ASR 模型进行转写
- Whisper（默认）：

```bash
python data_process/asr/run_whisper.py
```

- FunASR（中文体验更好，内置 VAD/断句/标点）：

```bash
python data_process/asr/run_funasr.py
```

- Dolphin（适合方言/小语种，注意：仅支持 < 30 秒音频片段）：

```bash
python data_process/asr/run_dolphin.py
```

- Qwen-ASR（云端服务，按量计费，多语种）：

```bash
python data_process/asr/run_qwen_asr.py
```

提示：
- 以上 ASR 脚本会自动遍历 `data/**/sliced` 目录，并在各自 `sliced` 目录下生成/增量更新 `transcription.json`（键为音频绝对路径，值为转写文本）。
- 使用 Qwen-ASR 前，请在 `data_process/asr/configs/qwen_asr.json` 填写 `api_key`，或设置环境变量 `DASHSCOPE_API_KEY`。

4) 统计切分后总时长

ASR 脚本会在处理完成后自动统计对应 `sliced` 目录的总时长。若需手动统计，可在交互式环境中：

```python
from tools.calculate_time import calculate_time
calculate_time("data")           # 统计 data/ 下所有受支持格式
calculate_time("data/专辑/sliced")  # 指定某个 sliced 目录
```

5) 可选：Demucs 声部分离（提取人声）

```bash
python data_process/uvr/run_demucs.py
```

6) 可选：LLM 文本润色（以 DeepSeek-Chat 为例）
`tools/llm_polish.py` 提供 `deepseek(json_file, client, fangyan)` 函数，你可以在交互式环境或脚本中调用：

```python
from openai import OpenAI
from tools.llm_polish import deepseek

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api.deepseek.com"  # 按实际服务商填写
)

deepseek(
    json_file="data/sliced/transcription.json",
    client=client,
    fangyan=True  # 方言文本润色可设为 True
)
```

7) 可选：导出标签文本（为每段音频生成同名 `.txt` 文本）

```bash
python tools/extract_label.py
```

该脚本会读取每个 `sliced/transcription.json`，在音频同目录写入同名 `.txt`。

8) 可选：根据映射批量合成语音（数据蒸馏 / 文本到语音）

- Fish-Audio SDK：

```bash
python data_process/distill/fish_audio/generate.py \
  --config data_process/distill/fish_audio/config.json \
  --mapping data/distill/mapping.json
```

- 火山引擎 Seed-TTS：

```bash
python data_process/distill/seed_tts/generate.py \
  --config data_process/distill/seed_tts/config.json \
  --mapping data/distill/mapping.json
```

说明：
- 两者均支持传入 `--mapping`（形如 `{ "filename": "text" }`），会在配置的 `OUT_DIR` 下按文件名批量合成；未传 `--mapping` 时默认使用配置中的单条 `TEXT`。
- 使用前请在各自 `config.json` 中填入服务的鉴权信息（API Key / APP_ID / ACCESS_TOKEN / VOICE_TYPE 等）。

### 各模块说明与注意事项

- 爬虫 `crawler/`
  - `qtfm/crawler.py`
    - 修改 `PROGRAM_URL`，脚本会自动翻页并下载到 `data/<专辑名>/`
    - 使用 HMAC MD5 构造签名并通过 302 重定向获取下载链接
  - `xmly/crawler.py`
    - 修改 `REQUEST_URL`，并确保本机安装 Node.js
    - 运行时将调用 `decode.js` 解密音频直链
    - 该站点有风控，务必控制请求频率
  - `voicewiki/crawler.py`
    - 修改 `character_url` 为角色主页，脚本自动发现语音子页面并下载 `.ogg` 与对应中文文本到 `data/<角色名>/`

- 切分 `data_process/slicer/`
  - `lets_slice.py`：批量处理 `data/` 下 `.wav`，在原目录生成 `sliced/` 子目录
  - `slicer.py`：核心静音切分逻辑（来自 openvpi/audio-slicer，MIT）

- ASR `data_process/asr/`
  - `run_whisper.py`：默认加载 `large-v3-turbo`，首轮会自动下载模型
  - `run_funasr.py`：使用 Paraformer-zh + FSMN-VAD + 标点恢复
  - `run_dolphin.py`：适合方言/小语种，单段音频需 < 30 秒
  - `run_qwen_asr.py`：调用阿里 DashScope Qwen-ASR，需提供 API Key
  - 输出统一增量写入各 `sliced/transcription.json`

- UVR `data_process/uvr/`
  - `run_demucs.py`：基于 Demucs 的人声分离，模型 `htdemucs_ft`

- 数据蒸馏/TTS `data_process/distill/`
  - `fish_audio/`：基于 Fish-Audio SDK；支持参考音频与采样控制
  - `seed_tts/`：基于火山引擎 WebSocket TTS；支持按映射批量生成

- 日志 `logs/`
  - `whisper_transcription.log`、`funasr_transcription.log`、`dolphin_transcription.log`、`demucs.log`、`llm_polish.log`

### 平台与依赖建议
- PyTorch/torchaudio：请严格按官方说明选择与 CUDA 匹配的版本安装
- `vLLM` 加速（可选）：目前仅支持 Linux。若要启用 `run_whisper.py` 的 vLLM 路径，请在 Linux 下手动安装 `vllm` 并将脚本开关设为 `use_vllm=True`
- Windows 上如不使用 vLLM，可忽略该依赖
- `xmly` 爬虫需 Node.js 支持
- Qwen-ASR 需安装 `dashscope`，并提供 `DASHSCOPE_API_KEY`

### 常见问题
- 首次运行下载模型耗时较长：请保持网络畅通并预留足够磁盘空间
- CUDA 不匹配导致 `torch` 无法使用 GPU：请重新按 PyTorch 官网上的指令安装对应版本
- `dolphin` 仅处理 < 30 秒片段：请先用切分工具缩短音频
- `voicewiki` 未自动识别到语音页面：脚本会在 `debug/` 下保存 HTML，便于排查
- `run_whisper.py` 的 vLLM 路径仅在 Linux 可用；Windows 会自动回退到基础推理

### 许可证与致谢
- 本项目遵循仓库根目录的 `LICENSE`
- `data_process/slicer/slicer.py` 来自 `openvpi/audio-slicer`（MIT）
