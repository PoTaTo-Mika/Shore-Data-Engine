# Shore-Data-Engine

为了构建TTS而准备的数据处理pipeline

## 爬虫

目前只提供了两个爬虫：蜻蜓FM和喜马拉雅FM，其中喜马拉雅FM极其容易被风控

## ASR

提供了两种ASR的模型：Whisper和Dolphin

## UVR

使用htdemucs

## 日志输出

所有处理模块的日志文件都会输出到 `logs/` 目录中：
- `logs/whisper_transcription.log` - Whisper ASR处理日志
- `logs/dolphin_transcription.log` - Dolphin ASR处理日志  
- `logs/demucs.log` - UVR音频分离日志
- `logs/llm_polish.log` - LLM文本润色日志
