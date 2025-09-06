# 懒狗脚本已一键配置
# bash tools/bash_scripts/lazy_run.sh

# 将所有的音频全部转换为wav格式
echo "Start converting..."
bash tools/bash_scripts/all_to_wav.sh data

# 进行demucs处理
echo "Start demucs..."
python -m data_process.uvr.run_demucs

# 开始切分
echo "Start slicing..."
python -m data_process.slicer.lets_slice
# python -m data_process.slicer.vad

# 开始asr处理
# python data_process/asr/run_whisper.py
python -m data_process.asr.run_funasr
# python data_process/asr/run_dolphin.py

# 最后提取label为一个个小文件
# python tools/extract_label.py
