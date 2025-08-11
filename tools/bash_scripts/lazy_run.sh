# 懒狗脚本已一键配置
# bash tools/bash_scripts/lazy_run.sh

# 将所有的音频全部转换为wav格式
bash tools/bash_scripts/all_to_wav.sh data

# 进行demucs处理
python data_process/uvr/run_demucs.py

# 开始切分
echo "Start Slicing..."
python data_process/slicer/lets_slice.py

# 开始asr处理
# python data_process/asr/run_whisper.py
python -m data_process.asr.run_funasr
# python data_process/asr/run_dolphin.py

# 最后提取label为一个个小文件
python tools/extract_label.py
