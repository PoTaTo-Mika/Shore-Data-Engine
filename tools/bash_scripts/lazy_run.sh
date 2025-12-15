# 懒狗脚本已一键配置
# bash tools/bash_scripts/lazy_run.sh
# 进行demucs处理
echo "Start demucs..."
python -m data_process.uvr.run_demucs

# 开始切分
echo "Start slicing..."
python -m data_process.slicer.silero_vad_
# python -m data_process.fsmn_vad

# 开始asr处理
# python data_process/asr/run_whisper.py
python -m data_process.asr.run_funasr
# python data_process/asr/run_funasr_chuan.py

# 最后提取label为一个个小文件
python tools/extract_label.py

# 检查data目录
# 归档并切分每个子目录下的 sliced 结果（固定每卷 10G）
echo "Start packaging..."

DATA_DIR="data"
UPDATE_DIR="$DATA_DIR/update"
CHUNK_SIZE="10G"

mkdir -p "$UPDATE_DIR"

for dir in "$DATA_DIR"/*/; do
    base=$(basename "$dir")
    sliced_dir="${dir}sliced"
    target_dir="${dir}${base}"

    # 若存在 sliced 则重命名为父目录名
    if [ -d "$sliced_dir" ]; then
        mv "$sliced_dir" "$target_dir"
    fi

    # 若不存在目标目录则跳过
    [ -d "$target_dir" ] || continue

    # 打包并分卷（纯归档，不压缩），随后移动到 update
    tar -C "$dir" -cf - "$base" | split -b "$CHUNK_SIZE" -d -a 3 - "${DATA_DIR}/${base}.tar."
    mv "${DATA_DIR}/${base}.tar."* "$UPDATE_DIR"/
done

echo "Done. Tar parts -> $UPDATE_DIR"

# 最后，我们把tar也安排好folder
echo "Organizing tar parts into per-album folders..."
for first_part in "$UPDATE_DIR"/*.tar.000; do
    [ -e "$first_part" ] || break
    fname=$(basename "$first_part")
    album="${fname%%.tar.*}"
    mkdir -p "$UPDATE_DIR/$album"
    mv "$UPDATE_DIR/${album}.tar."* "$UPDATE_DIR/$album"/
done
echo "Done. Albums are in $UPDATE_DIR/<album>/<album>.tar.000..."

export HF_ENDPOINT=https://hf-mirror.com

hf upload PoTaTo721/Shore-Lunch-box ./data/update TTS/Chinese/mandarin --repo-type=dataset