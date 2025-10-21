#!/bin/bash

# 强制设置脚本运行环境为UTF-8
export LC_ALL=C.UTF-8

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <目录路径> [并发线程数]"
    echo "默认并发线程数: 4"
    exit 1
fi

DIR="$1"
THREADS="${2:-4}"  # 默认4个线程

# 检查目录是否存在
if [ ! -d "$DIR" ]; then
    echo "错误: 目录 '$DIR' 不存在"
    exit 1
fi

# 检查线程数是否为数字
if ! [[ "$THREADS" =~ ^[0-9]+$ ]] || [ "$THREADS" -lt 1 ]; then
    echo "错误: 线程数必须是正整数"
    exit 1
fi

# 使用 realpath 将输入目录转换为绝对路径，避免所有相对路径问题
ABS_DIR=$(realpath "$DIR")

echo "将在绝对路径下查找并转换文件: $ABS_DIR"
echo "使用并发线程数: $THREADS"

# 创建临时目录用于进程控制
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# 创建命名管道用于控制并发
mkfifo "$TMP_DIR/queue"
exec 3<>"$TMP_DIR/queue"

# 初始化信号量
for ((i=0; i<THREADS; i++)); do
    echo >&3
done

# 转换单个文件的函数
convert_file() {
    local file="$1"
    local basename="${file%.*}"
    
    echo "转换: $file"
    
    # 转换为Opus格式，使用libopus编码器，设置音频质量
    if ffmpeg -nostdin -i "$file" -c:a libopus -b:a 128k -vbr on -y "$basename.opus" 2>/dev/null; then
        echo "完成: $basename.opus"
        rm "$file"
        echo "已删除源文件: $file"
    else
        echo "转换失败，保留源文件: $file"
    fi
}

# 导出函数以便在子shell中使用
export -f convert_file

echo "开始查找音频文件..."

# 查找所有音频文件并并行处理
find "$ABS_DIR" -type f \( -iname "*.mp3" -o -iname "*.flac" -o -iname "*.m4a" -o -iname "*.aac" -o -iname "*.ogg" -o -iname "*.wma" \) -print0 | \
while IFS= read -r -d '' file; do
    read -u 3  # 从信号量读取
    {
        convert_file "$file"
        echo >&3  # 释放信号量
    } &
done

# 等待所有后台任务完成
wait
exec 3>&-  # 关闭文件描述符

echo "所有转换任务完成！"