#!/bin/bash

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <目录路径>"
    exit 1
fi

DIR="$1"

# 检查目录是否存在
if [ ! -d "$DIR" ]; then
    echo "错误: 目录 '$DIR' 不存在"
    exit 1
fi

# 递归查找并转换音频文件
# 使用进程替换而不是管道来避免子shell问题
while IFS= read -r -d '' file; do
    # 获取不带扩展名的文件名
    basename="${file%.*}"
    
    # 转换为wav
    echo "转换: $file"
    if ffmpeg -i "$file" -y "$basename.wav" 2>/dev/null; then
        echo "完成: $basename.wav"
        rm "$file"
        echo "已删除源文件: $file"
    else
        echo "转换失败，保留源文件: $file"
    fi
done < <(find "$DIR" -type f \( -iname "*.mp3" -o -iname "*.flac" -o -iname "*.m4a" -o -iname "*.aac" -o -iname "*.ogg" -o -iname "*.wma" \) -print0)

echo "转换完成！"