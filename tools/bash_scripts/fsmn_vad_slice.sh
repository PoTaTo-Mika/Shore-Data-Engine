#!/bin/bash

# ============================================
# 音频批量切分脚本 (基于VAD时间戳)
# ============================================

set -euo pipefail

# ==================== 配置参数 ====================
INPUT_DIR="${1:-./data}"
OUTPUT_DIR="${2:-./data/sliced}"
PARALLEL_JOBS="${3:-32}"        # 并行任务数
OUTPUT_FORMAT="${4:-}"          # 输出格式(留空保持原格式)
DELETE_ORIGINAL="${5:-true}"   # 是否删除原文件

# ==================== 颜色输出 ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# ==================== 检查依赖 ====================
check_dependencies() {
    local missing_deps=()
    
    for cmd in ffmpeg jq parallel bc; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "缺少依赖: ${missing_deps[*]}"
        log_info "请安装: sudo apt-get install ffmpeg jq parallel bc"
        exit 1
    fi
}

# ==================== 获取编码参数 ====================
get_codec_params() {
    local input_ext="$1"
    local output_ext="$2"
    
    input_ext=$(echo "$input_ext" | tr '[:upper:]' '[:lower:]')
    output_ext=$(echo "$output_ext" | tr '[:upper:]' '[:lower:]')
    
    # 如果格式相同，直接复制
    if [ "$input_ext" = "$output_ext" ]; then
        echo "-c:a copy"
        return
    fi
    
    # 根据输出格式选择编码器
    case "$output_ext" in
        opus) echo "-c:a libopus" ;;
        wav)  echo "-c:a pcm_s16le" ;;
        mp3)  echo "-c:a libmp3lame" ;;
        m4a)  echo "-c:a aac" ;;
        flac) echo "-c:a flac" ;;
        aac)  echo "-c:a aac" ;;
        ogg)  echo "-c:a libvorbis" ;;
        wma)  echo "-c:a wmav2" ;;
        *)    echo "-c:a libopus" ;;
    esac
}

# ==================== 处理单个音频文件 ====================
process_single_audio() {
    local audio_path="$1"
    
    local timestamp_path="${audio_path}.timestamp"
    
    # 检查时间戳文件
    if [ ! -f "$timestamp_path" ]; then
        log_error "时间戳文件不存在: $timestamp_path"
        return 1
    fi
    
    # 读取时间戳
    local timestamps
    if ! timestamps=$(jq -c '.timestamps' "$timestamp_path" 2>/dev/null); then
        log_error "无法读取时间戳: $timestamp_path"
        return 1
    fi
    
    # 检查时间戳数量
    local timestamp_count
    timestamp_count=$(echo "$timestamps" | jq 'length')
    
    if [ "$timestamp_count" -eq 0 ]; then
        log_warning "时间戳为空，跳过: $audio_path"
        return 0
    fi
    
    # 计算输出目录 - 保持原始目录结构
    local audio_dir
    audio_dir=$(dirname "$audio_path")
    
    local relative_dir="${audio_dir#$INPUT_DIR}"
    relative_dir="${relative_dir#/}"
    
    local current_output_dir="$OUTPUT_DIR"
    if [ -n "$relative_dir" ]; then
        current_output_dir="$OUTPUT_DIR/$relative_dir"
    fi
    
    mkdir -p "$current_output_dir"
    
    # 获取基础文件名和扩展名
    local base_filename
    base_filename=$(basename "$audio_path")
    local input_ext="${base_filename##*.}"
    base_filename="${base_filename%.*}"
    
    # 确定输出扩展名
    local output_ext="$input_ext"
    if [ -n "$OUTPUT_FORMAT" ]; then
        output_ext="${OUTPUT_FORMAT#.}"
    else
        # 特殊格式转换
        case "$input_ext" in
            pcm|aiff) output_ext="wav" ;;
        esac
    fi
    
    # 获取编码参数
    local codec_params
    codec_params=$(get_codec_params "$input_ext" "$output_ext")
    
    # 遍历时间戳进行切分
    local success_count=0
    local i=0
    
    while IFS= read -r timestamp; do
        local start_ms end_ms
        start_ms=$(echo "$timestamp" | jq -r '.[0]')
        end_ms=$(echo "$timestamp" | jq -r '.[1]')
        
        # 转换为秒，确保格式正确（添加前导0）
        local start_sec end_sec
        start_sec=$(awk "BEGIN {printf \"%.3f\", $start_ms / 1000}")
        end_sec=$(awk "BEGIN {printf \"%.3f\", $end_ms / 1000}")
        
        # 构建输出文件名
        local output_filename
        output_filename=$(printf "%s/%s_%04d.%s" "$current_output_dir" "$base_filename" "$i" "$output_ext")
        
        # 执行FFmpeg切分
        if ffmpeg -i "$audio_path" \
                  -ss "$start_sec" \
                  -to "$end_sec" \
                  $codec_params \
                  -vn \
                  -nostdin \
                  -y \
                  -loglevel error \
                  "$output_filename" 2>&1; then
            ((success_count++))
        else
            log_error "FFmpeg切分失败: $output_filename (段 $i, ${start_sec}s-${end_sec}s)"
            return 1
        fi
        
        ((i++))
    done < <(echo "$timestamps" | jq -c '.[]')
    
    # 检查是否全部成功
    if [ "$success_count" -eq "$timestamp_count" ]; then
        log_success "切分完成: $audio_path ($success_count 段)"
        
        # 删除原文件(如果指定)
        if [ "$DELETE_ORIGINAL" = "true" ]; then
            if rm -f "$audio_path" "$timestamp_path" 2>/dev/null; then
                log_info "已删除原文件: $audio_path"
            fi
        fi
        
        return 0
    else
        log_error "切分不完整: $audio_path ($success_count/$timestamp_count)"
        return 1
    fi
}

# ==================== 导出函数和变量 ====================
export -f process_single_audio
export -f get_codec_params
export -f log_info
export -f log_success
export -f log_warning
export -f log_error
export INPUT_DIR OUTPUT_DIR OUTPUT_FORMAT DELETE_ORIGINAL
export RED GREEN YELLOW BLUE NC

# ==================== 主函数 ====================
main() {
    log_info "======================================"
    log_info "音频批量切分脚本"
    log_info "======================================"
    
    # 检查依赖
    check_dependencies
    
    # 规范化路径
    INPUT_DIR="${INPUT_DIR%/}"
    OUTPUT_DIR="${OUTPUT_DIR%/}"
    
    # 显示配置
    log_info "配置信息:"
    log_info "  输入目录: $INPUT_DIR"
    log_info "  输出目录: $OUTPUT_DIR"
    log_info "  并行任务数: $PARALLEL_JOBS"
    log_info "  输出格式: ${OUTPUT_FORMAT:-保持原格式}"
    log_info "  删除原文件: $DELETE_ORIGINAL"
    
    # 查找所有有时间戳的音频文件
    log_info "正在扫描音频文件..."
    
    local audio_files=()
    while IFS= read -r -d '' timestamp_file; do
        local audio_file="${timestamp_file%.timestamp}"
        if [ -f "$audio_file" ]; then
            audio_files+=("$audio_file")
        fi
    done < <(find "$INPUT_DIR" -type f -name "*.timestamp" -print0)
    
    local total_files=${#audio_files[@]}
    
    if [ "$total_files" -eq 0 ]; then
        log_warning "未找到任何带时间戳的音频文件"
        exit 0
    fi
    
    log_info "找到 $total_files 个待处理文件"
    
    # 创建临时文件列表
    local temp_file_list
    temp_file_list=$(mktemp)
    trap "rm -f $temp_file_list" EXIT
    
    printf "%s\n" "${audio_files[@]}" > "$temp_file_list"
    
    # 使用GNU parallel进行批量并行处理
    log_info "======================================"
    log_info "开始批量切分处理..."
    log_info "======================================"
    
    if [ "$DELETE_ORIGINAL" = "true" ]; then
        log_warning "⚠️  警告: 切分成功后将删除原始音频文件和时间戳文件"
        sleep 2
    fi
    
    # 执行并行处理
    local start_time
    start_time=$(date +%s)
    
    cat "$temp_file_list" | \
        parallel --progress \
                 --jobs "$PARALLEL_JOBS" \
                 --line-buffer \
                 --halt soon,fail=10% \
                 process_single_audio {}
    
    local exit_code=$?
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 统计结果
    log_info "======================================"
    if [ $exit_code -eq 0 ]; then
        log_success "所有处理完成！"
    else
        log_warning "处理完成，但有部分文件失败"
    fi
    log_info "总耗时: ${duration}秒"
    log_info "处理文件数: $total_files"
    log_info "输出目录: $OUTPUT_DIR"
    log_info "======================================"
    
    return $exit_code
}

# ==================== 使用说明 ====================
show_usage() {
    cat << EOF
使用方法:
    $0 [输入目录] [输出目录] [并行数] [输出格式] [删除原文件]

参数说明:
    输入目录      : 包含音频文件和.timestamp文件的目录 (默认: ./data)
    输出目录      : 切分后音频的输出目录 (默认: ./data/sliced)
    并行数        : 并行处理的任务数 (默认: 32)
    输出格式      : 输出音频格式,如 wav/mp3/opus (默认: 保持原格式)
    删除原文件    : true/false, 是否删除原始文件 (默认: false)

示例:
    # 使用默认参数
    $0

    # 指定输入输出目录
    $0 ./input ./output

    # 指定并行数
    $0 ./input ./output 64

    # 转换为opus格式
    $0 ./data ./data/sliced 32 opus false

    # 保持原格式并删除原文件
    $0 ./data ./data/sliced 32 "" true

依赖:
    - ffmpeg: 音频处理
    - jq: JSON解析
    - parallel: 并行处理
    - bc/awk: 浮点计算

安装依赖:
    sudo apt-get install ffmpeg jq parallel bc gawk
EOF
}

# ==================== 入口 ====================
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    show_usage
    exit 0
fi

main "$@"