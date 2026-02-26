import os
import json
import requests
import subprocess
import re
import time


CHANNELS_FILE = './crawler/ragtag/channels.json'
OUTPUT_ROOT = 'data'
API_URL = "https://archive.ragtag.moe/api/v1/search"
CONTENT_BASE_URL = "https://content.archive.ragtag.moe"

# 请求头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

FFMPEG_CMD_TEMPLATE = [
    'ffmpeg', 
    '-y', # 覆盖输出文件
    '-hide_banner', '-loglevel', 'error', # 减少日志输出
    '-i', '{input_url}', 
    '-vn', 
    '-c:a', 'libopus', 
    '-b:a', '48k', 
    '-ar', '48000', 
    '{output_path}'
]

def sanitize_filename(name):
    """清理文件名中的非法字符"""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def get_video_list(channel_id, limit=10):
    """
    获取指定频道的视频列表
    """
    videos = []
    params = {
        "channel_id": channel_id, 
        "sort": "upload_date",
        "sort_order": "desc",
        "size": limit,
        "from": 0
    }
    
    try:
        resp = requests.get(API_URL, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        # 提取 hits 列表
        if 'hits' in data and 'hits' in data['hits']:
            for hit in data['hits']['hits']:
                videos.append(hit['_source'])
    except Exception as e:
        print(f"  ❌ 获取视频列表失败: {e}")
    
    return videos

def process_channel(channel):
    channel_name = channel.get('channel_name', 'Unknown')
    channel_id = channel.get('channel_id')
    
    print(f"\n📺 正在处理频道: {channel_name} ({channel_id})")
    
    safe_channel_name = sanitize_filename(channel_name)
    channel_dir = os.path.join(OUTPUT_ROOT, safe_channel_name)
    os.makedirs(channel_dir, exist_ok=True)
    
    videos = get_video_list(channel_id, limit=5)
    print(f"  🔍 找到 {len(videos)} 个视频")
    
    for vid in videos:
        try:
            video_id = vid.get('video_id')
            title = vid.get('title', video_id)
            drive_base = vid.get('drive_base') 
            files = vid.get('files', [])

            target_file = None
            for f in files:
                if f['name'].endswith(('.mkv', '.mp4')):
                    target_file = f['name']
                    break
            
            if not target_file or not drive_base:
                print(f"  ⚠️ 跳过 {title}: 缺少文件信息")
                continue

            download_url = f"{CONTENT_BASE_URL}/{drive_base}/{video_id}/{target_file}"
            
            safe_title = sanitize_filename(f"{title}")

            if len(safe_title) > 100: 
                safe_title = safe_title[:100]
                
            output_filename = f"{video_id}_{safe_title}.opus"
            output_path = os.path.join(channel_dir, output_filename)
            
            if os.path.exists(output_path):
                print(f"  ⏭️ 已存在: {output_filename}")
                continue
                
            print(f"  ⬇️ 正在转码下载: {output_filename}")
            print(f"     🔗 源: {download_url}")

            cmd = [
                arg.format(input_url=download_url, output_path=output_path) 
                for arg in FFMPEG_CMD_TEMPLATE
            ]
            
            start_time = time.time()
            # 运行命令
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode == 0:
                elapsed = time.time() - start_time
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  ✅ 完成! 耗时: {elapsed:.1f}s, 大小: {size_mb:.2f}MB")
            else:
                print(f"  ❌ FFmpeg 失败 (可能链接无效或404)")
                # 打印错误详情（可选）
                print(result.stderr.decode())

        except Exception as e:
            print(f"  ❌ 处理视频出错: {e}")

def main():
    # 读取 channels.json
    if not os.path.exists(CHANNELS_FILE):
        print(f"❌ 找不到 {CHANNELS_FILE}，请先运行爬虫脚本。")
        return

    with open(CHANNELS_FILE, 'r', encoding='utf-8') as f:
        channels = json.load(f)
    
    print(f"📂 开始处理 {len(channels)} 个频道的数据...")
    
    for channel in channels:
        process_channel(channel)
        time.sleep(1)

if __name__ == "__main__":
    main()