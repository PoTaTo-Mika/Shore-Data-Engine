import os
import re
import time
import requests
from urllib.parse import urlparse, urlunparse

################### 配置参数 ###################

ORIGINAL_URL = "https://zeno.fm/api/podcasts/hkayat-mn-alktb/episodes?query&offset=10&limit=10&sortDir=desc"
BASE_DIR = "data"     # 根目录
BATCH_SIZE = 50       # 每次请求数量

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

################### 辅助工具 ###################

def get_clean_api_url(url):
    """去除查询参数，保留基础API路径"""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

def extract_album_id(url):
    """
    从URL中提取专辑英文ID。
    例如: https://.../podcasts/hkayat-mn-alktb/episodes...
    提取出: hkayat-mn-alktb
    """
    parsed = urlparse(url)
    path_parts = parsed.path.split('/')
    # URL结构通常是 /api/podcasts/{id}/episodes
    # 我们寻找 'podcasts' 后面紧跟的那一部分
    try:
        if 'podcasts' in path_parts:
            index = path_parts.index('podcasts')
            return path_parts[index + 1]
    except IndexError:
        pass
    return "unknown_album"

def sanitize_filename(filename):
    """清洗文件名"""
    safe_name = re.sub(r'[\\/*?:"<>|]', "", filename)
    return safe_name.strip()

def download_simple(url, filepath):
    """
    简单粗暴的下载逻辑：
    1. 既然内存大，直接请求内容到内存。
    2. 一次性写入文件。
    """
    try:
        print(f"  [↓] 正在下载...", end="", flush=True)
        # 不使用 stream=True，直接加载到内存
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        # 写入文件
        with open(filepath, 'wb') as f:
            f.write(response.content)
            
        print(" 完成！")
        return True
    except Exception as e:
        print(f" 失败: {e}")
        # 如果下载出错，确保不留下空文件
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

################### 主逻辑 ###################

def main():
    # 1. 解析基础信息
    base_api_url = get_clean_api_url(ORIGINAL_URL)
    album_id = extract_album_id(base_api_url)
    
    # 2. 构建保存路径: data/hkayat-mn-alktb/
    save_dir = os.path.join(BASE_DIR, album_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"目标专辑ID: {album_id}")
    print(f"保存目录: {save_dir}")
    print("开始任务...\n")

    offset = 0
    total_new_downloads = 0

    while True:
        # 3. 构造分页参数
        params = {
            "query": "",
            "offset": offset,
            "limit": BATCH_SIZE,
            "sortDir": "desc"
        }
        
        print(f"--- 请求列表: 偏移量 {offset} (获取 {BATCH_SIZE} 条) ---")
        
        try:
            response = requests.get(base_api_url, params=params, headers=HEADERS)
            response.raise_for_status()
            episodes = response.json()
            
            # 列表为空说明爬取结束
            if not episodes:
                print("列表为空，所有专辑已扫描完毕！")
                break
            
            for item in episodes:
                title = item.get('title', 'Unknown_Title')
                media_url = item.get('mediaUrl')
                
                if not media_url:
                    continue
                
                # 构建文件名
                filename = sanitize_filename(title) + ".mp3"
                filepath = os.path.join(save_dir, filename)
                
                print(f"检查: {filename}")
                
                # 优化点1：只检测文件名是否存在
                if os.path.exists(filepath):
                    print(f"  [√] 文件已存在，跳过。")
                else:
                    # 优化点2：直接下载
                    download_simple(media_url, filepath)
                    total_new_downloads += 1

            # 翻页
            offset += BATCH_SIZE
            time.sleep(1) # 稍微休息一下，防止被ban
            
        except Exception as e:
            print(f"请求列表失败: {e}")
            break

    print(f"\n任务全部结束！本次新下载了 {total_new_downloads} 个文件。")

if __name__ == "__main__":
    main()