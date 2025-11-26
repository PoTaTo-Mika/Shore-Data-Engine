import os
import re
import time
import requests
from urllib.parse import urlparse, urlunparse

################### 配置参数 ###################

# 你只需要提供原始 URL，不管后面带什么 offset/limit 参数，脚本都会自动处理
ORIGINAL_URL = "https://zeno.fm/api/podcasts/hkayat-mn-alktb/episodes?query&offset=10&limit=10&sortDir=desc"
DOWNLOAD_DIR = "data"
BATCH_SIZE = 50  # 每次请求获取多少个音频信息（建议 20-50，不要太大以免超时）

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

################### 核心逻辑 ###################

def get_clean_api_url(url):
    """
    智能处理 URL：去除 url 中的查询参数，只保留基础 API 路径。
    这样我们可以自己控制 offset 和 limit。
    """
    parsed = urlparse(url)
    # 重组 URL，去掉 query 部分
    clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, '', parsed.fragment))
    return clean_url

def sanitize_filename(filename):
    """清洗文件名，移除非法字符"""
    safe_name = re.sub(r'[\\/*?:"<>|]', "", filename)
    return safe_name.strip()

def get_remote_file_size(url):
    """
    发送 HEAD 请求获取远程文件大小，不下载内容。
    """
    try:
        response = requests.head(url, headers=HEADERS, allow_redirects=True)
        if response.status_code == 200:
            return int(response.headers.get('content-length', 0))
    except Exception:
        pass
    return 0

def download_file_smart(url, filepath):
    """
    智能下载：支持断点完整性校验
    """
    # 1. 获取远程文件大小
    remote_size = get_remote_file_size(url)
    
    # 2. 检查本地文件
    if os.path.exists(filepath):
        local_size = os.path.getsize(filepath)
        if remote_size > 0 and local_size == remote_size:
            print(f"  [√] 文件完整，跳过: {os.path.basename(filepath)}")
            return True
        else:
            print(f"  [!] 文件不完整或已更新 (本地: {local_size} vs 远程: {remote_size})，重新下载...")
    
    # 3. 开始下载
    try:
        print(f"  [↓] 正在下载...", end="", flush=True)
        with requests.get(url, headers=HEADERS, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(" 完成！")
        return True
    except Exception as e:
        print(f" 失败: {e}")
        # 下载失败如果留下了残缺文件，最好删掉，以免下次误判（虽然有大小校验，但删掉更干净）
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def main():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    # 1. 智能提取 Base URL
    base_api_url = get_clean_api_url(ORIGINAL_URL)
    print(f"解析出的基础API地址: {base_api_url}")
    print("开始全量爬取任务...\n")

    offset = 0
    total_downloaded = 0

    while True:
        # 2. 构造动态参数
        params = {
            "query": "",
            "offset": offset,
            "limit": BATCH_SIZE,
            "sortDir": "desc"
        }
        
        print(f"--- 正在请求列表: 偏移量 {offset} (获取 {BATCH_SIZE} 条) ---")
        
        try:
            response = requests.get(base_api_url, params=params, headers=HEADERS)
            response.raise_for_status()
            episodes = response.json()
            
            # 3. 终止条件：如果返回列表为空，说明爬完了
            if not episodes:
                print("列表为空，所有专辑已爬取完毕！")
                break
                
            print(f"本页获取到 {len(episodes)} 个音频，开始处理...")
            
            for item in episodes:
                title = item.get('title', 'Unknown_Title')
                media_url = item.get('mediaUrl')
                
                if not media_url:
                    continue
                
                # 构建文件名
                filename = sanitize_filename(title) + ".mp3"
                filepath = os.path.join(DOWNLOAD_DIR, filename)
                
                print(f"处理: {filename}")
                download_file_smart(media_url, filepath)
                total_downloaded += 1

            # 4. 更新 offset，准备下一页
            offset += BATCH_SIZE
            
            # 礼貌性延时
            time.sleep(1)
            
        except Exception as e:
            print(f"请求列表失败: {e}")
            print("尝试重试或退出...")
            break

    print(f"\n任务全部结束！共处理 {total_downloaded} 个文件。")

if __name__ == "__main__":
    main()