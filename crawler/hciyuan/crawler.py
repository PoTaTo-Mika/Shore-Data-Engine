import requests
import os
import subprocess
import shutil
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

# 从我们的核心处理器模块导入主函数
from decode import get_download_info_for_episode

# --- 全局设置 ---
BASE_URL = "https://h-ciyuan.com/category/%E9%87%8C%E7%95%AA/"
DATA_DIR = "./data"
TEMP_DIR = "./temp_video_chunks"

def download_and_merge_ts(playlist_content, base_ts_url, output_filename, referer_url):
    """根据二级M3U8的内容，下载所有ts分片并合并成mp4。"""
    try:
        ts_urls = [urljoin(base_ts_url, line.strip()) for line in playlist_content.splitlines() if line and not line.startswith('#')]
        if not ts_urls:
            print("  [错误] 未在二级M3U8文件中找到任何 .ts 分片链接。")
            return False

        print(f"  解析到 {len(ts_urls)} 个视频分片，开始下载...")
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': referer_url
        }
        
        ts_paths = []
        for i, ts_url in enumerate(tqdm(ts_urls, desc="  下载分片")):
            ts_path = os.path.join(TEMP_DIR, f"{i:05d}.ts")
            ts_paths.append(ts_path)
            try:
                ts_res = requests.get(ts_url, headers=headers, stream=True, timeout=30)
                if ts_res.status_code == 200:
                    with open(ts_path, 'wb') as f: shutil.copyfileobj(ts_res.raw, f)
                else: tqdm.write(f"  [警告] 下载分片 {i} 失败，状态码: {ts_res.status_code}")
            except Exception: tqdm.write(f"  [警告] 下载分片 {i} 时网络超时或出错。")

        print("  分片下载完成，开始使用 FFmpeg 合并...")

        file_list_path = os.path.join(TEMP_DIR, 'file_list.txt')
        with open(file_list_path, 'w', encoding='utf-8') as f:
            for ts_path in ts_paths: f.write(f"file '{os.path.abspath(ts_path).replace('//', '/')}'\n")

        output_path = os.path.join(DATA_DIR, output_filename)
        command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', file_list_path, '-c', 'copy', '-y', '-hide_banner', '-loglevel', 'error', output_path]
        
        subprocess.run(command, check=True)
        print(f"✓ 视频合并完成: {output_path}")
        return True

    except FileNotFoundError:
        print("\n[严重错误] 'ffmpeg' 命令未找到！请确保已正确安装并配置到环境变量。")
        return False
    except subprocess.CalledProcessError:
        print("  [错误] FFmpeg 合并失败。可能是部分分片下载不完整导致。")
        return False
    except Exception as e:
        print(f"  [错误] 下载与合并过程中发生意外: {e}")
        return False
    finally:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            print("  临时文件已清理。")

def crawl_and_download_videos():
    """主函数：爬取、处理并下载所有视频"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(">>> 任务开始：正在爬取剧集列表...")
    try:
        response = requests.get(BASE_URL)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"无法访问主页面 {BASE_URL}，请检查网络或URL。错误: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    episode_list = [{'title': a.text, 'link': a['href']} for a in soup.select('div.gridview-grid-post h3.gridview-grid-post-title a')]
    
    if not episode_list:
        print("未能从主页爬取到任何剧集信息。")
        return

    print(f">>> 成功爬取 {len(episode_list)} 集，开始逐一处理...")
    
    downloaded_count = 0
    for i, episode in enumerate(episode_list, 1):
        print(f"\n--- [处理第 {i}/{len(episode_list)} 集: {episode['title']}] ---")
        
        # 调用核心处理器，获取下载所需信息
        download_info = get_download_info_for_episode(episode['link'])
        
        if not download_info:
            print(f"  [失败] 处理本集失败，跳过。")
            continue
            
        safe_title = "".join(c for c in episode['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_title}.mp4"
        
        # 调用下载与合并模块
        if download_and_merge_ts(download_info['playlist_content'], download_info['base_ts_url'], filename, download_info['referer_url']):
            downloaded_count += 1
    
    print(f"\n==============================================")
    print(f"任务全部完成! 成功下载 {downloaded_count}/{len(episode_list)} 个视频。")
    print(f"文件保存在: {os.path.abspath(DATA_DIR)}")
    print(f"==============================================")

if __name__ == "__main__":
    if not shutil.which("ffmpeg"):
        print("[严重错误] 系统中未找到 'ffmpeg'。")
        print("请先安装 FFmpeg 并确保它在系统的 PATH 环境变量中。")
    else:
        crawl_and_download_videos()