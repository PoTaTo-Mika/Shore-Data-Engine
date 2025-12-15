import os
import re
import time
import requests
# 黑产完全体（逃
################### 配置参数 ###################

# 目标语言 (API参数区分大小写，通常首字母大写，如 Korean, Arabic, English)
TARGET_LANGUAGE = "Korean"

# 基础下载目录
BASE_DIR = "data"

# 每次请求单曲列表的数量 (limit)
EPISODE_BATCH_SIZE = 50

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

################### 通用工具函数 ###################

def sanitize_filename(filename):
    """清洗文件名，移除非法字符"""
    # 替换 Windows/Linux 文件系统非法字符
    safe_name = re.sub(r'[\\/*?:"<>|]', "", filename)
    # 替换连续空格
    safe_name = re.sub(r'\s+', " ", safe_name)
    return safe_name.strip()

def download_simple(url, filepath):
    """内存直接写入模式下载"""
    try:
        # 简单检查：文件存在则跳过
        if os.path.exists(filepath):
            print(f"    [√] 跳过 (已存在)")
            return True

        print(f"    [↓] 下载中...", end="", flush=True)
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
            
        print(" 完成！")
        return True
    except Exception as e:
        print(f" 失败: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

################### 第一层：获取专辑列表 ###################

def fetch_all_podcasts(language):
    """
    遍历分页，获取该语言下所有的 Podcast 信息 (slug 和 title)
    """
    print(f"=== 正在扫描 [{language}] 语言下的所有专辑... ===")
    
    podcasts = []
    page = 1
    
    while True:
        url = f"https://zeno.fm/api/podcasts/?limit=20&language={language}&page={page}"
        try:
            print(f"正在获取第 {page} 页列表...", end="")
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            items = data.get('data', [])
            if not items:
                print(" 空 (扫描结束)")
                break
            
            print(f" 发现 {len(items)} 个专辑")
            
            for item in items:
                podcasts.append({
                    'slug': item.get('slug'),
                    'title': item.get('title'),
                    'id': item.get('id')
                })
            
            # 翻页
            page += 1
            time.sleep(0.5) # 防止请求过快
            
        except Exception as e:
            print(f"\n获取专辑列表失败: {e}")
            break
            
    print(f"=== 扫描完成，共找到 {len(podcasts)} 个专辑 ===\n")
    return podcasts

################### 第二层：下载单个专辑 ###################

def process_podcast(podcast_info, language_path):
    """
    处理单个 Podcast：获取所有集数并下载
    """
    slug = podcast_info['slug']
    raw_title = podcast_info['title']
    safe_title = sanitize_filename(raw_title)
    
    # 如果标题是空的，用 slug 代替
    if not safe_title:
        safe_title = slug

    # 构造专辑目录：data/Korean/专辑名
    # 有些专辑名可能太长，截断一下防止系统报错
    album_dir = os.path.join(language_path, safe_title[:100])
    
    if not os.path.exists(album_dir):
        os.makedirs(album_dir)

    print(f"--> 开始处理专辑: {safe_title}")
    
    # 构造获取集数的 API URL 模板
    # 注意：这里我们不需要 query 参数，直接拼 slug
    base_episode_url = f"https://zeno.fm/api/podcasts/{slug}/episodes"
    
    offset = 0
    download_count = 0
    
    while True:
        # 构造参数
        params = {
            "query": "",
            "offset": offset,
            "limit": EPISODE_BATCH_SIZE,
            "sortDir": "desc"
        }
        
        try:
            resp = requests.get(base_episode_url, params=params, headers=HEADERS, timeout=10)
            
            # 有些专辑可能被删除了或者 slug 变了，处理 404
            if resp.status_code == 404:
                print(f"    [!] 警告: 无法找到该专辑的音频列表 (404)")
                break
                
            resp.raise_for_status()
            episodes = resp.json()
            
            if not episodes:
                break # 这一页没数据了，说明该专辑下载完了
                
            for ep in episodes:
                ep_title = ep.get('title', 'Unknown')
                media_url = ep.get('mediaUrl')
                
                if not media_url:
                    continue
                
                # 文件名: 标题.mp3
                ep_filename = sanitize_filename(ep_title) + ".mp3"
                filepath = os.path.join(album_dir, ep_filename)
                
                print(f"    文件: {ep_filename[:50]}...") # 只打印前50个字符
                download_simple(media_url, filepath)
                download_count += 1
            
            offset += EPISODE_BATCH_SIZE
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    [!] 获取集数列表出错: {e}")
            break
            
    print(f"<-- 专辑处理完毕，共处理 {download_count} 个文件。\n")

################### 主程序 ###################

def main():
    # 1. 准备语言目录
    language_dir = os.path.join(BASE_DIR, TARGET_LANGUAGE)
    if not os.path.exists(language_dir):
        os.makedirs(language_dir)
        
    # 2. 获取该语言下所有 Podcast 列表
    all_podcasts = fetch_all_podcasts(TARGET_LANGUAGE)
    
    if not all_podcasts:
        print("未找到任何专辑，请检查 TARGET_LANGUAGE 是否正确 (注意大小写)。")
        return

    # 3. 遍历列表，逐个下载
    total_podcasts = len(all_podcasts)
    
    for index, podcast in enumerate(all_podcasts, 1):
        print(f"正在处理第 {index}/{total_podcasts} 个专辑...")
        process_podcast(podcast, language_dir)
        
    print(f"\n全部任务结束！所有 [{TARGET_LANGUAGE}] 内容已归档。")

if __name__ == "__main__":
    main()