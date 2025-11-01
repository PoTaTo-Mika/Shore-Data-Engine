import requests
import os
import json
import time # 引入time模块，用于在请求之间添加延迟
from bs4 import BeautifulSoup

############### 步骤1: 初始化会话和请求头 ##############
# 我们将沿用原有的Session和请求头策略，这是非常正确的做法

session = requests.Session()
browser_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0',
    'Accept': 'application/json, text/javascript, */*; q=0.01', # 接受JSON，因为AJAX返回的是JSON
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'X-Requested-With': 'XMLHttpRequest', # 关键请求头，表明这是一个AJAX请求
    'Origin': 'https://www.hoyhablamos.com', # 关键请求头，表明请求来源
    'Referer': 'https://www.hoyhablamos.com/?fwp_categorias_podcast=podcast', # 关键请求头
}

# 访问一次主页以“预热”Session，获取必要的Cookies
base_url = "https://www.hoyhablamos.com/?fwp_categorias_podcast=podcast#podcast"
print(f"✅ 步骤1完成: 正在初始化会话，访问 -> {base_url}")
session.get(base_url, headers=browser_headers)
print("   - 会话初始化成功，Cookies已获取。")


############### 步骤2: 模拟AJAX翻页，获取所有剧集信息 ##############
print("\n✅ 步骤2开始: 模拟AJAX翻页，获取全站剧集信息...")

all_episodes_data = []
page = 1
# FacetWP插件的AJAX接口地址
ajax_url = "https://www.hoyhablamos.com/wp-json/facetwp/v1/refresh"

while True:
    print(f"   - 正在请求第 {page} 页的数据...")
    
    # 构造AJAX请求需要发送的载荷(payload)
    # 这是模拟浏览器翻页行为的关键
    payload = {
        "action": "facetwp_refresh",
        "data": {
            "facets": {
                "categorias_podcast": ["podcast"],
                "temas": [],
                "buscador": "",
                "load_more": str(page) 
            },
            "paged": page,
            "template": "blog_posts",
            "extras": {
                "sort": "default"
            }
        }
    }

    try:
        # 使用 session.post 发送AJAX请求，注意是 post 方法和 json=payload
        response = session.post(ajax_url, headers=browser_headers, json=payload, timeout=20)
        
        if response.status_code != 200:
            print(f"   ❌ 请求第 {page} 页失败，状态码: {response.status_code}。停止翻页。")
            break
            
        # 解析返回的JSON数据
        data = response.json()
        
        # 从JSON中提取包含剧集列表的HTML片段
        html_fragment = data.get('template', '')
        if not html_fragment:
            print("   - 未在响应中找到HTML内容，可能是最后一页。")
            break

        # 使用BeautifulSoup解析这个HTML片段
        soup = BeautifulSoup(html_fragment, 'lxml')
        episode_divs = soup.select('div.fwpl-result')

        # 如果当前页没有解析到任何剧集，说明已经到达末尾
        if not episode_divs:
            print(f"   - 第 {page} 页没有发现更多剧集，已到达最后一页。")
            break

        newly_found_episodes = []
        for episode_html in episode_divs:
            title_element = episode_html.select_one('div.el-p94mjr a')
            image_element = episode_html.select_one('div.el-0m0ly5 img')

            if title_element and image_element:
                newly_found_episodes.append({
                    "title": title_element.get_text(strip=True),
                    "url": title_element.get('href'),
                    "image_url": (image_element.get('data-src') or image_element.get('src'))
                })
        
        print(f"   - 第 {page} 页成功解析出 {len(newly_found_episodes)} 个剧集。")
        all_episodes_data.extend(newly_found_episodes)
        
        # 翻页并添加一个小的延迟，避免请求过快
        page += 1
        time.sleep(1) # 礼貌性延迟1秒

    except requests.exceptions.RequestException as e:
        print(f"   ❌ 请求第 {page} 页时发生网络错误: {e}。停止翻-页。")
        break
    except json.JSONDecodeError:
        print(f"   ❌ 解析第 {page} 页的JSON响应失败。可能页面已无内容。停止翻页。")
        break

print(f"\n✅ 步骤2完成: 共获取到 {len(all_episodes_data)} 个剧集信息。")


############### 步骤3: 遍历所有剧集，在内存中解析并下载MP3 ##############

print(f"\n✅ 步骤3开始: 准备处理和下载 {len(all_episodes_data)} 个剧集...")

# 确保data目录存在
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"   - 创建目录: {data_dir}")

# 计数器
downloaded_count = 0
skipped_count = 0
failed_count = 0

for index, episode in enumerate(all_episodes_data):
    
    print(f"\n--- 正在处理第 {index + 1} / {len(all_episodes_data)} 个: {episode['title']} ---")
    
    try:
        # --- 请求详情页 (内存化) ---
        target_url = episode['url']
        print(f"   - 请求详情页: {target_url}")
        detail_response = session.get(target_url, headers=browser_headers, timeout=10)
        
        if detail_response.status_code != 200:
            print(f"   ❌ 详情页请求失败，状态码: {detail_response.status_code}。跳过此剧集。")
            failed_count += 1
            continue
            
        # --- 解析详情页提取MP3链接 (内存化) ---
        soup = BeautifulSoup(detail_response.text, 'lxml')
        audio_link_element = soup.select_one('.sm2-playlist-bd li a')

        if not audio_link_element:
            print("   ❌ 未能在详情页中找到MP3链接。跳过此剧集。")
            failed_count += 1
            continue
        
        mp3_url = audio_link_element.get('href')
        print(f"   - 成功提取MP3链接: {mp3_url}")
        
        # --- 下载MP3文件 ---
        mp3_filename = os.path.basename(mp3_url.split('?')[0])
        mp3_filepath = os.path.join(data_dir, mp3_filename)
        
        # 优化：如果文件已存在，则跳过下载
        if os.path.exists(mp3_filepath):
            print(f"   - 文件已存在，跳过下载: {mp3_filepath}")
            skipped_count += 1
            continue
            
        print(f"   - 正在下载到: {mp3_filepath}")
        mp3_response = session.get(mp3_url, headers=browser_headers, stream=True)
        
        if mp3_response.status_code == 200:
            with open(mp3_filepath, 'wb') as f:
                for chunk in mp3_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(mp3_filepath)
            print(f"   🎉 下载成功！文件大小: {file_size / 1024 / 1024:.2f} MB")
            downloaded_count += 1
        else:
            print(f"   ❌ MP3下载失败，状态码: {mp3_response.status_code}")
            failed_count += 1

    except requests.exceptions.RequestException as e:
        print(f"   ❌ 处理过程中发生网络错误: {e}。跳过此剧集。")
        failed_count += 1
        continue
    except Exception as e:
        print(f"   ❌ 发生未知错误: {e}。跳过此剧集。")
        failed_count += 1
        continue

print("\n\n🎉🎉🎉 全部任务完成！🎉🎉🎉")
print("===================================")
print(f"  - 成功下载: {downloaded_count} 个文件")
print(f"  - 跳过(已存在): {skipped_count} 个文件")
print(f"  - 失败: {failed_count} 个剧集")
print("===================================")