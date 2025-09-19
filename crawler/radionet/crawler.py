import requests
import os
import time
import logging
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


INPUT_LINK = "https://prod.radio-api.net/podcasts/episodes/by-podcast-ids?podcastIds=the-mel-robbins-podcast&count=5&offset=0"

# 解析输入链接，准备分页抓取（每页最多100条）
parsed = urlparse(INPUT_LINK)
orig_qs = parse_qs(parsed.query)
podcast_id = (orig_qs.get("podcastIds") or ["unknown_podcast"])[0]
COUNT_PER_PAGE = 100

# 分页拉取
logging.info("开始获取所有节目列表...")
all_episodes = []
total_count = None
offset = 0
current_page = 1

while True:
    page_qs = {
        "podcastIds": [podcast_id],
        "count": [str(COUNT_PER_PAGE)],
        "offset": [str(offset)],
    }
    page_query = urlencode(page_qs, doseq=True)
    fetch_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, page_query, parsed.fragment))

    try:
        logging.info(f"正在获取第 {current_page} 页...")
        resp = requests.get(fetch_url)
        resp.raise_for_status()
        data = resp.json()

        if total_count is None:
            total_count = data.get("totalCount")

        episodes = data.get("episodes", [])
        if not episodes:
            logging.info(f"第 {current_page} 页没有数据，获取完成")
            break

        all_episodes.extend(episodes)
        logging.info(f"第 {current_page} 页获取到 {len(episodes)} 个节目")

        # 如果当前页节目数量小于页面大小，说明这是最后一页
        if len(episodes) < COUNT_PER_PAGE or (total_count is not None and len(all_episodes) >= total_count):
            logging.info(f"第 {current_page} 页是最后一页")
            break

        offset += len(episodes)
        current_page += 1
        time.sleep(1)

    except Exception as e:
        logging.error(f"获取第 {current_page} 页失败: {e}")
        break

logging.info(f"总共获取到 {len(all_episodes)} 个节目的信息")

if not all_episodes:
    logging.error("没有获取到任何节目信息")
    raise SystemExit(1)

# 准备下载目录，使用第一条的 parentTitle
first_parent_title = all_episodes[0].get("parentTitle") or podcast_id
safe_parent_title = first_parent_title.replace('/', '_').replace('\\', '_').strip()
download_dir = os.path.join("./data", safe_parent_title)
if not os.path.exists(download_dir):
    os.makedirs(download_dir)
logging.info(f"下载目录：{download_dir}")

# 逐集下载：处理可能的重定向并流式写入
total_for_log = total_count or len(all_episodes)
for idx, ep in enumerate(all_episodes, start=1):
    try:
        ep_title = (ep.get("title") or f"episode_{idx}").replace('/', '_').replace('\\', '_').strip()
        file_path = os.path.join(download_dir, f"{ep_title}.mp3")

        if os.path.exists(file_path):
            logging.info(f"[{idx}/{total_for_log}]文件 {file_path} 已存在，跳过下载")
            continue

        ep_url = ep.get("url")
        if not ep_url:
            logging.warning(f"[{idx}/{total_for_log}]缺少下载链接，跳过下载：{ep_title}")
            continue

        logging.info(f"[{idx}/{total_for_log}]准备下载文件 {file_path}")

        # 先试探是否为重定向
        redirect_resp = requests.get(ep_url, allow_redirects=False)
        download_link = ep_url
        if redirect_resp.status_code == 302:
            location = redirect_resp.headers.get('Location')
            if location:
                download_link = location
                logging.info(f"[{idx}/{total_for_log}]重定向成功，开始下载{file_path}")
            else:
                logging.error(f"[{idx}/{total_for_log}]重定向失败，状态码: {redirect_resp.status_code}")
        elif 300 <= redirect_resp.status_code < 400:
            logging.error(f"[{idx}/{total_for_log}]重定向失败，状态码: {redirect_resp.status_code}")

        audio_resp = requests.get(download_link, stream=True)
        audio_resp.raise_for_status()

        with open(file_path, 'wb') as f:
            for chunk in audio_resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logging.info(f"[{idx}/{total_for_log}]下载完成:{ep_title}")
        time.sleep(1)

    except Exception as e:
        logging.error(f"处理 {ep.get('title', '未知节目')} 时发生错误: {e}")
        time.sleep(5)

logging.info("所有节目下载完成")
