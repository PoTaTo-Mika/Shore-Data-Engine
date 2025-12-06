import requests
import json
import time
import logging
import hmac
import hashlib
import os
import re
from urllib.parse import urlparse, parse_qs
from crawl_tools.get_name import get_name
from crawl_tools.auth import refresh_token 

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

PROGRAM_URL = "https://i.qtfm.cn/capi/channel/284770/programs/a74dad695a140af9542611a12affb170?curpage=1&pagesize=30&order=asc"
QINGTING_ID = os.environ.get("QINGTING_ID")
INITIAL_REFRESH_TOKEN = os.environ.get("REFRESH_TOKEN")

if not QINGTING_ID or not INITIAL_REFRESH_TOKEN:
    logging.error("错误：请设置 QINGTING_ID 和 REFRESH_TOKEN 环境变量。")
    exit()

try:
    parsed_url = urlparse(PROGRAM_URL)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    ALBUM_ID = re.search(r'channel/(\d+)', PROGRAM_URL).group(1)
except (AttributeError, IndexError):
    logging.error(f"无法从 PROGRAM_URL '{PROGRAM_URL}' 中解析出 Album ID。请检查URL格式。")
    exit()

SECRECT_KEY = "7l8CZ)SgZgM_bkrw"
DOWNLOAD_FOLDER = './data/' + str(get_name(ALBUM_ID))

if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

current_access_token = None
current_refresh_token = INITIAL_REFRESH_TOKEN
token_expiry_time = 0

def get_valid_access_token():
    """
    获取一个有效的access_token。如果当前token不存在或即将过期，则自动刷新。
    """
    global current_access_token, token_expiry_time, current_refresh_token
    
    # 检查token是否即将过期 (设置5分钟的缓冲期)
    if not current_access_token or time.time() >= (token_expiry_time - 300):
        logging.info("Access token不存在或即将过期，开始执行刷新操作。")
        new_token_info = refresh_token(current_refresh_token, QINGTING_ID)
        
        if new_token_info and 'access_token' in new_token_info:
            current_access_token = new_token_info['access_token']
            # API可能会返回一个新的refresh_token，用于下一次刷新
            current_refresh_token = new_token_info.get('refresh_token', current_refresh_token)
            # 计算精确的过期时间戳
            token_expiry_time = time.time() + new_token_info.get('expires_in', 7200) # 默认2小时
        else:
            logging.error("无法刷新access_token，请检查你的REFRESH_TOKEN是否有效。程序将退出。")
            exit()
    
    return current_access_token

def get_all_programs():
    """获取所有页面的节目信息"""
    all_programs = []
    current_page = 1
    page_size = 100
    logging.info(f"开始获取专辑 {ALBUM_ID} 的所有节目列表...")
    while True:
        current_url = f"{base_url}?curpage={current_page}&pagesize={page_size}&order=asc"
        try:
            logging.info(f"正在获取第 {current_page} 页...")
            response = requests.get(current_url)
            response.raise_for_status()
            data = response.json()
            programs = data.get('data', {}).get('programs', [])
            if not programs:
                logging.info(f"第 {current_page} 页没有数据，获取完成。")
                break
            all_programs.extend(programs)
            logging.info(f"第 {current_page} 页获取到 {len(programs)} 个节目。")
            if len(programs) < page_size:
                logging.info(f"第 {current_page} 页是最后一页。")
                break
            current_page += 1
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"获取第 {current_page} 页失败: {e}")
            break
    logging.info(f"总共获取到 {len(all_programs)} 个节目的信息。")
    return all_programs

all_programs = get_all_programs()
if not all_programs:
    logging.warning("未能获取到任何节目信息，程序即将退出。")
    exit()

logging.info("================== 开始下载 ==================")
for index, program in enumerate(all_programs):
    try:
        program_id = program['id']
        program_title = program['title'].replace('/', '_').replace('\\', '_')
        file_path = os.path.join(DOWNLOAD_FOLDER, f"{program_title}.m4a")

        if os.path.exists(file_path):
            logging.info(f"[{index + 1}/{len(all_programs)}] 文件已存在，跳过: {program_title}")
            continue

        logging.info(f"[{index + 1}/{len(all_programs)}] 准备下载: {program_title}")

        access_token = get_valid_access_token()
        path_and_query = f"/audiostream/redirect/{ALBUM_ID}/{program_id}?access_token={access_token}&device_id=MOBILESITE&qingting_id={QINGTING_ID}"

        signature = hmac.new(
            SECRECT_KEY.encode('utf-8'), 
            path_and_query.encode('utf-8'), 
            hashlib.md5
        ).hexdigest()

        final_url = f"https://audio.qtfm.cn{path_and_query}&sign={signature}"

        redirect_response = requests.get(final_url, allow_redirects=False)
        redirect_response.raise_for_status()

        if redirect_response.status_code == 302:
            download_link = redirect_response.headers.get('Location')
            if not download_link:
                logging.error(f"[{index + 1}/{len(all_programs)}] 重定向成功但未找到Location头: {program_title}")
                continue
            
            logging.info(f"[{index + 1}/{len(all_programs)}] 重定向成功，开始下载...")
            audio_response = requests.get(download_link, stream=True)
            audio_response.raise_for_status()

            with open(file_path, 'wb') as f:
                for chunk in audio_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logging.info(f"[{index + 1}/{len(all_programs)}] 下载完成: {program_title}")
        else:
            logging.error(f"[{index + 1}/{len(all_programs)}] 重定向失败，状态码: {redirect_response.status_code} - {program_title}")

        time.sleep(1)
 
    except requests.exceptions.HTTPError as e:
        # 特别处理HTTP错误，例如401/403可能意味着token问题
        logging.error(f"处理 {program.get('title', '未知')} 时发生HTTP错误: {e.response.status_code} - {e.response.text}")
        if e.response.status_code in [401, 403]:
            logging.warning("收到认证错误，将在下次循环时强制刷新token。")
            current_access_token = None # 强制刷新
        time.sleep(5)
    except Exception as e:
        logging.error(f"处理 {program.get('title', '未知')} 时发生未知错误: {e}")
        time.sleep(5)

logging.info("================== 所有任务完成 ==================")