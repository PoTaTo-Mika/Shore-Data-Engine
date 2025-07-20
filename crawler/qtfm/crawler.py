import requests
import json
import time
import logging
import hmac
import hashlib
import os
import re
from get_name import get_name

# 配置logging，使其能在终端显示INFO级别的日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 输出到终端
    ]
)

# 蜻蜓FM采用的API接口是一个重定向的动态网页
# 目前这个脚本只能爬一个album，后续待更新到全自动爬取

PROGRAM_URL = "https://i.qtfm.cn/capi/channel/121170/programs/9a4ff4c800f208f564579039441b5062?curpage=5&pagesize=100&order=asc"

# 然后我们要从这个url当中获取到album_id

ALBUM_ID = re.search(r'channel/(\d+)', PROGRAM_URL).group(1)

# 额外的盐 不确定会不会每天改变

SECRECT_KEY = "7l8CZ)SgZgM_bkrw"

# 下载文件夹
DOWNLOAD_FOLDER = './data/' + str(get_name(ALBUM_ID))
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

################获取节目信息#################

logging.info(f"获取节目列表中...")

try:
    response = requests.get(PROGRAM_URL)
    response.raise_for_status()
    all_programs = response.json()['data']['programs']
    logging.info(f"成功获取到 {len(all_programs)} 个节目的信息")

except Exception as e:
    logging.error(f"获取节目列表失败: {e}")
    exit()

################开始下载#################

for index, program in enumerate(all_programs):
    try:
        program_id = program['id']
        program_title = program['title'].replace('/', '_').replace('\\', '_')
        file_path = os.path.join(DOWNLOAD_FOLDER, f"{program_title}.m4a")

        if os.path.exists(file_path):
            logging.info(f"[{index + 1}/{len(all_programs)}]文件 {file_path} 已存在，跳过下载")
            continue

        logging.info(f"[{index + 1}/{len(all_programs)}]准备下载文件 {file_path}")

################构造时间戳#################

        timestamp = int(time.time() * 1000)

        path_and_query = f"/audiostream/redirect/{ALBUM_ID}/{program_id}?access_token=&device_id=MOBILESITE&qingting_id={timestamp}"

        signature = hmac.new(
            SECRECT_KEY.encode('utf-8'), 
            path_and_query.encode('utf-8'), 
            hashlib.md5
        ).hexdigest()

        final_url = f"https://audio.qtfm.cn{path_and_query}&sign={signature}"

        logging.info(f"[{index + 1}/{len(all_programs)}]构造完毕{file_path}")

###############重定向#################

        redirect_response = requests.get(final_url, allow_redirects=False)
        redirect_response.raise_for_status()

        if redirect_response.status_code == 302:
            download_link = redirect_response.headers.get('Location')

            logging.info(f"[{index + 1}/{len(all_programs)}]重定向成功，开始下载{file_path}")
            audio_response = requests.get(download_link, stream=True)
            audio_response.raise_for_status()

            with open(file_path, 'wb') as f:
                for chunk in audio_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logging.info(f"[{index + 1}/{len(all_programs)}]下载完成{program_title}")

        else:
            logging.error(f"[{index + 1}/{len(all_programs)}]重定向失败，状态码: {redirect_response.status_code}")

        #别给我网站草死了
        time.sleep(1)
 
    except Exception as e:
        logging.error(f"处理 {program.get('title', '未知节目')} 时发生错误: {e}")
        time.sleep(5)

logging.info("所有节目下载完成")


        