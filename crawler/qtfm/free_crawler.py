import requests
import time
import logging
import hmac
import hashlib
import os
import re

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

# 默认专辑，直接填蜻蜓FM网页地址即可（脚本会从中解析出专辑 ID）
CHANNEL_URL = "https://www.qtfm.cn/channels/259433/"

# 从网页地址中解析出album_id
match = re.search(r'/channels/(\d+)', CHANNEL_URL)
if not match:
    logging.error(f"无法从 CHANNEL_URL '{CHANNEL_URL}' 中解析出专辑 ID")
    exit()
ALBUM_ID = match.group(1)

# 节目列表接口为 /capi/channel/{id}/programs/{v}，路径末尾的 v 是频道详情接口
# 返回的版本号，专辑更新后会变化，写死会失效，所以每次运行时实时获取
try:
    response = requests.get(f"https://i.qtfm.cn/capi/v3/channel/{ALBUM_ID}", timeout=15)
    response.raise_for_status()
    channel_info = response.json()['data']
    base_url = f"https://i.qtfm.cn/capi/channel/{ALBUM_ID}/programs/{channel_info['v']}"
    ALBUM_TITLE = channel_info.get('title', ALBUM_ID)
    logging.info(f"专辑: {ALBUM_TITLE} ({ALBUM_ID})，共 {channel_info.get('program_count', '未知')} 个节目")
except Exception as e:
    logging.error(f"获取专辑信息失败: {e}")
    exit()

# 额外的盐 不确定会不会每天改变
SECRECT_KEY = "7l8CZ)SgZgM_bkrw"

# 下载文件夹
DOWNLOAD_FOLDER = './data/' + str(ALBUM_TITLE)
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

################获取所有页面的节目信息#################

def get_all_programs():
    """获取所有页面的节目信息"""
    all_programs = []
    current_page = 1
    page_size = 100  # 服务端限制的最大值
    
    logging.info(f"开始获取所有节目列表...")
    
    while True:
        # 构建当前页面的URL
        current_url = f"{base_url}?curpage={current_page}&pagesize={page_size}&order=asc"
        
        try:
            logging.info(f"正在获取第 {current_page} 页...")
            response = requests.get(current_url)
            response.raise_for_status()
            
            data = response.json()
            programs = data['data']['programs']
            
            if not programs:  # 如果没有更多节目，退出循环
                logging.info(f"第 {current_page} 页没有数据，获取完成")
                break
                
            all_programs.extend(programs)
            logging.info(f"第 {current_page} 页获取到 {len(programs)} 个节目")
            
            # 如果当前页节目数量小于页面大小，说明这是最后一页
            if len(programs) < page_size:
                logging.info(f"第 {current_page} 页是最后一页")
                break
                
            current_page += 1
            time.sleep(0.5)  # 避免请求过快
            
        except Exception as e:
            logging.error(f"获取第 {current_page} 页失败: {e}")
            break
    
    logging.info(f"总共获取到 {len(all_programs)} 个节目的信息")
    return all_programs

# 获取所有节目
all_programs = get_all_programs()

if not all_programs:
    logging.error("没有获取到任何节目信息")
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
            
            logging.info(f"[{index + 1}/{len(all_programs)}]下载完成:{program_title}")

        else:
            logging.error(f"[{index + 1}/{len(all_programs)}]重定向失败，状态码: {redirect_response.status_code}")

        #别给我网站草死了
        time.sleep(1)
 
    except Exception as e:
        logging.error(f"处理 {program.get('title', '未知节目')} 时发生错误: {e}")
        time.sleep(5)

logging.info("所有节目下载完成")


        