import httpx
import json
import time
import os
import subprocess

# 喜马拉雅主要按专辑进行爬取，专辑的id在url中
# PROGRAM_URL = "https://www.ximalaya.com/album/79011323"
# 请求url，返回的内容是一个json，我们把它保存为一个json文件
REQUEST_URL = "https://www.ximalaya.com/revision/play/v1/show?id=681553954&sort=0&size=100&ptype=1" #到时候只需要修改这个就可以了
TIME_URL = "https://www.ximalaya.com/revision/time"

# 设置请求头，模拟真实浏览器
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://www.ximalaya.com/'
}

# 先获取一下时间戳
timestamp = httpx.get(TIME_URL, headers=headers).text # 返回的内容就是一个时间戳

# 然后请求主url，获得节目列表
album_info = httpx.get(REQUEST_URL, headers=headers).json()

# 创建输出目录
output_dir = f"./data/{album_info['data']['tracksAudioPlay'][0]['albumName']}"
os.makedirs(output_dir, exist_ok=True)

# 接下来我们要构建单集url请求，以获得加密音频链接了
for track in album_info['data']['tracksAudioPlay']:
    # 获得轨道ID
    trackId = track['trackId']
    # 单集url请求链接的构建：
    single_url = f"https://www.ximalaya.com/mobile-playpage/track/v3/baseInfo/{timestamp}?device=www2&trackId={trackId}&trackQualityLevel=1"
    # 请求这个url，获取加密音频链接
    single_response = httpx.get(single_url, headers=headers).json()
    print(single_response)
    # 获取最终加密链接
    encoded_url = single_response['trackInfo']['playUrlList'][0]['url']
    # 使用js脚本解密，原指令为: node decode.js {encoded_url}
    final_url = subprocess.run(['node', 'decode.js', encoded_url], capture_output=True, text=True).stdout.strip()
    # 直接下载音频
    response = httpx.get(final_url, headers=headers) #返回应该是m4a格式的音频

    with open(f'{output_dir}/{trackId}.m4a', 'wb') as f:
        f.write(response.content)
    
    time.sleep(1) # 这个网站我是真不担心草死，但是担心被ban

