import requests
import json
import time
import os

# 喜马拉雅主要按专辑进行爬取，专辑的id在url中
# PROGRAM_URL = "https://www.ximalaya.com/album/79011323"
# 请求url，返回的内容是一个json，我们把它保存为一个json文件
REQUEST_URL = "https://www.ximalaya.com/revision/play/v1/show?id=681553954&sort=0&size=30&ptype=1"

# 设置请求头，模拟真实浏览器
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://www.ximalaya.com/'
}

# 发送GET请求到REQUEST_URL，添加请求头
response = requests.get(REQUEST_URL, headers=headers)

json_data = response.json() # json内容样例同xmly_response.json

# 然后我们对另一个链接发送请求
REQUEST_SINGLE_URL = "https://www.ximalaya.com/mobile-playpage/track/v3/baseInfo/1753148210203?device=www2&trackId=681553954&trackQualityLevel=1"

response = requests.get(REQUEST_SINGLE_URL, headers=headers)
json_data_single = response.json()

with open('xmly_response_single.json', 'w', encoding='utf-8') as f:
    json.dump(json_data_single, f, ensure_ascii=False, indent=2)




