import requests
import json

base_url = "https://i.qtfm.cn/capi/v3/channel/"

def get_name(album_id):
    # 蜻蜓FM的剧集和简介是独立的
    url = base_url + album_id # 获取简介json文件
    response = requests.get(url)
    data = json.loads(response.text)
    # 从data.title字段提取剧集名称
    name = data['data']['title']
    return name
    
if __name__ == "__main__":
    album_id = '437133' # 大明皇帝朱元璋
    name = get_name(album_id)
    print(f"剧集名称: {name}")
