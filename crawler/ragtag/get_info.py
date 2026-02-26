import requests
from bs4 import BeautifulSoup
import json

url = "https://archive.ragtag.moe/channels"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()  # 如果状态码不是 200，抛出异常
    
    soup = BeautifulSoup(response.text, 'html.parser')
    script_tag = soup.find('script', id='__NEXT_DATA__')

    if script_tag:
        data = json.loads(script_tag.string)
        
        # 使用 .get() 级联获取，防止中间层级缺失导致崩溃
        props = data.get('props', {})
        page_props = props.get('pageProps', {})
        channels = page_props.get('channels', [])

        if channels:
            print(f"✅ 成功提取到 {len(channels)} 个频道信息")
            with open('channels.json', 'w', encoding='utf-8') as f:
                json.dump(channels, f, ensure_ascii=False, indent=2)
            print("💾 数据已保存到 channels.json")
        else:
            print("⚠️ 未在 JSON 中找到 channels 字段")
    else:
        print("❌ 未找到 __NEXT_DATA__ 标签，网站可能更改了渲染方式")

except requests.exceptions.RequestException as e:
    print(f"🚀 网络请求失败: {e}")
except json.JSONDecodeError:
    print("🚫 JSON 解析失败")


