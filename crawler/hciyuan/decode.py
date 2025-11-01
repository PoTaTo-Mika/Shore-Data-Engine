import requests
import json
import base64
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- 内部解密函数 ---

def _decode_rot13(s):
    result = []
    for char in s:
        code = ord(char)
        if 65 <= code <= 90: code = (code - 65 + 13) % 26 + 65
        elif 97 <= code <= 122: code = (code - 97 + 13) % 26 + 97
        result.append(chr(code))
    return "".join(result)

def _replace_special_chars(s):
    special_chars = ['@$', '^^', '~@', '%?', '*~', '!!', '#&']
    for char_pair in special_chars: s = s.replace(char_pair, '_')
    return s

def _remove_underscores(s): return s.replace('_', '')

def _char_code_subtract(s, key):
    return "".join([chr(ord(char) - key) for char in s])

def _reverse_string(s): return s[::-1]

def _decode_voe_string(encrypted_string):
    """主解密函数，模拟JS的解密流程"""
    try:
        s1 = _decode_rot13(encrypted_string)
        s2 = _replace_special_chars(s1)
        s3 = _remove_underscores(s2)
        s4 = base64.b64decode(s3 + '==' * (-len(s3) % 4)).decode('utf-8')
        s5 = _char_code_subtract(s4, 3)
        s6 = _reverse_string(s5)
        s7 = base64.b64decode(s6).decode('utf-8')
        return json.loads(s7)
    except Exception as e:
        print(f"  [错误] 解密失败: {e}")
        return None

# --- 模块对外暴露的主函数 ---

def get_download_info_for_episode(post_url):
    """
    接收一集的URL，返回二级M3U8的内容和下载所需信息。
    这是本模块的核心功能。
    """
    try:
        print("  (1/4) 正在获取剧集页面...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get(post_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        iframe = soup.select_one('div.hciyuan-9 iframe')
        if not (iframe and iframe.has_attr('src')):
            print("  [错误] 未找到 iframe 播放器链接。")
            return None
        player_page_url = iframe['src']
        if "voe.sx" in player_page_url:
            player_page_url = player_page_url.replace("voe.sx", "mikaylaarealike.com")

        print(f"  (2/4) 正在访问播放器页面: {player_page_url}")
        headers['Referer'] = post_url # Referer 应该是上一级页面
        response = requests.get(player_page_url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        json_tag = soup.find('script', type='application/json')
        if not json_tag:
            print("  [错误] 未在播放页找到加密的JSON数据。")
            return None
        
        encrypted_string = json.loads(json_tag.string)[0]
        decoded_config = _decode_voe_string(encrypted_string)
        if not (decoded_config and 'source' in decoded_config):
            print("  [错误] 解密失败或未找到 'source' 字段。")
            return None
        
        master_m3u8_url = decoded_config['source']
        print(f"  (3/4) 解密成功，正在获取主M3U8...")
        
        headers['Referer'] = player_page_url
        response = requests.get(master_m3u8_url, headers=headers, timeout=15)
        response.raise_for_status()
        master_content = response.text
        
        if "#EXT-X-STREAM-INF" in master_content:
            media_playlist_url = None
            for line in master_content.splitlines():
                if line and not line.startswith('#'):
                    media_playlist_url = urljoin(master_m3u8_url, line.strip())
                    break
            if not media_playlist_url:
                print("  [错误] 未能从主M3U8中解析出二级媒体列表链接。")
                return None
        else:
            media_playlist_url = master_m3u8_url

        print(f"  (4/4) 成功解析，正在获取二级M3U8内容...")
        response = requests.get(media_playlist_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        return {
            "playlist_content": response.text,
            "base_ts_url": media_playlist_url[:media_playlist_url.rfind('/') + 1],
            "referer_url": player_page_url
        }
    except requests.exceptions.RequestException as e:
        print(f"  [网络错误] 处理剧集URL时失败: {e}")
        return None
    except Exception as e:
        print(f"  [未知错误] 处理剧集URL时发生意外: {e}")
        return None

# --- 独立测试本模块的功能 ---
if __name__ == '__main__':
    print(">>> 正在独立测试 decode.py 模块...")
    # 请在这里输入一个有效的剧集URL进行测试
    test_episode_url = "https://mikaylaarealike.com/e/edsq1dg6isgj" 
    print(f"测试URL: {test_episode_url}")
    
    download_data = get_download_info_for_episode(test_episode_url)
    
    if download_data:
        print("\n>>> 测试成功！已获取到下载所需信息:")
        print(f"  Referer: {download_data['referer_url']}")
        print(f"  TS分片基础URL: {download_data['base_ts_url']}")
        print(f"  二级M3U8内容 (前100字符): {download_data['playlist_content'][:100].strip()}...")
    else:
        print("\n>>> 测试失败。未能获取下载信息。")