import httpx
import json
from bs4 import BeautifulSoup
import re
import os
import time
from pathlib import Path
from urllib.parse import urljoin, unquote

BASE_URL = "https://voicewiki.cn"

def get_voice_page_urls(character_page_url: str, session: httpx.Client) -> list[str]:
    """
    从角色的主介绍页面获取所有语音子页面的URL。
    """
    print(f"正在访问角色主页: {character_page_url}")
    try:
        response = session.get(character_page_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = soup.select('div[style*="background-color:rgba(49,53,72,0.7)"] a[href*="/"]')
        page_urls = set()
        for link in links:
            href = link.get('href')
            if href and href.startswith('/wiki/') and href.count('/') > 1:
                full_url = urljoin(BASE_URL, href)
                page_urls.add(full_url)
                
        if not page_urls:
            print("警告：在主页上未能自动找到任何语音子页面链接。请检查URL是否为角色主页。")
        else:
            print(f"在主页上找到了 {len(page_urls)} 个语音页面链接。")

        return list(page_urls)
        
    except httpx.RequestError as e:
        print(f"请求角色主页时出错: {e}")
        return []

def parse_voice_page(html_content: str, page_url: str) -> dict:
    """
    将语音子页面的HTML内容转换为结构化的字典数据。
    使用更健壮的选择器来定位音频条目。
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    title_element = soup.find('span', id='cosmos-title-text')
    page_title = title_element.text.strip() if title_element else "未知标题"
    
    audio_data = []
    
    # 【核心修改】
    # 1. 直接查找所有明确标识为音频播放器的div
    audio_divs = soup.find_all('div', class_='downloadable-audio')
    
    if audio_divs:
        # 2. 遍历找到的div，并以其父级table作为处理单元
        for div in audio_divs:
            table_container = div.find_parent('table')
            if not table_container:
                continue

            audio_link = table_container.find('a', class_='internal', href=re.compile(r'\.ogg$'))
            if audio_link:
                audio_url = urljoin(BASE_URL, audio_link['href'])
                filename = unquote(audio_link['title'])
                
                text_element = table_container.find('span', lang='zh-Hans-CN')
                chinese_text = text_element.text.strip() if text_element else "无文本"
                
                id_match = re.search(r'([A-F0-9]{8})\.0B2', filename) or re.search(r'/([^/]+?)\.ogg', unquote(audio_url))
                audio_id = id_match.group(1).split('/')[-1] if id_match else f"未知ID_{len(audio_data)}"
                
                audio_data.append({
                    "id": audio_id,
                    "filename": filename,
                    "audio_url": audio_url,
                    "chinese_text": chinese_text
                })
    else:
        # 调试功能依然保留，以防万一
        print(f"!! 警告: 在页面 '{page_title}' 中未找到任何 class='downloadable-audio' 的内容块。")
        print(f"   页面URL: {page_url}")
        
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        filename = f"debug_{page_title.replace('/', '_').replace(':', '_')}.html"
        debug_path = debug_dir / filename
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
        print(f"   该页面的HTML内容已保存到: {debug_path.absolute()}")

    return {
        "page_title": page_title,
        "total_count": len(audio_data),
        "audio_list": audio_data
    }

def download_audio_and_text(audio_item: dict, output_dir: Path, session: httpx.Client) -> bool:
    """
    下载单个音频文件并保存对应的文本文件。
    """
    audio_id = audio_item['id']
    audio_url = audio_item['audio_url']
    chinese_text = audio_item['chinese_text']
    
    audio_filename = f"{audio_id}.ogg"
    text_filename = f"{audio_id}.txt"
    
    audio_path = output_dir / audio_filename
    text_path = output_dir / text_filename
    
    try:
        if audio_path.exists() and text_path.exists():
            return True
            
        print(f"下载音频: {audio_id} | 文本: {chinese_text}")
        audio_response = session.get(audio_url)
        audio_response.raise_for_status()
        
        with open(audio_path, 'wb') as f:
            f.write(audio_response.content)
        
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(chinese_text)
        
        return True
        
    except Exception as e:
        print(f"下载失败 {audio_id}: {e}")
        if audio_path.exists(): os.remove(audio_path)
        if text_path.exists(): os.remove(text_path)
        return False

def main():
    """
    主函数，协调整个爬取和下载过程。
    """
    character_url = "https://voicewiki.cn/wiki/%E8%8E%B1%E5%9B%A0%E5%93%88%E7%89%B9%EF%BC%88%E5%AE%88%E6%9C%9B%E5%85%88%E9%94%8B%EF%BC%89"

    try:
        character_name_full = unquote(character_url.split('/wiki/')[-1])
        character_name = re.sub(r'（.*?）', '', character_name_full).strip()
    except IndexError:
        character_name = "character_data"

    output_dir = Path(f"data/{character_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"所有文件将被保存到: {output_dir.absolute()}")

    with httpx.Client(timeout=30.0, follow_redirects=True) as session:
        voice_page_urls = get_voice_page_urls(character_url, session)
        
        if not voice_page_urls:
            print("未能获取任何语音页面链接，程序退出。")
            return
        
        total_downloaded = 0
        total_failed = 0

        for page_url in voice_page_urls:
            try:
                print(f"\n{'='*20}\n正在处理页面: {unquote(page_url)}\n{'='*20}")
                page_response = session.get(page_url)
                page_response.raise_for_status()

                voice_data = parse_voice_page(page_response.text, page_url)
                
                if voice_data["total_count"] == 0:
                    continue

                print(f"页面 '{voice_data['page_title']}' 找到 {voice_data['total_count']} 条语音，开始下载...")

                for i, audio_item in enumerate(voice_data['audio_list'], 1):
                    print(f"[{i}/{voice_data['total_count']}] ", end="")
                    if download_audio_and_text(audio_item, output_dir, session):
                        total_downloaded += 1
                    else:
                        total_failed += 1
                    time.sleep(0.2)

            except Exception as e:
                print(f"\n处理页面 {page_url} 时发生严重错误: {e}")
                total_failed += 1
    
    print("\n\n🎉 全部任务完成！ 🎉")
    print(f"总计成功下载: {total_downloaded} 个文件")
    if total_failed > 0:
        print(f"总计下载失败: {total_failed} 个文件")
    print(f"数据存放在目录: {output_dir.absolute()}")

if __name__ == "__main__":
    main()