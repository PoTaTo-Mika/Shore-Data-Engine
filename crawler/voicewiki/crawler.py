import httpx
import json
from bs4 import BeautifulSoup
import re
import os
import time
from pathlib import Path

character_url = "https://voicewiki.cn/wiki/%E7%8B%82%E9%BC%A0%EF%BC%88%E5%AE%88%E6%9C%9B%E5%85%88%E9%94%8B%EF%BC%89/%E6%99%AE%E9%80%9A%E8%AF%AD%E9%9F%B3P1"

# 创建输出目录
output_dir = Path("data/voicewiki_junkrat")
output_dir.mkdir(parents=True, exist_ok=True)

print("正在获取页面内容...")
response = httpx.get(character_url) # 返回的是html内容

# 把html转化为json
def html_to_json(html_content):
    """
    将HTML内容转换为结构化的JSON数据
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 提取页面标题
    title_element = soup.find('span', id='cosmos-title-text')
    page_title = title_element.text if title_element else "未知标题"
    
    # 提取语音列表范围信息
    voice_range_element = soup.find('p', string=re.compile(r'语音列表：'))
    voice_range = voice_range_element.text if voice_range_element else "未知范围"
    
    # 提取所有音频数据
    audio_data = []
    tables = soup.find_all('table')
    
    for table in tables:
        # 查找音频链接
        audio_link = table.find('a', href=re.compile(r'\.ogg$'))
        if audio_link:
            # 提取音频URL
            audio_url = "https://voicewiki.cn" + audio_link['href']
            
            # 提取文件名
            filename = audio_link['title']
            
            # 提取中文文本
            text_element = table.find('span', lang='zh-Hans-CN')
            chinese_text = text_element.text if text_element else "无文本"
            
            # 提取音频ID（从文件名中）
            audio_id_match = re.search(r'([A-F0-9]{8})\.0B2', filename)
            audio_id = audio_id_match.group(1) if audio_id_match else "未知ID"
            
            audio_data.append({
                "id": audio_id,
                "filename": filename,
                "audio_url": audio_url,
                "chinese_text": chinese_text
            })
    
    # 构建最终的JSON结构
    result = {
        "page_title": page_title,
        "voice_range": voice_range,
        "total_count": len(audio_data),
        "audio_list": audio_data
    }
    
    return result

def download_audio_and_text(audio_item, output_dir, session):
    """
    下载音频文件并保存对应的文本文件
    """
    audio_id = audio_item['id']
    audio_url = audio_item['audio_url']
    chinese_text = audio_item['chinese_text']
    
    # 设置文件名（使用.ogg扩展名）
    audio_filename = f"{audio_id}.ogg"  
    text_filename = f"{audio_id}.txt"
    
    audio_path = output_dir / audio_filename
    text_path = output_dir / text_filename
    
    try:
        # 检查文件是否已存在
        if audio_path.exists() and text_path.exists():
            print(f"跳过已存在的文件: {audio_id}")
            return True
            
        # 下载音频文件
        print(f"下载音频: {audio_id}")
        audio_response = session.get(audio_url)
        audio_response.raise_for_status()
        
        # 保存音频文件
        with open(audio_path, 'wb') as f:
            f.write(audio_response.content)
        
        # 保存文本文件
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(chinese_text)
        
        print(f"成功保存: {audio_id} - {chinese_text}")
        return True
        
    except Exception as e:
        print(f"下载失败 {audio_id}: {str(e)}")
        return False

def main():
    """
    主函数：解析HTML并下载所有音频文件和文本
    """
    # 转换HTML为JSON（存储在内存中）
    print("正在解析HTML内容...")
    json_data = html_to_json(response.text)
    
    print(f"成功提取了{json_data['total_count']}条语音数据")
    print(f"页面标题: {json_data['page_title']}")
    print(f"语音范围: {json_data['voice_range']}")
    
    # 创建HTTP会话以复用连接
    with httpx.Client(timeout=30.0) as session:
        success_count = 0
        total_count = len(json_data['audio_list'])
        
        print(f"\n开始下载 {total_count} 个音频文件...")
        
        for i, audio_item in enumerate(json_data['audio_list'], 1):
            print(f"[{i}/{total_count}] ", end="")
            
            if download_audio_and_text(audio_item, output_dir, session):
                success_count += 1
            
            # 添加短暂延迟避免过于频繁的请求
            time.sleep(0.5)
        
        print(f"\n下载完成！成功: {success_count}/{total_count}")
        print(f"文件保存在: {output_dir.absolute()}")

if __name__ == "__main__":
    main()

