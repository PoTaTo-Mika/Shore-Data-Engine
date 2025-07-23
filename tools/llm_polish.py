import json
import os
from openai import OpenAI
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('llm_polish.log', encoding='utf-8')  # 输出到文件
    ]
)

def deepseek(json_file, client, fangyan):
    with open(json_file, 'r') as f:
        data = json.load(f)
        # json的每一个键值对都是 "路径":"转写文本"
        for key, value in data.items():

            messages = [
                {"role": "system", "content": "现在你是一个文本润色高手，用户将给你一些语音识别后不准确的结果，请根据拼音，语境，上下文等多个要素，润色出更准确的文本"},
                {"role": "system", "content": "为了保证文件结构，不管你润色后的内容合理与否，你只需要输出润色后的文本，不要输出任何其他内容"}
            ]
            if fangyan == True:
                messages.append({"role": "system", "content": "请注意，用户给你的文本是方言识别的结果，请根据方言的特点，润色出更准确的文本"})
            messages.append({"role": "user", "content": f"原始文本: {value}"})

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages = messages,
                stream = False
            )

            data[key] = response.choices[0].message.content
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logging.info("Successfully polished the text.")