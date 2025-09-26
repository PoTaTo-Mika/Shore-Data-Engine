import json

# 读取emotion和transcription结果
with open('data/emotion_results.json', 'r') as f:
    emotion_results = json.load(f)

with open('data/transcription.json', 'r') as f:
    transcription = json.load(f)

with open('data/description.json', 'r') as f:
    description = json.load(f)

# 合并数据
merged_data = {}
for abs_path in emotion_results:
    merged_data[abs_path] = {
        "emotion": emotion_results[abs_path],
        "transcription": transcription.get(abs_path, ""),
        "description": description.get(abs_path, "")
    }

# 保存合并后的结果
with open('data/merged_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"成功合并 {len(merged_data)} 条记录")
