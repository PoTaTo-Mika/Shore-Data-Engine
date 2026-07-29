import os
import re
import time
import json
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, unquote
from collections import defaultdict

################### 配置参数 ###################

# 国家代码 (top podcasts 是按 country 拉, 印度区=IN)
COUNTRY_CODE = "IN"

# 每个 genre 拉多少条 top podcasts (Apple 上限约 200)
TOP_LIMIT_PER_GENRE = 100

# 要扫的 genre id -> 名字 (None 表示 top podcasts 总榜, 已含所有 genre)
# 默认集合挑选了 TTS 训练最常用的: 谈话/访谈/新闻/故事/教育/宗教
GENRES = {
    None:   "Top",                # 总榜 top podcasts
    "1301": "Arts",
    "1303": "Comedy",             # 多是访谈
    "1304": "Education",
    "1314": "Religion & Spirituality",
    "1324": "Society & Culture",
    "1487": "Kids & Family",      # 讲故事
    "1489": "News",               # 发音标准
    "1501": "True Crime",         # 故事性强
    "1502": "Fiction",            # 有声书 / 故事
    "1510": "History",            # 讲故事
}

# 下载根目录, 最终结构只有两层:
#   data/<lang>/<podcast_name>__<episode_title>.<ext>
#   <lang> 由 RSS <language> 标签决定, 解析失败归入 unknown/
BASE_DIR = "data"

# RSS <language> 没有 / 解析不出来时 归到这个子目录
UNKNOWN_LANG_DIR = "unknown"

# 常见语言别名 → ISO 639-1 代码
LANG_ALIASES = {
    "hin": "hi", "hindi": "hi",
    "eng": "en", "english": "en",
    "spa": "es", "spanish": "es",
    "fra": "fr", "french": "fr",
    "deu": "de", "german": "de",
    "jpn": "ja", "japanese": "ja",
    "kor": "ko", "korean": "ko",
    "zho": "zh", "chinese": "zh",
    "por": "pt", "portuguese": "pt",
    "ita": "it", "italian": "it",
    "rus": "ru", "russian": "ru",
    "ara": "ar", "arabic": "ar",
}

# 单个播客最多下载多少集 (None / 0 = 不限制)
# TTS 训练一般 ~100 集就能覆盖说话人风格
MAX_EPISODES_PER_PODCAST = 100

MAX_FILENAME_LEN = 80  # 单集文件名主截断长度 (不含扩展名)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/121.0.0.0 Safari/537.36"
}

################### 通用工具函数 ###################

def sanitize_filename(filename):
    """清洗文件名, 移除非法字符, 不截断 (长度截断调用方决定)"""
    safe_name = re.sub(r'[\\/*?:"<>|]', "", filename)
    safe_name = re.sub(r'\s+', " ", safe_name).strip()
    return safe_name


def truncate(s, maxlen):
    """截断字符串到 maxlen, 尝试不在词中间断 (向前找空格)"""
    if len(s) <= maxlen:
        return s
    cut = s[:maxlen]
    # 在最后 1/3 内找一个空格, 在那里断
    last_space = cut.rfind(" ", int(maxlen * 0.6))
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip()


def guess_extension(url, content_type=None):
    """根据 url path / content-type 推断扩展名"""
    path = urlparse(url).path.lower()
    for ext in (".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".wav", ".flac"):
        if path.endswith(ext):
            return ext
    ct = (content_type or "").lower()
    if "mpeg" in ct or "mp3" in ct:
        return ".mp3"
    if "mp4" in ct or "m4a" in ct or "aac" in ct:
        return ".m4a"
    if "ogg" in ct:
        return ".ogg"
    if "opus" in ct:
        return ".opus"
    if "wav" in ct:
        return ".wav"
    return ".mp3"


def download_file(url, filepath):
    """流式下载, 已存在则跳过"""
    if os.path.exists(filepath):
        print(f"    [√] 已存在, 跳过")
        return True

    tmp = filepath + ".part"
    try:
        print(f"    [↓] 下载中...", end="", flush=True)
        with requests.get(url, headers=HEADERS, stream=True, timeout=60,
                          allow_redirects=True) as r:
            r.raise_for_status()
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        os.rename(tmp, filepath)
        print(" 完成!")
        return True
    except Exception as e:
        print(f" 失败: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


################### 第 1 步: 按 country 拉 top podcasts 列表 (不用 search) ###################

def fetch_top_podcasts_ids(country, genre_id=None, limit=100):
    """
    Apple RSS top lists (无需 key):
      https://itunes.apple.com/<country>/rss/toppodcasts/limit=N[/genre=G]/json
    返回 podcast id 列表.
    """
    if genre_id:
        url = f"https://itunes.apple.com/{country}/rss/toppodcasts/limit={limit}/genre={genre_id}/json"
    else:
        url = f"https://itunes.apple.com/{country}/rss/toppodcasts/limit={limit}/json"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        entries = data.get("feed", {}).get("entry", [])
        # limit=1 时 Apple 返回单个 dict 而非 list，归一化
        if isinstance(entries, dict):
            entries = [entries]
        ids = []
        for e in entries:
            pid = e.get("id", {}).get("attributes", {}).get("im:id")
            if pid:
                ids.append(pid)
        return ids
    except Exception as e:
        print(f"    [!] 拉 top podcasts 失败 (genre={genre_id}): {e}")
        return []


def lookup_podcasts_by_ids(ids, batch_size=50):
    """
    Apple lookup 支持批量: https://itunes.apple.com/lookup?id=<id1>,<id2>,...
    返回 {podcast_id: {name, feedUrl, artist, trackCount}}.
    """
    results = {}
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        try:
            resp = requests.get(
                "https://itunes.apple.com/lookup",
                params={"id": ",".join(batch)},
                headers=HEADERS, timeout=30,
            )
            resp.raise_for_status()
            for r in resp.json().get("results", []):
                if r.get("wrapperType") != "track" or r.get("kind") != "podcast":
                    continue
                pid = str(r.get("collectionId") or r.get("trackId"))
                feed = r.get("feedUrl")
                if not feed:
                    continue
                results[pid] = {
                    "name": r.get("collectionName") or "unknown",
                    "artist": r.get("artistName") or "",
                    "feedUrl": feed,
                    "trackCount": r.get("trackCount", 0),
                }
        except Exception as e:
            print(f"    [!] lookup 失败 (batch {i}-{i+len(batch)}): {e}")
        time.sleep(0.3)
    return results


def collect_podcasts_by_country(country, genres, limit_per_genre):
    """
    先按 genre 拉 top podcasts id, 合并去重, 再批量 lookup 拿 feedUrl.
    """
    print(f"=== 拉取 [{country}] 区 top podcasts ===")
    all_ids = []
    seen = set()

    for gid, gname in genres.items():
        ids = fetch_top_podcasts_ids(country, gid, limit_per_genre)
        new = [i for i in ids if i not in seen]
        seen.update(new)
        all_ids.extend(new)
        print(f"  genre={gname:30} 拉到 {len(ids):3} 条 (新增 {len(new):3})")
        time.sleep(0.5)

    print(f"=== 共 {len(all_ids)} 个去重的 podcast id ===")
    print(f"=== 批量 lookup 解析 feedUrl ===")
    podcasts = lookup_podcasts_by_ids(all_ids)
    print(f"=== {len(podcasts)} 个有 feedUrl ===\n")
    return podcasts


################### 第 2 步: 解析 RSS (读 <language> 标签 + 集列表) ###################

def parse_rss(feed_url, max_retries=2):
    """
    GET 一个 RSS feed. 返回 (language_iso, episodes_list).
    language_iso 可能是 None / 'hi' / 'hi-IN' / 'en' 等.
    """
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(feed_url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            channel = root.find("channel")
            if channel is None:
                return None, []

            lang_el = channel.find("language")
            lang = (lang_el.text or "").strip() if lang_el is not None else None

            episodes = []
            for item in channel.findall("item"):
                title_el = item.find("title")
                pub_el = item.find("pubDate")
                enc = item.find("enclosure")
                if enc is None or not enc.get("url"):
                    continue
                episodes.append({
                    "title": (title_el.text or "").strip() if title_el is not None else "",
                    "audio_url": enc.get("url"),
                    "content_type": enc.get("type"),
                    "pub_date": pub_el.text.strip() if pub_el is not None else "",
                })
            return lang, episodes

        except ET.ParseError as e:
            print(f"    [!] RSS 解析失败 (xml): {e}")
            return None, []
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"    [!] RSS 抓取失败, {wait}s 后重试: {e}")
                time.sleep(wait)
            else:
                print(f"    [!] RSS 抓取失败, 放弃: {e}")
                return None, []


def unwrap_anchor_url(url):
    """anchor.fm 包装地址 -> 真实 CDN 地址."""
    m = re.match(r"https?://anchor\.fm/s/[^/]+/podcast/play/\d+/(.+)$", url)
    if m:
        return unquote(m.group(1))
    return url


################### 第 3 步: 按语言分类 ###################

def classify_language(rss_lang):
    """
    按 RSS <language> 标签分桶:
      'hi' / 'hi-IN' / 'hindi' ... → 'hi'
      'en' / 'en-US' / 'english' .. → 'en'
      None / ''                    → 'unknown'
    """
    if not rss_lang:
        return UNKNOWN_LANG_DIR

    s = rss_lang.strip().lower()
    # 拆分主语言代码: 'hi-IN' → 'hi', 'english' → 'english'
    main = re.split(r'[-_.]', s)[0]

    # 别名映射 -> 标准 ISO 639-1
    return LANG_ALIASES.get(main, main)


################### 主流程: 下载到扁平结构 ###################

def make_flat_filename(podcast_name, episode_title, ext, max_total=MAX_FILENAME_LEN):
    """
    扁平文件名: <podcast>__<episode>.<ext>
    - 切开用双下划线, 下游 split('__') 就能拆出 podcast
    - 长度预算按 podcast / episode 实际长度动态分配, 谁先塞满谁先停
    """
    safe_pod = sanitize_filename(podcast_name) or "unknown"
    safe_ep  = sanitize_filename(episode_title) or "episode"

    # 占位: "__" (2 字符分隔符)
    budget = max_total - 2

    # 初始期望: podcast 拿 budget/3, 上限 40; episode 拿剩下的, 上限保底 30
    pod_max = min(len(safe_pod), max(20, budget // 3), 40)
    ep_max  = budget - pod_max

    # 如果 episode 给不到 30, 反过来压 podcast
    if ep_max < 30:
        ep_max = 30
        pod_max = max(15, budget - ep_max)

    safe_pod = truncate(safe_pod, pod_max)
    safe_ep  = truncate(safe_ep, ep_max)

    return f"{safe_pod}__{safe_ep}{ext}"


def download_bucket(pods, target_dir):
    """把一个语言桶里的所有播客的全部集, 下载到 target_dir (扁平, 不带子目录)."""
    os.makedirs(target_dir, exist_ok=True)
    used_filenames = set(os.listdir(target_dir))  # 已经下载过的文件名
    downloaded = 0
    skipped = 0
    failed = 0

    for pod in pods:
        pod_name = pod["name"]
        episodes = pod["episodes"]
        if MAX_EPISODES_PER_PODCAST:
            episodes = episodes[:MAX_EPISODES_PER_PODCAST]

        print(f"--> {pod_name} (rss_lang={pod['rss_lang'] or '-'})  {len(episodes)} 集")

        for idx, ep in enumerate(episodes, 1):
            real_url = unwrap_anchor_url(ep["audio_url"]) or ep["audio_url"]
            ext = guess_extension(real_url, ep.get("content_type"))
            filename = make_flat_filename(pod_name, ep["title"] or f"ep_{idx}", ext)

            # 防冲突: 同播客可能有重名 / 不同播客可能撞名字
            base, _ext = os.path.splitext(filename)
            final = filename
            counter = 1
            while final in used_filenames:
                final = f"{base}_{counter:02d}{_ext}"
                counter += 1
            used_filenames.add(final)

            filepath = os.path.join(target_dir, final)

            if os.path.exists(filepath):
                print(f"    [{idx}/{len(episodes)}] [√] {final[:60]}")
                skipped += 1
                continue

            if download_file(ep["audio_url"], filepath):
                downloaded += 1
            else:
                failed += 1
            time.sleep(0.2)

    return downloaded, skipped, failed


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    # Step 1: 按 country + genre 拉候选
    podcasts_by_id = collect_podcasts_by_country(
        COUNTRY_CODE, GENRES, TOP_LIMIT_PER_GENRE
    )
    if not podcasts_by_id:
        print("没拉到任何播客, 检查网络 / country code")
        return

    # Step 2: 对每个候选抓 RSS, 读 <language>, 分桶
    print(f"=== 抓 {len(podcasts_by_id)} 个 feed, 读取 <language> 标签 ===")
    buckets = defaultdict(list)

    for i, (pid, pod) in enumerate(podcasts_by_id.items(), 1):
        if i % 10 == 0 or i == len(podcasts_by_id):
            print(f"  进度: {i}/{len(podcasts_by_id)}")
        rss_lang, episodes = parse_rss(pod["feedUrl"])
        lang_dir = classify_language(rss_lang)
        pod["rss_lang"] = rss_lang
        pod["episodes"] = episodes
        buckets[lang_dir].append(pod)
        time.sleep(0.3)

    print()
    print("=== 分桶结果 ===")
    for lang, pods in sorted(buckets.items(), key=lambda x: -len(x[1])):
        ep_total = sum(min(len(p["episodes"]), MAX_EPISODES_PER_PODCAST or 10**9) for p in pods)
        print(f"  {lang:15}  {len(pods):4} 个播客, ~{ep_total:6} 集 (截断到 {MAX_EPISODES_PER_PODCAST}/podcast)")
    print()

    # 分桶结果存盘 (下次想只拉 unknown / 想重分类不用重新扫)
    summary = {lang: [{
        "name": p["name"], "feedUrl": p["feedUrl"],
        "rss_lang": p["rss_lang"], "trackCount": p["trackCount"],
        "episode_count": len(p["episodes"]),
    } for p in pods] for lang, pods in buckets.items()}
    summary_path = os.path.join(BASE_DIR, "_buckets.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"分桶结果已存盘到 {summary_path}\n")

    # Step 3: 按桶下载到扁平目录
    print(f"=== 开始下载到 {BASE_DIR}/<lang>/ ===\n")
    for lang_dir in sorted(buckets.keys(), key=lambda x: -len(buckets[x])):
        pods = buckets[lang_dir]
        target_dir = os.path.join(BASE_DIR, lang_dir)
        print(f"\n########## 桶 [{lang_dir}]: {len(pods)} 个播客 ##########")
        d, s, f = download_bucket(pods, target_dir)
        print(f"桶 [{lang_dir}] 完成: 新下载 {d}, 跳过 {s}, 失败 {f}\n")

    print("全部完成!")


if __name__ == "__main__":
    main()
