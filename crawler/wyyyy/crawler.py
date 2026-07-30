"""
网易云音乐 — 暴力遍历 song_id 抓取 YRC/LRC 转写
==============================================
从 song_id=0 开始递增，超高并发探测。
返回 200 + 有歌词 → 写入文件；404/空响应 → 跳过。
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path

import aiohttp

# ========== 配置 ==========
START_ID = 0
END_ID = 9_999_999_999          # 遍历范围上限
CONCURRENCY = 50_000             # 并发数（asyncio semaphore）
TIMEOUT = aiohttp.ClientTimeout(total=3, connect=2)
BATCH_SIZE = 100_000             # 每批上报一次进度
OUTPUT_DIR = Path("./data/wyyyy/transcripts")
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://music.163.com/",
}

# 统计
found: list[dict] = []
attempted = 0
start_time = time.time()


def safe_name(song: dict) -> str:
    name = f"{song.get('name','unknown')} - {song.get('artist','unknown')}"
    return name.replace("/", "_").replace("\\", "_").replace("\x00", "")[:200]


# ================================================================
#  YRC 解析
# ================================================================

def parse_yrc(yrc_text: str) -> list[list[dict]]:
    """
    "[00:04.260]遠[00:04.940]い[00:05.150]夏"
    → [[{"time_ms":4260,"text":"遠"}, ...], ...]
    """
    parsed = []
    for line in yrc_text.strip().split("\n"):
        if not line.strip():
            continue
        ts = re.findall(r'\[(\d{2}):(\d{2})\.(\d{2,3})\]', line)
        parts = re.split(r'\[\d{2}:\d{2}\.\d{2,3}\]', line)
        words = []
        for i, (m, s, ms) in enumerate(ts):
            minutes = int(m)
            seconds = int(s)
            millis = int(ms) * (10 if len(ms) == 2 else 1)
            time_ms = minutes * 60000 + seconds * 1000 + millis
            text = parts[i + 1] if i + 1 < len(parts) else ""
            if text:
                words.append({"time_ms": time_ms, "text": text})
        if words:
            parsed.append(words)
    return parsed


def lrc_has_ts(text: str) -> bool:
    return bool(re.search(r'\[\d{2}:\d{2}\.\d{2,3}\]', text))


# ================================================================
#  核心：获取单首歌的歌词
# ================================================================

async def fetch_one(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                    song_id: int) -> dict | None:
    """返回有歌词的歌曲信息 dict，否则返回 None。"""
    async with sem:
        url = "https://music.163.com/api/song/lyric"
        params = {"os": "pc", "id": song_id, "lv": -1, "kv": -1, "tv": -1}
        try:
            async with session.get(url, params=params, headers=HEADERS) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json(encoding="utf-8")
        except Exception:
            return None

    # 提取歌词
    def _safe(field: str) -> str:
        f = data.get(field)
        if f is None:
            return ""
        t = f.get("lyric")
        return t if t else ""

    lrc_text = _safe("lrc")
    yrc_text = _safe("yrc")
    klyric_text = _safe("klyric")
    tlrc_text = _safe("tlyric")
    word_text = yrc_text or klyric_text

    # 需要至少有一种带时间戳的歌词
    if not word_text and not (lrc_text and lrc_has_ts(lrc_text)):
        return None

    # --- 提取歌曲名+歌手（从 LRC 元数据或额外请求） ---
    name = "unknown"
    artist = "unknown"
    # LRC 前几行通常有 [ti:歌名] [ar:歌手]
    for line in lrc_text.split("\n")[:20]:
        m = re.match(r'\[ti:\s*(.+)\]', line)
        if m:
            name = m.group(1).strip()
        m = re.match(r'\[ar:\s*(.+)\]', line)
        if m:
            artist = m.group(1).strip()

    return {
        "id": song_id,
        "name": name,
        "artist": artist,
        "yrc": word_text,                    # 逐字原文（yrc 或 klyric）
        "yrc_parsed": parse_yrc(word_text) if word_text else [],
        "lrc": lrc_text if lrc_has_ts(lrc_text) else "",
        "tlrc": tlrc_text if tlrc_text else "",
    }


# ================================================================
#  写入 + 进度
# ================================================================

def save_one(song: dict):
    sname = safe_name(song)
    word_text = song["yrc"]
    yrc_parsed = song["yrc_parsed"]
    lrc_text = song["lrc"]

    if word_text:
        # YRC/klyric 逐字
        ext = ".yrc"
        (OUTPUT_DIR / f"{sname}{ext}").write_text(word_text, encoding="utf-8")
        if yrc_parsed:
            (OUTPUT_DIR / f"{sname}.yrc.json").write_text(
                json.dumps(yrc_parsed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    else:
        # LRC 兜底
        (OUTPUT_DIR / f"{sname}.lrc").write_text(lrc_text, encoding="utf-8")


# ================================================================
#  主循环
# ================================================================

async def main():
    global attempted, found, start_time

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(CONCURRENCY)

    connector = aiohttp.TCPConnector(limit=CONCURRENCY, force_close=True)
    async with aiohttp.ClientSession(connector=connector, timeout=TIMEOUT) as session:
        for batch_start in range(START_ID, END_ID + 1, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, END_ID + 1)
            ids = range(batch_start, batch_end)

            tasks = [asyncio.create_task(fetch_one(session, sem, sid)) for sid in ids]
            results = await asyncio.gather(*tasks)

            for r in results:
                if r is not None:
                    found.append(r)
                    save_one(r)
                    logging.info(f"  ★ [{r['id']}] {r['name']} — {r['artist']} "
                                 f"({'YRC' if r['yrc'] else 'LRC'})")

            attempted += BATCH_SIZE
            elapsed = time.time() - start_time
            rate = attempted / elapsed if elapsed > 0 else 0
            logging.info(f"--- 进度: {batch_end - 1:,}/{END_ID:,} "
                         f"| 命中: {len(found)} "
                         f"| 速率: {rate:,.0f} req/s ---")

        logging.info(f"遍历完成。共命中 {len(found)} 首。")


if __name__ == "__main__":
    asyncio.run(main())
