"""
Spotify 独占 / 付费播客下载 crawler (Spotify 源, 通道 B)
  - 上游: open.spotify.com (Mercury/ext-metadata, EPISODE_V4, AUDIO_FILES, PODCAST_SUBSCRIPTIONS)
  - 通道: episode.audio[] -> storage-resolve -> AES-CTR 加密 OGG/MP3 流
  - 需要: pip install librespot + Premium 账号 (或对应 podcast 订阅)
  - 输出: data/spotify_vip/<lang>/<show_name>/<episode_name>.ogg
"""

import json
import os
import re
import time
from collections import defaultdict

from librespot.audio import SuperAudioFormat
from librespot.audio.decoders import AudioQuality, VorbisOnlyAudioQuality
from librespot.core import Session
from librespot.metadata import EpisodeId, ShowId
from librespot.proto import ExtensionKind_pb2  # 仅作 ID 参考

################### 配置 ###################

SPOTIFY_USERNAME = os.environ.get("SPOTIFY_USERNAME", "")
SPOTIFY_PASSWORD = os.environ.get("SPOTIFY_PASSWORD", "")
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "credentials.json")

SHOW_URLS = [
    # "https://open.spotify.com/show/<spotify_original_or_paid_show_id>",
]

MAX_EPISODES_PER_SHOW = 200
BASE_DIR = "data/spotify_vip"
UNKNOWN_LANG_DIR = "unknown"
MAX_FILENAME_LEN = 120

LANG_ALIASES = {
    "hin": "hi", "eng": "en", "spa": "es", "fra": "fr",
    "deu": "de", "jpn": "ja", "kor": "ko", "zho": "zh",
    "por": "pt", "ita": "it", "rus": "ru", "ara": "ar",
}

################### 工具 (保持仓库风格) ###################

def sanitize_filename(filename):
    safe = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return re.sub(r"\s+", " ", safe).strip()


def truncate(s, maxlen):
    if len(s) <= maxlen:
        return s
    cut = s[:maxlen]
    sp = cut.rfind(" ", int(maxlen * 0.6))
    return cut[:sp].rstrip() if sp > 0 else cut.rstrip()


def classify_language(rss_lang):
    if not rss_lang:
        return UNKNOWN_LANG_DIR
    main = re.split(r"[-_.]", rss_lang.strip().lower())[0]
    return LANG_ALIASES.get(main, main)


################### Session ###################

def create_session():
    if os.path.exists(CREDENTIALS_FILE):
        try:
            builder = Session.Builder().stored_file(CREDENTIALS_FILE)
            s = builder.create()
            print(f"[ok] 复用 {CREDENTIALS_FILE}")
            return s
        except Exception as e:
            print(f"[!] 读 credentials 失败, 重新登录: {e}")

    if not SPOTIFY_USERNAME or not SPOTIFY_PASSWORD:
        raise RuntimeError("缺少 Spotify 账号; 请设置 SPOTIFY_USERNAME / SPOTIFY_PASSWORD")

    s = Session.Builder().user_pass(SPOTIFY_USERNAME, SPOTIFY_PASSWORD).create()
    try:
        import base64
        with open(CREDENTIALS_FILE, "w", encoding="utf-8") as f:
            obj = json.loads(base64.b64decode(s.stored()))
            json.dump(obj, f, indent=2)
        print(f"[ok] credentials 已写入 {CREDENTIALS_FILE}")
    except Exception as e:
        print(f"[!] 无法写 credentials: {e}")
    return s


################### Episode / Show 解析 ###################

EPISODE_RE = re.compile(
    r"(?:open\.spotify\.com/(?:intl-[a-z-]+/)?episode/|spotify:episode:)"
    r"([A-Za-z0-9]{22})"
)
SHOW_RE = re.compile(
    r"(?:open\.spotify\.com/(?:intl-[a-z-]+/)?show/|spotify:show:)"
    r"([A-Za-z0-9]{22})"
)


def parse_episode_id(text):
    m = EPISODE_RE.search(text)
    return m.group(1) if m else None


def parse_show_id(text):
    m = SHOW_RE.search(text)
    return m.group(1) if m else None


################### 付费预审 (EPISODE_PLAYABILITY) ###################

def is_episode_playable(session, episode_id):
    """
    用 ext-metadata EPISODE_PLAYABILITY (46) 检查该集是否对当前账号开放.
    """
    try:
        raw = session.api().get_ext_metadata(
            ExtensionKind_pb2.EPISODE_PLAYABILITY,
            episode_id.to_spotify_uri(),
        )
        # 返回 bytes, 一般是一段 JSON/protobuf; 试一下 json.loads
        try:
            meta = json.loads(raw)
            return bool(meta.get("playable", meta.get("is_playable", True)))
        except Exception:
            return True
    except Exception as e:
        print(f"      playability check err: {e}")
        return True   # 拿不到时放行, 真正下载时再决定


################### 核心: 付费下载 (通道 B) ###################

QUALITY_PICKER = VorbisOnlyAudioQuality(AudioQuality.VERY_HIGH)


def download_paid_episode(session, episode_id, target_dir, index=None, total=None):
    """
    下载一集独占/付费播客. 走 AES 解密流, 写 .ogg (或 .mp3 看 format).
    """
    prefix = f"[{index}/{total}]" if index else ""
    api = session.api()
    ep = api.get_metadata_4_episode(episode_id)

    if ep.external_url:
        print(f"    {prefix} [skip] 是免费 external, 请用 free_crawler.py")
        return "skipped_free"

    if not is_episode_playable(session, episode_id):
        print(f"    {prefix} [skip] {ep.name[:60]} playability=false (该账号无权限)")
        return "skipped_paywall"

    show_name = sanitize_filename(ep.show.name) if ep.show and ep.show.name else "unknown_show"
    ep_name = sanitize_filename(ep.name) or "episode"
    show_dir = os.path.join(target_dir, show_name)
    os.makedirs(show_dir, exist_ok=True)

    # 依据 format 推扩展名, 优先 Vorbis; 若无 Vorbis 库内部会自己退到其它可用格式
    dest = os.path.join(show_dir, f"{truncate(ep_name, MAX_FILENAME_LEN)}.ogg")
    if os.path.exists(dest):
        print(f"    {prefix} [√] {show_name}/{ep_name[:60]}")
        return "exists"

    t0 = time.time()
    tmp = dest + ".part"
    try:
        loaded = session.content_feeder().load_episode(
            episode_id, QUALITY_PICKER, preload=False, halt_listener=None,
        )
        streamer = loaded.input_stream
        istream = streamer.stream()
        # CdnFeedHelper.load_episode 已经帮我们 skip(0xA7), 可直读

        print(
            f"    {prefix} [↓] {show_name}/{ep_name[:60]}"
            f" size={istream.size()/1e6:.1f}MB codec={streamer.codec()}", end="", flush=True,
        )

        with open(tmp, "wb") as f:
            bytes_written = 0
            while True:
                buf = istream.read(128 * 1024)
                if not buf:
                    break
                f.write(buf)
                bytes_written += len(buf)
        os.rename(tmp, dest)
        dt = max(time.time() - t0, 0.001)
        print(f" 完成 ({bytes_written/1e6:.1f}MB, {bytes_written/dt/1e6:.2f}MB/s)")
    except Exception as e:
        print(f" 失败: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return "failed"
    return "downloaded"


################### 整 show 处理 ###################

def download_show(session, show_id, max_eps):
    api = session.api()
    show = api.get_metadata_4_show(ShowId.from_base62(show_id))
    if not show.episode:
        print(f"  [!] show {show_id} 无集数")
        return

    ep_ids = [EpisodeId.from_hex(ep.gid.hex()) for ep in show.episode]
    if max_eps:
        ep_ids = ep_ids[:max_eps]

    lang = classify_language(show.language or "")
    out_root = os.path.join(BASE_DIR, lang)
    os.makedirs(out_root, exist_ok=True)
    show_name = sanitize_filename(show.name) or "unknown_show"
    print(f"\n########## [{lang}/{show_name}] 共 {len(ep_ids)} 集 ##########")

    cnt = defaultdict(int)
    for i, eid in enumerate(ep_ids, 1):
        try:
            r = download_paid_episode(session, eid, out_root, i, len(ep_ids))
            cnt[r] += 1
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"    [{i}/{len(ep_ids)}] [x] 异常: {e}")
            cnt["failed"] += 1
        time.sleep(0.2)

    print(
        f"\n[{show_name}] downloaded={cnt['downloaded']} exists={cnt['exists']}"
        f" skipped_paywall={cnt['skipped_paywall']} failed={cnt['failed']}"
    )

    # 保存 show metadata
    try:
        meta_file = os.path.join(out_root, show_name, "_show_metadata.json")
        os.makedirs(os.path.dirname(meta_file), exist_ok=True)
        json.dump(
            {
                "show_id": show_id,
                "show_name": show.name,
                "language": show.language,
                "publisher": show.publisher,
                "media_type": str(show.media_type),
                "episode_count": len(ep_ids),
                "episodes": [e.hex_id() for e in ep_ids],
            },
            open(meta_file, "w", encoding="utf-8"),
            ensure_ascii=False, indent=2,
        )
        print(f"  show metadata -> {meta_file}")
    except Exception as e:
        print(f"  [!] 写 show metadata 失败: {e}")


################### 入口 ###################

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    session = create_session()

    manual_eps = []
    shows = []
    for u in SHOW_URLS:
        sid = parse_show_id(u)
        if sid:
            shows.append(sid)
            continue
        eid = parse_episode_id(u)
        if eid:
            manual_eps.append(EpisodeId.from_base62(eid))
            continue
        print(f"[!] 不认识的 URL: {u}")

    for sid in shows:
        download_show(session, sid, MAX_EPISODES_PER_SHOW)

    if manual_eps:
        for i, eid in enumerate(manual_eps, 1):
            api = session.api()
            try:
                ep = api.get_metadata_4_episode(eid)
                lang = classify_language((ep.show.language if ep.show else ""))
                download_paid_episode(session, eid, os.path.join(BASE_DIR, lang), i, len(manual_eps))
            except Exception as e:
                print(f"  [x] 单集失败: {e}")


if __name__ == "__main__":
    main()
