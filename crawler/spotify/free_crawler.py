"""
Spotify 免费播客下载 crawler (Spotify 源)
  - 上游: open.spotify.com (Mercury/ext-metadata, EPISODE_V4/SHOW_V4)
  - 通道: Episode.external_url -> 直接 HTTP(S) 下载 (无 AES)
  - 需要: pip install librespot + 一个 Spotify 账号 (免费即可)
  - 输出: data/<lang>/<show_name>/<episode_name>.<ext>
  - 仅参考仓库 bilibili/hciyuan 等脚本风格; 不使用 Apple RSS / 外部镜像
"""

import io
import json
import os
import re
import time
from collections import defaultdict
from urllib.parse import unquote, urlparse

from librespot.core import Session
from librespot.metadata import EpisodeId, ShowId

################### 配置参数 ###################

# Spotify 账号 (免费账号即可下免费播客, 走 OAuth 后自动缓存 credentials)
SPOTIFY_USERNAME = os.environ.get("SPOTIFY_USERNAME", "")
SPOTIFY_PASSWORD = os.environ.get("SPOTIFY_PASSWORD", "")
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "credentials.json")

# 要扫的 show 列表; 也支持直接传 open.spotify.com/episode/<id>
SHOW_URLS = [
    # "https://open.spotify.com/show/4rOoJ6Egrf8K2IrywzwOMk",   # JRE (example)
]

# 单个 show 最多下载多少集 (None / 0 = 不限制)
MAX_EPISODES_PER_SHOW = 100

# 下载根目录, 最终结构:
#   data/<lang>/<show_name>/<episode_name>.<ext>
BASE_DIR = "data/spotify_free"

# 解析不到语言时归到这个子目录
UNKNOWN_LANG_DIR = "unknown"

# 单集文件名主截断长度 (不含扩展名)
MAX_FILENAME_LEN = 120

# 常见语言别名 → ISO 639-1
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

################### 通用工具 (从仓库其他 crawler 拷贝) ###################

def sanitize_filename(filename):
    """清洗文件名, 移除非法字符, 不截断"""
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", filename)
    safe_name = re.sub(r'\s+', " ", safe_name).strip()
    return safe_name


def truncate(s, maxlen):
    """截断字符串到 maxlen, 不在词中间断开"""
    if len(s) <= maxlen:
        return s
    cut = s[:maxlen]
    last_space = cut.rfind(" ", int(maxlen * 0.6))
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip()


def guess_extension(url, content_type=None):
    """按 url path / content-type 推断扩展名"""
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


def classify_language(rss_lang):
    """把 'hi-IN'/'hindi'/None 这类 RSS 语言标签归一到 ISO 639-1"""
    if not rss_lang:
        return UNKNOWN_LANG_DIR
    s = rss_lang.strip().lower()
    main = re.split(r"[-_.]", s)[0]
    return LANG_ALIASES.get(main, main)


################### Spotify 会话 & 鉴权 ###################

def create_session():
    """
    创建 Spotify Session.
    优先复用 CREDENTIALS_FILE ({type, username, credentials}), 兼容 rust/python librespot 格式;
    失败则 user/pass 登录并写回 credentials.
    """
    if os.path.exists(CREDENTIALS_FILE):
        try:
            builder = Session.Builder().stored_file(CREDENTIALS_FILE)
            session = builder.create()
            print(f"[ok] 复用 {CREDENTIALS_FILE}")
            return session
        except Exception as e:
            print(f"[!] 读 credentials 失败, 重新登录: {e}")

    if not SPOTIFY_USERNAME or not SPOTIFY_PASSWORD:
        raise RuntimeError(
            "缺少 Spotify 账号; 请设置环境变量 SPOTIFY_USERNAME / SPOTIFY_PASSWORD"
        )

    session = (
        Session.Builder().user_pass(SPOTIFY_USERNAME, SPOTIFY_PASSWORD).create()
    )
    # 保存以便下次复用
    try:
        # session.stored() -> base64 字符串; Builder.stored(...) 会解析
        # 但 Builder.stored_file(...) 更通吃(兼容 rust/protox), 这里直接写 base64 payload
        with open(CREDENTIALS_FILE, "w", encoding="utf-8") as f:
            stored_b64 = session.stored().encode("utf-8")
            # 反解 base64 到 dict, 以便下次 stored_file 读
            import base64, json as _j
            obj = _j.loads(base64.b64decode(stored_b64))
            _j.dump(obj, f, indent=2)
        print(f"[ok] credentials 已写入 {CREDENTIALS_FILE}")
    except Exception as e:
        print(f"[!] 无法写入 credentials: {e}")
    return session


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


def fetch_show_episode_ids(session, show_id):
    """
    用 librespot 的 ext-metadata API 拉一档 show 的所有 episode.
    返回 [EpisodeId, ...]
    """
    api = session.api()
    show = api.get_metadata_4_show(ShowId.from_base62(show_id))
    print(f"  [Show] {show.name}  lang={show.language}  n_ep={len(show.episode)}")

    out = []
    for ep_ref in show.episode:
        gid_hex = ep_ref.gid.hex()
        out.append(EpisodeId.from_hex(gid_hex))
    return show, out


def get_episode_metadata(session, episode_id):
    return session.api().get_metadata_4_episode(episode_id)


################### 下载主流程 (通道 A: external_url) ###################

def download_episode(session, episode_id, target_dir, index=None, total=None):
    """
    下载一集免费播客. external_url 分支; 明文 HTTP(S), 直接 stream 落盘.
    """
    ep = get_episode_metadata(session, episode_id)
    show_name = sanitize_filename(ep.show.name) if ep.show and ep.show.name else "unknown_show"
    ep_name = sanitize_filename(ep.name) or "episode"

    prefix = f"[{index}/{total}]" if index else ""

    if not ep.external_url:
        print(f"    {prefix} [skip] {ep_name}: 非 external_url (独占/付费), 请用 vip_crawler.py")
        return "skipped_vip"

    show_dir = os.path.join(target_dir, show_name)
    os.makedirs(show_dir, exist_ok=True)
    dest = os.path.join(show_dir, f"{truncate(ep_name, MAX_FILENAME_LEN)}.mp3")

    if os.path.exists(dest):
        print(f"    {prefix} [√] {show_name}/{ep_name[:60]}")
        return "exists"

    # librespot 自动: HEAD external_url -> 跟随 302 -> stream_external_episode (NoopAudioDecrypt)
    from librespot.audio import SuperAudioFormat  # noqa: F401  # pylint: disable=W0611
    from librespot.audio.decoders import AudioQuality, VorbisOnlyAudioQuality
    loaded = session.content_feeder().load_episode(
        episode_id,
        VorbisOnlyAudioQuality(AudioQuality.NORMAL),
        preload=False,
        halt_listener=None,
    )
    istream = loaded.input_stream.stream()

    tmp = dest + ".part"
    size = 0
    t0 = time.time()
    try:
        print(f"    {prefix} [↓] {show_name}/{ep_name[:60]}  size={istream.size()/1e6:.1f}MB", end="", flush=True)
        with open(tmp, "wb") as f:
            while True:
                buf = istream.read(128 * 1024)
                if not buf:
                    break
                f.write(buf)
                size += len(buf)
        os.rename(tmp, dest)
        dt = time.time() - t0
        print(f" 完成 ({size/1e6:.1f}MB, {size/dt/1e6:.2f}MB/s)")
    except Exception as e:
        print(f" 失败: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return "failed"
    return "downloaded"


def download_show(session, show_id, max_eps):
    """把一个 show 的免费 episodes 整档下载"""
    show, ep_ids = fetch_show_episode_ids(session, show_id)
    if not ep_ids:
        print(f"  [!] show {show_id} 没拿到任何 episode")
        return

    lang_dir = classify_language(show.language or "")
    out_root = os.path.join(BASE_DIR, lang_dir)
    os.makedirs(out_root, exist_ok=True)

    show_name = sanitize_filename(show.name) or "unknown_show"
    print(f"\n########## [{lang_dir}/{show_name}] 共 {len(ep_ids)} 集 ##########")

    if max_eps:
        ep_ids = ep_ids[:max_eps]

    cnt = defaultdict(int)
    total = len(ep_ids)
    for i, eid in enumerate(ep_ids, 1):
        try:
            r = download_episode(session, eid, out_root, i, total)
            cnt[r] += 1
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"    [{i}/{total}] [x] 异常: {e}")
            cnt["failed"] += 1
        time.sleep(0.15)

    print(f"\n[{show_name}] downloaded={cnt['downloaded']} exists={cnt['exists']} "
          f"skipped_vip={cnt['skipped_vip']} failed={cnt['failed']}")

    # 存一个 metadata 之后方便追溯
    meta_file = os.path.join(out_root, show_name, "_show_metadata.json")
    try:
        show_name_dir = os.path.join(out_root, show_name)
        os.makedirs(show_name_dir, exist_ok=True)
        meta_file = os.path.join(show_name_dir, "_show_metadata.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "show_id": show_id,
                    "show_name": show.name,
                    "language": show.language,
                    "publisher": show.publisher,
                    "media_type": str(show.media_type),
                    "episode_count": len(ep_ids),
                    "episodes_scanned": [e.hex_id() for e in ep_ids],
                },
                f, ensure_ascii=False, indent=2,
            )
        print(f"  show metadata -> {meta_file}")
    except Exception as e:
        print(f"  [!] 写 show metadata 失败: {e}")


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    session = create_session()

    manual_eps = []   # 直接给的 episode 链接, 不属于任何 show
    shows = []

    for u in SHOW_URLS:
        m = parse_show_id(u)
        if m:
            shows.append(m)
            continue
        e = parse_episode_id(u)
        if e:
            manual_eps.append(e)
            continue
        print(f"[!] 不认识的 URL: {u}")

    # 1) 整 show
    for sid in shows:
        download_show(session, sid, MAX_EPISODES_PER_SHOW)

    # 2) 单集
    if manual_eps:
        hit = 0
        for i, eid in enumerate(manual_eps, 1):
            try:
                ep = get_episode_metadata(session, EpisodeId.from_base62(eid))
                lang = classify_language((ep.show.language if ep.show else ""))
                r = download_episode(session, EpisodeId.from_base62(eid),
                                     os.path.join(BASE_DIR, lang), i, len(manual_eps))
                hit += int(r in ("downloaded", "exists"))
            except Exception as e:
                print(f"    [x] 单集 {eid} 异常: {e}")
        print(f"单集下载: {hit}/{len(manual_eps)}")


if __name__ == "__main__":
    main()
