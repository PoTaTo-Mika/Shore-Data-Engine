import argparse
import asyncio
import json
import logging
import uuid
from pathlib import Path

import websockets

# 确保作为脚本直接运行时也能找到项目根目录（便于绝对导入）
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_process.distill.seed_tts.protocols import MsgType, full_client_request, receive_message


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cluster(voice: str) -> str:
    if voice.startswith("S_"):
        return "volcano_icl"
    return "volcano_tts"


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


async def synthesize_text(cfg: dict, text: str, out_stem: str | None = None) -> Path:
    appid = cfg.get("APP_ID", "")
    token = cfg.get("ACCESS_TOKEN", "")
    voice_type = cfg.get("VOICE_TYPE", "")
    encoding = cfg.get("ENCODING", "wav")
    endpoint = cfg.get(
        "ENDPOINT", "wss://openspeech.bytedance.com/api/v1/tts/ws_binary"
    )
    cluster = cfg.get("CLUSTER") or get_cluster(voice_type)

    if not all([appid, token, voice_type, text]):
        raise ValueError("APP_ID/ACCESS_TOKEN/VOICE_TYPE/TEXT 不能为空")

    headers = {
        "Authorization": f"Bearer;{token}",
    }

    logger.info(f"Connecting to {endpoint}")
    async with websockets.connect(
        endpoint, extra_headers=headers, max_size=10 * 1024 * 1024
    ) as websocket:
        logid = ""
        resp_headers = getattr(websocket, "response_headers", None)
        if resp_headers and isinstance(resp_headers, dict):
            logid = resp_headers.get("x-tt-logid", "")
        logger.info(f"Connected" + (f", Logid: {logid}" if logid else ""))

        request = {
            "app": {
                "appid": appid,
                "token": token,
                "cluster": cluster,
            },
            "user": {
                "uid": str(uuid.uuid4()),
            },
            "audio": {
                "voice_type": voice_type,
                "encoding": encoding,
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "operation": "submit",
                "with_timestamp": "1",
                "extra_param": json.dumps({"disable_markdown_filter": False}),
            },
        }

        await full_client_request(websocket, json.dumps(request).encode())

        audio_data = bytearray()
        while True:
            msg = await receive_message(websocket)
            if msg.type == MsgType.FrontEndResultServer:
                continue
            elif msg.type == MsgType.AudioOnlyServer:
                audio_data.extend(msg.payload)
                if msg.sequence < 0:
                    break
            else:
                raise RuntimeError(f"TTS failed: {msg}")

        if not audio_data:
            raise RuntimeError("No audio data received")

        out_dir = Path(cfg.get("OUT_DIR", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_stem if out_stem else voice_type
        out_file = out_dir / f"{stem}.{encoding}"
        with out_file.open("wb") as f:
            f.write(audio_data)
        logger.info(f"Audio received: {len(audio_data)}, saved to {out_file}")
        return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parents[3] / "configs" / "seed_tts.json"),
        help="Path to config.json",
    )
    parser.add_argument(
        "--mapping",
        default="data/distill/mapping.json",
        help="Path to JSON mapping: {\"filename\": \"text\", ...}",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    mapping_path = args.mapping.strip()

    if mapping_path:
        with Path(mapping_path).open("r", encoding="utf-8") as f:
            name_to_text = json.load(f)
        if not isinstance(name_to_text, dict):
            raise ValueError("--mapping JSON 必须是 {filename: text} 的字典")

        async def run_batch():
            for stem, text in name_to_text.items():
                if not isinstance(stem, str) or not isinstance(text, str):
                    raise ValueError("--mapping 中的 key 和 value 都必须是字符串")
                await synthesize_text(cfg, text, out_stem=stem.rsplit('.', 1)[0])

        asyncio.run(run_batch())
    else:
        text = cfg.get("TEXT", "")
        asyncio.run(synthesize_text(cfg, text))


if __name__ == "__main__":
    main()


