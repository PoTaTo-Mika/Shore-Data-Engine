"""Stage dispatch for ASR against an OpenAI-compatible transcription backend."""

import mimetypes
from collections.abc import Callable
from pathlib import Path
from typing import cast

import httpx

from data_process.data_reader import AudioRecord
from data_process.protocol import post

ENDPOINT = "/v1/audio/transcriptions"
FIELDS = frozenset({"language", "prompt", "response_format"})


def build_request(
    client: httpx.Client, base_url: str, model: dict[str, object], record: AudioRecord
) -> httpx.Request:
    endpoint = cast(str, model.get("endpoint", ENDPOINT))
    if endpoint != ENDPOINT:
        raise ValueError(f"ASR only supports {ENDPOINT}, got {endpoint}")
    transport = cast(str, model.get("transport", "multipart"))
    if transport != "multipart":
        raise ValueError(f"ASR only supports multipart transport, got {transport}")

    raw = cast(dict[str, object], model.get("params", {}))
    params = {key: value for key, value in raw.items() if value is not None}
    unknown = set(params) - FIELDS
    if unknown:
        raise ValueError(f"Unmapped ASR fields: {sorted(unknown)}")
    if params.get("response_format", "json") != "json":
        raise ValueError("ASR only accepts response_format=json")

    payload = {"model": cast(str, model["model"]), **params}
    filename = Path(record.audio_id).name
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return client.build_request(
        "POST",
        base_url + ENDPOINT,
        data={key: str(value) for key, value in payload.items()},
        files={"file": (filename, record.data, mime)},
    )


def connect(
    client: httpx.Client,
    base_url: str,
    model: dict[str, object],
    retry: dict[str, object],
) -> Callable[[AudioRecord], str]:
    def transcribe(record: AudioRecord) -> str:
        response = post(client, build_request(client, base_url, model, record), retry)
        return cast(str, response.json()["text"]).strip()

    return transcribe
