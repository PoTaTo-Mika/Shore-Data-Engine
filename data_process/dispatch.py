"""Stage dispatch: operator backends, the shared server entry point, and ASR client."""

import argparse
import mimetypes
from collections.abc import Callable
from pathlib import Path
from typing import cast

import httpx

from data_process.data_reader import AudioRecord
from data_process.protocol import OperatorBackend, post

DISPATCH_MODULE = "data_process.dispatch"

BACKENDS: dict[str, OperatorBackend] = {
    "uvr": OperatorBackend(
        module=DISPATCH_MODULE,
        endpoint="/v1/audio/separations",
        operators={
            "demucs": "data_process.uvr.demucs",
            "mdx_net": "data_process.uvr.mdx",
        },
    ),
    "slicer": OperatorBackend(
        module=DISPATCH_MODULE,
        endpoint="/v1/audio/segments",
        operators={
            "rms_vad": "data_process.vad.rms",
            "silero_vad": "data_process.vad.silero",
            "ten_vad": "data_process.vad.ten",
        },
    ),
}

OPERATOR_TASKS = {
    operator: task
    for task, backend in BACKENDS.items()
    for operator in backend.operators
}

ASR_ENDPOINT = "/v1/audio/transcriptions"
ASR_FIELDS = frozenset({"language", "prompt", "response_format"})


def asr_request(
    client: httpx.Client, base_url: str, model: dict[str, object], record: AudioRecord
) -> httpx.Request:
    """Build a multipart transcription request from the model's declared params."""
    endpoint = cast(str, model.get("endpoint", ASR_ENDPOINT))
    if endpoint != ASR_ENDPOINT:
        raise ValueError(f"ASR only supports {ASR_ENDPOINT}, got {endpoint}")
    transport = cast(str, model.get("transport", "multipart"))
    if transport != "multipart":
        raise ValueError(f"ASR only supports multipart transport, got {transport}")

    raw = cast(dict[str, object], model.get("params", {}))
    params = {key: value for key, value in raw.items() if value is not None}
    unknown = set(params) - ASR_FIELDS
    if unknown:
        raise ValueError(f"Unmapped ASR fields: {sorted(unknown)}")
    if params.get("response_format", "json") != "json":
        raise ValueError("ASR only accepts response_format=json")

    payload = {"model": cast(str, model["model"]), **params}
    filename = Path(record.audio_id).name
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return client.build_request(
        "POST",
        base_url + ASR_ENDPOINT,
        data={key: str(value) for key, value in payload.items()},
        files={"file": (filename, record.data, mime)},
    )


def asr_transcriber(
    client: httpx.Client,
    base_url: str,
    model: dict[str, object],
    retry: dict[str, object],
) -> Callable[[AudioRecord], str]:
    def transcribe(record: AudioRecord) -> str:
        response = post(client, asr_request(client, base_url, model, record), retry)
        return cast(str, response.json()["text"]).strip()

    return transcribe


def main() -> None:
    """Serve the backend that owns the requested operator."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--operator", required=True, choices=sorted(OPERATOR_TASKS))
    args, _ = parser.parse_known_args()
    BACKENDS[OPERATOR_TASKS[cast(str, args.operator)]].serve()


if __name__ == "__main__":
    main()
