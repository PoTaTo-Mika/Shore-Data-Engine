"""Uniform payload interface for local and HTTP model backends."""

import base64
import importlib
import logging
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import httpx

logger = logging.getLogger(__name__)

ALLOWED_FIELDS: dict[str, frozenset[str]] = {
    "/v1/audio/transcriptions": frozenset({"language", "prompt", "response_format"}),
    "/v1/chat/completions": frozenset(
        {"modalities", "max_tokens", "temperature", "top_p", "seed", "stop"}
    ),
}


@dataclass(kw_only=True)
class Payload:
    task: str
    scope: Path
    inputs: tuple[Path, ...]
    params: dict[str, object]


@dataclass(kw_only=True)
class Result:
    annotations: dict[str, str] = field(default_factory=dict)
    failures: tuple[Path, ...] = ()


class Backend(Protocol):
    def process(self, payload: Payload) -> Result: ...


@dataclass(frozen=True, kw_only=True)
class Request:
    url: str
    data: dict[str, str] | None = None
    files: dict[str, tuple[str, bytes, str]] | None = None
    body: dict[str, object] | None = None


def build_request(
    model: dict[str, object], audio: Path, base_url: str
) -> Request:
    endpoint = cast(str, model["endpoint"])
    allowed = ALLOWED_FIELDS.get(endpoint)
    if allowed is None:
        raise ValueError(f"Unsupported endpoint: {endpoint}")
    params = cast(dict[str, object], model.get("params", {}))
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"Unmapped fields for {endpoint}: {sorted(unknown)}")

    payload: dict[str, object] = {"model": cast(str, model["model"])}
    payload.update({k: v for k, v in params.items() if v is not None})
    url = base_url.rstrip("/") + endpoint
    transport = cast(str, model["transport"])

    if endpoint == "/v1/audio/transcriptions":
        if transport != "multipart":
            raise ValueError("Transcription endpoint only supports multipart transport")
        mime = mimetypes.guess_type(audio.name)[0] or "application/octet-stream"
        return Request(
            url=url,
            data={k: str(v) for k, v in payload.items()},
            files={"file": (audio.name, audio.read_bytes(), mime)},
        )

    if transport != "base64":
        raise ValueError("Chat endpoint only supports base64 transport")
    audio_format = audio.suffix.lower().lstrip(".")
    if audio_format not in {"wav", "mp3"}:
        raise ValueError(f"base64 transport expects wav or mp3, got {audio_format}")
    encoded = base64.b64encode(audio.read_bytes()).decode("ascii")
    body = {
        **payload,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded, "format": audio_format},
                    }
                ],
            }
        ],
    }
    return Request(url=url, body=body)


def post(
    client: httpx.Client, request: Request, retry: dict[str, object]
) -> dict[str, object]:
    attempts = cast(int, retry["max_attempts"])
    backoff = cast(float, retry["backoff_s"])
    retryable = set(cast(list[int], retry["status_codes"]))
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.post(
                request.url,
                data=request.data,
                files=request.files,
                json=request.body,
            )
        except httpx.TransportError as exc:
            if attempt >= attempts:
                raise
            logger.warning("Transport error on attempt %d/%d: %s", attempt, attempts, exc)
            time.sleep(backoff * 2 ** (attempt - 1))
            continue
        if response.status_code in retryable and attempt < attempts:
            logger.warning(
                "Retryable status %d on attempt %d/%d",
                response.status_code,
                attempt,
                attempts,
            )
            time.sleep(backoff * 2 ** (attempt - 1))
            continue
        response.raise_for_status()
        return cast(dict[str, object], response.json())


def extract_text(endpoint: str, payload: dict[str, object]) -> str:
    if endpoint == "/v1/audio/transcriptions":
        return cast(str, payload["text"]).strip()
    choices = cast(list[dict[str, object]], payload["choices"])
    message = cast(dict[str, object], choices[0]["message"])
    return cast(str, message["content"]).strip()


class LocalBackend:
    """Runs in-process model operators behind the same payload interface."""

    def process(self, payload: Payload) -> Result:
        if payload.task == "audio.separate":
            module = importlib.import_module("data_process.uvr.run_demucs")
            module.process_folder(str(payload.scope))
        elif payload.task == "audio.slice":
            module = importlib.import_module("data_process.slicer.silero_vad_")
            module.process_folder_recursive(
                str(payload.scope),
                max_workers=cast(int | None, payload.params.get("max_workers")),
            )
        elif payload.task == "label.export":
            module = importlib.import_module("tools.extract_label")
            module.process_one_sliced_dir(payload.scope)
        else:
            raise ValueError(f"Unknown local task: {payload.task}")
        return Result()


class HttpBackend:
    """Sends each payload item to an OpenAI-compatible model service."""

    def __init__(self, service: dict[str, object], model: dict[str, object]) -> None:
        self._service = service
        self._model = model
        self._base_url = cast(str, service["base_url"])

    def process(self, payload: Payload) -> Result:
        if not payload.inputs:
            return Result()
        key_env = cast(str, self._service.get("api_key_env", ""))
        api_key = os.environ.get(key_env) if key_env else None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        retry = cast(dict[str, object], self._service["retry"])
        annotations: dict[str, str] = {}
        failures: list[Path] = []
        with httpx.Client(
            timeout=cast(float, self._service["timeout_s"]), headers=headers
        ) as client:
            health_path = cast(str, self._service["health_path"])
            client.get(self._base_url.rstrip("/") + health_path).raise_for_status()
            with ThreadPoolExecutor(
                max_workers=cast(int, self._service["concurrency"])
            ) as pool:
                pairs = pool.map(
                    lambda audio: self._annotate_one(client, audio, retry),
                    payload.inputs,
                )
                for audio, text in pairs:
                    if text is None:
                        failures.append(audio)
                    else:
                        annotations[str(audio.absolute())] = text
        return Result(annotations=annotations, failures=tuple(failures))

    def _annotate_one(
        self,
        client: httpx.Client,
        audio: Path,
        retry: dict[str, object],
    ) -> tuple[Path, str | None]:
        request = build_request(self._model, audio, self._base_url)
        try:
            response = post(client, request, retry)
            text = extract_text(cast(str, self._model["endpoint"]), response)
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            logger.error("Annotation failed for %s: %s", audio, exc)
            return audio, None
        return audio, text


def build_backend(
    service: dict[str, object], model: dict[str, object]
) -> Backend:
    kind = cast(str, service["type"])
    if kind == "local":
        return LocalBackend()
    if kind == "http":
        return HttpBackend(service, model)
    raise ValueError(f"Unknown backend type: {kind}")
