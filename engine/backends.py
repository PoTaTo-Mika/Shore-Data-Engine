"""Model service lifecycle and byte-level ASR requests."""

import base64
import logging
import mimetypes
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import httpx

logger = logging.getLogger(__name__)

Config = dict[str, object]

ALLOWED_FIELDS: dict[str, frozenset[str]] = {
    "/v1/audio/transcriptions": frozenset({"language", "prompt", "response_format"}),
    "/v1/chat/completions": frozenset(
        {"modalities", "max_tokens", "temperature", "top_p", "seed", "stop"}
    ),
}


@dataclass(frozen=True, kw_only=True)
class Request:
    url: str
    data: dict[str, str] | None = None
    files: dict[str, tuple[str, bytes, str]] | None = None
    body: dict[str, object] | None = None


def build_request(
    model: Config, audio_id: str, data: bytes, base_url: str
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
    filename = Path(audio_id).name

    if endpoint == "/v1/audio/transcriptions":
        if transport != "multipart":
            raise ValueError("Transcription endpoint only supports multipart transport")
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return Request(
            url=url,
            data={k: str(v) for k, v in payload.items()},
            files={"file": (filename, data, mime)},
        )

    if transport != "base64":
        raise ValueError("Chat endpoint only supports base64 transport")
    audio_format = Path(filename).suffix.lower().lstrip(".")
    if audio_format not in {"wav", "mp3"}:
        raise ValueError(f"base64 transport expects wav or mp3, got {audio_format}")
    body = {
        **payload,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(data).decode("ascii"),
                            "format": audio_format,
                        },
                    }
                ],
            }
        ],
    }
    return Request(url=url, body=body)


def post(client: httpx.Client, request: Request, retry: Config) -> dict[str, object]:
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


@dataclass
class Service:
    name: str
    config: Config
    process: subprocess.Popen | None = None
    client: httpx.Client | None = None

    @property
    def base_url(self) -> str:
        return cast(str, self.config["base_url"]).rstrip("/")

    def start(self) -> None:
        command = self.config.get("command")
        if command is not None:
            argv = cast(list[str], command)
            logger.info("Starting service %s: %s", self.name, " ".join(argv))
            self.process = subprocess.Popen(
                argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        key_env = cast(str, self.config.get("api_key_env", ""))
        api_key = os.environ.get(key_env) if key_env else None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(
            timeout=cast(float, self.config["timeout_s"]), headers=headers
        )
        health_url = self.base_url + cast(str, self.config["health_path"])
        deadline = time.monotonic() + cast(float, self.config["startup_timeout_s"])
        while True:
            try:
                if self.client.get(health_url).status_code == 200:
                    logger.info("Service %s is ready", self.name)
                    return
            except httpx.TransportError:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Service {self.name} not healthy at {health_url}")
            time.sleep(1)

    def stop(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=30)
            logger.info("Stopped service %s", self.name)
            self.process = None


class Transcriber(Protocol):
    def transcribe(self, audio_id: str, data: bytes) -> str: ...


class HttpTranscriber:
    def __init__(self, service: Service, model: Config) -> None:
        assert service.client is not None
        self.client = service.client
        self.base_url = service.base_url
        self.model = model
        self.retry = cast(Config, service.config["retry"])

    def transcribe(self, audio_id: str, data: bytes) -> str:
        request = build_request(self.model, audio_id, data, self.base_url)
        response = post(self.client, request, self.retry)
        return extract_text(cast(str, self.model["endpoint"]), response)


def start_services(services: dict[str, Config], dry_run: bool = False) -> list[Service]:
    """Launch services in config order and wait for each one to become healthy."""
    started: list[Service] = []
    if dry_run:
        return started
    for name, config in services.items():
        service = Service(name, config)
        try:
            service.start()
        except (OSError, RuntimeError):
            stop_services(started)
            raise
        started.append(service)
    return started


def stop_services(services: list[Service]) -> None:
    for service in reversed(services):
        service.stop()
