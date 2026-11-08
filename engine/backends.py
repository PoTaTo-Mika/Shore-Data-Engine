"""Model service lifecycle and byte-level ASR requests."""

import logging
import mimetypes
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx

logger = logging.getLogger(__name__)

Config = dict[str, object]

ASR_ENDPOINT = "/v1/audio/transcriptions"
ASR_FIELDS = frozenset({"language", "prompt", "response_format"})


@dataclass(frozen=True, kw_only=True)
class Request:
    url: str
    data: dict[str, str] | None = None
    files: dict[str, tuple[str, bytes, str]] | None = None


def build_request(
    model: Config, audio_id: str, data: bytes, base_url: str
) -> Request:
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

    payload: dict[str, object] = {"model": cast(str, model["model"]), **params}
    filename = Path(audio_id).name
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return Request(
        url=base_url.rstrip("/") + ASR_ENDPOINT,
        data={key: str(value) for key, value in payload.items()},
        files={"file": (filename, data, mime)},
    )


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


def extract_text(payload: dict[str, object]) -> str:
    return cast(str, payload["text"]).strip()


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


class HttpTranscriber:
    def __init__(self, service: Service, model: Config) -> None:
        assert service.client is not None
        self.client = service.client
        self.base_url = service.base_url
        self.model = model
        self.retry = cast(Config, service.config["retry"])

    def transcribe(self, audio_id: str, data: bytes) -> str:
        request = build_request(self.model, audio_id, data, self.base_url)
        return extract_text(post(self.client, request, self.retry))


def start_services(services: dict[str, Config]) -> list[Service]:
    """Launch services in config order and wait for each one to become healthy."""
    started: list[Service] = []
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
