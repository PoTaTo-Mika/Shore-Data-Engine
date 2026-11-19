"""HTTP protocol between the engine and stage backends (client side)."""

import json
import logging
import sys
import tarfile
import time
from dataclasses import dataclass
from io import BytesIO
from typing import cast
from urllib.parse import urlsplit

import httpx

from data_process.data_reader import AudioRecord, Operator

logger = logging.getLogger(__name__)

HEALTH_PATH = "/health"


def post(
    client: httpx.Client, request: httpx.Request, retry: dict[str, object]
) -> httpx.Response:
    """Send with exponential backoff on transport errors and retryable statuses."""
    attempts = cast(int, retry["max_attempts"])
    backoff = cast(float, retry["backoff_s"])
    retryable = set(cast(list[int], retry["status_codes"]))
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.send(request)
        except httpx.TransportError as exc:
            if attempt >= attempts:
                raise
            logger.warning(
                "Transport error on attempt %d/%d: %s", attempt, attempts, exc
            )
        else:
            if response.status_code not in retryable or attempt >= attempts:
                if response.is_error:
                    raise httpx.HTTPStatusError(
                        f"{request.url} returned {response.status_code}: "
                        f"{response.text}",
                        request=request,
                        response=response,
                    )
                return response
            logger.warning(
                "Retryable status %d on attempt %d/%d",
                response.status_code,
                attempt,
                attempts,
            )
        time.sleep(backoff * 2 ** (attempt - 1))


@dataclass(frozen=True, kw_only=True)
class OperatorBackend:
    """A chain of local operators served over HTTP, one tar member per request.

    The request body is the member bytes; the response is a tar of output members.
    Each member flows through the chain in order, flat-mapping as operators fan out.
    """

    module: str
    endpoint: str
    catalog: dict[str, str]

    def command(self, model: dict[str, object], base_url: str) -> list[str]:
        url = urlsplit(base_url)
        assert url.hostname and url.port, f"base_url needs host and port: {base_url}"
        return [
            sys.executable,
            "-m",
            self.module,
            "--steps",
            json.dumps(model["operators"]),
            "--host",
            url.hostname,
            "--port",
            str(url.port),
        ]

    def connect(
        self, client: httpx.Client, base_url: str, retry: dict[str, object]
    ) -> Operator:
        url = base_url + self.endpoint

        def request(record: AudioRecord) -> list[AudioRecord]:
            response = post(
                client,
                client.build_request(
                    "POST",
                    url,
                    params={"tar_name": record.tar_name, "audio_id": record.audio_id},
                    content=record.data,
                ),
                retry,
            )
            outputs = []
            with tarfile.open(fileobj=BytesIO(response.content)) as tar:
                for member in tar:
                    handle = tar.extractfile(member)
                    assert handle is not None
                    outputs.append(
                        AudioRecord(
                            tar_name=record.tar_name,
                            audio_id=member.name,
                            data=handle.read(),
                        )
                    )
            return outputs

        return request
