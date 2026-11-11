"""HTTP protocol between the engine and stage backends."""

import argparse
import importlib
import json
import logging
import sys
import tarfile
import time
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import cast
from urllib.parse import parse_qs, urlsplit

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
    """A stage whose local operator is served over HTTP, one tar member per request.

    The request body is the member bytes; the response is a tar of output members.
    """

    module: str
    endpoint: str
    operators: dict[str, str]

    def command(self, model: dict[str, object], base_url: str) -> list[str]:
        url = urlsplit(base_url)
        assert url.hostname and url.port, f"base_url needs host and port: {base_url}"
        return [
            sys.executable,
            "-m",
            self.module,
            "--operator",
            cast(str, model["operator"]),
            "--params",
            json.dumps(model.get("params", {})),
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

    def serve(self) -> None:
        """Entry point of a stage dispatch module: load one operator and serve it."""
        parser = argparse.ArgumentParser(description=f"Serve {self.endpoint}.")
        parser.add_argument("--operator", required=True, choices=sorted(self.operators))
        parser.add_argument("--params", default="{}", help="JSON operator params")
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, required=True)
        args = parser.parse_args()
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
        )
        module = importlib.import_module(self.operators[args.operator])
        operator = cast(Operator, module.build_operator(json.loads(args.params)))
        endpoint = self.endpoint

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def reply(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                status = 200 if self.path == HEALTH_PATH else 404
                self.reply(status, b"", "text/plain")

            def do_POST(self) -> None:
                url = urlsplit(self.path)
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if url.path != endpoint:
                    self.reply(404, b"", "text/plain")
                    return
                query = parse_qs(url.query)
                record = AudioRecord(
                    tar_name=query["tar_name"][0],
                    audio_id=query["audio_id"][0],
                    data=body,
                )
                try:
                    outputs = list(operator(record))
                except Exception:
                    # Return the traceback so the engine reports why this member failed.
                    logger.exception("Operator failed on %s", record.audio_id)
                    self.reply(500, traceback.format_exc().encode(), "text/plain")
                    return
                buffer = BytesIO()
                with tarfile.open(fileobj=buffer, mode="w") as tar:
                    for output in outputs:
                        info = tarfile.TarInfo(output.audio_id)
                        info.size = len(output.data)
                        tar.addfile(info, BytesIO(output.data))
                self.reply(200, buffer.getvalue(), "application/x-tar")

            def log_message(self, format: str, *args: object) -> None:
                pass  # Per-request access logs would flood the stage log.

        server = ThreadingHTTPServer((args.host, args.port), Handler)
        logger.info("Serving %s on %s:%d", args.operator, args.host, args.port)
        server.serve_forever()
