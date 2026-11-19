"""Operator-chain backend: decode once, run the chain, encode the final members.

Chained operators pass decoded PCM to one another, so an isolation step feeds the
following slicer directly; only terminal members are encoded, by `ensure_encoded`.
"""

import argparse
import importlib
import json
import logging
import tarfile
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import cast
from urllib.parse import parse_qs, urlsplit

from data_process.codec import ensure_encoded
from data_process.data_reader import AudioRecord, Operator
from data_process.dispatch import OPERATOR_ENDPOINT, OPERATORS
from data_process.protocol import HEALTH_PATH

logger = logging.getLogger(__name__)


def build_chain(steps: list[dict[str, object]]) -> tuple[list[Operator], list[str]]:
    operators: list[Operator] = []
    names: list[str] = []
    for step in steps:
        name = cast(str, step["operator"])
        if name not in OPERATORS:
            raise ValueError(f"Unknown operator: {name}")
        module = importlib.import_module(OPERATORS[name])
        params = cast(dict[str, object], step.get("params", {}))
        operators.append(cast(Operator, module.build_operator(params)))
        names.append(name)
    return operators, names


def run_chain(operators: list[Operator], record: AudioRecord) -> list[AudioRecord]:
    records = [record]
    for operator in operators:
        records = [out for r in records for out in operator(r)]
    return [ensure_encoded(r) for r in records]


def serve(host: str, port: int, steps: list[dict[str, object]]) -> None:
    operators, names = build_chain(steps)

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
            if self.path.split("?")[0] != OPERATOR_ENDPOINT:
                self.reply(404, b"", "text/plain")
                return
            body = self.rfile.read(int(self.headers["Content-Length"]))
            query = parse_qs(urlsplit(self.path).query)
            record = AudioRecord(
                tar_name=query["tar_name"][0],
                audio_id=query["audio_id"][0],
                data=body,
            )
            try:
                outputs = run_chain(operators, record)
            except Exception:
                # Return the traceback so the engine reports why this member failed.
                logger.exception("Operator chain failed on %s", record.audio_id)
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

    server = ThreadingHTTPServer((host, port), Handler)
    logger.info("Serving %s on %s:%d", " -> ".join(names), host, port)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Serve {OPERATOR_ENDPOINT}.")
    parser.add_argument(
        "--steps", required=True, help="JSON list of {operator, params}"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    serve(args.host, args.port, cast(list[dict[str, object]], json.loads(args.steps)))


if __name__ == "__main__":
    main()
