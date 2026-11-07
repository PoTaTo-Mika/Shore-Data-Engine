"""Tests for the config-driven serial tar pipeline."""

import base64
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import httpx

from data_process.annotation import load_completed
from data_process.data_reader import Operator
from engine.backends import (
    Config,
    Request,
    Service,
    build_request,
    extract_text,
    post,
    start_services,
)
from engine.run import (
    engine_revision,
    expand_env,
    load_config,
    run_pipeline,
)

ASR_MODEL: Config = {
    "model": "Qwen/Qwen3-ASR-1.7B",
    "endpoint": "/v1/audio/transcriptions",
    "transport": "multipart",
    "params": {"language": "zh", "response_format": "json"},
}
RETRY: Config = {"max_attempts": 3, "backoff_s": 0.0, "status_codes": [503]}
SERVICE: Config = {
    "base_url": "http://svc",
    "health_path": "/health",
    "api_key_env": "",
    "timeout_s": 5.0,
    "startup_timeout_s": 1.0,
    "retry": RETRY,
}


def make_audio_tar(path: Path, names: tuple[str, ...]) -> None:
    with tarfile.open(path, "w") as tar:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = 4
            tar.addfile(info, io.BytesIO(b"RIFF"))


def pipeline_config(root: Path) -> Config:
    return {
        "data_root": str(root),
        "output_name": "annotations.jsonl",
        "audio_extensions": [".opus"],
        "services": {"omni": SERVICE},
        "models": {"demucs": {}, "rms_vad": {}, "asr": ASR_MODEL},
        "stages": [
            {"name": "stage1", "task": "uvr", "model": "demucs"},
            {"name": "stage2", "task": "slicer", "model": "rms_vad"},
            {"name": "stage3", "task": "asr", "service": "omni", "model": "asr"},
        ],
    }


class ConfigTests(unittest.TestCase):
    def test_expands_env_with_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                expand_env("${SHORE_TEST_URL:-http://localhost:9}"),
                "http://localhost:9",
            )
        with patch.dict(os.environ, {"SHORE_TEST_URL": "http://example:1"}):
            self.assertEqual(
                expand_env("${SHORE_TEST_URL:-http://localhost:9}"),
                "http://example:1",
            )

    def test_repository_config_declares_three_stages(self) -> None:
        cfg = load_config(
            Path(__file__).resolve().parents[1] / "engine" / "configs" / "run.yaml"
        )
        stages = cast(list[Config], cfg["stages"])
        self.assertEqual([s["name"] for s in stages], ["stage1", "stage2", "stage3"])
        self.assertEqual([s["task"] for s in stages], ["uvr", "slicer", "asr"])
        self.assertEqual(cfg["output_name"], "annotations.jsonl")

    def test_engine_revision_tracks_submodule(self) -> None:
        path = Path(__file__).resolve().parents[1] / "engine" / "sglang-omni"
        revision = engine_revision(path)
        if (path / ".git").exists():
            self.assertEqual(len(cast(str, revision)), 40)
        else:
            self.assertIsNone(revision)


class RequestTests(unittest.TestCase):
    def test_transcription_payload_uses_bytes(self) -> None:
        request = build_request(ASR_MODEL, "album/clip.opus", b"RIFF", "http://svc")
        self.assertEqual(
            request.data,
            {
                "model": "Qwen/Qwen3-ASR-1.7B",
                "language": "zh",
                "response_format": "json",
            },
        )
        files = cast(dict[str, tuple[str, bytes, str]], request.files)
        self.assertEqual(files["file"][0], "clip.opus")
        self.assertEqual(files["file"][1], b"RIFF")

    def test_unmapped_field_is_rejected(self) -> None:
        model = {**ASR_MODEL, "params": {"language": "zh", "temperature": 0}}
        with self.assertRaises(ValueError):
            build_request(model, "clip.wav", b"RIFF", "http://svc")

    def test_chat_request_embeds_base64_audio(self) -> None:
        model: Config = {
            "model": "qwen3-omni",
            "endpoint": "/v1/chat/completions",
            "transport": "base64",
            "params": {"modalities": ["text"]},
        }
        request = build_request(model, "clip.wav", b"RIFF", "http://svc")
        body = cast(dict[str, object], request.body)
        self.assertIn(base64.b64encode(b"RIFF").decode("ascii"), json.dumps(body))

    def test_extract_text_from_both_endpoints(self) -> None:
        self.assertEqual(
            extract_text("/v1/audio/transcriptions", {"text": " hi "}), "hi"
        )
        self.assertEqual(
            extract_text(
                "/v1/chat/completions",
                {"choices": [{"message": {"content": " ok "}}]},
            ),
            "ok",
        )

    def test_retries_service_unavailable(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if len(calls) < 3:
                return httpx.Response(503)
            return httpx.Response(200, json={"text": "done"})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            payload = post(
                client, Request(url="http://svc/v1/audio/transcriptions"), RETRY
            )
        self.assertEqual(payload, {"text": "done"})
        self.assertEqual(len(calls), 3)


class ServiceTests(unittest.TestCase):
    def test_services_start_in_config_order(self) -> None:
        with patch.object(Service, "start", autospec=True) as starter:
            services = start_services({"first": SERVICE, "second": SERVICE})
        self.assertEqual([s.name for s in services], ["first", "second"])
        self.assertEqual(starter.call_count, 2)

    def test_health_is_checked_before_use(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return httpx.Response(200)

        real_client = httpx.Client
        with patch(
            "engine.backends.httpx.Client",
            side_effect=lambda **kwargs: real_client(
                transport=httpx.MockTransport(handler), **kwargs
            ),
        ):
            service = Service("omni", SERVICE)
            service.start()
            service.stop()
        self.assertEqual(calls, ["/health"])


class PipelineTests(unittest.TestCase):
    def test_stages_run_serially_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tar_path = root / "en_000.tar"
            make_audio_tar(tar_path, ("album/a.opus", "album/b.opus"))
            cfg = pipeline_config(root)
            calls: list[str] = []

            def handler(request: httpx.Request) -> httpx.Response:
                calls.append(request.url.path)
                return httpx.Response(200, json={"text": "hello"})

            def fake_start(services: dict[str, Config]) -> list[Service]:
                client = httpx.Client(transport=httpx.MockTransport(handler))
                return [Service("omni", SERVICE, client=client)]

            operator_calls: list[str] = []

            def fake_build_operator(task: str, model: Config) -> Operator:
                operator_calls.append(task)
                return lambda record: [record]

            with patch("engine.run.start_services", side_effect=fake_start), patch(
                "engine.run.build_operator", side_effect=fake_build_operator
            ):
                run_pipeline(cfg, None, 0, False)

            self.assertEqual(operator_calls, ["uvr", "slicer"])
            output = root / "annotations.jsonl"
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {r["audio_id"] for r in records},
                {"album/a.opus", "album/b.opus"},
            )
            self.assertTrue(
                all(
                    r["tar_name"] == "en_000.tar" and r["asr_text"] == "hello"
                    for r in records
                )
            )
            self.assertEqual(
                load_completed(output),
                {("en_000.tar", "album/a.opus"), ("en_000.tar", "album/b.opus")},
            )
            transcribed = calls.count("/v1/audio/transcriptions")
            self.assertEqual(transcribed, 2)

            with patch("engine.run.start_services", side_effect=fake_start), patch(
                "engine.run.build_operator", side_effect=fake_build_operator
            ):
                run_pipeline(cfg, None, 0, False)
            self.assertEqual(calls.count("/v1/audio/transcriptions"), transcribed)
            self.assertEqual(operator_calls, ["uvr", "slicer"])

    def test_stage_filter_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus", "b.opus"))
            cfg = pipeline_config(root)
            calls: list[str] = []

            def handler(request: httpx.Request) -> httpx.Response:
                calls.append(request.url.path)
                return httpx.Response(200, json={"text": "hello"})

            def fake_start(services: dict[str, Config]) -> list[Service]:
                client = httpx.Client(transport=httpx.MockTransport(handler))
                return [Service("omni", SERVICE, client=client)]

            with patch("engine.run.start_services", side_effect=fake_start), patch(
                "engine.run.build_operator"
            ) as build:
                run_pipeline(cfg, "stage3", 1, False)
            build.assert_not_called()
            lines = (root / "annotations.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)

    def test_dry_run_starts_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus",))
            cfg = pipeline_config(root)
            with patch("engine.run.start_services") as start, patch(
                "engine.run.build_operator"
            ) as build:
                run_pipeline(cfg, None, 0, True)
            start.assert_not_called()
            build.assert_not_called()
            self.assertFalse((root / "annotations.jsonl").exists())
