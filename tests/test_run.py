"""Tests for the payload-based annotation pipeline."""

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import httpx

from data_process.backends import (
    HttpBackend,
    LocalBackend,
    Payload,
    Request,
    build_backend,
    build_request,
    extract_text,
    post,
)
from data_process.run import (
    Config,
    build_payloads,
    engine_revision,
    expand_env,
    load_config,
    run_pipeline,
)

ASR_MODEL: Config = {
    "backend": "omni",
    "task": "audio.transcribe",
    "model": "Qwen/Qwen3-ASR-1.7B",
    "endpoint": "/v1/audio/transcriptions",
    "transport": "multipart",
    "params": {"language": "zh", "response_format": "json"},
}
RETRY: Config = {"max_attempts": 3, "backoff_s": 0.0, "status_codes": [503]}
SERVICE: Config = {
    "type": "http",
    "base_url": "http://svc",
    "health_path": "/health",
    "api_key_env": "",
    "timeout_s": 5.0,
    "concurrency": 2,
    "retry": RETRY,
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

    def test_loads_repository_config(self) -> None:
        cfg = load_config(
            Path(__file__).resolve().parents[1] / "data_process" / "configs" / "run.yaml"
        )
        self.assertEqual(cfg["data_root"], "data")
        self.assertIn("qwen3_asr", cast(dict[str, object], cfg["models"]))
        self.assertIn("omni", cast(dict[str, object], cfg["backends"]))

    def test_engine_revision_tracks_submodule(self) -> None:
        path = Path(__file__).resolve().parents[1] / "engine" / "sglang-omni"
        revision = engine_revision(path)
        if (path / ".git").exists():
            self.assertEqual(len(cast(str, revision)), 40)
        else:
            self.assertIsNone(revision)

    def test_build_payloads_respects_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "album" / "sliced").mkdir(parents=True)
            (root / "album" / "a.wav").write_bytes(b"RIFF")
            (root / "album" / "sliced" / "b.wav").write_bytes(b"RIFF")
            cfg: Config = {
                "data_root": str(root),
                "sliced_dirname": "sliced",
                "audio_extensions": [".wav"],
            }
            dataset = build_payloads(
                {"scope": "dataset"}, {"task": "audio.separate"}, cfg
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0].scope, root)
            self.assertEqual([p.name for p in dataset[0].inputs], ["a.wav"])
            sliced = build_payloads(
                {"scope": "sliced"}, {"task": "label.export"}, cfg
            )
            self.assertEqual(len(sliced), 1)
            self.assertEqual([p.name for p in sliced[0].inputs], ["b.wav"])


class RequestTests(unittest.TestCase):
    def test_transcription_payload_only_sends_declared_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "clip.wav"
            audio.write_bytes(b"RIFF")
            model = {
                **ASR_MODEL,
                "params": {
                    **cast(dict[str, object], ASR_MODEL["params"]),
                    "prompt": None,
                },
            }
            request = build_request(model, audio, "http://svc")
            self.assertEqual(
                request.data,
                {
                    "model": "Qwen/Qwen3-ASR-1.7B",
                    "language": "zh",
                    "response_format": "json",
                },
            )
            files = cast(dict[str, tuple[str, bytes, str]], request.files)
            self.assertEqual(files["file"][0], "clip.wav")
            self.assertEqual(files["file"][1], b"RIFF")

    def test_unmapped_field_is_rejected(self) -> None:
        model = {**ASR_MODEL, "params": {"language": "zh", "temperature": 0}}
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "clip.wav"
            audio.write_bytes(b"RIFF")
            with self.assertRaises(ValueError):
                build_request(model, audio, "http://svc")

    def test_chat_request_embeds_base64_audio(self) -> None:
        model: Config = {
            "model": "qwen3-omni",
            "endpoint": "/v1/chat/completions",
            "transport": "base64",
            "params": {"modalities": ["text"]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "clip.wav"
            audio.write_bytes(b"RIFF")
            request = build_request(model, audio, "http://svc")
        body = cast(dict[str, object], request.body)
        self.assertEqual(body["modalities"], ["text"])
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


class BackendTests(unittest.TestCase):
    def test_local_backend_dispatches_by_task(self) -> None:
        fake = SimpleNamespace(
            process_folder=MagicMock(),
            process_folder_recursive=MagicMock(),
            process_one_sliced_dir=MagicMock(),
        )
        with patch(
            "data_process.backends.importlib.import_module", return_value=fake
        ) as importer:
            backend = LocalBackend()
            separate = backend.process(
                Payload(
                    task="audio.separate",
                    scope=Path("/data"),
                    inputs=(),
                    params={},
                )
            )
            importer.assert_called_with("data_process.uvr.run_demucs")
            fake.process_folder.assert_called_once_with("/data")
            self.assertEqual(separate.annotations, {})
            backend.process(
                Payload(
                    task="audio.slice",
                    scope=Path("/data"),
                    inputs=(),
                    params={"max_workers": 8},
                )
            )
            fake.process_folder_recursive.assert_called_once_with(
                "/data", max_workers=8
            )
            backend.process(
                Payload(
                    task="label.export",
                    scope=Path("/data/sliced"),
                    inputs=(),
                    params={},
                )
            )
            fake.process_one_sliced_dir.assert_called_once_with(Path("/data/sliced"))

    def test_unknown_backend_and_task_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_backend({"type": "other"}, ASR_MODEL)
        with self.assertRaises(ValueError):
            LocalBackend().process(
                Payload(task="unknown", scope=Path("/data"), inputs=(), params={})
            )

    def test_http_backend_returns_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "clip.wav"
            audio.write_bytes(b"RIFF")
            calls: list[str] = []

            def handler(request: httpx.Request) -> httpx.Response:
                calls.append(request.url.path)
                if request.url.path == "/health":
                    return httpx.Response(200, json={"status": "healthy"})
                return httpx.Response(200, json={"text": "hello"})

            transport = httpx.MockTransport(handler)
            real_client = httpx.Client
            with patch(
                "data_process.backends.httpx.Client",
                side_effect=lambda **kwargs: real_client(transport=transport, **kwargs),
            ):
                backend = HttpBackend(SERVICE, ASR_MODEL)
                result = backend.process(
                    Payload(
                        task="audio.transcribe",
                        scope=Path(tmp),
                        inputs=(audio,),
                        params={},
                    )
                )
            self.assertEqual(result.annotations, {str(audio.absolute()): "hello"})
            self.assertEqual(result.failures, ())
            self.assertEqual(calls, ["/health", "/v1/audio/transcriptions"])

    def test_pipeline_resumes_without_repeating_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sliced = root / "album" / "sliced"
            sliced.mkdir(parents=True)
            for name in ("a.wav", "b.wav"):
                (sliced / name).write_bytes(b"RIFF")
            calls: list[str] = []

            def handler(request: httpx.Request) -> httpx.Response:
                calls.append(request.url.path)
                if request.url.path == "/health":
                    return httpx.Response(200, json={"status": "healthy"})
                return httpx.Response(200, json={"text": "hello"})

            transport = httpx.MockTransport(handler)
            cfg: Config = {
                "data_root": str(root),
                "sliced_dirname": "sliced",
                "output_name": "transcription.json",
                "audio_extensions": [".wav"],
                "backends": {"omni": SERVICE},
                "models": {"asr": ASR_MODEL},
                "pipeline": [
                    {"stage": "asr", "model": "asr", "scope": "sliced", "resume": True}
                ],
            }
            real_client = httpx.Client
            with patch(
                "data_process.backends.httpx.Client",
                side_effect=lambda **kwargs: real_client(transport=transport, **kwargs),
            ):
                run_pipeline(cfg, "asr", 0, False)
                pairs = cast(
                    dict[str, str],
                    json.loads((sliced / "transcription.json").read_text(encoding="utf-8")),
                )
                self.assertEqual(sorted(Path(k).name for k in pairs), ["a.wav", "b.wav"])
                self.assertEqual(set(pairs.values()), {"hello"})
                run_pipeline(cfg, "asr", 0, False)
                self.assertEqual(
                    calls.count("/v1/audio/transcriptions"), 2
                )

    def test_dry_run_does_not_call_backends(self) -> None:
        cfg: Config = {
            "data_root": "data",
            "sliced_dirname": "sliced",
            "output_name": "transcription.json",
            "audio_extensions": [".wav"],
            "backends": {"local": {"type": "local"}, "omni": SERVICE},
            "models": {
                "demucs": {"backend": "local", "task": "audio.separate"},
                "asr": ASR_MODEL,
            },
            "pipeline": [
                {"stage": "uvr", "model": "demucs", "scope": "dataset"},
                {"stage": "asr", "model": "asr", "scope": "sliced", "resume": True},
            ],
        }
        with patch.object(LocalBackend, "process") as local_process:
            run_pipeline(cfg, None, 0, True)
            local_process.assert_not_called()
