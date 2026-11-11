"""Tests for the config-driven pipeline and its stage backends."""

import io
import json
import os
import socket
import sys
import tarfile
import tempfile
import unittest
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import httpx
import numpy as np
import soundfile as sf

from data_process.annotation import Annotation, append_annotation, load_completed
from data_process.asr.dispatch import build_request
from data_process.data_reader import AudioRecord
from data_process.protocol import post
from data_process.state import (
    CURSOR_NAME,
    append_record,
    iter_records,
    load_cursors,
    pipeline_lock,
)
from data_process.uvr.dispatch import BACKEND as UVR_BACKEND
from data_process.vad.dispatch import BACKEND as VAD_BACKEND
from engine.backends import Config, Service, launch
from engine.run import (
    OPERATOR_BACKENDS,
    engine_revision,
    expand_env,
    load_config,
    model_task,
    run_pipeline,
    service_config,
    show_status,
)

REPO_CONFIG = Path(__file__).resolve().parents[1] / "engine" / "configs" / "run.yaml"
ASR_MODEL: Config = {
    "model": "Qwen/Qwen3-ASR-1.7B",
    "params": {"language": "zh", "response_format": "json"},
}
RETRY: Config = {"max_attempts": 3, "backoff_s": 0.0, "status_codes": [503]}


def service(base_url: str) -> Config:
    return {
        "base_url": base_url,
        "health_path": "/health",
        "timeout_s": 5.0,
        "startup_timeout_s": 10.0,
        "retry": RETRY,
    }


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])


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
        "stage_workers": 3,
        "services": {
            "uvr": service("http://uvr:1"),
            "vad": service("http://vad:2"),
            "omni": service("http://omni:3"),
        },
        "models": {
            "demucs": {"operator": "demucs"},
            "mdx_net": {"operator": "mdx_net"},
            "rms_vad": {"operator": "rms_vad"},
            "asr": {
                **ASR_MODEL,
                "command": ["sgl-omni", "serve", "--model-path", "Qwen/Qwen3-ASR-1.7B"],
            },
        },
        "stages": [
            {"name": "stage1", "task": "uvr", "service": "uvr", "model": "demucs"},
            {"name": "stage2", "task": "slicer", "service": "vad", "model": "rms_vad"},
            {"name": "stage3", "task": "asr", "service": "omni", "model": "asr"},
        ],
    }


def tar_body(records: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in records:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


Handler = Callable[[str, httpx.Request], httpx.Response]


def echo(events: list[str]) -> Handler:
    """Backend stub: operator stages echo the member, ASR returns `hello`."""

    def handle(name: str, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/audio/transcriptions":
            events.append(f"{name}:asr")
            return httpx.Response(200, json={"text": " hello "})
        audio_id = request.url.params["audio_id"]
        events.append(f"{name}:{request.url.params['tar_name']}:{audio_id}")
        return httpx.Response(200, content=tar_body([(audio_id, request.content)]))

    return handle


def fake_launch(
    handler: Handler, started: list[tuple[str, Config]]
) -> Callable[[str, Config], Iterator[Service]]:
    @contextmanager
    def launch_stub(name: str, config: Config) -> Iterator[Service]:
        started.append((name, config))
        transport = httpx.MockTransport(lambda request: handler(name, request))
        with httpx.Client(transport=transport) as client:
            yield Service(
                name=name,
                base_url=cast(str, config["base_url"]),
                client=client,
                retry=RETRY,
            )
        started.append((f"stop:{name}", config))

    return launch_stub


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

    def test_repository_config_binds_every_stage_to_a_service(self) -> None:
        cfg = load_config(REPO_CONFIG)
        stages = cast(list[Config], cfg["stages"])
        self.assertEqual([s["name"] for s in stages], ["stage1", "stage2", "stage3"])
        self.assertEqual([s["task"] for s in stages], ["uvr", "slicer", "asr"])
        self.assertEqual([s["service"] for s in stages], ["uvr", "vad", "omni"])
        self.assertEqual(cfg["output_name"], "annotations.jsonl")

    def test_repository_config_operators_match_dispatch_modules(self) -> None:
        models = cast(dict[str, Config], load_config(REPO_CONFIG)["models"])
        operators = {
            key: entry["operator"]
            for key, entry in models.items()
            if "operator" in entry
        }
        self.assertEqual(
            set(operators.values()),
            set(UVR_BACKEND.operators) | set(VAD_BACKEND.operators),
        )
        self.assertEqual(model_task(models["mdx_net"]), "uvr")
        self.assertEqual(model_task(models["ten_vad"]), "slicer")

    def test_repository_config_declares_all_asr_models(self) -> None:
        models = cast(dict[str, Config], load_config(REPO_CONFIG)["models"])
        asr = {key: entry for key, entry in models.items() if "model" in entry}
        self.assertEqual(
            set(asr),
            {
                "qwen3_asr",
                "fun_asr_nano",
                "ark_asr",
                "nemotron_3_5_asr",
                "nemotron_transcribe",
                "ming3_asr",
                "whisper_large_v3",
                "moss_td",
            },
        )
        for key, entry in asr.items():
            with self.subTest(model=key):
                self.assertIn("--model-path", cast(list[str], entry["command"]))
                self.assertEqual(
                    cast(dict[str, object], entry["params"])["response_format"],
                    "json",
                )

    def test_stage_model_task_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.yaml"
            cfg = pipeline_config(Path(tmp))
            cast(list[Config], cfg["stages"])[0]["model"] = "rms_vad"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_unknown_stage_service_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.yaml"
            cfg = pipeline_config(Path(tmp))
            cast(list[Config], cfg["stages"])[1]["service"] = "missing"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_engine_revision_tracks_submodule(self) -> None:
        path = Path(__file__).resolve().parents[1] / "engine" / "sglang-omni"
        revision = engine_revision(path)
        if (path / ".git").exists():
            self.assertEqual(len(cast(str, revision)), 40)
        else:
            self.assertIsNone(revision)

    def test_operator_service_command_runs_stage_dispatch(self) -> None:
        model: Config = {"operator": "mdx_net", "params": {"stem": "vocals"}}
        config = service_config("uvr", service("http://127.0.0.1:8001"), model)
        command = cast(list[str], config["command"])
        self.assertEqual(command[1:3], ["-m", "data_process.uvr.dispatch"])
        self.assertEqual(command[command.index("--operator") + 1], "mdx_net")
        self.assertEqual(
            json.loads(command[command.index("--params") + 1]), {"stem": "vocals"}
        )
        self.assertEqual(command[command.index("--port") + 1], "8001")

    def test_asr_service_command_comes_from_model(self) -> None:
        config = service_config(
            "asr", service("http://omni:3"), {**ASR_MODEL, "command": ["x"]}
        )
        self.assertEqual(config["command"], ["x"])
        self.assertNotIn(
            "command", service_config("asr", service("http://omni:3"), ASR_MODEL)
        )


class RequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = httpx.Client()
        self.record = AudioRecord(
            tar_name="en.tar", audio_id="album/clip.opus", data=b"RIFF"
        )

    def tearDown(self) -> None:
        self.client.close()

    def test_transcription_payload_uses_bytes(self) -> None:
        request = build_request(self.client, "http://svc", ASR_MODEL, self.record)
        self.assertEqual(str(request.url), "http://svc/v1/audio/transcriptions")
        body = request.read()
        self.assertIn(b'name="language"\r\n\r\nzh', body)
        self.assertIn(b'filename="clip.opus"', body)
        self.assertIn(b"RIFF", body)

    def test_unmapped_field_is_rejected(self) -> None:
        model = {**ASR_MODEL, "params": {"language": "zh", "temperature": 0}}
        with self.assertRaises(ValueError):
            build_request(self.client, "http://svc", model, self.record)

    def test_non_asr_endpoint_is_rejected(self) -> None:
        model: Config = {**ASR_MODEL, "endpoint": "/v1/chat/completions"}
        with self.assertRaises(ValueError):
            build_request(self.client, "http://svc", model, self.record)

    def test_non_text_response_format_is_rejected(self) -> None:
        model: Config = {**ASR_MODEL, "params": {"response_format": "verbose_json"}}
        with self.assertRaises(ValueError):
            build_request(self.client, "http://svc", model, self.record)

    def test_retries_service_unavailable(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if len(calls) < 3:
                return httpx.Response(503)
            return httpx.Response(200, json={"text": "done"})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            response = post(client, client.build_request("POST", "http://svc/x"), RETRY)
        self.assertEqual(response.json(), {"text": "done"})
        self.assertEqual(len(calls), 3)

    def test_backend_error_body_is_reported(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Traceback: boom")

        with (
            httpx.Client(transport=httpx.MockTransport(handler)) as client,
            self.assertRaisesRegex(httpx.HTTPStatusError, "boom"),
        ):
            post(client, client.build_request("POST", "http://svc/x"), RETRY)


class BackendTests(unittest.TestCase):
    def test_launch_waits_for_health_and_stops_process(self) -> None:
        port = free_port()
        script = (
            "import http.server as h\n"
            "class H(h.BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        self.send_response(200); self.end_headers()\n"
            f"h.HTTPServer(('127.0.0.1', {port}), H).serve_forever()\n"
        )
        config = {
            **service(f"http://127.0.0.1:{port}"),
            "command": [sys.executable, "-c", script],
        }
        with launch("stub", config) as svc:
            self.assertEqual(svc.client.get(svc.base_url + "/health").status_code, 200)
        with self.assertRaises(httpx.ConnectError):
            httpx.get(f"http://127.0.0.1:{port}/health")

    def test_launch_fails_fast_when_process_exits(self) -> None:
        config = {
            **service(f"http://127.0.0.1:{free_port()}"),
            "command": [sys.executable, "-c", "raise SystemExit(3)"],
        }
        with (
            self.assertRaisesRegex(RuntimeError, "exited with code 3"),
            launch("dead", config),
        ):
            pass

    def test_operator_backend_roundtrip_over_http(self) -> None:
        port = free_port()
        model: Config = {"operator": "rms_vad", "params": {"min_length": 1000}}
        config = service_config("slicer", service(f"http://127.0.0.1:{port}"), model)
        tone = np.sin(2 * np.pi * 440 * np.arange(3 * 44100) / 44100) * 0.5
        buffer = io.BytesIO()
        sf.write(
            buffer, np.concatenate([tone, np.zeros(44100), tone]), 44100, format="WAV"
        )
        record = AudioRecord(
            tar_name="en.tar", audio_id="a/b.wav", data=buffer.getvalue()
        )
        with launch("vad", config) as svc:
            operator = VAD_BACKEND.connect(svc.client, svc.base_url, svc.retry)
            outputs = operator(record)
            with self.assertRaises(httpx.HTTPStatusError):
                operator(replace(record, data=b"not audio"))
        self.assertEqual(
            [r.audio_id for r in outputs], ["a/b_000.opus", "a/b_001.opus"]
        )
        self.assertTrue(
            all(r.tar_name == "en.tar" and r.data[:4] == b"OggS" for r in outputs)
        )


class PipelineTests(unittest.TestCase):
    def run_with_backends(
        self,
        cfg: Config,
        handler: Handler,
        *,
        only: str | None = None,
        limit: int = 0,
        overrides: list[str] | None = None,
    ) -> list[tuple[str, Config]]:
        started: list[tuple[str, Config]] = []
        with patch("engine.run.launch", fake_launch(handler, started)):
            run_pipeline(cfg, only, limit, False, overrides)
        return started

    def test_each_stage_runs_its_backend_and_covers_all_tars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus",))
            make_audio_tar(root / "en_001.tar", ("b.opus",))
            events: list[str] = []
            started = self.run_with_backends(pipeline_config(root), echo(events))

            self.assertEqual(
                [name for name, _ in started],
                ["uvr", "stop:uvr", "vad", "stop:vad", "omni", "stop:omni"],
            )
            self.assertEqual(
                events,
                [
                    "uvr:en_000.tar:a.opus",
                    "uvr:en_001.tar:b.opus",
                    "vad:en_000.tar:a.opus",
                    "vad:en_001.tar:b.opus",
                    "omni:asr",
                    "omni:asr",
                ],
            )
            uvr_command = cast(list[str], started[0][1]["command"])
            self.assertIn("data_process.uvr.dispatch", uvr_command)
            self.assertEqual(
                started[4][1]["command"],
                ["sgl-omni", "serve", "--model-path", "Qwen/Qwen3-ASR-1.7B"],
            )

    def test_stages_resume_without_restarting_backends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("album/a.opus", "album/b.opus"))
            cfg = pipeline_config(root)
            events: list[str] = []
            self.run_with_backends(cfg, echo(events))

            output = root / "annotations.jsonl"
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {r["audio_id"] for r in records}, {"album/a.opus", "album/b.opus"}
            )
            self.assertTrue(all(r["asr_text"] == "hello" for r in records))
            self.assertEqual(list(root.glob(".*.rewrite")), [])
            cursor_events = list(iter_records(root / CURSOR_NAME))
            self.assertEqual(
                {(e["stage"], e["tar_name"]) for e in cursor_events if e.get("done")},
                {("stage1", "en_000.tar"), ("stage2", "en_000.tar")},
            )
            self.assertEqual(load_cursors(root / CURSOR_NAME), {})

            events.clear()
            self.assertEqual(self.run_with_backends(cfg, echo(events)), [])
            self.assertEqual(events, [])

    def test_stage_filter_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus", "b.opus"))
            events: list[str] = []
            started = self.run_with_backends(
                pipeline_config(root), echo(events), only="stage3", limit=1
            )
            self.assertEqual([name for name, _ in started], ["omni", "stop:omni"])
            lines = (
                (root / "annotations.jsonl").read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(len(lines), 1)

    def test_dry_run_starts_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus",))
            with patch("engine.run.launch") as start:
                run_pipeline(pipeline_config(root), None, 0, True)
            start.assert_not_called()
            self.assertFalse((root / "annotations.jsonl").exists())

    def test_overrides_select_backend_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus",))
            cfg = pipeline_config(root)
            alt: Config = {
                **ASR_MODEL,
                "model": "openai/whisper-large-v3",
                "command": [
                    "sgl-omni",
                    "serve",
                    "--model-path",
                    "checkpoints/whisper-large-v3",
                ],
            }
            cast(dict[str, Config], cfg["models"])["alt"] = alt
            bodies: list[bytes] = []
            handle = echo([])

            def handler(name: str, request: httpx.Request) -> httpx.Response:
                bodies.append(request.read())
                return handle(name, request)

            started = self.run_with_backends(cfg, handler, overrides=["mdx_net", "alt"])
            uvr_command = cast(list[str], started[0][1]["command"])
            self.assertEqual(
                uvr_command[uvr_command.index("--operator") + 1], "mdx_net"
            )
            self.assertEqual(started[4][1]["command"], alt["command"])
            self.assertIn(b"openai/whisper-large-v3", bodies[-1])

    def test_unknown_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            run_pipeline(pipeline_config(Path(tmp)), None, 0, True, ["missing"])

    def test_override_without_matching_stage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            run_pipeline(pipeline_config(Path(tmp)), "stage3", 0, True, ["mdx_net"])

    def test_stage_without_pending_work_starts_no_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                self.run_with_backends(pipeline_config(Path(tmp)), echo([])), []
            )

    def test_stage_workers_and_cursor_log_are_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus",))
            started: list[tuple[str, Config]] = []
            with (
                patch("engine.run.launch", fake_launch(echo([]), started)),
                patch("engine.run.transform_tar", return_value=1) as transform,
            ):
                run_pipeline(pipeline_config(root), "stage1", 0, False)
            kwargs = transform.call_args.kwargs
            self.assertEqual(kwargs["workers"], 3)
            self.assertEqual(kwargs["cursor_path"], root / CURSOR_NAME)

    def test_asr_progress_is_streamed_to_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus", "b.opus"))
            cfg = pipeline_config(root)
            output = root / "annotations.jsonl"
            transcribed: list[str] = []

            def paused(name: str, request: httpx.Request) -> httpx.Response:
                transcribed.append(name)
                if len(transcribed) > 1:
                    return httpx.Response(400, text="paused")
                return httpx.Response(200, json={"text": "hello"})

            with self.assertRaises(httpx.HTTPStatusError):
                self.run_with_backends(cfg, paused, only="stage3")
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                [json.loads(line)["audio_id"] for line in lines], ["a.opus"]
            )

            events: list[str] = []
            self.run_with_backends(cfg, echo(events), only="stage3")
            self.assertEqual(len(load_completed(output)), 2)
            self.assertEqual(events, ["omni:asr"])

    def test_status_streams_cursors_and_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "annotations.jsonl"
            with (root / CURSOR_NAME).open("a", encoding="utf-8") as handle:
                append_record(
                    handle,
                    {
                        "tar_name": "en.tar",
                        "stage": "stage1",
                        "member": 2,
                        "written": 2,
                    },
                )
            with output.open("a", encoding="utf-8") as handle:
                append_annotation(
                    handle,
                    Annotation(tar_name="en.tar", audio_id="a.opus", asr_text="hi"),
                )
            with self.assertLogs("engine.run", level="INFO") as logs:
                show_status(root, output)
            self.assertTrue(any("member=2 written=2" in line for line in logs.output))
            self.assertTrue(any("Annotations: 1" in line for line in logs.output))
            self.assertTrue(any("('en.tar', 'a.opus')" in line for line in logs.output))

    def test_hard_kill_leftover_is_removed_on_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = root / ".en_000.tar.rewrite"
            stale.write_bytes(b"partial tar bytes")
            self.run_with_backends(pipeline_config(root), echo([]), only="stage1")
            self.assertFalse(stale.exists())

    def test_second_pipeline_instance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with pipeline_lock(root), self.assertRaises(RuntimeError):
                run_pipeline(pipeline_config(root), "stage1", 0, False)

    def test_operator_backends_cover_operator_tasks(self) -> None:
        self.assertEqual(set(OPERATOR_BACKENDS), {"uvr", "slicer"})
