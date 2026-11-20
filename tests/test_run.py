"""Tests for the config-driven pipeline and its stage backends."""

import io
import json
import os
import socket
import sys
import tarfile
import tempfile
import time
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

from data_process.data_reader import MANIFEST_NAME, AudioRecord
from data_process.dispatch import OPERATORS, PREPROCESS, asr_request
from data_process.protocol import post
from data_process.state import (
    CURSOR_NAME,
    Annotation,
    annotation_path,
    append_annotation,
    append_record,
    iter_records,
    load_cursors,
    load_done_names,
    pipeline_lock,
)
from engine.backends import Config, Service, launch
from engine.run import (
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


def make_audio_tar(
    path: Path, names: tuple[str, ...], stages: tuple[str, ...] = ()
) -> None:
    with tarfile.open(path, "w") as tar:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = 4
            tar.addfile(info, io.BytesIO(b"RIFF"))
        if stages:
            payload = json.dumps({"stages": list(stages)}).encode("utf-8")
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def health_server(port: int) -> str:
    return (
        "import http.server as h\n"
        "class H(h.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers()\n"
        f"h.HTTPServer(('127.0.0.1', {port}), H).serve_forever()\n"
    )


def pipeline_config(root: Path) -> Config:
    return {
        "data_root": str(root),
        "audio_extensions": [".opus"],
        "stage_workers": 3,
        "services": {
            "preprocess": service("http://preprocess:1"),
            "omni": service("http://omni:3"),
        },
        "models": {
            "clean": {"operators": [{"operator": "demucs"}, {"operator": "rms_vad"}]},
            "alt_clean": {
                "operators": [
                    {"operator": "mdx_net", "params": {"stem": "vocals"}},
                    {"operator": "silero_vad"},
                ]
            },
            "asr": {
                **ASR_MODEL,
                "command": ["sgl-omni", "serve", "--model-path", "Qwen/Qwen3-ASR-1.7B"],
            },
        },
        "stages": [
            {
                "name": "stage1",
                "task": "preprocess",
                "service": "preprocess",
                "model": "clean",
            },
            {"name": "stage2", "task": "asr", "service": "omni", "model": "asr"},
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
) -> Callable[[str, Config, Path], Iterator[Service]]:
    @contextmanager
    def launch_stub(name: str, config: Config, log_path: Path) -> Iterator[Service]:
        assert log_path.parent.name == "_shore_logs"
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
        self.assertEqual([s["name"] for s in stages], ["stage1", "stage2"])
        self.assertEqual([s["task"] for s in stages], ["preprocess", "asr"])
        self.assertEqual([s["service"] for s in stages], ["preprocess", "omni"])

    def test_repository_config_chains_clean_then_slice(self) -> None:
        models = cast(dict[str, Config], load_config(REPO_CONFIG)["models"])
        chains = {
            key: [
                cast(str, step["operator"])
                for step in cast(list[Config], entry["operators"])
            ]
            for key, entry in models.items()
            if "operators" in entry
        }
        self.assertEqual(chains["demucs_rms"], ["demucs", "rms_vad"])
        for key, operators in chains.items():
            with self.subTest(model=key):
                self.assertEqual(model_task(models[key]), "preprocess")
                self.assertTrue(set(operators) <= set(OPERATORS))
                self.assertIn(operators[0], {"demucs", "mdx_net"})
                self.assertIn(operators[-1], {"rms_vad", "silero_vad", "ten_vad"})

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
            cast(list[Config], cfg["stages"])[0]["model"] = "asr"
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
        path = Path(__file__).resolve().parents[1] / "third-party" / "sglang-omni"
        revision = engine_revision(path)
        if (path / ".git").exists():
            self.assertEqual(len(cast(str, revision)), 40)
        else:
            self.assertIsNone(revision)

    def test_operator_service_command_runs_stage_dispatch(self) -> None:
        model: Config = {
            "operators": [
                {"operator": "mdx_net", "params": {"stem": "vocals"}},
                {"operator": "rms_vad"},
            ]
        }
        config = service_config("preprocess", service("http://127.0.0.1:8001"), model)
        command = cast(list[str], config["command"])
        self.assertEqual(command[1:3], ["-m", "data_process.backend"])
        self.assertEqual(
            json.loads(command[command.index("--steps") + 1]), model["operators"]
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
        request = asr_request(self.client, "http://svc", ASR_MODEL, self.record)
        self.assertEqual(str(request.url), "http://svc/v1/audio/transcriptions")
        body = request.read()
        self.assertIn(b'name="language"\r\n\r\nzh', body)
        self.assertIn(b'filename="clip.opus"', body)
        self.assertIn(b"RIFF", body)

    def test_unmapped_field_is_rejected(self) -> None:
        model = {**ASR_MODEL, "params": {"language": "zh", "temperature": 0}}
        with self.assertRaises(ValueError):
            asr_request(self.client, "http://svc", model, self.record)

    def test_non_asr_endpoint_is_rejected(self) -> None:
        model: Config = {**ASR_MODEL, "endpoint": "/v1/chat/completions"}
        with self.assertRaises(ValueError):
            asr_request(self.client, "http://svc", model, self.record)

    def test_non_text_response_format_is_rejected(self) -> None:
        model: Config = {**ASR_MODEL, "params": {"response_format": "verbose_json"}}
        with self.assertRaises(ValueError):
            asr_request(self.client, "http://svc", model, self.record)

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
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.log_path = Path(self.tmp.name) / "logs" / "service.log"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def assert_port_closes(self, port: int) -> None:
        deadline = time.monotonic() + 5
        while True:
            try:
                httpx.get(f"http://127.0.0.1:{port}/health")
            except httpx.ConnectError:
                return
            self.assertLess(time.monotonic(), deadline, "backend outlived the stage")
            time.sleep(0.1)

    def test_launch_waits_for_health_and_stops_process(self) -> None:
        port = free_port()
        config = {
            **service(f"http://127.0.0.1:{port}"),
            "command": [sys.executable, "-c", health_server(port)],
        }
        with launch("stub", config, self.log_path) as svc:
            self.assertEqual(svc.client.get(svc.base_url + "/health").status_code, 200)
        self.assert_port_closes(port)

    def test_launch_stops_backend_children(self) -> None:
        port = free_port()
        parent = (
            "import subprocess, sys\n"
            f"subprocess.Popen([sys.executable, '-c', {health_server(port)!r}]).wait()\n"
        )
        config = {
            **service(f"http://127.0.0.1:{port}"),
            "command": [sys.executable, "-c", parent],
        }
        with launch("stub", config, self.log_path) as svc:
            self.assertEqual(svc.client.get(svc.base_url + "/health").status_code, 200)
        self.assert_port_closes(port)

    def test_launch_fails_fast_and_keeps_backend_log(self) -> None:
        script = "import sys; sys.stderr.write('cuda oom\\n'); raise SystemExit(3)"
        config = {
            **service(f"http://127.0.0.1:{free_port()}"),
            "command": [sys.executable, "-c", script],
        }
        with (
            self.assertRaisesRegex(RuntimeError, "exited with code 3"),
            launch("dead", config, self.log_path),
        ):
            pass
        self.assertIn("cuda oom", self.log_path.read_text(encoding="utf-8"))

    def two_tone_wav(self) -> bytes:
        tone = np.sin(2 * np.pi * 440 * np.arange(3 * 44100) / 44100) * 0.5
        buffer = io.BytesIO()
        sf.write(
            buffer, np.concatenate([tone, np.zeros(44100), tone]), 44100, format="WAV"
        )
        return buffer.getvalue()

    def test_operator_backend_roundtrip_over_http(self) -> None:
        port = free_port()
        model: Config = {
            "operators": [{"operator": "rms_vad", "params": {"min_length": 1000}}]
        }
        config = service_config(
            "preprocess", service(f"http://127.0.0.1:{port}"), model
        )
        record = AudioRecord(
            tar_name="en.tar", audio_id="a/b.wav", data=self.two_tone_wav()
        )
        with launch("preprocess", config, self.log_path) as svc:
            operator = PREPROCESS.connect(svc.client, svc.base_url, svc.retry)
            outputs = operator(record)
            with self.assertRaises(httpx.HTTPStatusError):
                operator(replace(record, data=b"not audio"))
        self.assertEqual(
            [r.audio_id for r in outputs], ["a/b_000.opus", "a/b_001.opus"]
        )
        self.assertTrue(
            all(r.tar_name == "en.tar" and r.data[:4] == b"OggS" for r in outputs)
        )

    def test_operator_chain_runs_steps_in_order(self) -> None:
        port = free_port()
        model: Config = {
            "operators": [
                {"operator": "rms_vad", "params": {"min_length": 1000}},
                {"operator": "rms_vad", "params": {"min_length": 1000}},
            ]
        }
        config = service_config(
            "preprocess", service(f"http://127.0.0.1:{port}"), model
        )
        record = AudioRecord(
            tar_name="en.tar", audio_id="a/b.wav", data=self.two_tone_wav()
        )
        with launch("preprocess", config, self.log_path) as svc:
            outputs = PREPROCESS.connect(svc.client, svc.base_url, svc.retry)(record)
        # First slicer splits into two clips; the second slices each clip again.
        self.assertEqual(
            [r.audio_id for r in outputs],
            ["a/b_000_000.opus", "a/b_001_000.opus"],
        )
        self.assertTrue(all(r.data[:4] == b"OggS" for r in outputs))


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
                ["preprocess", "stop:preprocess", "omni", "stop:omni"],
            )
            self.assertEqual(
                events,
                [
                    "preprocess:en_000.tar:a.opus",
                    "preprocess:en_001.tar:b.opus",
                    "omni:asr",
                    "omni:asr",
                ],
            )
            clean_command = cast(list[str], started[0][1]["command"])
            self.assertIn("data_process.backend", clean_command)
            self.assertEqual(
                json.loads(clean_command[clean_command.index("--steps") + 1]),
                [{"operator": "demucs"}, {"operator": "rms_vad"}],
            )
            self.assertEqual(
                started[2][1]["command"],
                ["sgl-omni", "serve", "--model-path", "Qwen/Qwen3-ASR-1.7B"],
            )

    def test_stages_resume_without_restarting_backends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("album/a.opus", "album/b.opus"))
            cfg = pipeline_config(root)
            events: list[str] = []
            self.run_with_backends(cfg, echo(events))

            output = root / "en_000.jsonl"
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {r["name"] for r in records}, {"album/a.opus", "album/b.opus"}
            )
            self.assertTrue(all(r["asr_text"] == "hello" for r in records))
            self.assertEqual(list(root.glob(".*.rewrite")), [])
            cursor_events = list(iter_records(root / CURSOR_NAME))
            self.assertEqual(
                {(e["stage"], e["tar_name"]) for e in cursor_events if e.get("done")},
                {("stage1", "en_000.tar")},
            )
            self.assertEqual(load_cursors(root / CURSOR_NAME), {})

            events.clear()
            self.assertEqual(self.run_with_backends(cfg, echo(events)), [])
            self.assertEqual(events, [])

    def test_stage_filter_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus", "b.opus"), ("stage1",))
            events: list[str] = []
            started = self.run_with_backends(
                pipeline_config(root), echo(events), only="stage2", limit=1
            )
            self.assertEqual([name for name, _ in started], ["omni", "stop:omni"])
            lines = (
                (root / "en_000.jsonl").read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(len(lines), 1)

    def test_dry_run_starts_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus",))
            with patch("engine.run.launch") as start:
                run_pipeline(pipeline_config(root), None, 0, True)
            start.assert_not_called()
            self.assertFalse((root / "en_000.jsonl").exists())

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

            started = self.run_with_backends(
                cfg, handler, overrides=["alt_clean", "alt"]
            )
            clean_command = cast(list[str], started[0][1]["command"])
            steps = json.loads(clean_command[clean_command.index("--steps") + 1])
            self.assertEqual(steps[0]["operator"], "mdx_net")
            self.assertEqual(started[2][1]["command"], alt["command"])
            self.assertIn(b"openai/whisper-large-v3", bodies[-1])

    def test_unknown_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            run_pipeline(pipeline_config(Path(tmp)), None, 0, True, ["missing"])

    def test_override_without_matching_stage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            run_pipeline(pipeline_config(Path(tmp)), "stage2", 0, True, ["clean"])

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

    def test_asr_skips_tars_with_pending_earlier_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus",))
            make_audio_tar(root / "en_001.tar", ("b.opus",), ("stage1",))
            events: list[str] = []
            with self.assertLogs("engine.run", level="WARNING") as logs:
                self.run_with_backends(
                    pipeline_config(root), echo(events), only="stage2"
                )
            self.assertTrue(any("en_000.tar" in line for line in logs.output))
            self.assertFalse((root / "en_000.jsonl").exists())
            self.assertEqual(load_done_names(root / "en_001.jsonl"), {"b.opus"})

    def test_asr_progress_is_streamed_to_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus", "b.opus"), ("stage1",))
            cfg = pipeline_config(root)
            output = root / "en_000.jsonl"
            transcribed: list[str] = []

            def paused(name: str, request: httpx.Request) -> httpx.Response:
                transcribed.append(name)
                if len(transcribed) > 1:
                    return httpx.Response(400, text="paused")
                return httpx.Response(200, json={"text": "hello"})

            with self.assertRaises(httpx.HTTPStatusError):
                self.run_with_backends(cfg, paused, only="stage2")
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line)["name"] for line in lines], ["a.opus"])

            events: list[str] = []
            self.run_with_backends(cfg, echo(events), only="stage2")
            self.assertEqual(len(load_done_names(output)), 2)
            self.assertEqual(events, ["omni:asr"])

    def test_status_streams_cursors_and_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en.tar", ("a.opus",))
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
            with annotation_path(root / "en.tar").open("a", encoding="utf-8") as handle:
                append_annotation(handle, Annotation(name="a.opus", asr_text="hi"))
            with self.assertLogs("engine.run", level="INFO") as logs:
                show_status(root)
            self.assertTrue(any("member=2 written=2" in line for line in logs.output))
            self.assertTrue(any("Annotations: 1" in line for line in logs.output))
            self.assertTrue(any("('en.tar', 'a.opus')" in line for line in logs.output))

    def test_status_leaves_in_flight_append_intact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en.tar", ("a.opus",))
            output = annotation_path(root / "en.tar")
            content = (
                '{"name": "a.opus", "asr_text": "hi"}\n'
                '{"name": "b.op'
            )
            output.write_text(content, encoding="utf-8")
            with self.assertLogs("engine.run", level="INFO") as logs:
                show_status(root)
            self.assertTrue(any("Annotations: 1" in line for line in logs.output))
            self.assertEqual(output.read_text(encoding="utf-8"), content)

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

    def test_catalog_covers_chainable_operators(self) -> None:
        self.assertEqual(
            set(PREPROCESS.catalog),
            {"demucs", "mdx_net", "rms_vad", "silero_vad", "ten_vad"},
        )
        self.assertEqual(set(OPERATORS), set(PREPROCESS.catalog))
