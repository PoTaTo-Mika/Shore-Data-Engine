"""Tests for the config-driven serial tar pipeline."""

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

from data_process.annotation import Annotation, append_annotation, load_completed
from data_process.data_reader import AudioRecord, Operator
from data_process.state import (
    CURSOR_NAME,
    append_record,
    iter_records,
    load_cursors,
    pipeline_lock,
)
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
    show_status,
)

ASR_MODEL: Config = {
    "model": "Qwen/Qwen3-ASR-1.7B",
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
        "stage_workers": 3,
        "services": {"omni": SERVICE},
        "models": {
            "demucs": {},
            "rms_vad": {},
            "asr": {
                **ASR_MODEL,
                "command": ["sgl-omni", "serve", "--model-path", "Qwen/Qwen3-ASR-1.7B"],
            },
        },
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

    def test_repository_config_declares_all_asr_models(self) -> None:
        cfg = load_config(
            Path(__file__).resolve().parents[1] / "engine" / "configs" / "run.yaml"
        )
        models = cast(dict[str, Config], cfg["models"])
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

    def test_non_asr_endpoint_is_rejected(self) -> None:
        model: Config = {**ASR_MODEL, "endpoint": "/v1/chat/completions"}
        with self.assertRaises(ValueError):
            build_request(model, "clip.wav", b"RIFF", "http://svc")

    def test_non_text_response_format_is_rejected(self) -> None:
        model: Config = {**ASR_MODEL, "params": {"response_format": "verbose_json"}}
        with self.assertRaises(ValueError):
            build_request(model, "clip.wav", b"RIFF", "http://svc")

    def test_extract_text_returns_plain_text(self) -> None:
        self.assertEqual(extract_text({"text": " hi "}), "hi")

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
            self.assertFalse((root / "_shore_state.json").exists())
            self.assertEqual(list(root.glob("*.partial")), [])
            self.assertEqual(list(root.glob(".*.rewrite")), [])
            events = list(iter_records(root / CURSOR_NAME))
            self.assertEqual(
                {(e["stage"], e["tar_name"]) for e in events if e.get("done")},
                {("stage1", "en_000.tar"), ("stage2", "en_000.tar")},
            )
            self.assertEqual(load_cursors(root / CURSOR_NAME), {})

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

    def test_selected_asr_model_supplies_service_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus",))
            cfg = pipeline_config(root)
            models = cast(dict[str, Config], cfg["models"])
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
            models["alt"] = alt
            started: list[dict[str, Config]] = []
            bodies: list[bytes] = []

            def handler(request: httpx.Request) -> httpx.Response:
                bodies.append(request.content)
                return httpx.Response(200, json={"text": "hello"})

            def fake_start(services: dict[str, Config]) -> list[Service]:
                started.append(services)
                client = httpx.Client(transport=httpx.MockTransport(handler))
                return [Service("omni", SERVICE, client=client)]

            with patch("engine.run.start_services", side_effect=fake_start):
                run_pipeline(cfg, "stage3", 0, False, "alt")

            self.assertEqual(started[0]["omni"]["command"], alt["command"])
            self.assertIn(b"openai/whisper-large-v3", bodies[0])

    def test_non_asr_stage_starts_no_services(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = pipeline_config(Path(tmp))
            with patch("engine.run.start_services", return_value=[]) as start:
                run_pipeline(cfg, "stage1", 0, False)
            start.assert_not_called()

    def test_unknown_asr_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = pipeline_config(Path(tmp))
            with self.assertRaises(ValueError):
                run_pipeline(cfg, None, 0, True, "demucs")

    def test_each_stage_covers_all_tars_before_the_next(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus",))
            make_audio_tar(root / "en_001.tar", ("b.opus",))
            cfg = pipeline_config(root)
            events: list[str] = []

            def handler(request: httpx.Request) -> httpx.Response:
                events.append("asr_request")
                return httpx.Response(200, json={"text": "hello"})

            def fake_start(services: dict[str, Config]) -> list[Service]:
                events.append("start_services")
                client = httpx.Client(transport=httpx.MockTransport(handler))
                return [Service("omni", SERVICE, client=client)]

            def fake_build_operator(task: str, model: Config) -> Operator:
                events.append(f"build:{task}")

                def apply(record: AudioRecord) -> list[AudioRecord]:
                    events.append(f"{task}:{record.tar_name}")
                    return [record]

                return apply

            with patch("engine.run.start_services", side_effect=fake_start), patch(
                "engine.run.build_operator", side_effect=fake_build_operator
            ):
                run_pipeline(cfg, None, 0, False)

            self.assertEqual(
                events,
                [
                    "build:uvr",
                    "uvr:en_000.tar",
                    "uvr:en_001.tar",
                    "build:slicer",
                    "slicer:en_000.tar",
                    "slicer:en_001.tar",
                    "start_services",
                    "asr_request",
                    "asr_request",
                ],
            )

    def test_stage_workers_and_cursor_log_are_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus",))
            cfg = pipeline_config(root)
            with patch("engine.run.transform_tar", return_value=1) as transform, patch(
                "engine.run.build_operator", return_value=lambda record: [record]
            ):
                run_pipeline(cfg, "stage1", 0, False)
            kwargs = transform.call_args.kwargs
            self.assertEqual(kwargs["workers"], 3)
            self.assertEqual(kwargs["cursor_path"], root / CURSOR_NAME)
            self.assertNotIn("state", kwargs)

    def test_asr_progress_is_streamed_to_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_audio_tar(root / "en_000.tar", ("a.opus", "b.opus"))
            cfg = pipeline_config(root)
            output = root / "annotations.jsonl"
            transcribed: list[str] = []

            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json={"text": "hello"})

            def fake_start(services: dict[str, Config]) -> list[Service]:
                client = httpx.Client(transport=httpx.MockTransport(handler))
                return [Service("omni", SERVICE, client=client)]

            class PausedTranscriber:
                def __init__(self, service: Service, model: Config) -> None:
                    pass

                def transcribe(self, audio_id: str, data: bytes) -> str:
                    transcribed.append(audio_id)
                    if len(transcribed) > 1:
                        raise RuntimeError("paused")
                    return "hello"

            class HealthyTranscriber:
                def __init__(self, service: Service, model: Config) -> None:
                    pass

                def transcribe(self, audio_id: str, data: bytes) -> str:
                    transcribed.append(audio_id)
                    return "hello"

            with patch("engine.run.start_services", side_effect=fake_start), patch(
                "engine.run.HttpTranscriber", PausedTranscriber
            ):
                with self.assertRaises(RuntimeError):
                    run_pipeline(cfg, "stage3", 0, False)

            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["audio_id"], "a.opus")
            self.assertFalse((root / "_shore_state.json").exists())
            self.assertFalse(Path(str(output) + ".bad").exists())
            self.assertEqual(list(root.glob("*.partial")), [])

            with patch("engine.run.start_services", side_effect=fake_start), patch(
                "engine.run.HttpTranscriber", HealthyTranscriber
            ):
                run_pipeline(cfg, "stage3", 0, False)

            self.assertEqual(len(load_completed(output)), 2)
            self.assertEqual(transcribed.count("a.opus"), 1)
            self.assertEqual(transcribed.count("b.opus"), 2)

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
            self.assertTrue(
                any("('en.tar', 'a.opus')" in line for line in logs.output)
            )

    def test_hard_kill_leftover_is_removed_on_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = root / ".en_000.tar.rewrite"
            stale.write_bytes(b"partial tar bytes")
            run_pipeline(pipeline_config(root), "stage1", 0, False)
            self.assertFalse(stale.exists())

    def test_second_pipeline_instance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = pipeline_config(root)
            with pipeline_lock(root):
                with self.assertRaises(RuntimeError):
                    run_pipeline(cfg, "stage1", 0, False)
