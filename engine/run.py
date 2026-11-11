"""Config-driven serial pipeline; each stage launches its backend and sends requests."""

import argparse
import logging
import os
import re
import subprocess
from contextlib import ExitStack
from pathlib import Path
from typing import Literal, cast, get_args

import yaml

from data_process.annotation import Annotation, append_annotation, load_completed
from data_process.asr import dispatch as asr_dispatch
from data_process.data_reader import (
    Operator,
    completed_stages,
    iter_audio,
    iter_tars,
    transform_tar,
)
from data_process.protocol import OperatorBackend
from data_process.state import CURSOR_NAME, iter_records, load_cursors, pipeline_lock
from data_process.uvr.dispatch import BACKEND as UVR_BACKEND
from data_process.vad.dispatch import BACKEND as VAD_BACKEND
from engine.backends import Config, Service, launch

logger = logging.getLogger(__name__)

Task = Literal["uvr", "slicer", "asr"]

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "run.yaml"
DEFAULT_STAGE_WORKERS = 4
ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")
REQUIRED_KEYS = (
    "data_root",
    "output_name",
    "audio_extensions",
    "services",
    "models",
    "stages",
)
OPERATOR_BACKENDS: dict[Task, OperatorBackend] = {
    "uvr": UVR_BACKEND,
    "slicer": VAD_BACKEND,
}


def expand_env(value: object) -> object:
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda m: os.environ.get(m[1], m[2] or ""), value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def model_task(entry: Config) -> Task:
    """ASR entries name a served `model`; operator entries name an `operator`."""
    if "model" in entry:
        return "asr"
    for task, backend in OPERATOR_BACKENDS.items():
        if entry.get("operator") in backend.operators:
            return task
    raise ValueError(f"Unknown operator: {entry.get('operator')}")


def load_config(path: Path) -> Config:
    with path.open(encoding="utf-8") as handle:
        cfg = cast(Config, expand_env(yaml.safe_load(handle)))
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise ValueError(f"Missing config key: {key}")
    models = cast(dict[str, Config], cfg["models"])
    services = cast(dict[str, Config], cfg["services"])
    for stage in cast(list[Config], cfg["stages"]):
        model = cast(str, stage["model"])
        if model not in models or model_task(models[model]) != stage["task"]:
            raise ValueError(
                f"Stage {stage['name']} ({stage['task']}) cannot use model {model}"
            )
        if stage["service"] not in services:
            raise ValueError(f"Unknown service: {stage['service']}")
    return cfg


def engine_revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("Cannot read engine revision from %s: %s", path, exc)
        return None
    return result.stdout.strip()


def show_status(root: Path, output: Path) -> None:
    cursors = load_cursors(root / CURSOR_NAME)
    for cursor in cursors.values():
        logger.info(
            "Cursor %s %s: member=%d written=%d",
            cursor.tar_name,
            cursor.stage,
            cursor.member,
            cursor.written,
        )
    if not cursors:
        logger.info("No stage cursors in progress")
    lines = 0
    tars: set[str] = set()
    last: tuple[str, str] | None = None
    for record in iter_records(output):
        tar_name = cast(str, record["tar_name"])
        tars.add(tar_name)
        last = (tar_name, cast(str, record["audio_id"]))
        lines += 1
    logger.info("ASR tars annotated: %d", len(tars))
    logger.info("Last annotation: %s", last)
    logger.info("Annotations: %d", lines)


def service_config(task: Task, service: Config, model: Config) -> Config:
    """Service settings with the backend command for the selected model."""
    config = dict(service)
    if task == "asr":
        if "command" in model:
            config["command"] = model["command"]
    else:
        base_url = cast(str, config["base_url"])
        config["command"] = OPERATOR_BACKENDS[task].command(model, base_url)
    return config


def run_pipeline(
    cfg: Config,
    only: str | None,
    limit: int,
    dry_run: bool,
    overrides: list[str] | None = None,
) -> None:
    """Run stages in order; `overrides` swaps models into stages of the same task."""
    engine_path = cfg.get("engine_path")
    if engine_path is not None:
        path = Path(cast(str, engine_path))
        if not path.exists():
            path = Path(__file__).resolve().parents[1] / path
        revision = engine_revision(path)
        if revision:
            logger.info("Engine revision %s", revision)

    root = Path(cast(str, cfg["data_root"]))
    extensions = set(cast(list[str], cfg["audio_extensions"]))
    services = cast(dict[str, Config], cfg["services"])
    models = cast(dict[str, Config], cfg["models"])
    output = root / cast(str, cfg["output_name"])
    workers = int(cast(str | int, cfg.get("stage_workers", DEFAULT_STAGE_WORKERS)))
    if workers < 1:
        raise ValueError(f"stage_workers must be >= 1, got {workers}")
    stages = [
        dict(stage)
        for stage in cast(list[Config], cfg["stages"])
        if not only or stage["name"] == only
    ]

    for override in overrides or []:
        if override not in models:
            raise ValueError(
                f"Unknown model: {override}; available: {', '.join(sorted(models))}"
            )
        task = model_task(models[override])
        matched = [stage for stage in stages if stage["task"] == task]
        if not matched:
            raise ValueError(f"No selected {task} stage for model {override}")
        for stage in matched:
            stage["model"] = override

    if dry_run:
        for stage in stages:
            logger.info(
                "Stage %s task=%s service=%s model=%s",
                stage["name"],
                stage["task"],
                stage["service"],
                stage["model"],
            )
        for task in get_args(Task):
            keys = sorted(
                key for key, entry in models.items() if model_task(entry) == task
            )
            logger.info("Models for %s: %s", task, ", ".join(keys))
        logger.info("Dry run: %d tars, %d stage workers", len(iter_tars(root)), workers)
        return

    with pipeline_lock(root):
        # A hard kill can leave a transient tar target behind; no staging data is kept.
        for stale in root.rglob(".*.rewrite"):
            stale.unlink(missing_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        completed = load_completed(output)
        cursor_path = root / CURSOR_NAME
        annotated = 0
        for stage in stages:
            name = cast(str, stage["name"])
            task = cast(Task, stage["task"])
            model = models[cast(str, stage["model"])]
            config = service_config(task, services[cast(str, stage["service"])], model)
            tars = iter_tars(root)
            logger.info("Stage %s processing %d tars", name, len(tars))
            # The backend starts on first pending work and stops when the stage ends.
            with ExitStack() as stack:
                service: Service | None = None

                if task != "asr":
                    operator: Operator | None = None
                    for tar_path in tars:
                        if name in completed_stages(tar_path):
                            logger.info(
                                "Stage %s already done for %s", name, tar_path.name
                            )
                            continue
                        if operator is None:
                            service = stack.enter_context(
                                launch(cast(str, stage["service"]), config)
                            )
                            operator = OPERATOR_BACKENDS[task].connect(
                                service.client, service.base_url, service.retry
                            )
                        members = transform_tar(
                            tar_path,
                            operator,
                            extensions,
                            name,
                            cursor_path=cursor_path,
                            workers=workers,
                        )
                        logger.info(
                            "Stage %s rewrote %s with %d audio members",
                            name,
                            tar_path.name,
                            members,
                        )
                    continue

                with output.open("a", encoding="utf-8") as handle:
                    for tar_path in tars:
                        if limit and annotated >= limit:
                            break
                        for record in iter_audio(tar_path, extensions):
                            key = (tar_path.name, record.audio_id)
                            if key in completed:
                                continue
                            if limit and annotated >= limit:
                                break
                            if service is None:
                                service = stack.enter_context(
                                    launch(cast(str, stage["service"]), config)
                                )
                                transcribe = asr_dispatch.connect(
                                    service.client,
                                    service.base_url,
                                    model,
                                    service.retry,
                                )
                            append_annotation(
                                handle,
                                Annotation(
                                    tar_name=tar_path.name,
                                    audio_id=record.audio_id,
                                    asr_text=transcribe(record),
                                ),
                            )
                            completed.add(key)
                            annotated += 1
                        logger.info("Stage %s annotated %s", name, tar_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the serial tar-based annotation pipeline."
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", help="Run a single stage by name")
    parser.add_argument(
        "--model",
        action="append",
        help="Use this models key for the stage of its task; repeatable",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max annotations, 0 means all"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true", help="Show cursors and exit")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # httpx logs every request at INFO, which floods per-member stage logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config(args.config)
    if args.status:
        root = Path(cast(str, cfg["data_root"]))
        show_status(root, root / cast(str, cfg["output_name"]))
        return
    run_pipeline(cfg, args.stage, args.limit, args.dry_run, args.model)


if __name__ == "__main__":
    main()
