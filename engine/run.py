"""Config-driven serial pipeline that reads and writes audio inside tars."""

import argparse
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Literal, cast, get_args

import yaml

from data_process.annotation import Annotation, append_annotation, load_completed
from data_process.data_reader import (
    Operator,
    completed_stages,
    iter_audio,
    iter_tars,
    transform_tar,
)
from data_process.state import (
    CURSOR_NAME,
    iter_records,
    load_cursors,
    pipeline_lock,
)
from data_process.stages import build_slicer, build_uvr
from engine.backends import (
    Config,
    HttpTranscriber,
    Service,
    start_services,
    stop_services,
)

logger = logging.getLogger(__name__)

Task = Literal["uvr", "slicer", "asr"]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "run.yaml"
ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")
TASKS = frozenset(get_args(Task))


def expand_env(value: object) -> object:
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda m: os.environ.get(m[1], m[2] or ""), value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def load_config(path: Path) -> Config:
    with path.open(encoding="utf-8") as handle:
        cfg = cast(Config, expand_env(yaml.safe_load(handle)))
    for key in (
        "data_root",
        "output_name",
        "audio_extensions",
        "services",
        "models",
        "stages",
    ):
        if key not in cfg:
            raise ValueError(f"Missing config key: {key}")
    for stage in cast(list[Config], cfg["stages"]):
        if cast(str, stage["task"]) not in TASKS:
            raise ValueError(f"Unknown stage task: {stage['task']}")
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
    if cursors:
        for cursor in cursors.values():
            logger.info(
                "Cursor %s %s: member=%d written=%d",
                cursor.tar_name,
                cursor.stage,
                cursor.member,
                cursor.written,
            )
    else:
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


def build_operator(task: Task, model: Config) -> Operator:
    params = cast(dict[str, object], model.get("params", {}))
    if task == "uvr":
        return build_uvr(params)
    if task == "slicer":
        return build_slicer(params)
    raise ValueError(f"Task has no local operator: {task}")


def run_pipeline(
    cfg: Config,
    only: str | None,
    limit: int,
    dry_run: bool,
    model: str | None = None,
) -> None:
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
    all_services = cast(dict[str, Config], cfg["services"])
    models = cast(dict[str, Config], cfg["models"])
    output = root / cast(str, cfg["output_name"])
    workers = int(cast(str | int, cfg.get("stage_workers", 4)))
    if workers < 1:
        raise ValueError(f"stage_workers must be >= 1, got {workers}")
    stages = [
        dict(stage)
        for stage in cast(list[Config], cfg["stages"])
        if not only or stage["name"] == only
    ]
    asr_keys = sorted(key for key, entry in models.items() if "model" in entry)

    if model is not None:
        if model not in asr_keys:
            raise ValueError(
                f"Unknown ASR model: {model}; available: {', '.join(asr_keys)}"
            )
        for stage in stages:
            if stage["task"] == "asr":
                stage["model"] = model

    if dry_run:
        for stage in stages:
            logger.info(
                "Stage %s task=%s model=%s",
                stage["name"],
                stage["task"],
                stage["model"],
            )
        logger.info("ASR models: %s", ", ".join(asr_keys))
        logger.info(
            "Dry run: %d tars, %d stage workers", len(list(iter_tars(root))), workers
        )
        return

    services_config: dict[str, Config] = {}
    for stage in stages:
        if stage["task"] != "asr":
            continue
        service_name = cast(str, stage["service"])
        entry = models[cast(str, stage["model"])]
        if "model" not in entry:
            raise ValueError(f"Not an ASR model: {stage['model']}")
        config = dict(all_services[service_name])
        if "command" in entry:
            config["command"] = entry["command"]
        services_config[service_name] = config

    with pipeline_lock(root):
        # A hard kill can leave a transient tar target behind; no staging data is kept.
        for stale in root.rglob(".*.rewrite"):
            stale.unlink(missing_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        completed = load_completed(output)
        cursor_path = root / CURSOR_NAME
        services: list[Service] | None = None
        annotated = 0
        try:
            for stage in stages:
                name = cast(str, stage["name"])
                task = cast(Task, stage["task"])
                model_config = models[cast(str, stage["model"])]
                tars = list(iter_tars(root))
                logger.info("Stage %s processing %d tars", name, len(tars))

                if task != "asr":
                    operator: Operator | None = None
                    for tar_path in tars:
                        if name in completed_stages(tar_path):
                            logger.info(
                                "Stage %s already done for %s", name, tar_path.name
                            )
                            continue
                        if operator is None:
                            operator = build_operator(task, model_config)
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

                if services is None:
                    services = start_services(services_config)
                service_by_name = {service.name: service for service in services}
                service_name = cast(str, stage["service"])
                if service_name not in service_by_name:
                    raise ValueError(f"Unknown service: {service_name}")
                transcriber = HttpTranscriber(
                    service_by_name[service_name], model_config
                )

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
                            text = transcriber.transcribe(
                                record.audio_id, record.data
                            )
                            append_annotation(
                                handle,
                                Annotation(
                                    tar_name=tar_path.name,
                                    audio_id=record.audio_id,
                                    asr_text=text,
                                ),
                            )
                            completed.add(key)
                            annotated += 1
                        logger.info("Stage %s annotated %s", name, tar_path.name)
        finally:
            stop_services(services or [])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the serial tar-based annotation pipeline."
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", help="Run a single stage by name")
    parser.add_argument(
        "--model", help="Override the ASR model for stage3 (key in models)"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max annotations, 0 means all"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true", help="Show cursors and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.status:
        root = Path(cast(str, cfg["data_root"]))
        show_status(root, root / cast(str, cfg["output_name"]))
        return
    run_pipeline(
        cfg, args.stage, args.limit, args.dry_run, args.model
    )


if __name__ == "__main__":
    main()
