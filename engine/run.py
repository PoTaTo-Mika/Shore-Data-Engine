"""Config-driven serial pipeline that reads and writes audio inside tars."""

import argparse
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import cast

import yaml

from data_process.annotation import Annotation, append_annotation, load_completed
from data_process.data_reader import (
    Operator,
    completed_stages,
    iter_audio,
    iter_tars,
    transform_tar,
)
from data_process.stages import build_slicer, build_uvr
from engine.backends import (
    Config,
    HttpTranscriber,
    Transcriber,
    start_services,
    stop_services,
)

logger = logging.getLogger(__name__)

ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "run.yaml"
TASKS = frozenset({"uvr", "slicer", "asr"})


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


def build_operator(task: str, model: Config) -> Operator:
    params = cast(dict[str, object], model.get("params", {}))
    if task == "uvr":
        return build_uvr(params)
    if task == "slicer":
        return build_slicer(params)
    raise ValueError(f"Task has no local operator: {task}")


def run_pipeline(
    cfg: Config, only: str | None, limit: int, dry_run: bool
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
    stages = cast(list[Config], cfg["stages"])
    models = cast(dict[str, Config], cfg["models"])
    output = root / cast(str, cfg["output_name"])

    if dry_run:
        for stage in stages:
            logger.info(
                "Stage %s task=%s model=%s",
                stage["name"],
                stage["task"],
                stage["model"],
            )
        logger.info("Dry run: %d tars", len(list(iter_tars(root))))
        return

    services = start_services(cast(dict[str, Config], cfg["services"]))
    try:
        service_by_name = {service.name: service for service in services}
        transcribers: dict[str, Transcriber] = {}
        operators: dict[str, Operator] = {}
        completed = load_completed(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        annotated = 0

        with output.open("a", encoding="utf-8") as handle:
            for tar_path in iter_tars(root):
                if limit and annotated >= limit:
                    break
                for stage in stages:
                    name = cast(str, stage["name"])
                    task = cast(str, stage["task"])
                    if only and name != only:
                        continue
                    model = models[cast(str, stage["model"])]

                    if task == "asr":
                        if name not in transcribers:
                            service_name = cast(str, stage["service"])
                            if service_name not in service_by_name:
                                raise ValueError(f"Unknown service: {service_name}")
                            transcribers[name] = HttpTranscriber(
                                service_by_name[service_name], model
                            )
                        for record in iter_audio(tar_path, extensions):
                            key = (tar_path.name, record.audio_id)
                            if key in completed:
                                continue
                            if limit and annotated >= limit:
                                break
                            text = transcribers[name].transcribe(
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
                        continue

                    if name in completed_stages(tar_path):
                        logger.info("Stage %s already done for %s", name, tar_path.name)
                        continue
                    if name not in operators:
                        operators[name] = build_operator(task, model)
                    members = transform_tar(
                        tar_path, operators[name], extensions, name
                    )
                    logger.info(
                        "Stage %s rewrote %s with %d audio members",
                        name,
                        tar_path.name,
                        members,
                    )
    finally:
        stop_services(services)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the serial tar-based annotation pipeline."
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", help="Run a single stage by name")
    parser.add_argument(
        "--limit", type=int, default=0, help="Max annotations, 0 means all"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pipeline(load_config(args.config), args.stage, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
