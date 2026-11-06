"""Config-driven annotation pipeline routed through payload-based backends."""

import argparse
import json
import logging
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import cast

import yaml

from data_process.backends import Backend, Payload, build_backend

logger = logging.getLogger(__name__)

ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")
Config = dict[str, object]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "run.yaml"


def expand_env(value: object) -> object:
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda m: os.environ.get(m[1], m[2] or ""), value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def load_config(path: Path) -> Config:
    with path.open(encoding="utf-8") as f:
        cfg = cast(Config, expand_env(yaml.safe_load(f)))
    for key in (
        "data_root",
        "sliced_dirname",
        "output_name",
        "audio_extensions",
        "backends",
        "models",
        "pipeline",
    ):
        if key not in cfg:
            raise ValueError(f"Missing config key: {key}")
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


def iter_sliced_dirs(root: Path, name: str) -> list[Path]:
    return sorted(p for p in root.rglob(name) if p.is_dir())


def audio_files_in(directory: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def source_audio_files(
    root: Path, extensions: set[str], sliced_name: str
) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in extensions
        and sliced_name not in p.relative_to(root).parts
    )


def load_annotations(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return cast(dict[str, str], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, pairs: dict[str, str]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_payloads(stage: Config, model: Config, cfg: Config) -> list[Payload]:
    root = Path(cast(str, cfg["data_root"]))
    extensions = set(cast(list[str], cfg["audio_extensions"]))
    sliced_name = cast(str, cfg["sliced_dirname"])
    task = cast(str, model["task"])
    params = cast(dict[str, object], model.get("params", {}))
    scope = cast(str, stage["scope"])

    if scope == "dataset":
        return [
            Payload(
                task=task,
                scope=root,
                inputs=tuple(source_audio_files(root, extensions, sliced_name)),
                params=params,
            )
        ]
    if scope == "sliced":
        return [
            Payload(
                task=task,
                scope=directory,
                inputs=tuple(audio_files_in(directory, extensions)),
                params=params,
            )
            for directory in iter_sliced_dirs(root, sliced_name)
        ]
    raise ValueError(f"Unknown scope: {scope}")


def run_pipeline(cfg: Config, only: str | None, limit: int, dry_run: bool) -> None:
    engine_path = cfg.get("engine_path")
    if engine_path is not None:
        path = Path(cast(str, engine_path))
        if not path.exists():
            path = Path(__file__).resolve().parents[1] / path
        revision = engine_revision(path)
        if revision:
            logger.info("Engine revision %s", revision)

    backends = cast(dict[str, Config], cfg["backends"])
    models = cast(dict[str, Config], cfg["models"])
    output_name = cast(str, cfg["output_name"])
    failures = 0
    remaining = limit
    for stage in cast(list[Config], cfg["pipeline"]):
        name = cast(str, stage["stage"])
        if only and name != only:
            continue
        model_name = cast(str, stage["model"])
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        model = models[model_name]
        backend_name = cast(str, model["backend"])
        if backend_name not in backends:
            raise ValueError(f"Unknown backend: {backend_name}")
        backend: Backend = build_backend(backends[backend_name], model)
        payloads = build_payloads(stage, model, cfg)
        logger.info(
            "Stage %s model=%s backend=%s scope=%s files=%d",
            name,
            model_name,
            backend_name,
            cast(str, stage["scope"]),
            sum(len(p.inputs) for p in payloads),
        )
        if dry_run:
            continue

        for payload in payloads:
            resume = bool(stage.get("resume", False))
            existing: dict[str, str] = {}
            inputs = payload.inputs
            if resume:
                existing = load_annotations(payload.scope / output_name)
                inputs = tuple(
                    p for p in inputs if str(p.absolute()) not in existing
                )
                if limit:
                    if remaining <= 0:
                        break
                    inputs = inputs[:remaining]
                if not inputs:
                    continue
            result = backend.process(replace(payload, inputs=inputs))
            if result.annotations:
                existing.update(result.annotations)
                write_json(payload.scope / output_name, existing)
                if limit:
                    remaining -= len(inputs)
            failures += len(result.failures)

    if failures:
        raise RuntimeError(f"{failures} audio files failed; rerun to retry them")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the config-driven annotation pipeline."
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", help="Run a single stage by name")
    parser.add_argument(
        "--limit", type=int, default=0, help="Max annotation inputs, 0 means all"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pipeline(load_config(args.config), args.stage, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
