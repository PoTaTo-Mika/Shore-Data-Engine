"""Stage backend process lifecycle."""

import logging
import os
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx

logger = logging.getLogger(__name__)

Config = dict[str, object]

HEALTH_TIMEOUT_S = 5.0
STOP_TIMEOUT_S = 30.0


@dataclass(frozen=True, kw_only=True)
class Service:
    name: str
    base_url: str
    client: httpx.Client
    retry: Config


@contextmanager
def launch(name: str, config: Config, log_path: Path) -> Iterator[Service]:
    """Start `command` if configured, wait until healthy, and stop it on exit.

    The backend runs in its own process group with output appended to `log_path`.
    """
    base_url = cast(str, config["base_url"]).rstrip("/")
    command = cast(list[str] | None, config.get("command"))
    process: subprocess.Popen[bytes] | None = None
    if command is not None:
        logger.info("Starting service %s: %s", name, " ".join(command))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    key_env = cast(str, config.get("api_key_env", ""))
    api_key = os.environ.get(key_env) if key_env else None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(
            timeout=cast(float, config["timeout_s"]), headers=headers
        ) as client:
            health_url = base_url + cast(str, config["health_path"])
            deadline = time.monotonic() + cast(float, config["startup_timeout_s"])
            while True:
                try:
                    response = client.get(health_url, timeout=HEALTH_TIMEOUT_S)
                    if response.status_code == 200:
                        break
                except httpx.TransportError:
                    pass  # Server is still booting; keep polling until the deadline.
                if process is not None and process.poll() is not None:
                    raise RuntimeError(
                        f"Service {name} exited with code {process.returncode}; "
                        f"see {log_path}"
                    )
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Service {name} not healthy at {health_url}")
                time.sleep(1)
            logger.info("Service %s is ready", name)
            yield Service(
                name=name,
                base_url=base_url,
                client=client,
                retry=cast(Config, config["retry"]),
            )
    finally:
        if process is not None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                logger.warning("Service %s ignored SIGTERM; killing it", name)
            # Sweep the group so backend workers never outlive the stage.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            logger.info("Stopped service %s", name)
