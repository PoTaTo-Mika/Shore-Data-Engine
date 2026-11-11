"""Stage backend process lifecycle."""

import logging
import os
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import httpx

logger = logging.getLogger(__name__)

Config = dict[str, object]


@dataclass(frozen=True, kw_only=True)
class Service:
    name: str
    base_url: str
    client: httpx.Client
    retry: Config


@contextmanager
def launch(name: str, config: Config) -> Iterator[Service]:
    """Start `command` if configured, wait until healthy, and stop it on exit."""
    base_url = cast(str, config["base_url"]).rstrip("/")
    command = cast(list[str] | None, config.get("command"))
    process: subprocess.Popen[bytes] | None = None
    if command is not None:
        logger.info("Starting service %s: %s", name, " ".join(command))
        process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
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
                    if client.get(health_url).status_code == 200:
                        break
                except httpx.TransportError:
                    pass  # Server is still booting; keep polling until the deadline.
                if process is not None and process.poll() is not None:
                    raise RuntimeError(
                        f"Service {name} exited with code {process.returncode}"
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
            process.terminate()
            process.wait(timeout=30)
            logger.info("Stopped service %s", name)
