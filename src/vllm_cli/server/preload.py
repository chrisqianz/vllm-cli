#!/usr/bin/env python3
"""
Weight cache preload daemon management (vLLM >= 0.31.0 `vllm preload`).

The `vllm preload` subcommand starts a weight-cache daemon that loads model
weights from disk once and keeps them in shared memory, so later
`vllm serve --load-format ipc_cache` launches can attach to the cached
weights instead of reading from disk again.

This module manages that daemon as a subprocess, mirroring the conventions of
VLLMServer / ProxyServerProcess: new session, piped logs, log monitor thread.
"""
import logging
import os
import signal
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from ..config import ConfigManager
from .utils import get_next_available_port

logger = logging.getLogger(__name__)

# Marker printed once by vLLM when every rank of the daemon is ready
# (vllm/entrypoints/cli/preload.py, logger.info_once).
READY_MARKER = "Weight cache daemon READY"

# Dedicated flags of `vllm preload` itself (not part of EngineArgs).
WEIGHT_CACHE_FLAG_KEYS = {
    "weight_cache_socket_dir": "--weight-cache-socket-dir",
    "weight_cache_master_port": "--weight-cache-master-port",
    "weight_cache_draft_master_port": "--weight-cache-draft-master-port",
    "weight_cache_health_port": "--weight-cache-health-port",
    "weight_cache_health_host": "--weight-cache-health-host",
}

# Config keys that are vllm-cli metadata, never CLI arguments.
METADATA_KEYS = {"name", "description", "icon", "profile_environment"}

# Frontend-only args (vllm/entrypoints/launchers/cli_args.py FrontendArgs +
# make_arg_parser extras) and async-engine-only args. `vllm preload` builds its
# parser from EngineArgs.add_cli_args only, so these would be rejected.
PRELOAD_UNSUPPORTED_DESTS = {
    # FrontendArgs dataclass fields
    "host",
    "port",
    "data_parallel_supervisor_port",
    "dp_supervisor_probe_interval_s",
    "dp_supervisor_probe_timeout_s",
    "dp_supervisor_probe_failure_threshold",
    "uds",
    "uvicorn_log_level",
    "disable_uvicorn_access_log",
    "disable_access_log_for_endpoints",
    "allow_credentials",
    "allowed_origins",
    "allowed_methods",
    "allowed_headers",
    "api_key",
    "ssl_keyfile",
    "ssl_certfile",
    "ssl_ca_certs",
    "enable_ssl_refresh",
    "ssl_cert_reqs",
    "ssl_ciphers",
    "root_path",
    "middleware",
    "enable_request_id_headers",
    "disable_fastapi_docs",
    "h11_max_incomplete_event_size",
    "h11_max_header_count",
    "enable_offline_docs",
    "enable_flash_late_interaction",
    # make_arg_parser extras for `vllm serve`
    "model_tag",
    "api_server_count",
    "config",
    "grpc",
    "headless",
    # AsyncEngineArgs-only
    "enable_log_requests",
    # Own flags handled separately / not engine args
    "extra_args",
    "lora_modules",
}


def build_preload_args(config: dict[str, Any]) -> list[str]:
    """Build the `vllm preload` argument list (without the executable).

    Filters a vllm-cli config dict down to arguments accepted by
    `vllm preload` (= EngineArgs + the five --weight-cache-* flags).

    Raises:
        ValueError: if the model is missing, or load_format is ipc_cache
            (the daemon itself must load weights from disk).
    """
    schema_manager = _get_schema_manager()
    model = config.get("model")
    if isinstance(model, dict):
        model = model.get("model")
    if not model or not isinstance(model, str):
        raise ValueError("model is required to start a weight cache daemon")

    args: list[str] = ["preload", "--model", model]

    for key, value in config.items():
        if key == "model" or value is None:
            continue
        if key in METADATA_KEYS or key in WEIGHT_CACHE_FLAG_KEYS:
            continue
        if key in PRELOAD_UNSUPPORTED_DESTS:
            logger.debug(f"Ignoring non-preload argument for weight cache: {key}")
            continue
        if key == "load_format":
            if str(value).startswith("ipc_cache"):
                raise ValueError(
                    "The weight cache daemon itself must load from disk; "
                    "load_format 'ipc_cache' cannot be used here"
                )
            args.extend(["--load-format", str(value)])
            continue

        entry = schema_manager.get_argument_info(key) if schema_manager else None
        if not entry or not entry.get("cli_flag"):
            logger.debug(f"Unknown argument, skipping for weight cache: {key}")
            continue

        flag = entry["cli_flag"]
        arg_type = entry.get("type")
        if arg_type == "boolean":
            if value:
                args.append(flag)
        elif arg_type == "list":
            if isinstance(value, (list, tuple)):
                if value:
                    args.extend([flag, ",".join(str(v) for v in value)])
            else:
                args.extend([flag, str(value)])
        elif arg_type == "dict" or isinstance(value, dict):
            import json as _json

            args.extend([flag, _json.dumps(value)] if value else [])
        else:
            args.extend([flag, str(value)])

    # Dedicated --weight-cache-* flags
    for key, flag in WEIGHT_CACHE_FLAG_KEYS.items():
        value = config.get(key)
        if value is None or value == "":
            continue
        args.extend([flag, str(value)])

    return args


@lru_cache(maxsize=1)
def _get_schema_manager():
    try:
        from ..config.schemas import SchemaManager

        return SchemaManager()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Schema unavailable for weight cache arg filtering: {exc}")
        return None


class WeightCacheDaemon:
    """Manage one `vllm preload` weight-cache daemon subprocess."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        model = config.get("model", "unknown")
        if isinstance(model, dict):
            model = model.get("model", "unknown")
        self.model: str = model
        self.process: subprocess.Popen | None = None
        self.log_queue: Queue[str] = Queue()
        self.log_thread: threading.Thread | None = None
        self._recent_logs: deque = deque(maxlen=1000)
        self.start_time: datetime | None = None
        self._ready_seen = False

        socket_dir = config.get("weight_cache_socket_dir") or "default (tempdir)"
        self.socket_dir: str = str(socket_dir)
        self.health_port: int | None = config.get("weight_cache_health_port")

        log_dir = Path.home() / ".vllm-cli" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"preload_{self.model.replace('/', '_')}_{timestamp}.log"

    # ------------------------------------------------------------------ start
    def _build_command(self) -> list[str]:
        return ["vllm"] + build_preload_args(self.config)

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        try:
            config_manager = ConfigManager()
            universal_env = config_manager.config.get("universal_environment", {}) or {}
            for key, value in universal_env.items():
                env[str(key)] = str(value)
            hf_token = config_manager.config.get("hf_token")
            if hf_token:
                env["HF_TOKEN"] = hf_token
                env["HUGGING_FACE_HUB_TOKEN"] = hf_token
        except Exception as exc:  # pragma: no cover - settings are optional
            logger.debug(f"Skipping settings environment for weight cache: {exc}")
        return env

    def start(self) -> bool:
        """Start the daemon. Returns True if the process survived launch."""
        if self.is_running():
            logger.warning("Weight cache daemon is already running")
            return False
        try:
            cmd = self._build_command()
        except ValueError as exc:
            logger.error(f"Invalid weight cache configuration: {exc}")
            print(f"  ✗ {exc}")
            return False

        logger.info(f"Starting weight cache daemon: {' '.join(cmd)}")
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,
                universal_newlines=True,
                env=self._build_env(),
                start_new_session=True,
            )
        except FileNotFoundError:
            print("  ✗ 'vllm' executable not found on PATH")
            return False
        except OSError as exc:
            logger.error(f"Failed to start weight cache daemon: {exc}")
            print(f"  ✗ Failed to start: {exc}")
            return False

        self._start_log_monitor()
        self.start_time = datetime.now()
        time.sleep(1)
        if not self.is_running():
            logger.error("Weight cache daemon exited immediately after launch")
            return False
        _register(self)
        return True

    def wait_until_ready(self, timeout: float = 1800.0) -> bool | None:
        """Wait for the READY marker. True=ready, False=process died, None=timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._ready_seen:
                return True
            if self.process is not None and self.process.poll() is not None:
                return False
            time.sleep(0.5)
        return None

    # -------------------------------------------------------------------- log
    def _start_log_monitor(self) -> None:
        self.log_thread = threading.Thread(target=self._log_monitor_worker, daemon=True)
        self.log_thread.start()

    def _log_monitor_worker(self) -> None:
        try:
            with open(self.log_path, "w", buffering=1) as log_file:
                while True:
                    if self.process is not None and self.process.stdout is not None:
                        line = self.process.stdout.readline()
                        if not line:
                            break
                        log_file.write(line)
                        self._recent_logs.append(line.rstrip())
                        self.log_queue.put(line.rstrip())
                        if READY_MARKER in line:
                            self._ready_seen = True
                    else:
                        break
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Weight cache log monitor ended: {exc}")

    def get_recent_logs(self, n: int = 30) -> list[str]:
        return list(self._recent_logs)[-n:]

    def tail_logs(self, timeout: float = 0.1):
        """Yield log lines as they arrive until timeout elapses with no line."""
        while True:
            try:
                yield self.log_queue.get(timeout=timeout)
            except Empty:
                return

    # ------------------------------------------------------------------- stop
    def stop(self) -> bool:
        if not self.process:
            return False
        try:
            pgid = self.process.pid
            os.killpg(pgid, signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self.process.wait()
        except ProcessLookupError:
            pass
        except PermissionError:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        finally:
            _unregister(self)
        return True

    # ----------------------------------------------------------------- status
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def is_ready(self) -> bool:
        return self.is_running() and self._ready_seen

    def exit_code(self) -> int | None:
        return self.process.poll() if self.process is not None else None

    def uptime_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0

    def get_status(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "running": self.is_running(),
            "ready": self._ready_seen,
            "pid": self.process.pid if self.process else None,
            "socket_dir": self.socket_dir,
            "health_port": self.health_port,
            "log_path": str(self.log_path),
            "uptime": self.uptime_seconds(),
            "exit_code": self.exit_code(),
        }

    def __str__(self) -> str:
        return f"WeightCacheDaemon({self.model}, pid={self.process.pid if self.process else None})"


# --------------------------------------------------------------------- registry
_active_daemons: list[WeightCacheDaemon] = []
_registry_lock = threading.Lock()


def _register(daemon: WeightCacheDaemon) -> None:
    with _registry_lock:
        if daemon not in _active_daemons:
            _active_daemons.append(daemon)


def _unregister(daemon: WeightCacheDaemon) -> None:
    with _registry_lock:
        if daemon in _active_daemons:
            _active_daemons.remove(daemon)


def get_active_preload_daemons() -> list[WeightCacheDaemon]:
    """Return live weight cache daemons, dropping dead entries' visibility."""
    with _registry_lock:
        return [d for d in _active_daemons if d.is_running() or d.process is not None]


def get_running_preload_daemons() -> list[WeightCacheDaemon]:
    with _registry_lock:
        return [d for d in _active_daemons if d.is_running()]


def auto_pick_health_port(preferred: int = 0) -> int:
    """Pick a free port for the daemon health endpoint."""
    if preferred:
        from .utils import is_port_available

        if is_port_available(preferred):
            return preferred
    return get_next_available_port(29051)
