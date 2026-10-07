#!/usr/bin/env python3
"""Weight cache (vllm preload) menu - manage shared-memory weight cache daemons."""
import os
import shlex
import shutil
import time
from typing import Any

import inquirer  # type: ignore[import-not-found]

from ..config import ConfigManager
from ..i18n import tr
from ..server.preload import (
    WeightCacheDaemon,
    auto_pick_health_port,
    build_preload_args,
    get_running_preload_daemons,
)
from .model_manager import select_model
from .navigation import prompt_choice

# Weight loading of large models can legitimately take a long time; we stream
# logs for up to 30 minutes and then leave the daemon loading in background.
WAIT_READY_SECONDS = 1800


def _format_uptime(seconds: float) -> str:
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h {s % 3600 // 60}m {s % 60}s"
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


def _explain_weight_cache() -> None:
    """Print a short explanation of the weight cache subsystem."""
    print(f"\n{tr('preload_ui.explain_title', 'What is the weight cache?')}")
    print("-" * 60)
    print(tr(
        "preload_ui.explain_body",
        "vLLM 0.31+ can keep model weights in shared memory so restarts stop\n"
        "re-reading weights from disk:\n\n"
        "  1. Start a daemon:  vllm preload --model <model>\n"
        "     (loads weights once, prints 'Weight cache daemon READY')\n"
        "  2. Launch servers:  vllm serve <model> --load-format ipc_cache\n"
        "     (attaches to the daemon, skips disk read)\n"
        "  3. Many restarts:   second and later servers load in seconds\n"
        "     (e.g. 74GB model: ~37s instead of ~70s)\n"
        "  4. Stop the daemon last: it holds the cached weights.\n\n"
        "Requirements: vLLM >= 0.31.0 (this build: {ver}); same model path in\n"
        "serve and preload; TP size must fit between daemon startup and the\n"
        "first server. When all servers exit, GPU memory is released."
    ).format(ver=_vllm_version_hint()))
    print(tr(
        "preload_ui.explain_footnote",
        "Note: this is not a replacement for sleep mode or swapping, and LoRA\n"
        "adapter weights are not cached."
    ))


def _vllm_version_hint() -> str:
    try:
        from ..config.cli_args_sync import SUPPORTED_VLLM_VERSIONS

        return SUPPORTED_VLLM_VERSIONS[0].lstrip("v")
    except Exception:
        return "unknown"


def _watch_until_ready(daemon: WeightCacheDaemon) -> None:
    """Stream daemon logs until READY, the process dies, or we give up."""
    print(tr(
        "preload_ui.watching",
        "Streaming daemon output (Ctrl+C leaves it loading in background)..."
    ))
    print("-" * 60)
    deadline = time.monotonic() + WAIT_READY_SECONDS
    try:
        while time.monotonic() < deadline:
            for line in daemon.tail_logs(timeout=1.0):
                print(line)
            if daemon.is_ready():
                print("-" * 60)
                print(f"✓ {tr('preload_ui.ready', 'Weight cache is READY.')}")
                print(tr(
                    "preload_ui.ready_hint",
                    "Now launch this model from the menu with\n"
                    "load_format = ipc_cache (profile / custom config / import),\n"
                    "and it will attach to the cache instead of reading disk.\n"
                    "Keep this daemon running while any server uses the cache."
                ))
                return
            if not daemon.is_running():
                print("-" * 60)
                print(tr(
                    "preload_ui.died",
                    "Daemon exited with code {code}. Last log lines:"
                ).format(code=daemon.exit_code()))
                for line in daemon.get_recent_logs(15):
                    print(line)
                return
    except KeyboardInterrupt:
        print()
        print(tr(
            "preload_ui.left_running",
            "Left running in background - check it later under 'Manage daemons'."
        ))


def _tune_engine_option(config: dict) -> None:
    """Optionally set tensor_parallel_size on a manual config."""
    answer = inquirer.text(
        tr(
            "preload_ui.tp_prompt",
            "Tensor parallel size (leave empty for default)"
        ),
        default="",
    )
    answer = str(answer or "").strip()
    if not answer:
        return
    try:
        tp = int(answer)
        if tp >= 1:
            config["tensor_parallel_size"] = tp
            return
    except ValueError:
        pass
    print(tr("preload_ui.tp_invalid", "Not a positive integer - ignored."))


def _collect_config() -> dict | None:
    """Collect the daemon configuration from last config / profile / manual."""
    cm = ConfigManager()
    last = cm.get_last_config()
    profiles = cm.get_all_profiles()

    choices = []
    if last and last.get("model"):
        choices.append(("last", tr(
            "preload_ui.src_last",
            "Reuse last serve configuration ({model})", model=last.get("model")
        )))
    choices.append(("manual", tr(
        "preload_ui.src_manual", "Pick a model, minimal options"
    )))
    if profiles:
        choices.append(("profile", tr(
            "preload_ui.src_profile", "Load a saved profile"
        )))

    src = prompt_choice(
        "preload_config_source",
        tr("preload_ui.src_title", "Configuration source"),
        choices,
    )
    if src in (None, "BACK"):
        return None

    config: dict[str, Any]
    if src == "last":
        if not last:
            return None
        config = dict(last)
    elif src == "profile":
        names = sorted(profiles.keys())
        name = prompt_choice(
            "preload_profile",
            tr("preload_ui.profile_title", "Select profile"),
            [(n, n) for n in names],
        )
        if name in (None, "BACK"):
            return None
        profile = cm.get_profile(name)
        if not profile:
            print(tr("preload_ui.profile_missing", "Profile not found."))
            return None
        config = dict(profile)
    else:
        model = select_model()
        if isinstance(model, dict):
            model = model.get("model")
        if not model:
            return None
        config = {"model": model}
        _tune_engine_option(config)

    # The daemon itself must load weights from disk, never from the cache.
    if str(config.get("load_format", "")).startswith("ipc_cache"):
        config["load_format"] = "auto"
        print(tr(
            "preload_ui.ipc_cache_reset",
            "load_format 'ipc_cache' refers to a cache the daemon provides; "
            "switched to 'auto' for the daemon itself."
        ))

    if inquirer.confirm(
        tr("preload_ui.health_confirm", "Expose a /health endpoint for the daemon?"),
        default=True,
    ):
        config["weight_cache_health_port"] = auto_pick_health_port()
        config["weight_cache_health_host"] = "127.0.0.1"

    socket_dir = inquirer.text(
        tr(
            "preload_ui.socket_dir_prompt",
            "Socket directory (empty = auto temp dir; set the same value when serving)"
        ),
        default="",
    )
    socket_dir = str(socket_dir or "").strip()
    if socket_dir:
        config["weight_cache_socket_dir"] = os.path.expanduser(socket_dir)

    return config


def _start_flow() -> None:
    """Interactive flow to launch a new weight cache daemon."""
    if get_running_preload_daemons():
        print(tr(
            "preload_ui.already_running",
            "A weight cache daemon is already running - manage it first."
        ))
        return
    if not shutil.which("vllm"):
        print(tr(
            "preload_ui.err_vllm_missing",
            "The 'vllm' executable was not found on PATH. Install vLLM >= 0.31.0 "
            "in this environment first."
        ))
        return

    config = _collect_config()
    if not config:
        return

    try:
        cmd = ["vllm"] + build_preload_args(config)
    except ValueError as exc:
        print(f"✗ {exc}")
        return

    print(f"\n{tr('preload_ui.preview_title', 'Command preview')}")
    print("-" * 60)
    print(shlex.join(cmd))
    print("-" * 60)
    if not inquirer.confirm(
        tr("preload_ui.start_confirm", "Start the weight cache daemon?"),
        default=True,
    ):
        return

    daemon = WeightCacheDaemon(config)
    if not daemon.start():
        print(tr("preload_ui.start_failed", "Failed to start the daemon."))
        for line in daemon.get_recent_logs(10):
            print(line)
        return
    print(tr(
        "preload_ui.started",
        "Daemon started (pid {pid}); log file: {log}",
        pid=daemon.process.pid if daemon.process else "?",
        log=daemon.log_path,
    ))
    _watch_until_ready(daemon)


def _daemon_status_line(daemon: WeightCacheDaemon) -> str:
    status = daemon.get_status()
    state = tr("preload_ui.state_ready", "READY") if status["ready"] else tr(
        "preload_ui.state_loading", "loading"
    )
    return f"{status['model']}  [{state}]  pid {status['pid']}  {_format_uptime(status['uptime'])}"


def _show_status(daemon: WeightCacheDaemon) -> None:
    s = daemon.get_status()
    print()
    print(f"  {tr('preload_ui.f_model', 'Model')}: {s['model']}")
    print(f"  {tr('preload_ui.f_state', 'State')}: " + (
        tr("preload_ui.state_ready", "READY")
        if s["ready"] else tr("preload_ui.state_loading", "loading")
    ))
    print(f"  {tr('preload_ui.f_pid', 'PID')}: {s['pid']}")
    print(f"  {tr('preload_ui.f_socket_dir', 'Socket dir')}: {s['socket_dir']}")
    print(f"  {tr('preload_ui.f_health_port', 'Health port')}: {s['health_port'] or '-'}")
    print(f"  {tr('preload_ui.f_uptime', 'Uptime')}: {_format_uptime(s['uptime'])}")
    print(f"  {tr('preload_ui.f_log', 'Log file')}: {s['log_path']}")


def _follow_logs(daemon: WeightCacheDaemon) -> None:
    print(tr(
        "preload_ui.logs_tail_hint",
        "Last lines (Ctrl+C stops following, the daemon keeps running):"
    ))
    print("-" * 60)
    for line in daemon.get_recent_logs(30):
        print(line)
    try:
        while daemon.is_running():
            for line in daemon.tail_logs(timeout=1.0):
                print(line)
    except KeyboardInterrupt:
        print()


def _manage_flow() -> None:
    """Pick a running daemon and inspect / stop it."""
    while True:
        daemons = get_running_preload_daemons()
        if not daemons:
            print(tr("preload_ui.no_daemons", "No weight cache daemons are running."))
            return
        print()
        picked = prompt_choice(
            "preload_daemon",
            tr("preload_ui.daemon_pick_title", "Running weight cache daemons"),
            [(str(i), _daemon_status_line(d)) for i, d in enumerate(daemons)],
        )
        if picked in (None, "BACK"):
            return
        daemon = daemons[int(picked)]

        action = prompt_choice(
            "preload_daemon_action",
            _daemon_status_line(daemon),
            [
                ("status", tr("preload_ui.act_status", "Show details")),
                ("logs", tr("preload_ui.act_logs", "Follow logs")),
                ("stop", tr("preload_ui.act_stop", "Stop daemon")),
            ],
        )
        if action in (None, "BACK"):
            continue
        if action == "status":
            _show_status(daemon)
            input(tr("common.press_enter", "Press Enter to continue..."))
        elif action == "logs":
            _follow_logs(daemon)
            input(tr("common.press_enter", "Press Enter to continue..."))
        elif action == "stop":
            _stop_flow(daemon)


def _stop_flow(daemon: WeightCacheDaemon) -> None:
    """Confirm and stop one daemon, releasing its shared memory."""
    if not inquirer.confirm(
        tr(
            "preload_ui.stop_confirm",
            "Stop this daemon? Cached weights are released and any "
            "in-flight loading is lost."
        ),
        default=False,
    ):
        return
    if not daemon.stop():
        return
    print(tr(
        "preload_ui.stopped",
        "Daemon stopped; shared memory released."
    ))
    if get_running_preload_daemons():
        input(tr("common.press_enter", "Press Enter to continue..."))


def handle_weight_cache_menu(i18n_manager=None) -> str:
    """Main-menu entry for the vLLM weight cache (vllm preload)."""
    while True:
        running = get_running_preload_daemons()
        choices = [("start", tr("preload_ui.menu_start", "Start weight cache daemon"))]
        if running:
            choices.append(("manage", tr(
                "preload_ui.menu_manage", "Manage daemons ({count})", count=len(running)
            )))
        choices.append(("what_is", tr(
            "preload_ui.menu_what_is", "What is the weight cache?"
        )))

        action = prompt_choice(
            "preload_menu",
            tr("preload_ui.menu_title", "Weight Cache Preload (vllm preload)"),
            choices,
        )
        if action in (None, "BACK"):
            return "continue"

        print()
        if action == "start":
            _start_flow()
        elif action == "manage":
            _manage_flow()
        elif action == "what_is":
            _explain_weight_cache()
        input(tr("common.press_enter", "Press Enter to continue..."))
