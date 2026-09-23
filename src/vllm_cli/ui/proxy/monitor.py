#!/usr/bin/env python3
"""
Proxy monitoring UI module for vLLM CLI.

Handles UI display for monitoring proxy servers and model engines.
"""
import logging
import re
import threading
import time
from threading import Thread
from typing import TYPE_CHECKING, Dict, List, Optional

from rich import box
from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ...config import ConfigManager
from ...i18n import tr
from ...system import get_gpu_info
from ..common import console, create_panel
from ..gpu_utils import calculate_gpu_panel_size, create_gpu_status_panel
from ..navigation import prompt_choice
from .components import create_model_registry_table

if TYPE_CHECKING:
    from ...proxy.manager import ProxyManager
    from ...server import VLLMServer

logger = logging.getLogger(__name__)

# Color palette for distinct model names
MODEL_COLORS = ["cyan", "magenta", "green", "yellow", "blue", "red"]


def parse_sleep_wake_logs(log_line: str) -> Optional[Dict]:
    """
    Parse sleep/wake completion logs for key information.

    Args:
        log_line: Single log line to parse

    Returns:
        Dict with parsed information or None if no pattern matched
    """
    # Check for sleep start
    # Pattern: "POST /sleep?level=1"
    if "POST /sleep" in log_line:
        return {"type": "sleep_start", "message": "Sleep operation initiated"}

    # Check for wake start
    # Pattern: "wake up the engine with tags: None"
    if "wake up the engine" in log_line:
        return {"type": "wake_start", "message": "Wake operation initiated"}

    # Check for sleep completion
    # Pattern: "It took 17.893898 seconds to fall asleep."
    sleep_match = re.search(r"It took ([\d.]+) seconds to fall asleep", log_line)
    if sleep_match:
        return {
            "type": "sleep_complete",
            "time": float(sleep_match.group(1)),
            "message": f"Sleep completed in {sleep_match.group(1)} seconds",
        }

    # Check for wake completion
    # Pattern: "It took 24.368552 seconds to wake up tags {'kv_cache', 'weights'}."
    wake_match = re.search(r"It took ([\d.]+) seconds to wake up", log_line)
    if wake_match:
        tags = None
        tags_match = re.search(r"tags ({[^}]+})", log_line)
        if tags_match:
            tags = tags_match.group(1)
        return {
            "type": "wake_complete",
            "time": float(wake_match.group(1)),
            "tags": tags,
            "message": f"Wake completed in {wake_match.group(1)} seconds",
        }

    # Check for memory info during sleep
    # Pattern: "Sleep mode freed 90.40 GiB memory, 3.30 GiB memory is still in use."
    memory_match = re.search(
        r"Sleep mode freed ([\d.]+) GiB memory, ([\d.]+) GiB memory is still in use",
        log_line,
    )
    if memory_match:
        return {
            "type": "memory_info",
            "freed": float(memory_match.group(1)),
            "remaining": float(memory_match.group(2)),
            "message": f"Freed {memory_match.group(1)} GiB, {memory_match.group(2)} GiB still in use",
        }

    return None


def create_models_log_panel(
    proxy_manager: "ProxyManager",
    startup_status: dict = None,
    total_lines: int = None,
    show_status: bool = True,
    available_height: int = None,
    filter_models: set = None,
) -> list:
    """
    Create consistent model logs panel for monitoring.

    Args:
        proxy_manager: The ProxyManager instance
        startup_status: Optional dict with startup status for each model
        total_lines: Total number of log lines to display across all models
        show_status: Whether to show status indicators
        available_height: Available terminal height for dynamic calculation
        filter_models: Optional set of model names to display (filters active_models)

    Returns:
        List of Rich Text objects for display
    """
    models_content = []

    # Get active models
    active_models = [m for m in proxy_manager.proxy_config.models if m.enabled]

    # Filter to specific models if requested
    if filter_models is not None:
        active_models = [m for m in active_models if m.name in filter_models]

    num_models = len(active_models)

    # Calculate lines_per_model dynamically based on available space
    if available_height and num_models > 0:
        # Reserve space for separators and headers
        # Each model after first needs 2 lines for separator
        separator_lines = (num_models - 1) * 2 if num_models > 1 else 0
        # Each model needs 1 line for header
        header_lines = num_models

        # Calculate truly available space for logs
        overhead = separator_lines + header_lines
        available_for_logs = max(0, available_height - overhead)

        if available_for_logs > 0:
            # Distribute available space equally among all models
            # No arbitrary caps - use what's available
            lines_per_model = available_for_logs // num_models

            # Ensure at least 1 line per model if we have any space
            lines_per_model = max(1, lines_per_model)
        else:
            # No space for logs, just show headers
            lines_per_model = 0
    elif total_lines is not None:
        # Use provided total_lines
        lines_per_model = total_lines // num_models if num_models > 0 else total_lines
    else:
        # No height info provided, use conservative fallback
        lines_per_model = 50 // num_models if num_models > 0 else 50

    for idx, model_config in enumerate(active_models):
        model_name = model_config.name
        model_color = MODEL_COLORS[idx % len(MODEL_COLORS)]

        # Add separator if not first model
        if idx > 0:
            models_content.append(Text())  # Empty line
            models_content.append(Rule("─" * 60, style="dim"))

        # Create model header with distinct color
        header = Text()
        header.append(f"[{model_name}]", style=f"bold {model_color}")

        # Add status if provided and requested
        if show_status and startup_status:
            status = startup_status.get(model_name, "unknown")
            header.append(" - ")

            if status == "ready":
                header.append(
                    tr("proxy_monitor.status_ready", "✓ Ready"), style="green"
                )
            elif status == "failed":
                header.append(
                    tr("proxy_monitor.status_failed", "✗ Failed"), style="red"
                )
            elif status == "starting":
                header.append(
                    tr("proxy_monitor.status_starting", "⠋ Starting..."),
                    style="yellow",
                )
            elif status == "pending":
                header.append(
                    tr("proxy_monitor.status_pending", "Pending"), style="dim"
                )
            else:
                # For overview mode, check if server is running
                if model_name in proxy_manager.vllm_servers:
                    server = proxy_manager.vllm_servers[model_name]
                    if server.is_running():
                        header.append(
                            tr("proxy_monitor.status_running_dot", "● Running"),
                            style="green",
                        )
                    else:
                        header.append(
                            tr("proxy_monitor.status_stopped_dot", "○ Stopped"),
                            style="red",
                        )

        # Add port and GPU info
        header.append(
            tr(
                "proxy_monitor.port_fragment",
                " (Port: {port}",
                port=model_config.port,
            ),
            style="dim",
        )
        if model_config.gpu_ids:
            gpu_str = ",".join(str(g) for g in model_config.gpu_ids)
            header.append(
                tr("proxy_monitor.gpu_fragment", ", GPU: {gpu}", gpu=gpu_str),
                style="dim",
            )
        header.append(")", style="dim")

        models_content.append(header)

        # Get and display logs
        if model_name in proxy_manager.vllm_servers:
            server = proxy_manager.vllm_servers[model_name]
            recent_logs = server.get_recent_logs(lines_per_model)

            if recent_logs:
                for log in recent_logs:
                    # Rich will automatically wrap long lines
                    log_text = Text(f"  {log}", style="dim white")
                    models_content.append(log_text)
            else:
                waiting_text = Text(
                    f"  {tr('proxy_monitor.waiting_for_logs', 'Waiting for logs...')}",
                    style="dim",
                )
                models_content.append(waiting_text)
        else:
            pending_text = Text(
                "  "
                + tr(
                    "proxy_monitor.server_not_started",
                    "Server not started yet...",
                ),
                style="dim",
            )
            models_content.append(pending_text)

    return models_content


def monitor_startup_progress(proxy_manager: "ProxyManager") -> bool:
    """
    Monitor the startup progress of all models with live log display.

    Shows real-time logs from all models as they start up concurrently,
    and registers them with the proxy as they become ready.

    Args:
        proxy_manager: The ProxyManager instance

    Returns:
        True if all models started successfully, False otherwise
    """
    console.print(
        tr(
            "proxy_monitor.startup_monitor_title",
            "\n[bold cyan]Model Startup Monitor[/bold cyan]",
        )
    )
    console.print(
        tr(
            "proxy_monitor.startup_monitor_subtitle",
            "[dim]Showing real-time logs from all models...[/dim]\n",
        )
    )

    # Get UI preferences
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()
    monitor_refresh_rate = ui_prefs.get("monitor_refresh_rate", 1.0)

    # Track startup status for each model
    startup_status = {}
    startup_complete = threading.Event()

    def check_startup_completion():
        """Background thread to check startup completion and register models."""
        models_to_register = []

        while not startup_complete.is_set():
            all_ready = True

            for model_config in proxy_manager.proxy_config.models:
                if not model_config.enabled:
                    continue

                model_name = model_config.name

                # Skip if already processed
                if model_name in startup_status and startup_status[model_name] in [
                    "ready",
                    "failed",
                ]:
                    continue

                # Check if server exists
                if model_name not in proxy_manager.vllm_servers:
                    startup_status[model_name] = "pending"
                    all_ready = False
                    continue

                server = proxy_manager.vllm_servers[model_name]

                # Check if server is running
                if not server.is_running():
                    startup_status[model_name] = "failed"
                    logger.error(f"Server {model_name} failed to start")
                    continue

                # Check logs for startup completion
                recent_logs = server.get_recent_logs(20)
                if recent_logs:
                    for log in recent_logs:
                        if "application startup complete" in log.lower():
                            if model_name not in models_to_register:
                                startup_status[model_name] = "ready"
                                models_to_register.append(model_config)
                                logger.info(f"Model {model_name} is ready")
                            break
                    else:
                        startup_status[model_name] = "starting"
                        all_ready = False
                else:
                    startup_status[model_name] = "starting"
                    all_ready = False

            # Register ready models with proxy
            for model_config in models_to_register[:]:
                if (
                    proxy_manager.proxy_process
                    and proxy_manager.proxy_process.is_running()
                ):
                    # Register the model
                    model_data = {
                        "name": model_config.name,
                        "port": model_config.port,
                        "model_path": model_config.model_path,
                        "gpu_ids": model_config.gpu_ids,
                        **model_config.config_overrides,
                    }

                    if proxy_manager._proxy_api_request(
                        "POST", "/proxy/add_model", model_data
                    ):
                        logger.debug(
                            f"Registered model '{model_config.name}' with proxy"
                        )

                        # Also register aliases
                        aliases = model_config.config_overrides.get("aliases", [])
                        for alias in aliases:
                            alias_data = model_data.copy()
                            alias_data["name"] = alias
                            proxy_manager._proxy_api_request(
                                "POST", "/proxy/add_model", alias_data
                            )

                    models_to_register.remove(model_config)

            if all_ready:
                startup_complete.set()
                break

            time.sleep(1)

    # Start background thread to check completion
    check_thread = Thread(target=check_startup_completion, daemon=True)
    check_thread.start()

    # Get GPU info for layout
    gpu_info = get_gpu_info()
    gpu_panel_size = calculate_gpu_panel_size(len(gpu_info) if gpu_info else 0)

    # Create layout for monitoring with GPU panel
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="status", size=4),
        Layout(name="gpu", size=gpu_panel_size),
        Layout(name="divider", size=1),
        Layout(name="models"),
        Layout(name="footer", size=1),
    )

    # Header
    layout["header"].update(
        create_panel(
            tr(
                "proxy_monitor.starting_model_servers",
                "[bold cyan]Starting Model Servers...[/bold cyan]",
            ),
            title=tr("proxy_monitor.title_proxy_startup", "Proxy Startup"),
            border_style="cyan",
        )
    )

    # Initial GPU panel
    layout["gpu"].update(create_gpu_status_panel())

    # Divider
    layout["divider"].update(
        Rule(tr("proxy_monitor.rule_model_logs", "Model Logs"), style="cyan")
    )

    # Footer
    layout["footer"].update(
        Align.center(
            Text(
                tr("proxy_monitor.press_ctrl_c_cancel", "Press Ctrl+C to cancel"),
                style="dim yellow",
            )
        )
    )

    # Track if we've shown the final state
    final_state_shown = False

    try:
        with Live(layout, console=console, refresh_per_second=monitor_refresh_rate):
            while not startup_complete.is_set() or not final_state_shown:
                # Recalculate terminal height in case of resize
                current_height = console.height

                # Calculate available space for logs
                # Fixed: header(3) + status(4) + gpu(varies) + divider(1) + footer(1)
                fixed_height = 9 + gpu_panel_size

                # Available height for logs with padding
                available_log_height = max(20, current_height - fixed_height - 2)

                # Update status
                status_table = Table(show_header=False, box=None)
                status_table.add_column(
                    tr("proxy_monitor.column_model", "Model"), style="cyan"
                )
                status_table.add_column(
                    tr("proxy_monitor.column_status", "Status"), style="green"
                )

                ready_count = sum(1 for s in startup_status.values() if s == "ready")
                total_count = len(
                    [m for m in proxy_manager.proxy_config.models if m.enabled]
                )

                status_table.add_row(
                    tr("proxy_monitor.progress_label", "Progress"),
                    tr(
                        "proxy_monitor.progress_value",
                        "{ready}/{total} models ready",
                        ready=ready_count,
                        total=total_count,
                    ),
                )

                layout["status"].update(
                    create_panel(
                        status_table,
                        title=tr("proxy_monitor.title_status", "Status"),
                        border_style=(
                            "green" if ready_count == total_count else "yellow"
                        ),
                    )
                )

                # Update GPU panel periodically
                layout["gpu"].update(create_gpu_status_panel())

                # Use common function to create models log panel with dynamic height
                models_content = create_models_log_panel(
                    proxy_manager,
                    startup_status=startup_status,
                    available_height=available_log_height,  # Use dynamic height
                    show_status=True,
                )

                # Update models panel
                if models_content:
                    layout["models"].update(Padding(Group(*models_content), (1, 2)))
                else:
                    layout["models"].update(
                        Padding(
                            Text(
                                tr(
                                    "proxy_monitor.no_models_starting",
                                    "[dim]No models starting...[/dim]",
                                )
                            ),
                            (1, 2),
                        )
                    )

                # Check if all models are ready and mark final state shown
                if startup_complete.is_set() and not final_state_shown:
                    # Give one more iteration to show the final "Ready" state
                    final_state_shown = True
                    time.sleep(1.0)  # Show final state for 1 second
                elif not startup_complete.is_set():
                    time.sleep(0.5)

    except KeyboardInterrupt:
        console.print(
            tr(
                "proxy_monitor.startup_cancelled",
                "\n[yellow]Startup cancelled by user.[/yellow]",
            )
        )
        startup_complete.set()
        return False

    # Wait for check thread to finish
    check_thread.join(timeout=1)

    # Final status
    success_count = sum(1 for s in startup_status.values() if s == "ready")
    fail_count = sum(1 for s in startup_status.values() if s == "failed")

    # Get list of failed models
    failed_models = [
        name for name, status in startup_status.items() if status == "failed"
    ]

    if fail_count > 0:
        console.print(
            tr(
                "proxy_monitor.models_failed_to_start",
                "\n[red]✗ {count} model(s) failed to start[/red]",
                count=fail_count,
            )
        )

        # Offer to view logs for failed models
        if failed_models:
            handle_failed_models(proxy_manager, failed_models)

        return False
    else:
        console.print(
            tr(
                "proxy_monitor.all_models_started",
                "\n[green]✓ All {count} model(s) started successfully[/green]",
                count=success_count,
            )
        )
        return True


def monitor_priority_group(
    proxy_manager: "ProxyManager",
    models_in_group: List,
    priority_label: str,
    group_index: int,
    total_groups: int,
) -> bool:
    """
    Monitor the startup progress of models in a single priority group.

    Reuses the monitoring infrastructure from monitor_startup_progress but
    filters to show only models in the current priority group.

    Args:
        proxy_manager: The ProxyManager instance
        models_in_group: List of ModelConfig instances in this priority group
        priority_label: Human-readable label (e.g., "Priority 1")
        group_index: Current group number (1-indexed)
        total_groups: Total number of priority groups

    Returns:
        True if all models in group started successfully, False otherwise
    """
    console.print(
        tr(
            "proxy_monitor.priority_group_header",
            "\n[bold cyan]{label} (Group {index}/{total})[/bold cyan]",
            label=priority_label,
            index=group_index,
            total=total_groups,
        )
    )
    console.print(
        tr(
            "proxy_monitor.starting_group_models",
            "[dim]Starting {count} model(s) - showing real-time logs...[/dim]\n",
            count=len(models_in_group),
        )
    )

    # Get UI preferences
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()
    monitor_refresh_rate = ui_prefs.get("monitor_refresh_rate", 1.0)

    # Track startup status for models in this group
    startup_status = {}
    model_names_in_group = {m.name for m in models_in_group}
    startup_complete = threading.Event()

    def check_startup_completion():
        """Background thread to check startup completion and register models."""
        models_to_register = []

        while not startup_complete.is_set():
            all_ready = True

            for model_config in models_in_group:
                model_name = model_config.name

                # Skip if already processed
                if model_name in startup_status and startup_status[model_name] in [
                    "ready",
                    "failed",
                ]:
                    continue

                # Check if server exists
                if model_name not in proxy_manager.vllm_servers:
                    startup_status[model_name] = "pending"
                    all_ready = False
                    continue

                server = proxy_manager.vllm_servers[model_name]

                # Check if server is running
                if not server.is_running():
                    startup_status[model_name] = "failed"
                    logger.error(f"Server {model_name} failed to start")
                    continue

                # Check logs for startup completion
                recent_logs = server.get_recent_logs(20)
                if recent_logs:
                    for log in recent_logs:
                        if "application startup complete" in log.lower():
                            if model_name not in models_to_register:
                                startup_status[model_name] = "ready"
                                models_to_register.append(model_config)
                                logger.info(f"Model {model_name} is ready")
                            break
                    else:
                        startup_status[model_name] = "starting"
                        all_ready = False
                else:
                    startup_status[model_name] = "starting"
                    all_ready = False

            # Register ready models with proxy
            for model_config in models_to_register[:]:
                if proxy_manager.wait_and_register_model(model_config):
                    models_to_register.remove(model_config)

            if all_ready:
                startup_complete.set()
                break

            time.sleep(1)

    # Start background thread
    check_thread = Thread(target=check_startup_completion, daemon=True)
    check_thread.start()

    # Get GPU info for layout
    gpu_info = get_gpu_info()
    gpu_panel_size = calculate_gpu_panel_size(len(gpu_info) if gpu_info else 0)

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="status", size=4),
        Layout(name="gpu", size=gpu_panel_size),
        Layout(name="divider", size=1),
        Layout(name="models"),
        Layout(name="footer", size=1),
    )

    # Header
    layout["header"].update(
        create_panel(
            tr(
                "proxy_monitor.priority_group_panel_header",
                "[bold cyan]{label} - Group {index}/{total}[/bold cyan]",
                label=priority_label,
                index=group_index,
                total=total_groups,
            ),
            title=tr(
                "proxy_monitor.title_sequential_model_loading",
                "Sequential Model Loading",
            ),
            border_style="cyan",
        )
    )

    # Initial GPU panel
    layout["gpu"].update(create_gpu_status_panel())

    # Divider
    layout["divider"].update(
        Rule(tr("proxy_monitor.rule_model_logs", "Model Logs"), style="cyan")
    )

    # Footer
    layout["footer"].update(
        Align.center(
            Text(
                tr("proxy_monitor.press_ctrl_c_cancel", "Press Ctrl+C to cancel"),
                style="dim yellow",
            )
        )
    )

    # Track if we've shown the final state
    final_state_shown = False

    try:
        with Live(layout, console=console, refresh_per_second=monitor_refresh_rate):
            while not startup_complete.is_set() or not final_state_shown:
                # Recalculate terminal height
                current_height = console.height
                fixed_height = 9 + gpu_panel_size
                available_log_height = max(20, current_height - fixed_height - 2)

                # Update status
                status_table = Table(show_header=False, box=None)
                status_table.add_column(
                    tr("proxy_monitor.column_model", "Model"), style="cyan"
                )
                status_table.add_column(
                    tr("proxy_monitor.column_status", "Status"), style="green"
                )

                ready_count = sum(1 for s in startup_status.values() if s == "ready")
                total_count = len(models_in_group)

                status_table.add_row(
                    tr("proxy_monitor.progress_label", "Progress"),
                    tr(
                        "proxy_monitor.progress_value",
                        "{ready}/{total} models ready",
                        ready=ready_count,
                        total=total_count,
                    ),
                )

                layout["status"].update(
                    create_panel(
                        status_table,
                        title=tr("proxy_monitor.title_status", "Status"),
                        border_style=(
                            "green" if ready_count == total_count else "yellow"
                        ),
                    )
                )

                # Update GPU panel
                layout["gpu"].update(create_gpu_status_panel())

                # Create models log panel (only show models in this group)
                models_content = create_models_log_panel(
                    proxy_manager,
                    startup_status=startup_status,
                    available_height=available_log_height,
                    show_status=True,
                    filter_models=model_names_in_group,  # Filter to group
                )

                # Update models panel
                if models_content:
                    layout["models"].update(Padding(Group(*models_content), (1, 2)))
                else:
                    layout["models"].update(
                        Padding(
                            Text(
                                tr(
                                    "proxy_monitor.no_models_starting",
                                    "[dim]No models starting...[/dim]",
                                )
                            ),
                            (1, 2),
                        )
                    )

                # Check if all models are ready
                if startup_complete.is_set() and not final_state_shown:
                    final_state_shown = True
                    time.sleep(1.0)
                elif not startup_complete.is_set():
                    time.sleep(0.5)

    except KeyboardInterrupt:
        console.print(
            tr(
                "proxy_monitor.startup_cancelled",
                "\n[yellow]Startup cancelled by user.[/yellow]",
            )
        )
        startup_complete.set()
        return False

    # Wait for check thread
    check_thread.join(timeout=1)

    # Final status
    success_count = sum(1 for s in startup_status.values() if s == "ready")
    fail_count = sum(1 for s in startup_status.values() if s == "failed")

    # Get list of failed models
    failed_models = [
        name for name, status in startup_status.items() if status == "failed"
    ]

    if fail_count > 0:
        console.print(
            tr(
                "proxy_monitor.group_models_failed",
                "\n[red]✗ {count} model(s) in {label} failed to start[/red]",
                count=fail_count,
                label=priority_label,
            )
        )
        if failed_models:
            handle_failed_models(proxy_manager, failed_models)
        return False
    else:
        console.print(
            tr(
                "proxy_monitor.group_models_started",
                "\n[green]✓ All {count} model(s) in {label} started successfully[/green]",
                count=success_count,
                label=priority_label,
            )
        )
        return True


def handle_failed_models(
    proxy_manager: "ProxyManager", failed_models: List[str]
) -> None:
    """
    Handle failed models by offering log viewing options.

    Args:
        proxy_manager: The ProxyManager instance
        failed_models: List of model names that failed to start
    """
    from ..log_viewer import show_log_menu

    for model_name in failed_models:
        if model_name in proxy_manager.vllm_servers:
            server = proxy_manager.vllm_servers[model_name]
            console.print(
                tr(
                    "proxy_monitor.model_failed_to_start",
                    "\n[yellow]Model '{name}' failed to start[/yellow]",
                    name=model_name,
                )
            )

            # Show last few log lines
            recent_logs = server.get_recent_logs(5)
            if recent_logs:
                console.print(
                    tr("proxy_monitor.last_logs_header", "[bold]Last logs:[/bold]")
                )
                for log in recent_logs:
                    console.print(f"  {log}")

            # Offer to view full logs
            view_logs = (
                input(
                    tr(
                        "proxy_monitor.view_full_logs_for",
                        "\nView full logs for {name}? (y/N): ",
                        name=model_name,
                    )
                )
                .strip()
                .lower()
            )
            if view_logs in ["y", "yes"]:
                show_log_menu(server)
            else:
                if server.log_path:
                    console.print(
                        tr(
                            "proxy_monitor.log_file",
                            "[dim]Log file: {path}[/dim]",
                            path=server.log_path,
                        )
                    )


def refresh_model_registry(proxy_manager: "ProxyManager"):
    """
    Refresh the model registry by scanning and registering all configured models.

    Args:
        proxy_manager: The ProxyManager instance
    """
    console.print(
        tr(
            "proxy_monitor.refreshing_registry_title",
            "\n[bold cyan]Refreshing Model Registry[/bold cyan]",
        )
    )
    console.print(
        tr(
            "proxy_monitor.refreshing_registry_subtitle",
            "[dim]Scanning for models to register...[/dim]\n",
        )
    )

    # Call the refresh method
    with console.status(
        tr(
            "proxy_monitor.refreshing_registry_status",
            "[cyan]Refreshing model registrations...[/cyan]",
        )
    ):
        result = proxy_manager.refresh_model_registrations()

    if result.get("status") == "error":
        console.print(
            tr(
                "proxy_monitor.error_prefix",
                "[red]✗ Error: {message}[/red]",
                message=result.get("message", "Unknown error"),
            )
        )
    else:
        # Display results
        summary = result.get("summary", {})
        details = result.get("details", {})

        # Create a summary table for what changed
        summary_table = Table(box=box.ROUNDED)
        summary_table.add_column(
            tr("proxy_monitor.column_status", "Status"), style="cyan"
        )
        summary_table.add_column(
            tr("proxy_monitor.column_count", "Count"), style="white"
        )
        summary_table.add_column(
            tr("proxy_monitor.column_models", "Models"), style="dim"
        )

        # Only add rows if there were changes
        has_changes = False
        if summary.get("registered", 0) > 0:
            has_changes = True
            registered = details.get("newly_registered", [])
            summary_table.add_row(
                tr("proxy_monitor.row_newly_registered", "✓ Newly Registered"),
                str(summary.get("registered", 0)),
                ", ".join(registered) if registered else "-",
            )

        if summary.get("failed", 0) > 0:
            has_changes = True
            failed = details.get("newly_failed", [])
            summary_table.add_row(
                tr("proxy_monitor.row_newly_failed", "✗ Newly Failed"),
                str(summary.get("failed", 0)),
                ", ".join(failed) if failed else "-",
            )

        if summary.get("removed", 0) > 0:
            has_changes = True
            removed = details.get("removed", [])
            summary_table.add_row(
                tr("proxy_monitor.row_removed_stopped", "[×] Removed (stopped)"),
                str(summary.get("removed", 0)),
                ", ".join(removed) if removed else "-",
            )

        # Show the summary table only if there were changes
        if has_changes:
            summary_panel = Panel(
                summary_table,
                title=tr(
                    "proxy_monitor.registration_changes_title",
                    "[bold cyan]Registration Changes[/bold cyan]",
                ),
                border_style="cyan",
                padding=(1, 2),
            )
            console.print(summary_panel)

        # Show status for already available models
        if summary.get("already_available", 0) > 0:
            available_list = details.get("already_available", [])
            if available_list:
                console.print(
                    tr(
                        "proxy_monitor.models_verified_available",
                        "\n[green]✓ {count} model(s) verified as available:[/green] {names}",
                        count=len(available_list),
                        names=", ".join(available_list),
                    )
                )

        # Get full registry status to show current state of all models
        registry_status = proxy_manager.get_proxy_registry_status()
        if registry_status:
            # Use shared component to create table
            status_table = create_model_registry_table(
                registry_status, proxy_manager, include_title=True
            )

            # Display if we have models
            if registry_status.get("models") or (
                proxy_manager and proxy_manager.proxy_config.models
            ):
                console.print()  # Add spacing
                console.print(status_table)

        # Show detailed failure reasons if any
        if details.get("failed"):
            console.print(
                tr(
                    "proxy_monitor.failed_registration_details",
                    "\n[yellow]Failed Registration Details:[/yellow]",
                )
            )
            for failure in details["failed"]:
                console.print(
                    "  • "
                    + tr(
                        "proxy_monitor.failure_reason",
                        "{name}: [dim]{reason}[/dim]",
                        name=failure.get("name", "unknown"),
                        reason=failure.get("reason", "Unknown reason"),
                    )
                )

    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")


def monitor_model_logs_menu(proxy_manager: "ProxyManager") -> str:
    """
    Submenu for selecting model monitoring mode.

    Args:
        proxy_manager: The ProxyManager instance

    Returns:
        Navigation command string ('back' or 'stop')
    """
    console.print(
        tr(
            "proxy_monitor.monitor_model_logs_title",
            "[bold cyan]Monitor Model Logs[/bold cyan]\n",
        )
    )

    options = [
        (
            "overview",
            tr("proxy_monitor.option_overview", "Overview - Monitor all models"),
        ),
        (
            "individual",
            tr(
                "proxy_monitor.option_individual_model",
                "Individual - Monitor specific model",
            ),
        ),
    ]

    choice = prompt_choice(
        "model_monitoring_mode",
        tr("proxy_monitor.select_monitoring_mode", "Select monitoring mode"),
        options,
        allow_back=True,
    )

    if choice == "BACK" or not choice:
        return "back"

    if choice == "overview":
        return monitor_proxy_overview(proxy_manager)
    elif choice == "individual":
        return monitor_individual_model(proxy_manager)

    return "back"


def monitor_proxy(proxy_manager: "ProxyManager") -> str:
    """
    Monitor the proxy server and all model engines.

    Provides options to view overall status or individual model logs.

    Args:
        proxy_manager: The ProxyManager instance

    Returns:
        Navigation command string
    """
    while True:
        console.clear()
        console.print(
            tr(
                "proxy_monitor.proxy_monitor_title",
                "[bold cyan]Multi-Model Proxy Monitor[/bold cyan]",
            )
        )
        console.print(
            tr(
                "proxy_monitor.press_ctrl_c_exit",
                "[dim]Press Ctrl+C to exit monitoring[/dim]\n",
            )
        )

        # Get UI preferences
        config_manager = ConfigManager()
        ui_prefs = config_manager.get_ui_preferences()
        ui_prefs.get("show_gpu_in_monitor", True)
        ui_prefs.get("monitor_refresh_rate", 1.0)

        # Show monitoring options
        options = [
            (
                "overview",
                tr("proxy_monitor.option_overview", "Overview - Monitor all models"),
            ),
            (
                "individual",
                tr(
                    "proxy_monitor.option_individual_logs",
                    "Individual - Monitor specific model logs",
                ),
            ),
            (
                "proxy_logs",
                tr(
                    "proxy_monitor.option_proxy_logs",
                    "Proxy Server Logs - View proxy server logs",
                ),
            ),
            (
                "refresh_registry",
                tr(
                    "proxy_monitor.option_refresh_registry",
                    "Refresh Model Registry - Scan and register models",
                ),
            ),
            (
                "status",
                tr("proxy_monitor.option_status", "Status - Show current status"),
            ),
            (
                "back_menu",
                tr("proxy_monitor.option_back_to_menu", "Back to proxy menu"),
            ),
        ]

        choice = prompt_choice(
            "proxy_monitor",
            tr("proxy_monitor.select_monitoring_mode", "Select monitoring mode"),
            options,
            allow_back=True,
        )

        if choice == "BACK" or choice == "back_menu" or not choice:
            return "back"

        result = None
        if choice == "overview":
            result = monitor_proxy_overview(proxy_manager)
        elif choice == "individual":
            result = monitor_individual_model(proxy_manager)
        elif choice == "proxy_logs":
            result = monitor_proxy_logs(proxy_manager)
        elif choice == "refresh_registry":
            refresh_model_registry(proxy_manager)
            continue  # Loop back to menu
        elif choice == "status":
            show_proxy_status(proxy_manager)
            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            continue  # Loop back to menu

        # Handle navigation results from monitoring functions
        if result == "back":
            return "back"  # Exit to proxy running menu
        elif result == "menu":
            continue  # Loop back to monitoring menu
        # If monitoring function returns another monitoring function,
        # it will handle the transition internally
        elif result:
            # For any other result, assume we should exit
            return result

    return "back"


def monitor_proxy_overview(proxy_manager: "ProxyManager") -> str:
    """
    Monitor overview of all models in the proxy.

    Shows status of all models and aggregated metrics.
    """
    console.print(
        tr(
            "proxy_monitor.overview_monitor_title",
            "[bold cyan]Proxy Overview Monitor[/bold cyan]",
        )
    )
    console.print(
        tr(
            "proxy_monitor.press_ctrl_c_menu_line",
            "[dim]Press Ctrl+C for menu options[/dim]\n",
        )
    )

    # Get UI preferences
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()
    show_gpu = ui_prefs.get("show_gpu_in_monitor", True)
    monitor_refresh_rate = ui_prefs.get("monitor_refresh_rate", 1.0)

    try:
        # Get GPU info for panel sizing
        gpu_info = get_gpu_info() if show_gpu else None
        gpu_panel_size = (
            calculate_gpu_panel_size(len(gpu_info) if gpu_info else 0)
            if show_gpu
            else 0
        )

        # Create layout - simplified without models table
        layout = Layout()
        if show_gpu:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="proxy_status", size=4),
                Layout(name="gpu", size=gpu_panel_size),
                Layout(name="log_divider", size=1),
                Layout(name="logs"),  # Takes remaining space
                Layout(name="footer", size=1),
            )
        else:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="proxy_status", size=4),
                Layout(name="log_divider", size=1),
                Layout(name="logs"),  # Takes remaining space
                Layout(name="footer", size=1),
            )

        # Header
        header_text = Text(
            tr(
                "proxy_monitor.overview_header",
                "Multi-Model Proxy Monitor - {count} Models",
                count=len(proxy_manager.vllm_servers),
            ),
            style="bold cyan",
            justify="center",
        )
        layout["header"].update(Padding(header_text, (1, 0)))

        # Footer
        layout["footer"].update(
            Align.center(
                Text(
                    tr(
                        "proxy_monitor.press_ctrl_c_menu",
                        "Press Ctrl+C for menu options",
                    ),
                    style="dim cyan",
                )
            )
        )

        with Live(layout, console=console, refresh_per_second=monitor_refresh_rate):
            while True:
                # Recalculate terminal height in case of resize
                current_height = console.height

                # Calculate available space for logs
                # Fixed: header(3) + proxy_status(4) + divider(1) + footer(1) = 9
                fixed_height = 9
                if show_gpu:
                    fixed_height += gpu_panel_size

                # Available height for logs with padding
                available_log_height = max(20, current_height - fixed_height - 2)

                # Update proxy status
                proxy_status = Table(show_header=False, box=None)
                proxy_status.add_column(
                    tr("proxy_monitor.column_key", "Key"), style="cyan"
                )
                proxy_status.add_column(
                    tr("proxy_monitor.column_value", "Value"), style="magenta"
                )

                proxy_status.add_row(
                    tr("proxy_monitor.row_proxy_status", "Proxy Status"),
                    (
                        tr("proxy_monitor.running_markup", "[green]Running[/green]")
                        if proxy_manager.proxy_process
                        and proxy_manager.proxy_process.is_running()
                        else tr("proxy_monitor.stopped_markup", "[red]Stopped[/red]")
                    ),
                )
                proxy_status.add_row(
                    tr("proxy_monitor.row_proxy_port", "Proxy Port"),
                    str(proxy_manager.proxy_config.port),
                )
                proxy_status.add_row(
                    tr("proxy_monitor.row_active_models", "Active Models"),
                    str(len(proxy_manager.vllm_servers)),
                )

                layout["proxy_status"].update(
                    create_panel(
                        proxy_status,
                        title=tr(
                            "proxy_monitor.title_proxy_server", "Proxy Server"
                        ),
                        border_style=(
                            "green"
                            if proxy_manager.proxy_process
                            and proxy_manager.proxy_process.is_running()
                            else "red"
                        ),
                    )
                )

                # Update GPU panel if enabled
                if show_gpu:
                    gpu_panel = create_gpu_status_panel()
                    layout["gpu"].update(gpu_panel)

                # Update log divider
                layout["log_divider"].update(
                    Rule(
                        tr("proxy_monitor.rule_model_logs", "Model Logs"),
                        style="cyan",
                    )
                )

                # Use common function to create models log panel with dynamic height
                models_log_content = create_models_log_panel(
                    proxy_manager,
                    startup_status=None,  # No startup status in overview mode
                    available_height=available_log_height,  # Pass dynamic height
                    show_status=True,  # Show status since we removed the table
                )

                if models_log_content:
                    logs_content = Padding(Group(*models_log_content), (1, 2))
                else:
                    logs_content = Padding(
                        Text(
                            tr(
                                "proxy_monitor.waiting_for_logs",
                                "Waiting for logs...",
                            ),
                            style="dim yellow",
                        ),
                        (1, 2),
                    )

                layout["logs"].update(logs_content)

                time.sleep(0.5)

    except KeyboardInterrupt:
        pass

    console.print(
        tr(
            "proxy_monitor.monitoring_stopped",
            "\n[yellow]Monitoring stopped.[/yellow]",
        )
    )

    try:
        input(
            tr(
                "proxy_monitor.press_enter_return",
                "\nPress Enter to return to monitoring menu...",
            )
        )
        return "back"
    except KeyboardInterrupt:
        # User wants to stop proxy
        stop_confirm = prompt_choice(
            "confirm_stop_overview",
            tr(
                "proxy_monitor.stop_all_servers_confirm",
                "Stop all proxy servers?",
            ),
            [
                (
                    "yes",
                    tr(
                        "proxy_monitor.yes_stop_all_servers",
                        "Yes, stop all servers",
                    ),
                ),
                ("no", tr("proxy_monitor.no_keep_running", "No, keep running")),
            ],
            allow_back=False,
        )
        if stop_confirm == "yes":
            return "stop"
        return "back"


def monitor_individual_model_by_name(
    proxy_manager: "ProxyManager", model_name: str
) -> str:
    """
    Monitor logs of a specific model by name.

    Args:
        proxy_manager: The ProxyManager instance
        model_name: Name of the model to monitor

    Returns:
        Navigation command string
    """
    if model_name not in proxy_manager.vllm_servers:
        console.print(
            tr(
                "proxy_monitor.model_not_found_running",
                "[red]Model '{name}' not found or not running[/red]",
                name=model_name,
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return "back"

    server = proxy_manager.vllm_servers[model_name]
    return monitor_model_engine(server, model_name, proxy_manager)


def monitor_individual_model(proxy_manager: "ProxyManager") -> str:
    """
    Monitor logs of a specific model engine.

    Allows user to select a model and view its logs in real-time.
    """
    if not proxy_manager.vllm_servers:
        console.print(
            tr(
                "proxy_monitor.no_models_running",
                "[yellow]No models are currently running.[/yellow]",
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return "back"

    # List available models
    console.print(
        tr(
            "proxy_monitor.select_model_title",
            "\n[bold cyan]Select Model to Monitor[/bold cyan]",
        )
    )

    model_options = []
    for model_name, server in proxy_manager.vllm_servers.items():
        status = (
            tr("proxy_monitor.engine_running", "Running")
            if server.is_running()
            else tr("proxy_monitor.engine_stopped", "Stopped")
        )
        model_options.append(
            (
                model_name,
                tr(
                    "proxy_monitor.model_option_label",
                    "{name} (Port {port}, {status})",
                    name=model_name,
                    port=server.port,
                    status=status,
                ),
            )
        )

    model_name = prompt_choice(
        "select_model",
        tr("proxy_monitor.select_model_prompt", "Select a model to monitor"),
        model_options,
        allow_back=True,
    )

    if model_name == "BACK" or not model_name:
        return "back"

    if model_name not in proxy_manager.vllm_servers:
        console.print(
            tr(
                "proxy_monitor.model_not_found",
                "[red]Model '{name}' not found.[/red]",
                name=model_name,
            )
        )
        return monitor_individual_model(proxy_manager)

    server = proxy_manager.vllm_servers[model_name]

    # Monitor this specific model (reuse existing monitor function)
    return monitor_model_engine(server, model_name, proxy_manager)


def monitor_model_engine(
    server: "VLLMServer", model_name: str, proxy_manager: "ProxyManager"
) -> str:
    """
    Monitor a specific model engine with its logs.

    Similar to single-model monitoring but within proxy context.
    """

    # Get UI preferences
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()
    show_gpu = ui_prefs.get("show_gpu_in_monitor", True)
    monitor_refresh_rate = ui_prefs.get("monitor_refresh_rate", 1.0)

    try:
        # Get GPU info
        gpu_info = get_gpu_info() if show_gpu else None
        gpu_panel_size = (
            calculate_gpu_panel_size(len(gpu_info) if gpu_info else 0)
            if show_gpu
            else 0
        )

        # Create layout
        layout = Layout()
        if show_gpu:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="status", size=6),
                Layout(name="gpu", size=gpu_panel_size),
                Layout(name="logs"),
                Layout(name="footer", size=2),
            )
        else:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="status", size=6),
                Layout(name="logs"),
                Layout(name="footer", size=2),
            )

        # Header
        header_text = Text(
            tr(
                "proxy_monitor.engine_monitor_header",
                "Model Engine Monitor - {name}",
                name=model_name,
            ),
            style="bold cyan",
            justify="center",
        )
        layout["header"].update(Padding(header_text, (1, 0)))

        # Footer
        footer_text = tr(
            "proxy_monitor.press_ctrl_c_menu", "Press Ctrl+C for menu options"
        )
        layout["footer"].update(Align.center(Text(footer_text, style="dim cyan")))

        # Track operation state for completion detection
        operation_state = {
            "type": None,  # 'sleeping', 'waking', or None
            "started_this_session": False,  # Track if we saw the start in THIS monitoring session
            "memory_freed": None,
            "memory_remaining": None,
            "completion_time": None,
            "completed": False,
            "notification_shown_until": None,  # Show notification for 5 seconds
            "exit_after": None,  # Track when to auto-exit after completion
        }

        # Flag to track if server failed during monitoring
        server_failed = False

        with Live(layout, console=console, refresh_per_second=monitor_refresh_rate):
            while True:
                # Update status
                status_table = Table(show_header=False, box=None)
                status_table.add_column(
                    tr("proxy_monitor.column_key", "Key"), style="cyan"
                )
                status_table.add_column(
                    tr("proxy_monitor.column_value", "Value"), style="magenta"
                )

                status_table.add_row(
                    tr("proxy_monitor.column_status", "Status"),
                    (
                        tr("proxy_monitor.running_markup", "[green]Running[/green]")
                        if server.is_running()
                        else tr("proxy_monitor.stopped_markup", "[red]Stopped[/red]")
                    ),
                )
                status_table.add_row(
                    tr("proxy_monitor.column_model", "Model"), model_name
                )
                status_table.add_row(
                    tr("proxy_monitor.column_port", "Port"), str(server.port)
                )
                status_table.add_row(
                    tr("proxy_monitor.column_pid", "PID"),
                    str(server.process.pid) if server.process else "N/A",
                )

                # Add GPU info
                model_config = proxy_manager._get_model_config_by_name(model_name)
                if model_config and model_config.gpu_ids:
                    gpu_str = ",".join(str(g) for g in model_config.gpu_ids)
                    status_table.add_row(
                        tr("proxy_monitor.column_gpus", "GPUs"), gpu_str
                    )

                layout["status"].update(
                    create_panel(
                        status_table,
                        title=tr(
                            "proxy_monitor.title_engine_status", "Engine Status"
                        ),
                        border_style="green" if server.is_running() else "red",
                    )
                )

                # Update GPU panel
                if show_gpu:
                    gpu_panel = create_gpu_status_panel()
                    layout["gpu"].update(gpu_panel)

                # Update logs dynamically based on available space
                # Calculate available height for logs
                current_height = console.height
                # Fixed: header(3) + status(6) + footer(2) = 11
                fixed_height = 11
                if show_gpu:
                    fixed_height += gpu_panel_size

                # Account for logs panel overhead: Rule(1) + Padding(1) + margin(1)
                logs_panel_overhead = 3

                # Available lines for logs with proper accounting
                available_lines = max(
                    10, current_height - fixed_height - logs_panel_overhead
                )
                recent_logs = server.get_recent_logs(available_lines)

                # Check for sleep/wake operations in logs
                if recent_logs:
                    for log_line in recent_logs:
                        parsed = parse_sleep_wake_logs(log_line)
                        if parsed:
                            # Process start patterns - track if we see them in this session
                            if parsed["type"] == "sleep_start":
                                # Only process if we haven't already started tracking this operation
                                if (
                                    not operation_state.get("started_this_session")
                                    or operation_state["type"] != "sleeping"
                                ):
                                    # New sleep operation in this session
                                    operation_state["type"] = "sleeping"
                                    operation_state["started_this_session"] = True
                                    operation_state["completed"] = False
                                    operation_state["memory_freed"] = None
                                    operation_state["memory_remaining"] = None
                                    operation_state["completion_time"] = None
                                    operation_state["notification_shown_until"] = None
                                    operation_state["exit_after"] = None
                            elif parsed["type"] == "wake_start":
                                # Only process if we haven't already started tracking this operation
                                if (
                                    not operation_state.get("started_this_session")
                                    or operation_state["type"] != "waking"
                                ):
                                    # New wake operation in this session
                                    operation_state["type"] = "waking"
                                    operation_state["started_this_session"] = True
                                    operation_state["completed"] = False
                                    operation_state["completion_time"] = None
                                    operation_state["notification_shown_until"] = None
                                    operation_state["memory_freed"] = None
                                    operation_state["memory_remaining"] = None
                                    operation_state["exit_after"] = None

                            # Always process completion and info patterns
                            if parsed["type"] == "memory_info":
                                # Store memory info for sleep operation
                                operation_state["memory_freed"] = parsed["freed"]
                                operation_state["memory_remaining"] = parsed[
                                    "remaining"
                                ]
                            elif parsed["type"] == "sleep_complete":
                                # Only process completion if we saw the start in this session
                                if (
                                    operation_state.get("started_this_session")
                                    and operation_state["type"] == "sleeping"
                                ):
                                    # Sleep operation completed
                                    operation_state["completion_time"] = parsed["time"]
                                    operation_state["completed"] = True
                                    operation_state["notification_shown_until"] = (
                                        time.time() + 5
                                    )  # Show for 5 seconds
                                    operation_state["exit_after"] = (
                                        time.time() + 8
                                    )  # Exit 3 seconds after notification ends
                                    break  # Exit the log checking loop immediately
                            elif parsed["type"] == "wake_complete":
                                # Only process completion if we saw the start in this session
                                if (
                                    operation_state.get("started_this_session")
                                    and operation_state["type"] == "waking"
                                ):
                                    # Wake operation completed
                                    operation_state["completion_time"] = parsed["time"]
                                    operation_state["completed"] = True
                                    operation_state["notification_shown_until"] = (
                                        time.time() + 5
                                    )
                                    operation_state["exit_after"] = (
                                        time.time() + 8
                                    )  # Exit 3 seconds after notification ends
                                    break  # Exit the log checking loop immediately

                # Build logs display with notification if needed
                if operation_state.get("type") and not operation_state.get("completed"):
                    # Operation in progress
                    if operation_state["type"] == "sleeping":
                        progress_text = tr(
                            "proxy_monitor.sleep_in_progress",
                            "[yellow]⏳ Sleep operation in progress...[/yellow]",
                        )
                    else:  # waking
                        progress_text = tr(
                            "proxy_monitor.wake_in_progress",
                            "[yellow]⏳ Wake operation in progress...[/yellow]",
                        )

                    # Show progress indicator with logs
                    log_text = Text(
                        (
                            "\n".join(recent_logs)
                            if recent_logs
                            else tr(
                                "proxy_monitor.waiting_for_logs",
                                "Waiting for logs...",
                            )
                        ),
                        style="dim white",
                    )
                    logs_content = Group(
                        Panel(progress_text, border_style="yellow", box=box.ROUNDED),
                        Rule(
                            tr(
                                "proxy_monitor.rule_engine_logs",
                                "Engine Logs - {name}",
                                name=model_name,
                            ),
                            style="yellow",
                        ),
                        Padding(log_text, (0, 2)),
                    )
                elif operation_state.get("completed") and operation_state.get(
                    "notification_shown_until"
                ):
                    if time.time() < operation_state["notification_shown_until"]:
                        # Check if this notification is still relevant (no new operation started)
                        show_notification = True

                        # Check recent logs for new operations that would invalidate this notification
                        for log_line in (
                            recent_logs[-5:] if len(recent_logs) > 5 else recent_logs
                        ):
                            # Check for new wake operation after sleep completion
                            if (
                                "wake up the engine" in log_line
                                and operation_state["type"] == "sleeping"
                            ):
                                show_notification = False
                                operation_state["notification_shown_until"] = None
                                break
                            # Check for new sleep operation after wake completion
                            elif (
                                "POST /sleep" in log_line
                                and operation_state["type"] == "waking"
                            ):
                                show_notification = False
                                operation_state["notification_shown_until"] = None
                                break

                        if show_notification:
                            # Show completion notification
                            notification_lines = []
                            elapsed = f"{operation_state['completion_time']:.2f}"

                            if operation_state["type"] == "sleeping":
                                notification_lines.append(
                                    tr(
                                        "proxy_monitor.sleep_completed_badge",
                                        "[bold green]✓ SLEEP COMPLETED[/bold green]",
                                    )
                                )
                                notification_lines.append(
                                    tr(
                                        "proxy_monitor.time_taken",
                                        "[cyan]Time taken: {seconds} seconds[/cyan]",
                                        seconds=elapsed,
                                    )
                                )
                                if operation_state.get("memory_freed"):
                                    notification_lines.append(
                                        tr(
                                            "proxy_monitor.memory_freed",
                                            "[yellow]Memory freed: {size} GiB[/yellow]",
                                            size=f"{operation_state['memory_freed']:.2f}",
                                        )
                                    )
                                    notification_lines.append(
                                        tr(
                                            "proxy_monitor.memory_in_use",
                                            "[dim]Memory in use: {size} GiB[/dim]",
                                            size=f"{operation_state['memory_remaining']:.2f}",
                                        )
                                    )
                            elif operation_state["type"] == "waking":
                                notification_lines.append(
                                    tr(
                                        "proxy_monitor.wake_completed_badge",
                                        "[bold green]✓ WAKE COMPLETED[/bold green]",
                                    )
                                )
                                notification_lines.append(
                                    tr(
                                        "proxy_monitor.time_taken",
                                        "[cyan]Time taken: {seconds} seconds[/cyan]",
                                        seconds=elapsed,
                                    )
                                )
                                notification_lines.append(
                                    tr(
                                        "proxy_monitor.model_fully_operational",
                                        "[green]Model is now fully operational[/green]",
                                    )
                                )

                            # Create notification panel
                            notification_panel = Panel(
                                "\n".join(notification_lines),
                                title=tr(
                                    "proxy_monitor.operation_complete_title",
                                    "[bold yellow]Operation Complete[/bold yellow]",
                                ),
                                border_style="green",
                                box=box.DOUBLE,
                            )

                            # Show logs below notification
                            log_text = Text(
                                "\n".join(recent_logs[-10:]), style="dim white"
                            )  # Show last 10 lines

                            logs_content = Group(
                                notification_panel,
                                Rule(
                                    tr(
                                        "proxy_monitor.rule_engine_logs",
                                        "Engine Logs - {name}",
                                        name=model_name,
                                    ),
                                    style="yellow",
                                ),
                                Padding(log_text, (0, 2)),
                            )
                        else:
                            # Notification was invalidated by new operation, show normal logs
                            log_text = Text(
                                (
                                    "\n".join(recent_logs)
                                    if recent_logs
                                    else tr(
                                        "proxy_monitor.waiting_for_logs",
                                        "Waiting for logs...",
                                    )
                                ),
                                style="dim white",
                            )
                            logs_content = Group(
                                Rule(
                                    tr(
                                        "proxy_monitor.rule_engine_logs",
                                        "Engine Logs - {name}",
                                        name=model_name,
                                    ),
                                    style="yellow",
                                ),
                                Padding(log_text, (0, 2)),
                            )
                    else:
                        # Notification expired, clear the completed flag
                        operation_state["completed"] = False
                        operation_state["notification_shown_until"] = None

                        # Normal log display
                        log_text = Text("\n".join(recent_logs), style="dim white")
                        logs_content = Group(
                            Rule(
                                tr(
                                    "proxy_monitor.rule_engine_logs",
                                    "Engine Logs - {name}",
                                    name=model_name,
                                ),
                                style="yellow",
                            ),
                            Padding(log_text, (0, 2)),
                        )
                else:
                    # Normal log display
                    if recent_logs:
                        log_text = Text("\n".join(recent_logs), style="dim white")
                    else:
                        log_text = Text(
                            tr(
                                "proxy_monitor.waiting_for_logs",
                                "Waiting for logs...",
                            ),
                            style="dim yellow",
                        )

                    logs_content = Group(
                        Rule(
                            tr(
                                "proxy_monitor.rule_engine_logs",
                                "Engine Logs - {name}",
                                name=model_name,
                            ),
                            style="yellow",
                        ),
                        Padding(log_text, (0, 2)),
                    )

                layout["logs"].update(logs_content)

                # Auto-exit after completion and notification display
                if operation_state.get("exit_after"):
                    if time.time() > operation_state["exit_after"]:
                        # Time to exit after showing completion
                        break

                # Check if server is still running
                if not server.is_running():
                    # Set flag to handle failure after exiting Live display
                    server_failed = True
                    break

                time.sleep(0.5)

    except KeyboardInterrupt:
        pass

    # Handle server failure after Live display has ended
    if server_failed:
        console.print(
            tr(
                "proxy_monitor.model_engine_stopped",
                "\n[red]Model engine '{name}' has stopped.[/red]",
                name=model_name,
            )
        )

        # Show last few log lines to help diagnose
        recent_logs = server.get_recent_logs(10)
        if recent_logs:
            console.print(
                tr(
                    "proxy_monitor.last_logs_before_failure",
                    "\n[bold]Last logs before failure:[/bold]",
                )
            )
            for log in recent_logs[-5:]:  # Show last 5 lines
                console.print(f"  {log}")

        # Offer to view full logs
        view_logs = (
            input(
                tr(
                    "proxy_monitor.view_full_logs_for",
                    "\nView full logs for {name}? (y/N): ",
                    name=model_name,
                )
            )
            .strip()
            .lower()
        )
        if view_logs in ["y", "yes"]:
            from ..log_viewer import show_log_menu

            show_log_menu(server)

    # Display operation-specific completion summary
    if operation_state.get("completed"):
        elapsed = f"{operation_state['completion_time']:.2f}"
        if operation_state["type"] == "sleeping":
            console.print(
                tr(
                    "proxy_monitor.sleep_summary",
                    "\n[bold green]✓ Sleep operation completed successfully![/bold green]",
                )
            )
            console.print(
                tr(
                    "proxy_monitor.time_taken",
                    "[cyan]Time taken: {seconds} seconds[/cyan]",
                    seconds=elapsed,
                )
            )
            if operation_state.get("memory_freed"):
                console.print(
                    tr(
                        "proxy_monitor.memory_freed",
                        "[yellow]Memory freed: {size} GiB[/yellow]",
                        size=f"{operation_state['memory_freed']:.2f}",
                    )
                )
                console.print(
                    tr(
                        "proxy_monitor.memory_still_in_use",
                        "[dim]Memory still in use: {size} GiB[/dim]",
                        size=f"{operation_state['memory_remaining']:.2f}",
                    )
                )
            console.print(
                tr(
                    "proxy_monitor.model_sleep_mode",
                    "\n[green]Model '{name}' is now in sleep mode[/green]",
                    name=model_name,
                )
            )
        elif operation_state["type"] == "waking":
            console.print(
                tr(
                    "proxy_monitor.wake_summary",
                    "\n[bold green]✓ Wake operation completed successfully![/bold green]",
                )
            )
            console.print(
                tr(
                    "proxy_monitor.time_taken",
                    "[cyan]Time taken: {seconds} seconds[/cyan]",
                    seconds=elapsed,
                )
            )
            console.print(
                tr(
                    "proxy_monitor.model_now_operational",
                    "\n[green]Model '{name}' is now fully operational[/green]",
                    name=model_name,
                )
            )

        # Auto-return to menu after brief pause
        console.print(
            tr(
                "proxy_monitor.returning_to_menu",
                "\n[dim]Returning to proxy management menu in 3 seconds...[/dim]",
            )
        )
        time.sleep(3)
        return "back"
    else:
        # Manual exit via Ctrl+C
        console.print(
            tr(
                "proxy_monitor.monitoring_stopped",
                "\n[yellow]Monitoring stopped.[/yellow]",
            )
        )
        console.print(
            tr(
                "proxy_monitor.model_continues_running",
                "[green]✓ Model '{name}' continues running[/green]",
                name=model_name,
            )
        )

        try:
            input(
                tr(
                    "proxy_monitor.press_enter_return",
                    "\nPress Enter to return to monitoring menu...",
                )
            )
            return "back"
        except KeyboardInterrupt:
            # User wants to stop proxy
            stop_confirm = prompt_choice(
                "confirm_stop_model",
                tr(
                    "proxy_monitor.stop_all_servers_confirm",
                    "Stop all proxy servers?",
                ),
                [
                    (
                        "yes",
                        tr(
                            "proxy_monitor.yes_stop_all_servers",
                            "Yes, stop all servers",
                        ),
                    ),
                    (
                        "no",
                        tr("proxy_monitor.no_keep_running", "No, keep running"),
                    ),
                ],
                allow_back=False,
            )
            if stop_confirm == "yes":
                return "stop"
            return "back"


def show_proxy_status(proxy_manager: "ProxyManager"):
    """
    Display current status of proxy and all models.
    """
    console.print(
        tr(
            "proxy_monitor.proxy_server_status_title",
            "\n[bold cyan]Proxy Server Status[/bold cyan]",
        )
    )

    # Proxy status
    proxy_table = Table(show_header=False, box=None)
    proxy_table.add_column(
        tr("proxy_monitor.column_property", "Property"), style="cyan"
    )
    proxy_table.add_column(
        tr("proxy_monitor.column_value", "Value"), style="magenta"
    )

    proxy_table.add_row(
        tr("proxy_monitor.row_proxy_server", "Proxy Server"),
        (
            tr("proxy_monitor.running_markup", "[green]Running[/green]")
            if proxy_manager.proxy_process and proxy_manager.proxy_process.is_running()
            else tr("proxy_monitor.not_running_markup", "[red]Not Running[/red]")
        ),
    )
    proxy_table.add_row(
        tr("proxy_monitor.column_host", "Host"), proxy_manager.proxy_config.host
    )
    proxy_table.add_row(
        tr("proxy_monitor.column_port", "Port"),
        str(proxy_manager.proxy_config.port),
    )
    proxy_table.add_row(
        tr("proxy_monitor.row_cors", "CORS"),
        tr("proxy_monitor.enabled", "Enabled")
        if proxy_manager.proxy_config.enable_cors
        else tr("proxy_monitor.disabled", "Disabled"),
    )
    proxy_table.add_row(
        tr("proxy_monitor.row_metrics", "Metrics"),
        tr("proxy_monitor.enabled", "Enabled")
        if proxy_manager.proxy_config.enable_metrics
        else tr("proxy_monitor.disabled", "Disabled"),
    )

    console.print(
        create_panel(
            proxy_table,
            title=tr(
                "proxy_monitor.title_proxy_configuration", "Proxy Configuration"
            ),
        )
    )

    # Models status
    if proxy_manager.vllm_servers:
        console.print(
            tr("proxy_monitor.model_engines_header", "\n[bold]Model Engines:[/bold]")
        )

        models_table = Table()
        models_table.add_column(
            tr("proxy_monitor.column_model", "Model"), style="cyan"
        )
        models_table.add_column(
            tr("proxy_monitor.column_port", "Port"), style="magenta"
        )
        models_table.add_column(
            tr("proxy_monitor.column_status", "Status"), style="green"
        )
        models_table.add_column(
            tr("proxy_monitor.column_gpus_short", "GPU(s)"), style="yellow"
        )
        models_table.add_column(
            tr("proxy_monitor.column_profile", "Profile"), style="blue"
        )

        for model_name, server in proxy_manager.vllm_servers.items():
            model_config = proxy_manager._get_model_config_by_name(model_name)

            gpu_str = "N/A"
            profile_str = "N/A"

            if model_config:
                if model_config.gpu_ids:
                    gpu_str = ",".join(str(g) for g in model_config.gpu_ids)
                if model_config.profile:
                    profile_str = model_config.profile

            models_table.add_row(
                model_name,
                str(server.port),
                (
                    tr("proxy_monitor.running_markup", "[green]Running[/green]")
                    if server.is_running()
                    else tr("proxy_monitor.stopped_markup", "[red]Stopped[/red]")
                ),
                gpu_str,
                profile_str,
            )

        console.print(models_table)
    else:
        console.print(
            tr(
                "proxy_monitor.no_models_currently_running",
                "\n[yellow]No models currently running.[/yellow]",
            )
        )


def monitor_proxy_logs(proxy_manager: "ProxyManager") -> str:
    """
    Monitor proxy server logs and statistics.

    Shows proxy server logs, request statistics, and routing info.

    Args:
        proxy_manager: The ProxyManager instance

    Returns:
        Navigation command string
    """
    console.print(
        tr(
            "proxy_monitor.proxy_logs_title",
            "[bold cyan]Proxy Server Logs & Statistics[/bold cyan]\n",
        )
    )

    # Get UI preferences
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()
    show_gpu = ui_prefs.get("show_gpu_in_monitor", True)
    monitor_refresh_rate = ui_prefs.get("monitor_refresh_rate", 1.0)

    try:
        # Get GPU info for panel sizing
        gpu_info = get_gpu_info() if show_gpu else None
        gpu_panel_size = (
            calculate_gpu_panel_size(len(gpu_info) if gpu_info else 0)
            if show_gpu
            else 0
        )

        # Calculate dynamic size for backends based on number of models
        model_count = (
            len(proxy_manager.vllm_servers) if proxy_manager.vllm_servers else 1
        )
        # Size: 6 overhead (borders, title, header, separator, padding) + model rows
        # Minimum 7 lines for display, maximum 15 to prevent tall panels
        backends_size = max(7, min(6 + model_count, 15))

        # Create layout
        layout = Layout()
        if show_gpu:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="proxy_info", size=6),
                Layout(name="backends", size=backends_size),
                Layout(name="gpu", size=gpu_panel_size),
                Layout(name="logs"),  # Takes remaining space
                Layout(name="footer", size=1),
            )
        else:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="proxy_info", size=6),
                Layout(
                    name="backends", size=backends_size
                ),  # Same size, calculation already includes overhead
                Layout(name="logs"),  # Takes remaining space
                Layout(name="footer", size=1),
            )

        # Header
        header_text = Text(
            tr(
                "proxy_monitor.proxy_logs_header",
                "Proxy Server Logs & Statistics",
            ),
            style="bold cyan",
            justify="center",
        )
        layout["header"].update(Padding(header_text, (1, 0)))

        # Footer
        layout["footer"].update(
            Align.center(
                Text(
                    tr(
                        "proxy_monitor.press_ctrl_c_stop",
                        "Press Ctrl+C to stop monitoring",
                    ),
                    style="dim cyan",
                )
            )
        )

        # Smart registry refresh for pending models
        registry_refresh_interval = 5.0  # Start with 5 seconds
        max_refresh_interval = 30.0  # Max 30 seconds between refreshes
        last_registry_update = time.time()
        pending_model_count = 0
        refresh_attempts = 0
        max_refresh_attempts = 10  # Stop trying after 10 attempts

        # Initial registry fetch
        try:
            initial_registry_status = proxy_manager.get_proxy_registry_status()
            if initial_registry_status:
                # Count pending models
                models = initial_registry_status.get("models", [])
                pending_model_count = sum(
                    1 for m in models if m.get("registration_status") == "pending"
                )
                cached_registry_table = create_model_registry_table(
                    initial_registry_status, proxy_manager, include_title=False
                )

                if pending_model_count > 0:
                    logger.debug(
                        f"Found {pending_model_count} pending models, will refresh registry"
                    )
            else:
                cached_registry_table = Table(box=box.ROUNDED)
                cached_registry_table.add_column(
                    tr("proxy_monitor.column_status", "Status"), style="yellow"
                )
                cached_registry_table.add_row(
                    tr(
                        "proxy_monitor.registry_status_unavailable",
                        "Registry status unavailable",
                    )
                )
        except Exception as e:
            logger.warning(f"Error fetching initial registry: {e}")
            cached_registry_table = Table(box=box.ROUNDED)
            cached_registry_table.add_column(
                tr("proxy_monitor.column_error", "Error"), style="red"
            )
            cached_registry_table.add_row(f"Error: {str(e)[:50]}")

        with Live(layout, console=console, refresh_per_second=monitor_refresh_rate):
            while True:
                # Proxy server information
                if (
                    proxy_manager.proxy_process
                    and proxy_manager.proxy_process.is_running()
                ):
                    proxy_table = Table(show_header=False, box=None)
                    proxy_table.add_column(
                        tr("proxy_monitor.column_key", "Key"), style="cyan"
                    )
                    proxy_table.add_column(
                        tr("proxy_monitor.column_value", "Value"), style="magenta"
                    )

                    proxy_table.add_row(
                        tr("proxy_monitor.column_status", "Status"),
                        (
                            tr(
                                "proxy_monitor.running_markup",
                                "[green]Running[/green]",
                            )
                            if proxy_manager.proxy_process
                            and proxy_manager.proxy_process.is_running()
                            else tr(
                                "proxy_monitor.stopped_markup", "[red]Stopped[/red]"
                            )
                        ),
                    )
                    proxy_table.add_row(
                        tr("proxy_monitor.column_host", "Host"),
                        proxy_manager.proxy_config.host,
                    )
                    proxy_table.add_row(
                        tr("proxy_monitor.column_port", "Port"),
                        str(proxy_manager.proxy_config.port),
                    )
                    proxy_table.add_row(
                        tr("proxy_monitor.row_cors", "CORS"),
                        (
                            tr(
                                "proxy_monitor.enabled_markup",
                                "[green]Enabled[/green]",
                            )
                            if proxy_manager.proxy_config.enable_cors
                            else tr(
                                "proxy_monitor.disabled_markup",
                                "[yellow]Disabled[/yellow]",
                            )
                        ),
                    )
                    proxy_table.add_row(
                        tr("proxy_monitor.row_metrics", "Metrics"),
                        (
                            tr(
                                "proxy_monitor.enabled_markup",
                                "[green]Enabled[/green]",
                            )
                            if proxy_manager.proxy_config.enable_metrics
                            else tr(
                                "proxy_monitor.disabled_markup",
                                "[yellow]Disabled[/yellow]",
                            )
                        ),
                    )

                    # Calculate uptime if proxy process has start_time
                    if proxy_manager.proxy_process and hasattr(
                        proxy_manager.proxy_process, "start_time"
                    ):
                        uptime = (
                            time.time()
                            - proxy_manager.proxy_process.start_time.timestamp()
                        )
                        hours, remainder = divmod(int(uptime), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        proxy_table.add_row(
                            tr("proxy_monitor.column_uptime", "Uptime"),
                            f"{hours:02d}:{minutes:02d}:{seconds:02d}",
                        )

                    # Request statistics are available via /proxy/status endpoint
                    # but not needed for basic monitoring display
                else:
                    proxy_table = Table(show_header=False, box=None)
                    proxy_table.add_column("", style="red")
                    proxy_table.add_row(
                        tr(
                            "proxy_monitor.proxy_server_not_running",
                            "Proxy server not running",
                        )
                    )

                layout["proxy_info"].update(
                    create_panel(
                        proxy_table,
                        title=tr(
                            "proxy_monitor.title_proxy_server_status",
                            "Proxy Server Status",
                        ),
                        border_style=(
                            "green"
                            if proxy_manager.proxy_process
                            and proxy_manager.proxy_process.is_running()
                            else "red"
                        ),
                    )
                )

                # Smart refresh: Only update registry if there are pending models
                current_time = time.time()
                should_refresh = False

                if pending_model_count > 0 and refresh_attempts < max_refresh_attempts:
                    # Check if enough time has passed for next refresh
                    if (
                        current_time - last_registry_update
                    ) >= registry_refresh_interval:
                        should_refresh = True
                if should_refresh:
                    try:
                        # Fetch updated registry
                        updated_registry = proxy_manager.get_proxy_registry_status()
                        if updated_registry:
                            # Count pending models in update
                            models = updated_registry.get("models", [])
                            new_pending_count = sum(
                                1
                                for m in models
                                if m.get("registration_status") == "pending"
                            )
                            # Update the table
                            cached_registry_table = create_model_registry_table(
                                updated_registry, proxy_manager, include_title=False
                            )

                            # Check if we made progress
                            if new_pending_count < pending_model_count:
                                # Progress made, reset interval
                                registry_refresh_interval = 5.0
                                refresh_attempts = 0
                                logger.debug(
                                    f"Registry refresh: {pending_model_count} -> {new_pending_count} pending"
                                )
                            else:
                                # No progress, increase interval (exponential backoff)
                                registry_refresh_interval = min(
                                    registry_refresh_interval * 1.5,
                                    max_refresh_interval,
                                )
                                refresh_attempts += 1

                            pending_model_count = new_pending_count
                            last_registry_update = current_time

                            # Stop refreshing if all models are registered
                            if pending_model_count == 0:
                                logger.debug(
                                    "All models registered, stopping registry refresh"
                                )

                    except Exception as e:
                        logger.warning(f"Error refreshing registry: {e}")
                        # On error, increase interval
                        registry_refresh_interval = min(
                            registry_refresh_interval * 2, max_refresh_interval
                        )
                        refresh_attempts += 1

                # Use the cached (possibly updated) registry table
                layout["backends"].update(
                    create_panel(
                        cached_registry_table,
                        title=tr(
                            "proxy_monitor.title_current_model_registry",
                            "Current Model Registry",
                        ),
                        border_style="blue",
                    )
                )

                # GPU panel if enabled
                if show_gpu:
                    gpu_panel = create_gpu_status_panel()
                    layout["gpu"].update(gpu_panel)

                # Display proxy server logs dynamically
                # Calculate available height for logs
                current_height = console.height
                # Calculate fixed elements
                fixed_height = (
                    3 + 6 + backends_size + 1
                )  # header + proxy_info + backends + footer
                if show_gpu:
                    fixed_height += gpu_panel_size

                # Available lines for logs
                available_lines = max(20, current_height - fixed_height - 2)

                # Get recent logs from proxy process if available
                recent_logs = []
                if proxy_manager.proxy_process and hasattr(
                    proxy_manager.proxy_process, "get_recent_logs"
                ):
                    recent_logs = proxy_manager.proxy_process.get_recent_logs(
                        available_lines
                    )

                if recent_logs:
                    log_text = Text("\n".join(recent_logs), style="dim white")
                else:
                    log_text = Text(
                        tr(
                            "proxy_monitor.waiting_for_proxy_logs",
                            "Waiting for proxy server logs...\n",
                        ),
                        style="dim yellow",
                    )
                    log_text.append(
                        tr(
                            "proxy_monitor.request_logs_note",
                            "\nNote: Request logs will appear here when the proxy receives requests.",
                        ),
                        style="dim",
                    )

                logs_content = Group(
                    Rule(
                        tr(
                            "proxy_monitor.rule_proxy_server_logs",
                            "Proxy Server Logs",
                        ),
                        style="yellow",
                    ),
                    Padding(log_text, (1, 2)),
                )
                layout["logs"].update(logs_content)

                time.sleep(0.5)

    except KeyboardInterrupt:
        pass

    console.print(
        tr(
            "proxy_monitor.monitoring_stopped",
            "\n[yellow]Monitoring stopped.[/yellow]",
        )
    )

    try:
        input(
            tr(
                "proxy_monitor.press_enter_return",
                "\nPress Enter to return to monitoring menu...",
            )
        )
        return "back"
    except KeyboardInterrupt:
        # User wants to stop proxy
        stop_confirm = prompt_choice(
            "confirm_stop_proxy",
            tr(
                "proxy_monitor.stop_all_servers_confirm",
                "Stop all proxy servers?",
            ),
            [
                (
                    "yes",
                    tr(
                        "proxy_monitor.yes_stop_all_servers",
                        "Yes, stop all servers",
                    ),
                ),
                ("no", tr("proxy_monitor.no_keep_running", "No, keep running")),
            ],
            allow_back=False,
        )
        if stop_confirm == "yes":
            return "stop"
        return "back"
