#!/usr/bin/env python3
"""
Server monitoring module for vLLM CLI.

Handles monitoring of running vLLM servers and displaying their status.
"""
import logging
import time

import inquirer
from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ..config import ConfigManager
from ..i18n import tr
from ..server import VLLMServer, get_active_servers
from ..system import get_gpu_info
from .common import console, create_panel
from .gpu_utils import calculate_gpu_panel_size, create_gpu_status_panel
from .log_viewer import show_log_menu
from .navigation import prompt_choice

logger = logging.getLogger(__name__)


def monitor_server(server: VLLMServer) -> str:
    """
    Monitor a running vLLM server.
    """
    # Clear console for clean display
    console.clear()
    console.print(f"[bold cyan]{tr('srv_monitor.monitoring_title', 'Monitoring Server')}[/bold cyan]\n")

    # Get UI preferences
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()
    show_gpu = ui_prefs.get("show_gpu_in_monitor", True)
    monitor_refresh_rate = ui_prefs.get("monitor_refresh_rate", 1.0)

    try:
        # Get GPU info to calculate optimal panel size
        gpu_info = get_gpu_info() if show_gpu else None
        gpu_panel_size = (
            calculate_gpu_panel_size(len(gpu_info) if gpu_info else 0)
            if show_gpu
            else 0
        )

        # Create layout for monitoring
        layout = Layout()
        if show_gpu:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="status", size=5),
                Layout(
                    name="gpu", size=gpu_panel_size
                ),  # Dynamic size based on GPU count
                Layout(name="logs"),  # Takes remaining space
                Layout(name="footer", size=1),
            )
        else:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="status", size=6),
                Layout(name="logs"),
                Layout(name="footer", size=1),
            )

        # Header
        header_text = Text(
            tr(
                "srv_monitor.header_title",
                "vLLM Server Monitor - {model}",
                model=server.model,
            ),
            style="bold cyan",
            justify="center",
        )
        layout["header"].update(Padding(header_text, (1, 0)))

        # Footer with exit instructions
        layout["footer"].update(
            Align.center(
                Text(
                    tr(
                        "srv_monitor.footer_stop_monitoring",
                        "Press Ctrl+C to stop monitoring • Server continues in background",
                    ),
                    style="dim cyan",
                )
            )
        )

        with Live(layout, console=console, refresh_per_second=monitor_refresh_rate):
            while True:
                # Update status
                status_table = Table(show_header=False, box=None)
                status_table.add_column(tr("srv_monitor.col_key", "Key"), style="cyan")
                status_table.add_column(
                    tr("srv_monitor.col_value", "Value"), style="magenta"
                )

                status_table.add_row(
                    tr("srv_monitor.row_status", "Status"),
                    (
                        tr("srv_monitor.status_running", "[green]Running[/green]")
                        if server.is_running()
                        else tr("srv_monitor.status_stopped", "[red]Stopped[/red]")
                    ),
                )
                status_table.add_row(tr("srv_monitor.row_port", "Port"), str(server.port))
                status_table.add_row(
                    "PID",
                    str(server.process.pid) if server.process else "N/A",
                )
                status_table.add_row(
                    tr("srv_monitor.row_model", "Model"),
                    (
                        server.model[:40] + "..."
                        if len(server.model) > 40
                        else server.model
                    ),
                )

                # Update status
                layout["status"].update(
                    create_panel(
                        status_table,
                        title=tr("srv_monitor.panel_server_status", "Server Status"),
                        border_style="green" if server.is_running() else "red",
                    )
                )

                # Update GPU panel if enabled
                if show_gpu:
                    gpu_panel = create_gpu_status_panel()
                    layout["gpu"].update(gpu_panel)

                # Update logs with integrated divider
                monitor_log_lines = ui_prefs.get("log_lines_monitor", 50)
                recent_logs = server.get_recent_logs(monitor_log_lines)
                if recent_logs:
                    log_text = Text("\n".join(recent_logs), style="dim white")
                    logs_content = Group(
                        Rule(tr("srv_monitor.recent_logs_rule", "Recent Logs"), style="yellow"), Padding(log_text, (1, 2))
                    )
                else:
                    # Check if this is an external server
                    if not hasattr(server.process, "poll"):
                        log_text = Text(
                            tr(
                                "srv_monitor.external_server_logs",
                                "This server was started externally.\n"
                                "Logs are not available through vLLM CLI.\n\n"
                                "To view logs, check the terminal where the server was started,\n"
                                "or look for log files in the vLLM working directory.",
                            ),
                            style="dim yellow",
                        )
                    else:
                        log_text = Text(
                            tr("srv_monitor.waiting_for_logs", "Waiting for logs..."),
                            style="dim yellow",
                        )

                    logs_content = Group(
                        Rule(tr("srv_monitor.recent_logs_rule", "Recent Logs"), style="yellow"), Padding(log_text, (1, 2))
                    )

                layout["logs"].update(logs_content)

                # Check if server is still running
                if not server.is_running():
                    console.print(
                        f"\n[red]{tr('srv_monitor.server_stopped_unexpectedly', 'Server has stopped unexpectedly.')}[/red]"
                    )
                    break

                # Small sleep to prevent high CPU usage
                time.sleep(0.5)

    except KeyboardInterrupt:
        # Clean exit via Ctrl+C
        pass  # Exit the Live context cleanly

    # After exiting monitoring (via Ctrl+C)
    console.print(
        f"\n[yellow]{tr('srv_monitor.monitoring_stopped', 'Monitoring stopped.')}[/yellow]"
    )
    console.print(
        f"\n[green]{tr('srv_monitor.server_continues_background', '✓ Server continues running in background')}[/green]"
    )
    console.print(
        tr(
            "srv_monitor.server_endpoint",
            "[dim]Server endpoint: http://localhost:{port}[/dim]",
            port=server.port,
        )
    )
    console.print("")

    # Show options with navigation
    monitor_options = [
        ("resume", tr("srv_monitor.option_resume_monitoring", "Resume monitoring")),
        ("view_logs", tr("srv_monitor.option_view_server_logs", "View server logs")),
        ("stop", tr("srv_monitor.option_stop_server", "Stop server")),
        ("main_menu", tr("srv_monitor.option_return_main_menu", "Return to main menu")),
    ]

    choice = prompt_choice(
        "post_monitor",
        tr("srv_monitor.what_would_you_like", "What would you like to do?"),
        monitor_options,
        allow_back=False,
    )

    if choice == "resume":
        return monitor_server(server)
    elif choice == "view_logs":
        show_log_menu(server)
        return monitor_server(server)  # Return to monitoring after viewing logs
    elif choice == "stop":
        console.print(f"[yellow]{tr('srv_monitor.stopping_server', 'Stopping server...')}[/yellow]")
        server.stop()
        console.print(f"[green]{tr('srv_monitor.server_stopped', '✓ Server stopped.')}[/green]")
        input("\n" + tr("common.press_enter", "Press Enter to continue..."))
        return "continue"
    else:
        # Return to main menu
        console.print(
            f"[dim]{tr('srv_monitor.server_will_continue', 'Server will continue running.')}[/dim]"
        )
        console.print(
            f"[dim]{tr('srv_monitor.monitor_again_hint', 'You can monitor it again from the main menu.')}[/dim]"
        )
        # Log server status for debugging
        if server.is_running():
            logger.info(
                f"Server {server.model} on port {server.port} is still running after exiting monitor"
            )
        else:
            logger.warning(
                f"Server {server.model} on port {server.port} appears to have stopped"
            )
        input("\n" + tr("common.press_enter", "Press Enter to continue..."))
        return "continue"


def monitor_active_servers() -> str:
    """
    Monitor all active vLLM servers.
    """
    servers = get_active_servers()
    if not servers:
        console.print(f"[yellow]{tr('srv_monitor.no_active_servers', 'No active servers found.')}[/yellow]")
        input("\n" + tr("common.press_enter", "Press Enter to continue..."))
        return "continue"

    # Show active servers
    table = Table(
        title=tr(
            "srv_monitor.active_servers_title",
            "[bold green]Active vLLM Servers[/bold green]",
        ),
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column(tr("srv_monitor.col_model", "Model"), style="magenta")
    table.add_column(tr("srv_monitor.col_port", "Port"), style="yellow")
    table.add_column("PID", style="green")
    table.add_column(tr("srv_monitor.col_status", "Status"), style="white")

    for i, server in enumerate(servers, 1):
        status = (
            tr("srv_monitor.status_running", "[green]Running[/green]")
            if server.is_running()
            else tr("srv_monitor.status_stopped", "[red]Stopped[/red]")
        )
        table.add_row(
            str(i),
            server.model,
            str(server.port),
            str(server.process.pid) if server.process else "N/A",
            status,
        )

    console.print(table)

    # Select server to monitor or manage
    choices = [
        (
            f"monitor_{i}",
            tr("srv_monitor.monitor_server_number", "Monitor Server {number}", number=i),
        )
        for i in range(1, len(servers) + 1)
    ]
    choices.append(
        ("view_logs", tr("srv_monitor.view_server_logs_option", "View Server Logs"))
    )
    choices.append(
        ("stop_all", tr("srv_monitor.stop_all_servers_option", "Stop All Servers"))
    )

    action = prompt_choice(
        "server_action",
        tr("srv_monitor.server_management", "Server Management"),
        choices,
        allow_back=True,
    )

    if action == "BACK" or not action:
        return "continue"
    elif action == "stop_all":
        confirm = inquirer.confirm(
            tr(
                "srv_monitor.stop_all_confirm",
                "Stop all {count} servers?",
                count=len(servers),
            ),
            default=False,
        )
        if confirm:
            # Import here to avoid circular dependency
            from .proxy.menu import get_active_proxy

            # Check if there's an active proxy manager
            active_proxy_manager, _ = get_active_proxy()
            if active_proxy_manager:
                # Use proxy manager to stop all proxy-managed servers
                console.print(
                    f"[yellow]{tr('srv_monitor.stopping_proxy_and_servers', 'Stopping proxy and all managed servers...')}[/yellow]"
                )
                active_proxy_manager.stop_proxy()
            else:
                # No proxy manager, use the standard approach
                for server in servers:
                    server.stop()

            console.print(
                f"[green]{tr('srv_monitor.all_servers_stopped', 'All servers stopped.')}[/green]"
            )
            input("\n" + tr("common.press_enter", "Press Enter to continue..."))
    elif action.startswith("monitor_"):
        server_idx = int(action.rsplit("_", 1)[-1]) - 1
        return monitor_server(servers[server_idx])
    elif action == "view_logs":
        # Import here to avoid circular dependencies
        from .log_viewer import select_server_for_logs, show_log_menu

        server = select_server_for_logs()
        if server:
            show_log_menu(server)
        return monitor_active_servers()  # Return to this menu after viewing logs

    return "continue"
