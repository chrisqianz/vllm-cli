#!/usr/bin/env python3
"""
Proxy menu module for vLLM CLI.

Handles all proxy-related menu functionality.
"""

import logging
import threading
import time

import inquirer

from ...i18n import tr
from ...proxy import ProxyManager
from ...proxy.config import ProxyConfigManager
from ..common import console
from ..navigation import prompt_choice, unified_prompt
from .control import configure_model_for_proxy, configure_proxy_interactively
from .monitor import (
    monitor_individual_model_by_name,
    monitor_model_logs_menu,
    monitor_priority_group,
    monitor_proxy_logs,
    monitor_startup_progress,
    refresh_model_registry,
)

logger = logging.getLogger(__name__)

# Module-level variables to track active proxy
_active_proxy_manager = None
_active_proxy_config = None


def get_active_proxy():
    """Get the currently active proxy manager and config."""
    return _active_proxy_manager, _active_proxy_config


def manage_models_menu(proxy_manager, proxy_config, i18n_manager=None) -> None:
    """
    Model management submenu for adding new models or managing existing ones.

    Args:
        proxy_manager: The ProxyManager instance
        proxy_config: The ProxyConfig instance
        i18n_manager: Optional I18nManager for translations
    """
    def t(key: str, default: str = None, **kwargs) -> str:
        if i18n_manager:
            result = i18n_manager.t(key, **kwargs)
            return result if result != key else (default if default else key)
        return default if default else key

    while True:
        console.print(f"\n[bold cyan]{t('proxy.model_management', 'Model Management')}[/bold cyan]\n")

        options = [
            t("proxy.add_new_model", "Add New Model"),
            t("proxy.manage_existing_models", "Manage Existing Models"),
            t("proxy.back_to_proxy_menu", "← Back to proxy menu"),
        ]

        choice = unified_prompt(
            "model_management", t("proxy.select_action", "Select action"), options, allow_back=False
        )

        if choice == "BACK" or choice == t("proxy.back_to_proxy_menu", "← Back to proxy menu"):
            return

        if choice == t("proxy.add_new_model", "Add New Model"):
            add_new_model_to_proxy(proxy_manager, proxy_config, i18n_manager)
        elif choice == t("proxy.manage_existing_models", "Manage Existing Models"):
            manage_existing_models(proxy_manager, proxy_config, i18n_manager)


def add_new_model_to_proxy(proxy_manager, proxy_config, i18n_manager=None) -> None:
    """
    Add a new model to the running proxy using the unified configuration UI.

    Args:
        proxy_manager: The ProxyManager instance
        proxy_config: The ProxyConfig instance
        i18n_manager: Optional I18nManager for translations
    """
    def t(key: str, default: str = None, **kwargs) -> str:
        if i18n_manager:
            result = i18n_manager.t(key, **kwargs)
            return result if result != key else (default if default else key)
        return default if default else key

    # Use the unified function for consistent UI experience
    new_model = configure_model_for_proxy(
        index=len(proxy_config.models),
        existing_models=proxy_config.models,
        is_running_proxy=True,
        proxy_manager=proxy_manager,
    )

    if not new_model:
        return

    # Check if we need to handle port conflicts with stopped models
    # (The unified function only checks for conflicts, doesn't handle reuse)
    for existing_model in proxy_config.models[:]:
        if existing_model.port == new_model.port:
            # The unified function should have prevented this for running models
            # This handles the case of reusing a stopped model's port
            is_running = (
                existing_model.name in proxy_manager.vllm_servers
                and proxy_manager.vllm_servers[existing_model.name].is_running()
            )
            if not is_running:
                console.print(
                    tr(
                        "proxy_menu2.removing_stopped_model",
                        "[yellow]Removing stopped model '{name}' from port {port}[/yellow]",
                        name=existing_model.name,
                        port=existing_model.port,
                    )
                )
                proxy_config.models.remove(existing_model)

    # Check GPU memory and show warnings
    if new_model.gpu_ids:
        from ..gpu_utils import check_gpu_memory_warnings

        warnings = check_gpu_memory_warnings(new_model.gpu_ids)
        if warnings:
            console.print(f"\n[yellow]{t('proxy.gpu_memory_warning', 'GPU Memory Warnings')}:[/yellow]")
            for warning in warnings:
                console.print(warning)
            console.print(
                f"\n[yellow]{t('proxy.oom_warning', 'Starting additional models may cause Out-Of-Memory errors.')}[/yellow]"
            )

            # Ask user if they want to continue
            if not inquirer.confirm(t("proxy.continue_anyway", "Continue anyway?"), default=False):
                return

    # Add to proxy config
    proxy_config.models.append(new_model)

    # Start the model
    console.print(
        f"\n[cyan]{t('proxy.starting_model', 'Starting')} {new_model.name} on port {new_model.port}...[/cyan]"
    )
    if proxy_manager.start_model(new_model):
        console.print(f"[green]✓ {t('proxy.model_started', 'Model process started')}[/green]")

        # Start registration in background thread
        registration_thread = threading.Thread(
            target=proxy_manager.wait_and_register_model, args=(new_model,), daemon=True
        )
        registration_thread.start()

        # Immediately show monitoring - user sees logs right away!
        console.print(f"\n[cyan]{t('proxy.monitoring_startup', 'Monitoring')} {new_model.name} {t('proxy.starting_model', 'startup')}...[/cyan]\n")

        # Monitor shows real-time logs during startup
        result = monitor_individual_model_by_name(proxy_manager, new_model.name)

        # Ensure registration thread completes
        registration_thread.join(timeout=1.0)

        return result
    else:
        console.print(
            tr(
                "proxy_menu2.failed_to_start_model",
                "[red]✗ Failed to start model {name}[/red]",
                name=new_model.name,
            )
        )

        # Offer to view logs if server was created
        if new_model.name in proxy_manager.vllm_servers:
            server = proxy_manager.vllm_servers[new_model.name]

            # Show last few log lines
            recent_logs = server.get_recent_logs(5)
            if recent_logs:
                console.print(
                    tr("proxy_menu2.last_logs_header", "\n[bold]Last logs:[/bold]")
                )
                for log in recent_logs:
                    console.print(f"  {log}")

            # Offer to view full logs
            view_logs = (
                input(
                    tr(
                        "proxy_menu2.view_full_logs_for",
                        "\nView full logs for {name}? (y/N): ",
                        name=new_model.name,
                    )
                )
                .strip()
                .lower()
            )
            if view_logs in ["y", "yes"]:
                from ..log_viewer import show_log_menu

                show_log_menu(server)
            else:
                if server.log_path:
                    console.print(
                        tr(
                            "proxy_menu2.log_file",
                            "[dim]Log file: {path}[/dim]",
                            path=server.log_path,
                        )
                    )
                input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")

        # Remove from config if failed
        proxy_config.models.remove(new_model)


def manage_existing_models(proxy_manager, proxy_config, i18n_manager=None) -> None:
    """
    Manage existing models with sleep/stop/start options.

    Args:
        proxy_manager: The ProxyManager instance
        proxy_config: The ProxyConfig instance
        i18n_manager: Optional I18nManager for translations
    """
    def t(key: str, default: str = None, **kwargs) -> str:
        if i18n_manager:
            result = i18n_manager.t(key, **kwargs)
            return result if result != key else (default if default else key)
        return default if default else key

    console.print(f"\n[bold cyan]{t('proxy.manage_existing_models_title', 'Manage Existing Models')}[/bold cyan]\n")

    if not proxy_config.models:
        console.print(f"[yellow]{t('proxy.no_models_configured', 'No models configured')}[/yellow]")
        input(f"\n{t('common.press_enter', 'Press Enter to continue...')}")
        return

    # Build current state with sleep status
    model_states = []
    for model in proxy_config.models:
        if model.enabled:
            # Check if model is running
            is_running = (
                model.name in proxy_manager.vllm_servers
                and proxy_manager.vllm_servers[model.name].is_running()
            )
            is_sleeping = False

            # Get sleep state from registry if proxy is running
            if (
                is_running
                and proxy_manager.proxy_process
                and proxy_manager.proxy_process.is_running()
            ):
                registry_status = proxy_manager.get_proxy_registry_status()
                if registry_status:
                    models_info = registry_status.get("models", [])
                    for model_info in models_info:
                        if model_info.get("port") == model.port:
                            is_sleeping = model_info.get("state") == "sleeping"
                            break

            # Build status label with emoji
            if is_running and not is_sleeping:
                status = t("proxy.running_status", "[●] Running")
            elif is_sleeping:
                status = t("proxy.sleeping_status", "[z] Sleeping")
            else:
                status = t("proxy.stopped_status", "[○] Stopped")

            label = f"{model.name} (Port: {model.port})"
            if model.gpu_ids:
                label += f" GPU: {','.join(map(str, model.gpu_ids))}"
            label += f" - {status}"

            model_states.append(
                {
                    "label": label,
                    "model": model,
                    "running": is_running,
                    "sleeping": is_sleeping,
                }
            )

    if not model_states:
        console.print(f"[yellow]{t('proxy.no_enabled_models', 'No enabled models found')}[/yellow]")
        input(f"\n{t('common.press_enter', 'Press Enter to continue...')}")
        return

    # Let user select a model to manage
    model_choices = [state["label"] for state in model_states]
    model_choices.append(t("messages.back", "← Back"))

    selected = unified_prompt(
        "select_model", t("proxy.select_model_manage", "Select a model to manage"), model_choices, allow_back=False
    )

    if selected == t("messages.back", "← Back") or not selected:
        return

    # Find the selected model
    selected_state = None
    for state in model_states:
        if state["label"] == selected:
            selected_state = state
            break

    if not selected_state:
        return

    model = selected_state["model"]
    is_running = selected_state["running"]
    is_sleeping = selected_state["sleeping"]

    # Show action options based on model state
    actions = []
    if is_running and not is_sleeping:
        actions = [
            (
                "sleep",
                tr(
                    "proxy_menu2.action_sleep",
                    "[z] Put to sleep (free GPU, keep port)",
                ),
            ),
            (
                "stop",
                tr(
                    "proxy_menu2.action_stop",
                    "[■] Stop completely (free GPU and port)",
                ),
            ),
            ("restart", tr("proxy_menu2.action_restart", "↻ Restart model")),
        ]
    elif is_sleeping:
        actions = [
            ("wake", tr("proxy_menu2.action_wake", "[!] Wake up model")),
            (
                "stop",
                tr(
                    "proxy_menu2.action_stop_port",
                    "[■] Stop completely (free port)",
                ),
            ),
        ]
    else:  # Stopped
        actions = [
            ("start", tr("proxy_menu2.action_start", "► Start model")),
            ("remove", tr("proxy_menu2.action_remove", "[×] Remove model")),
        ]

    action = prompt_choice(
        "model_action",
        tr(
            "proxy_menu2.action_for_model", "Action for {name}", name=model.name
        ),
        actions,
        allow_back=True,
    )

    if action == "BACK" or not action:
        return manage_existing_models(
            proxy_manager, proxy_config
        )  # Go back to model list

    # Process the selected action
    if action == "sleep":
        console.print(
            tr(
                "proxy_menu2.putting_to_sleep",
                "\n[cyan]Putting {name} to sleep...[/cyan]",
                name=model.name,
            )
        )
        console.print(
            tr(
                "proxy_menu2.operation_may_take_minutes",
                "[dim]This may take several minutes for large models[/dim]",
            )
        )

        # Initiate sleep operation (returns immediately)
        if proxy_manager.sleep_model(model.name):
            console.print(
                tr(
                    "proxy_menu2.sleep_command_sent",
                    "[yellow]Sleep command sent. Monitoring progress...[/yellow]\n",
                )
            )

            # Monitor the model logs - handles completion detection and notifications
            return monitor_individual_model_by_name(proxy_manager, model.name)
        else:
            console.print(
                tr(
                    "proxy_menu2.failed_to_sleep",
                    "[red]✗ Failed to initiate sleep for {name}[/red]",
                    name=model.name,
                )
            )

    elif action == "wake":
        console.print(
            tr(
                "proxy_menu2.waking_up",
                "\n[cyan]Waking up {name}...[/cyan]",
                name=model.name,
            )
        )
        console.print(
            tr(
                "proxy_menu2.operation_may_take_minutes",
                "[dim]This may take several minutes for large models[/dim]",
            )
        )

        # Initiate wake operation (returns immediately)
        if proxy_manager.wake_model(model.name):
            console.print(
                tr(
                    "proxy_menu2.wake_command_sent",
                    "[yellow]Wake command sent. Monitoring progress...[/yellow]\n",
                )
            )

            # Monitor the model logs - handles completion detection and notifications
            return monitor_individual_model_by_name(proxy_manager, model.name)
        else:
            console.print(
                tr(
                    "proxy_menu2.failed_to_wake",
                    "[red]✗ Failed to initiate wake for {name}[/red]",
                    name=model.name,
                )
            )

    elif action == "stop":
        console.print(
            tr(
                "proxy_menu2.stopping_model",
                "\n[yellow]Stopping {name}...[/yellow]",
                name=model.name,
            )
        )
        if proxy_manager.stop_model(model.name):
            console.print(
                tr(
                    "proxy_menu2.model_stopped_free_port",
                    "[green]✓ {name} stopped (port {port} is now free)[/green]",
                    name=model.name,
                    port=model.port,
                )
            )
            # Remove from configuration
            proxy_config.models = [
                m for m in proxy_config.models if m.name != model.name
            ]
        else:
            console.print(
                tr(
                    "proxy_menu2.failed_to_stop",
                    "[red]✗ Failed to stop {name}[/red]",
                    name=model.name,
                )
            )

    elif action == "start":
        console.print(
            tr(
                "proxy_menu2.starting_model",
                "\n[cyan]Starting {name}...[/cyan]",
                name=model.name,
            )
        )
        if proxy_manager.start_model(model):
            console.print(
                tr(
                    "proxy_menu2.model_process_started",
                    "[green]✓ Model process started[/green]",
                )
            )

            # Background registration
            registration_thread = threading.Thread(
                target=proxy_manager.wait_and_register_model, args=(model,), daemon=True
            )
            registration_thread.start()

            # Immediate monitoring
            console.print(
                tr(
                    "proxy_menu2.monitoring_startup",
                    "\n[cyan]Monitoring {name} startup...[/cyan]\n",
                    name=model.name,
                )
            )

            return monitor_individual_model_by_name(proxy_manager, model.name)
        else:
            console.print(
                tr(
                    "proxy_menu2.failed_to_start",
                    "[red]✗ Failed to start {name}[/red]",
                    name=model.name,
                )
            )

            # Offer to view logs if server was created
            if model.name in proxy_manager.vllm_servers:
                server = proxy_manager.vllm_servers[model.name]

                # Show last few log lines
                recent_logs = server.get_recent_logs(5)
                if recent_logs:
                    console.print(
                        tr(
                            "proxy_menu2.last_logs_header", "\n[bold]Last logs:[/bold]"
                        )
                    )
                    for log in recent_logs:
                        console.print(f"  {log}")

                # Offer to view full logs
                view_logs = (
                    input(
                        tr(
                            "proxy_menu2.view_full_logs",
                            "\nView full logs? (y/N): ",
                        )
                    )
                    .strip()
                    .lower()
                )
                if view_logs in ["y", "yes"]:
                    from ..log_viewer import show_log_menu

                    show_log_menu(server)
                    return manage_existing_models(proxy_manager, proxy_config)

    elif action == "restart":
        console.print(
            tr(
                "proxy_menu2.restarting_model",
                "\n[cyan]Restarting {name}...[/cyan]",
                name=model.name,
            )
        )
        proxy_manager.stop_model(model.name)
        console.print(
            tr("proxy_menu2.waiting_for_cleanup", "[dim]Waiting for cleanup...[/dim]")
        )
        time.sleep(2)

        if proxy_manager.start_model(model):
            console.print(
                tr(
                    "proxy_menu2.model_process_restarted",
                    "[green]✓ Model process restarted[/green]",
                )
            )

            # Background registration
            registration_thread = threading.Thread(
                target=proxy_manager.wait_and_register_model, args=(model,), daemon=True
            )
            registration_thread.start()

            # Immediate monitoring
            console.print(
                tr(
                    "proxy_menu2.monitoring_restart",
                    "\n[cyan]Monitoring {name} restart...[/cyan]\n",
                    name=model.name,
                )
            )

            return monitor_individual_model_by_name(proxy_manager, model.name)
        else:
            console.print(
                tr(
                    "proxy_menu2.failed_to_restart",
                    "[red]✗ Failed to restart {name}[/red]",
                    name=model.name,
                )
            )

            # Offer to view logs if server was created
            if model.name in proxy_manager.vllm_servers:
                server = proxy_manager.vllm_servers[model.name]

                # Show last few log lines
                recent_logs = server.get_recent_logs(5)
                if recent_logs:
                    console.print(
                        tr(
                            "proxy_menu2.last_logs_header", "\n[bold]Last logs:[/bold]"
                        )
                    )
                    for log in recent_logs:
                        console.print(f"  {log}")

                # Offer to view full logs
                view_logs = (
                    input(
                        tr(
                            "proxy_menu2.view_full_logs",
                            "\nView full logs? (y/N): ",
                        )
                    )
                    .strip()
                    .lower()
                )
                if view_logs in ["y", "yes"]:
                    from ..log_viewer import show_log_menu

                    show_log_menu(server)
                    return manage_existing_models(proxy_manager, proxy_config)

    elif action == "remove":
        if inquirer.confirm(
            tr(
                "proxy_menu2.remove_from_proxy_confirm",
                "\nRemove {name} from proxy?",
                name=model.name,
            ),
            default=False,
        ):
            # Unregister from proxy registry to prevent stale entries
            proxy_manager.unregister_model_from_proxy(model.port)

            # Remove from configuration
            proxy_config.models = [
                m for m in proxy_config.models if m.name != model.name
            ]
            console.print(
                tr(
                    "proxy_menu2.removed_from_proxy",
                    "[green]✓ {name} removed from proxy[/green]",
                    name=model.name,
                )
            )
            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            return

    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
    # After any action, go back to the model list
    return manage_existing_models(proxy_manager, proxy_config)


def manage_running_proxy(proxy_manager, proxy_config, i18n_manager=None) -> None:
    """
    Simplified proxy management interface focusing on monitoring.

    Args:
        proxy_manager: The ProxyManager instance
        proxy_config: The proxy configuration
        i18n_manager: Optional I18nManager for translations
    """
    def t(key: str, default: str = None, **kwargs) -> str:
        if i18n_manager:
            result = i18n_manager.t(key, **kwargs)
            return result if result != key else (default if default else key)
        return default if default else key

    # Track the active proxy globally
    global _active_proxy_manager, _active_proxy_config
    _active_proxy_manager = proxy_manager
    _active_proxy_config = proxy_config

    time.sleep(2)

    while True:
        console.print(f"\n[bold cyan]{t('proxy.proxy_running', 'Proxy Server Running')}[/bold cyan]")
        console.print(
            tr(
                "proxy_menu2.access_at",
                "Access at: {url}",
                url=f"http://{proxy_config.host}:{proxy_config.port}",
            )
        )
        console.print(
            tr(
                "proxy_menu2.ctrl_c_hint",
                "[dim]Use Ctrl+C in monitoring views to return here[/dim]\n",
            )
        )

        opt_proxy_logs = t("proxy.monitor_proxy_logs", "Monitor proxy logs")
        opt_model_logs = t("proxy.monitor_model_logs", "Monitor model logs")
        opt_manage_models = t("proxy.manage_models", "Manage Models")
        opt_refresh_registry = t("proxy.refresh_registry", "Refresh Model Registry")
        opt_stop_all = t("proxy.stop_all_servers", "Stop all servers")

        management_options = [
            opt_proxy_logs,
            opt_model_logs,
            opt_manage_models,
            opt_refresh_registry,
            opt_stop_all,
        ]

        mgmt_choice = unified_prompt(
            "proxy_monitoring",
            t("proxy.select_action", "Select action"),
            management_options,
            allow_back=True,
        )

        if mgmt_choice == "BACK":
            # Exit to main menu, proxy continues running
            console.print(f"\n[green]✓ {t('proxy.proxy_running_background', 'Proxy continues running in background')}[/green]")
            console.print(
                tr(
                    "proxy_menu2.access_at",
                    "Access at: {url}",
                    url=f"http://{proxy_config.host}:{proxy_config.port}",
                )
            )
            console.print(
                tr(
                    "proxy_menu2.manage_later_hint",
                    "[dim]Return to Multi-Model Proxy menu to manage it later[/dim]",
                )
            )
            time.sleep(2)
            break

        if mgmt_choice is None:
            # Prompt cancelled (Ctrl+C / EOF): leave the menu like BACK instead
            # of spinning forever on a repeated None selection.
            break

        if mgmt_choice == opt_proxy_logs:
            result = monitor_proxy_logs(proxy_manager)
            if result == "stop":
                # User requested to stop proxy
                console.print(
                    tr(
                        "proxy_menu2.stopping_proxy_and_models",
                        "\n[yellow]Stopping proxy server and all models...[/yellow]",
                    )
                )
                proxy_manager.stop_proxy()
                console.print(
                    tr(
                        "proxy_menu2.all_servers_stopped",
                        "[green]✓ All servers stopped[/green]",
                    )
                )

                # Clear the global tracking variables
                _active_proxy_manager = None
                _active_proxy_config = None
                time.sleep(1)
                break
        elif mgmt_choice == opt_model_logs:
            result = monitor_model_logs_menu(proxy_manager)
            if result == "stop":
                # User requested to stop proxy
                console.print(
                    tr(
                        "proxy_menu2.stopping_proxy_and_models",
                        "\n[yellow]Stopping proxy server and all models...[/yellow]",
                    )
                )
                proxy_manager.stop_proxy()
                console.print(
                    tr(
                        "proxy_menu2.all_servers_stopped",
                        "[green]✓ All servers stopped[/green]",
                    )
                )

                # Clear the global tracking variables
                _active_proxy_manager = None
                _active_proxy_config = None
                time.sleep(1)
                break
        elif mgmt_choice == opt_manage_models:
            manage_models_menu(proxy_manager, proxy_config, i18n_manager)
            # Continue to show menu after managing models
        elif mgmt_choice == opt_refresh_registry:
            refresh_model_registry(proxy_manager)
            # Continue to show menu after refresh
        elif mgmt_choice == opt_stop_all:
            # Confirm before stopping
            stop_confirm = prompt_choice(
                "confirm_stop_servers",
                tr(
                    "proxy_menu2.stop_all_servers_confirm",
                    "Stop all proxy servers?",
                ),
                [
                    (
                        "yes",
                        tr(
                            "proxy_menu2.yes_stop_all_servers",
                            "Yes, stop all servers",
                        ),
                    ),
                    (
                        "no",
                        tr("proxy_menu2.no_keep_running", "No, keep running"),
                    ),
                ],
                allow_back=False,
            )
            if stop_confirm == "yes":
                console.print(
                    tr(
                        "proxy_menu2.stopping_proxy_and_models",
                        "\n[yellow]Stopping proxy server and all models...[/yellow]",
                    )
                )
                proxy_manager.stop_proxy()
                console.print(
                    tr(
                        "proxy_menu2.all_servers_stopped",
                        "[green]✓ All servers stopped[/green]",
                    )
                )

                # Clear the global tracking variables
                _active_proxy_manager = None
                _active_proxy_config = None

                time.sleep(1)
                break
            # If user cancels, continue to show menu


def handle_multi_model_proxy(i18n_manager=None) -> str:
    """
    Handle multi-model proxy configuration and management.

    Args:
        i18n_manager: Optional I18nManager for translations
    """

    # Helper function for translation
    def t(key: str, default: str = None, **kwargs) -> str:
        if i18n_manager:
            result = i18n_manager.t(key, **kwargs)
            return result if result != key else (default if default else key)
        return default if default else key

    console.print(
        f"\n[bold cyan]{t('menu.proxy.title', 'Multi-Model Proxy Server')}[/bold cyan]"
    )

    # Check if there are saved configurations
    config_manager = ProxyConfigManager()
    saved_configs = config_manager.list_saved_configs()

    options = []

    # Add saved config option if configs exist
    if saved_configs:
        options.append(t("proxy.start_saved_config", "Start from saved configuration"))

    # Always show configure new proxy option
    options.append(t("proxy.configure_new_proxy", "Configure new proxy"))

    choice = unified_prompt("proxy_menu", t("proxy.select_action", "Select action"), options, allow_back=True)

    if choice == "BACK":
        return "continue"

    if choice == t("proxy.start_saved_config", "Start from saved configuration"):
        # Quick start from saved configuration
        if len(saved_configs) == 1:
            # Only one config, start it directly
            config_name = list(saved_configs.keys())[0]
            console.print(
                tr(
                    "proxy_menu2.starting_saved_config",
                    "\n[cyan]Starting saved configuration: '{name}'[/cyan]",
                    name=config_name,
                )
            )
            proxy_config = config_manager.load_named_config(config_name)
        else:
            # Multiple configs, let user choose
            console.print(
                tr(
                    "proxy_menu2.select_config_to_start",
                    "\n[bold]Select configuration to start:[/bold]",
                )
            )
            cancel_label = tr("messages.cancel", "Cancel")
            config_choices = []
            for name, info in saved_configs.items():
                models_str = tr(
                    "proxy_menu2.models_count",
                    "{count} model(s)",
                    count=info["models"],
                )
                preview = ", ".join(info["model_names"][:2])
                if len(info["model_names"]) > 2:
                    preview += ", ..."
                config_choices.append(f"{name} ({models_str}: {preview})")
            config_choices.append(cancel_label)

            selected = unified_prompt(
                "select_config",
                tr("proxy_menu2.choose_configuration", "Choose configuration"),
                config_choices,
                allow_back=False,
            )

            if selected == cancel_label:
                return "continue"

            config_name = selected.split(" (")[0]
            proxy_config = config_manager.load_named_config(config_name)

        if proxy_config:
            # Display configuration summary
            console.print(
                tr(
                    "proxy_menu2.configuration_summary_header",
                    "\n[bold]Configuration Summary:[/bold]",
                )
            )
            console.print(
                tr(
                    "proxy_menu2.summary_port", "Port: {port}", port=proxy_config.port
                )
            )
            console.print(
                tr(
                    "proxy_menu2.summary_models",
                    "Models: {count}",
                    count=len(proxy_config.models),
                )
            )
            for model in proxy_config.models:
                gpu_str = (
                    ",".join(str(g) for g in model.gpu_ids)
                    if model.gpu_ids
                    else tr("proxy_wizard.auto", "Auto")
                )
                console.print(
                    tr(
                        "proxy_menu2.model_summary_line",
                        "  • {name} (Port {port}, GPU {gpu})",
                        name=model.name,
                        port=model.port,
                        gpu=gpu_str,
                    )
                )

            # Confirm start - use inquirer.confirm for inline prompt
            console.print()  # Add blank line before prompt
            if inquirer.confirm(
                tr("proxy.start_config_confirm", "Start this proxy configuration?"),
                default=True,
            ):
                # Start the proxy
                proxy_manager = ProxyManager(proxy_config)

                # Check if any models have loading priorities
                has_priorities = any(
                    m.loading_priority is not None
                    for m in proxy_config.models
                    if m.enabled
                )

                if has_priorities:
                    # Use sequential loading with priorities
                    console.print(
                        tr(
                            "proxy_menu2.starting_sequential_loading",
                            "\n[cyan]Starting models with sequential loading...[/cyan]",
                        )
                    )
                    console.print(
                        tr(
                            "proxy_menu2.priority_order_hint",
                            "[dim]Models will load in priority order to ensure proper GPU memory allocation[/dim]\n",
                        )
                    )

                    # Track results across all groups
                    total_started = 0
                    total_failed = []

                    # Iterate through priority groups
                    for group_info in proxy_manager.start_models_with_priorities():
                        # Monitor this priority group with live display
                        success = monitor_priority_group(
                            proxy_manager,
                            group_info["started_models"],
                            group_info["priority_label"],
                            group_info["group_index"],
                            group_info["total_groups"],
                        )

                        # Track results
                        total_started += len(group_info["started_models"])
                        total_failed.extend(group_info["failed_models"])

                        # Stop if this group failed
                        if not success:
                            console.print(
                                tr(
                                    "proxy_menu2.priority_group_failed",
                                    "\n[red]Priority group {label} failed to start[/red]",
                                    label=group_info["priority_label"],
                                )
                            )
                            if not inquirer.confirm(
                                tr(
                                    "proxy_menu2.continue_remaining_groups",
                                    "Continue with remaining groups?",
                                ),
                                default=False,
                            ):
                                console.print(
                                    tr(
                                        "proxy_menu2.stopping_all_servers",
                                        "[yellow]Stopping all servers...[/yellow]",
                                    )
                                )
                                proxy_manager.stop_proxy()
                                return "continue"

                    # Check overall results
                    if total_failed:
                        console.print(
                            tr(
                                "proxy_menu2.models_failed_to_start",
                                "\n[yellow]Warning: {count} model(s) failed to start:[/yellow]",
                                count=len(total_failed),
                            )
                        )
                        for model_name in total_failed:
                            console.print(f"  • {model_name}")

                        if total_started == 0:
                            console.print(
                                tr(
                                    "proxy_menu2.no_models_started",
                                    "[red]No models started successfully. Cannot start proxy.[/red]",
                                )
                            )
                            input(
                                f"\n{tr('common.press_enter', 'Press Enter to continue...')}"
                            )
                            return "continue"

                        if not inquirer.confirm(
                            tr(
                                "proxy_menu2.continue_with_available_count",
                                "Continue with {count} available model(s)?",
                                count=total_started,
                            ),
                            default=False,
                        ):
                            console.print(
                                tr(
                                    "proxy_menu2.stopping_all_servers",
                                    "[yellow]Stopping all servers...[/yellow]",
                                )
                            )
                            proxy_manager.stop_proxy()
                            return "continue"
                else:
                    # Use parallel loading (original behavior)
                    console.print(
                        tr(
                            "proxy_menu2.launching_model_servers",
                            "\n[cyan]Launching model servers...[/cyan]",
                        )
                    )
                    launched = proxy_manager.start_all_models_no_wait()

                    if launched > 0:
                        # Monitor startup progress with live logs
                        all_started = monitor_startup_progress(proxy_manager)

                        if not all_started:
                            console.print(
                                tr(
                                    "proxy_menu2.some_models_failed",
                                    "[yellow]Some models failed to start.[/yellow]",
                                )
                            )
                            if not inquirer.confirm(
                                tr(
                                    "proxy_menu2.continue_with_available",
                                    "Continue with available models?",
                                ),
                                default=False,
                            ):
                                console.print(
                                    tr(
                                        "proxy_menu2.stopping_all_servers",
                                        "[yellow]Stopping all servers...[/yellow]",
                                    )
                                )
                                proxy_manager.stop_proxy()
                                return "continue"

                if proxy_manager.start_proxy():
                    console.print(
                        tr(
                            "proxy_menu2.proxy_running_at",
                            "\n[green]✓ Proxy server running at {url}[/green]",
                            url=f"http://{proxy_config.host}:{proxy_config.port}",
                        )
                    )

                    # Enter simplified proxy management
                    manage_running_proxy(proxy_manager, proxy_config)

    elif choice == t("proxy.configure_new_proxy", "Configure new proxy"):
        proxy_config = configure_proxy_interactively()
        if not proxy_config:
            return "continue"

        # Validate configuration
        config_manager = ProxyConfigManager()
        errors = config_manager.validate_config(proxy_config)
        if errors:
            console.print(
                tr(
                    "proxy_menu2.configuration_errors",
                    "[red]Configuration errors:[/red]",
                )
            )
            for error in errors:
                console.print(f"  • {error}")
            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            return "continue"

        # Ask what to do with the configuration
        console.print(
            tr(
                "proxy_menu2.configuration_complete",
                "\n[bold cyan]Configuration Complete[/bold cyan]",
            )
        )
        action_options = [
            ("save_start", tr("proxy_menu2.action_save_start", "Save and start now")),
            ("save_later", tr("proxy_menu2.action_save_later", "Save for later use")),
            (
                "start_no_save",
                tr("proxy_menu2.action_start_without_saving", "Start without saving"),
            ),
            ("cancel", tr("messages.cancel", "Cancel")),
        ]

        action = prompt_choice(
            "config_action",
            tr(
                "proxy_menu2.what_with_configuration",
                "What would you like to do with this configuration?",
            ),
            action_options,
            allow_back=False,
        )

        if action == "cancel":
            return "continue"

        # Handle saving if requested
        if action in ["save_start", "save_later"]:
            console.print(
                tr(
                    "proxy_menu2.enter_config_name",
                    "\nEnter a name for this configuration:",
                )
            )
            config_name = (
                input(
                    tr(
                        "proxy_menu2.config_name_prompt",
                        "Name (default: 'default'): ",
                    )
                )
                .strip()
                or "default"
            )
            config_manager.save_named_config(proxy_config, config_name)
            console.print(
                tr(
                    "proxy_menu2.config_saved_as",
                    "[green]✓ Configuration saved as '{name}'[/green]",
                    name=config_name,
                )
            )

            if action == "save_later":
                console.print(
                    tr(
                        "proxy_menu2.start_later_hint",
                        "\n[dim]You can start this configuration later from the proxy menu.[/dim]",
                    )
                )
                input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
                return "continue"

        # Start the proxy if requested
        if action in ["save_start", "start_no_save"]:
            # Start proxy
            proxy_manager = ProxyManager(proxy_config)

            # Check if any models have loading priorities
            has_priorities = any(
                m.loading_priority is not None for m in proxy_config.models if m.enabled
            )

            if has_priorities:
                # Use sequential loading with priorities
                console.print(
                    tr(
                        "proxy_menu2.starting_sequential_loading",
                        "\n[cyan]Starting models with sequential loading...[/cyan]",
                    )
                )
                console.print(
                    tr(
                        "proxy_menu2.priority_order_hint",
                        "[dim]Models will load in priority order to ensure proper GPU memory allocation[/dim]\n",
                    )
                )

                # Track results across all groups
                total_started = 0
                total_failed = []

                # Iterate through priority groups
                for group_info in proxy_manager.start_models_with_priorities():
                    # Monitor this priority group with live display
                    success = monitor_priority_group(
                        proxy_manager,
                        group_info["started_models"],
                        group_info["priority_label"],
                        group_info["group_index"],
                        group_info["total_groups"],
                    )

                    # Track results
                    total_started += len(group_info["started_models"])
                    total_failed.extend(group_info["failed_models"])

                    # Stop if this group failed
                    if not success:
                        console.print(
                            tr(
                                "proxy_menu2.priority_group_failed",
                                "\n[red]Priority group {label} failed to start[/red]",
                                label=group_info["priority_label"],
                            )
                        )
                        if not inquirer.confirm(
                            tr(
                                "proxy_menu2.continue_remaining_groups",
                                "Continue with remaining groups?",
                            ),
                            default=False,
                        ):
                            console.print(
                                tr(
                                    "proxy_menu2.stopping_all_servers",
                                    "[yellow]Stopping all servers...[/yellow]",
                                )
                            )
                            proxy_manager.stop_proxy()
                            return "continue"

                # Check overall results
                if total_failed:
                    console.print(
                        tr(
                            "proxy_menu2.models_failed_to_start",
                            "\n[yellow]Warning: {count} model(s) failed to start:[/yellow]",
                            count=len(total_failed),
                        )
                    )
                    for model_name in total_failed:
                        console.print(f"  • {model_name}")

                    if total_started == 0:
                        console.print(
                            tr(
                                "proxy_menu2.no_models_started",
                                "[red]No models started successfully. Cannot start proxy.[/red]",
                            )
                        )
                        input(
                            f"\n{tr('common.press_enter', 'Press Enter to continue...')}"
                        )
                        return "continue"

                    if not inquirer.confirm(
                        tr(
                            "proxy_menu2.continue_with_available_count",
                            "Continue with {count} available model(s)?",
                            count=total_started,
                        ),
                        default=False,
                    ):
                        console.print(
                            tr(
                                "proxy_menu2.stopping_all_servers",
                                "[yellow]Stopping all servers...[/yellow]",
                            )
                        )
                        proxy_manager.stop_proxy()
                        return "continue"
            else:
                # Use parallel loading (original behavior)
                console.print(
                    tr(
                        "proxy_menu2.launching_model_servers",
                        "\n[cyan]Launching model servers...[/cyan]",
                    )
                )
                launched = proxy_manager.start_all_models_no_wait()

                if launched > 0:
                    # Monitor startup progress with live logs
                    all_started = monitor_startup_progress(proxy_manager)

                    if not all_started:
                        console.print(
                            tr(
                                "proxy_menu2.some_models_failed",
                                "[yellow]Some models failed to start.[/yellow]",
                            )
                        )
                        if not inquirer.confirm(
                            tr(
                                "proxy_menu2.continue_with_available",
                                "Continue with available models?",
                            ),
                            default=False,
                        ):
                            console.print(
                                tr(
                                    "proxy_menu2.stopping_all_servers",
                                    "[yellow]Stopping all servers...[/yellow]",
                                )
                            )
                            proxy_manager.stop_proxy()
                            return "continue"

            if proxy_manager.start_proxy():
                console.print(
                    tr(
                        "proxy_menu2.proxy_running_at",
                        "\n[green]✓ Proxy server running at {url}[/green]",
                        url=f"http://{proxy_config.host}:{proxy_config.port}",
                    )
                )

                # Enter simplified proxy management
                manage_running_proxy(proxy_manager, proxy_config)

    return "continue"
