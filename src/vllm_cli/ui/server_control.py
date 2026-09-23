#!/usr/bin/env python3
"""
Server control module for vLLM CLI.

Handles server configuration and startup operations.
"""

import logging
import time
from typing import Any, Dict

import inquirer
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.rule import Rule
from rich.text import Text

from ..config import ConfigManager
from ..i18n import tr
from ..server import VLLMServer
from ..system import get_gpu_info
from .common import console, create_panel
from .display import display_config, select_profile
from .gpu_utils import calculate_gpu_panel_size, create_gpu_status_panel
from .log_viewer import show_log_menu
from .model_manager import select_model
from .navigation import prompt_choice, unified_prompt

logger = logging.getLogger(__name__)


def handle_quick_serve(i18n_manager=None) -> str:
    """
    Quick serve with shortcuts or last used configuration.

    Args:
        i18n_manager: Optional I18nManager for translations
    """

    # Helper function for translation
    def t(key: str, default: str = None, **kwargs) -> str:
        if i18n_manager:
            result = i18n_manager.t(key, **kwargs)
            return result if result != key else (default if default else key)
        return default if default else key

    from .shortcuts import serve_with_shortcut

    config_manager = ConfigManager()

    # Build quick serve options
    options = []

    # Add last config if available
    last_config = config_manager.get_last_config()
    if last_config:
        model_name = last_config.get("model", "unknown")
        if isinstance(model_name, dict):
            model_name = model_name.get("model", "unknown")
        # Truncate long model names
        if len(str(model_name)) > 40:
            model_display = "..." + str(model_name)[-37:]
        else:
            model_display = str(model_name)
        options.append(f"{t('server_control.last_config')}: {model_display}")

    # Add shortcuts
    # Get recent shortcuts first (those with last_used timestamps)
    recent_shortcuts = config_manager.get_recent_shortcuts(5)

    # If we have fewer than 5 recent shortcuts, supplement with unused ones
    if len(recent_shortcuts) < 5:
        all_shortcuts = config_manager.list_shortcuts()
        # Filter out the ones we already have in recent
        recent_names = {s["name"] for s in recent_shortcuts}
        unused_shortcuts = [s for s in all_shortcuts if s["name"] not in recent_names]
        # Add unused shortcuts to fill up to 10 total
        shortcuts = recent_shortcuts + unused_shortcuts[: 10 - len(recent_shortcuts)]
    else:
        shortcuts = recent_shortcuts

    if shortcuts:
        if options:  # If we have last config, add separator
            shortcuts_title = t("server_control.shortcuts", "Shortcuts")
            options.append(f"───────── {shortcuts_title} ─────────")
        for shortcut in shortcuts:
            name = shortcut["name"]
            model = shortcut["model"]
            profile = shortcut["profile"]
            # Truncate long model names
            if len(str(model)) > 30:
                model_display = "..." + str(model)[-27:]
            else:
                model_display = str(model)
            options.append(f"{name}: {model_display} [{profile}]")

    # If no options available
    if not options:
        console.print(f"[yellow]{t('server_control.no_options_available')}[/yellow]")
        console.print(t("server_control.create_shortcuts_first"))
        input(f"\n{t('common.press_enter')}")
        return "continue"

    # Show quick serve menu
    console.print(f"\n[bold cyan]{t('server_control.quick_serve_options', 'Quick Serve Options')}[/bold cyan]")
    selected = unified_prompt(
        "quick_serve", t("server_control.select_config", "Select configuration to launch"), options, allow_back=True
    )

    if selected == "BACK" or not selected:
        return "continue"

    # Handle selection
    last_config_label = t("server_control.last_config", "Last Config")
    if selected.startswith(f"{last_config_label}:"):
        # Serve with last config
        # Apply dynamic defaults for display
        profile_manager = config_manager.profile_manager
        config_with_defaults = profile_manager.apply_dynamic_defaults(last_config)

        # Show last configuration
        display_config(config_with_defaults, title=t("server_control.last_config", "Last Configuration"))

        # Confirm
        console.print()  # Add blank line for spacing
        confirm = inquirer.confirm(
            t("server_control.start_config_confirm", "Start server with this configuration?"), default=True
        )

        if confirm:
            return start_server_with_config(config_with_defaults)

    elif (
        ":" in selected and "[" in selected and not selected.startswith(f"{last_config_label}:")
    ):
        # Extract shortcut name from the selection
        # Format is "ShortcutName: model [profile]"
        shortcut_name = selected.split(":")[0].strip()
        return serve_with_shortcut(shortcut_name)

    return "continue"


def handle_serve_with_profile(i18n_manager=None) -> str:
    """
    Serve a model with a pre-configured profile.

    Args:
        i18n_manager: Optional I18nManager for translations
    """

    # Helper function for translation
    def t(key: str, default: str = None, **kwargs) -> str:
        if i18n_manager:
            result = i18n_manager.t(key, **kwargs)
            return result if result != key else (default if default else key)
        return default if default else key

    # Select model (can return string or dict with LoRA config)
    model_selection = select_model()
    if not model_selection:
        return "continue"

    # Handle different model selection formats
    lora_modules = None
    if isinstance(model_selection, dict):
        if "type" in model_selection and model_selection["type"] == "shortcut":
            # Shortcut selected - use its model and profile directly
            model = model_selection["model"]
            profile_name = model_selection["profile"]
            shortcut_name = model_selection["name"]
            config_overrides = model_selection.get("config_overrides", {})
            model_config = None

            console.print(f"\n[bold cyan]{t('server_control.shortcuts', 'Using Shortcut')}: {shortcut_name}[/bold cyan]")
            console.print(f"  {t('arguments.descriptions.model', 'Model')}: {model}")
            console.print(f"  {t('settings.name', 'Profile')}: {profile_name}")

            # Skip profile selection since shortcut includes it
        elif "type" in model_selection and model_selection["type"] == "ollama_model":
            # Ollama/GGUF model configuration
            model = model_selection["model"]  # Path to GGUF file
            model_config = model_selection  # Keep full config for GGUF setup
            logger.info(f"Selected Ollama model: {model_selection.get('name', model)}")

            # Select profile for non-shortcut
            profile_name = select_profile()
            if not profile_name:
                return "continue"
        elif "lora_modules" in model_selection:
            # LoRA configuration
            model = model_selection["model"]
            lora_modules = model_selection["lora_modules"]
            model_config = model_selection

            # Select profile for non-shortcut
            profile_name = select_profile()
            if not profile_name:
                return "continue"
        else:
            # Other dict format - use as is
            model = model_selection.get("model", model_selection)
            model_config = model_selection

            # Select profile for non-shortcut
            profile_name = select_profile()
            if not profile_name:
                return "continue"
    else:
        # Simple string model name
        model = model_selection
        model_config = None

        # Select profile for non-shortcut
        profile_name = select_profile()
        if not profile_name:
            return "continue"

    # Get profile configuration
    config_manager = ConfigManager()
    profile = config_manager.get_profile(profile_name)
    if not profile:
        console.print(
            tr(
                "server_ui.profile_not_found",
                "[red]Profile '{name}' not found.[/red]",
                name=profile_name,
            )
        )
        return "continue"

    config = profile.get("config", {}).copy()

    # Include environment variables from the profile
    profile_env = profile.get("environment", {})
    if profile_env:
        config["profile_environment"] = profile_env

    # If we have LoRA modules, update the config
    if model_config:
        config["model"] = model_config  # Pass the full dict with LoRA info
    else:
        config["model"] = model

    # Apply config overrides from shortcut if present
    if isinstance(model_selection, dict) and model_selection.get("type") == "shortcut":
        config_overrides = model_selection.get("config_overrides", {})
        if config_overrides:
            config.update(config_overrides)

    # Apply dynamic defaults for display
    profile_manager = config_manager.profile_manager
    config_with_defaults = profile_manager.apply_dynamic_defaults(config)

    # Show configuration
    display_config(config_with_defaults, title=t("server_control.profile_configuration", "Profile Configuration"))

    # Get universal environment variables to show complete picture
    universal_env = config_manager.config.get("universal_environment", {})

    # Show environment variables with their sources
    total_env_vars = {}
    env_sources = {}

    # Add universal variables
    for key, value in universal_env.items():
        total_env_vars[key] = value
        env_sources[key] = tr("server_ui.env_source_universal", "universal")

    # Add/override with profile variables
    for key, value in profile_env.items():
        total_env_vars[key] = value
        env_sources[key] = (
            tr("server_ui.env_source_profile", "profile")
            if key not in universal_env
            else tr(
                "server_ui.env_source_profile_override",
                "profile (overrides universal)",
            )
        )

    # Show environment variables if present
    if total_env_vars:
        console.print(f"\n[cyan]{t('server_control.env_vars_count', count=len(total_env_vars))}[/cyan]")
        for key, value in total_env_vars.items():
            source = env_sources[key]
            # Hide sensitive values
            if "KEY" in key.upper() or "TOKEN" in key.upper():
                console.print(f"  • {key}: <hidden> [dim]({source})[/dim]")
            else:
                console.print(f"  • {key}: {value} [dim]({source})[/dim]")

    # Show LoRA adapters if present
    if lora_modules:
        console.print(f"\n[cyan]{t('server_control.lora_adapter_info', count=len(lora_modules))}[/cyan]")
        for lora in lora_modules:
            name = lora.get("name", "unknown")
            rank = lora.get("rank", 16)
            path = lora.get("path", "")
            console.print(f"  • {name} (rank={rank})")
            console.print(f"    {tr('server_ui.path_label', 'Path')}: {path}")

    # Confirm and start
    console.print()  # Add blank line for spacing
    confirm = inquirer.confirm(t("server_control.start_config_confirm", "Start server with this configuration?"), default=True)

    if confirm:
        # Save as last config
        config_manager.save_last_config(config_with_defaults)

        return start_server_with_config(config_with_defaults)

    return "continue"


def handle_custom_config(i18n_manager=None) -> str:
    """
    Create a custom configuration for serving using category-based approach.

    Args:
        i18n_manager: Optional I18nManager for translations
    """

    # Helper function for translation
    def t(key: str, default: str = None, **kwargs) -> str:
        if i18n_manager:
            result = i18n_manager.t(key, **kwargs)
            return result if result != key else (default if default else key)
        return default if default else key

    from .custom_config import configure_by_categories

    # Select model (can return string or dict with LoRA config)
    model_selection = select_model()
    if not model_selection:
        return "continue"

    # Handle different model selection formats
    lora_modules = None
    if isinstance(model_selection, dict):
        if "type" in model_selection and model_selection["type"] == "ollama_model":
            # Ollama/GGUF model configuration
            model = model_selection["model"]  # Path to GGUF file
            model_config = model_selection  # Keep full config for GGUF setup
            logger.info(f"Selected Ollama model: {model_selection.get('name', model)}")
        elif "lora_modules" in model_selection:
            # LoRA configuration
            model = model_selection["model"]
            lora_modules = model_selection["lora_modules"]
            model_config = model_selection
        else:
            # Other dict format - use as is
            model = model_selection.get("model", model_selection)
            model_config = model_selection
    else:
        # Simple string model name
        model = model_selection
        model_config = None

    console.print(f"\n[bold cyan]{t('arguments.categories.advanced', 'Custom Configuration')}[/bold cyan]")

    # Use category-based configuration
    config = configure_by_categories({"model": model})

    # Handle session environment variables if configured
    session_env = config.pop("session_environment", {})
    if session_env:
        # Store as profile_environment for the server to use
        config["profile_environment"] = session_env

    # If we have LoRA modules, update the config
    if model_config:
        config["model"] = model_config  # Pass the full dict with LoRA info
    else:
        config["model"] = model

    # Apply dynamic defaults for display
    config_manager = ConfigManager()
    profile_manager = config_manager.profile_manager
    config_with_defaults = profile_manager.apply_dynamic_defaults(config)

    # Show configuration summary
    display_config(config_with_defaults, title=t("server_control.configuration_summary", "Configuration Summary"))

    # Show session environment variables if configured
    if session_env:
        console.print(
            f"\n[cyan]{t('server_control.session_environment_variables', 'Session Environment Variables')} ({len(session_env)}):[/cyan]"
        )
        for key, value in session_env.items():
            if "KEY" in key.upper() or "TOKEN" in key.upper():
                console.print(f"  • {key}: <hidden>")
            else:
                console.print(f"  • {key}: {value}")

    # Show LoRA adapters if present
    if lora_modules:
        lora_header = tr(
            "server_ui.lora_adapters_header",
            "LoRA Adapters ({count})",
            count=len(lora_modules),
        )
        console.print(f"\n[cyan]{lora_header}[/cyan]")
        for lora in lora_modules:
            name = lora.get("name", "unknown")
            rank = lora.get("rank", 16)
            path = lora.get("path", "")
            console.print(f"  • {name} (rank={rank})")
            console.print(f"    {tr('server_ui.path_label', 'Path')}: {path}")

    # Option to add raw custom vLLM arguments
    # Note: configure_by_categories already provides comprehensive configuration,
    # so we don't need to ask about customizing further
    console.print()  # Add blank line for spacing
    add_raw_args = inquirer.confirm(t("server_control.add_raw_args", "Add raw custom vLLM arguments?"), default=False)

    if add_raw_args:
        console.print(f"\n[yellow]{t('server_control.custom_arguments', 'Custom vLLM Arguments')}[/yellow]")
        console.print(
            tr(
                "server_ui.raw_args_instructions",
                "Enter additional vLLM arguments exactly as you would on the command line.",
            )
        )
        console.print(tr("server_ui.examples_label", "Examples:"))
        console.print("  --seed 42 --enable-prefix-caching")
        console.print("  --max-num-seqs 256 --disable-log-stats")
        console.print("  --lora-modules name=/path/to/lora")

        extra_args = input(
            f"\n{t('server_control.raw_args_prompt', 'Enter custom arguments (or press Enter to skip): ')}"
        ).strip()
        if extra_args:
            config["extra_args"] = extra_args
            console.print(
                tr(
                    "server_ui.custom_args_applied",
                    "[green]Custom arguments: {args}[/green]",
                    args=extra_args,
                )
            )

    # Ask about saving configuration
    console.print()  # Add blank line for spacing
    save_profile = inquirer.confirm(
        t("server_control.save_as_profile", "Save this configuration as a profile for future use?"), default=False
    )

    profile_name = None
    if save_profile:
        profile_name = input(f"{t('settings.name', 'Profile')} name: ").strip()
        if profile_name:
            config_manager = ConfigManager()

            # Extract environment variables from config (stored as profile_environment)
            env_vars = config.pop("profile_environment", {})

            # Create clean config without environment variables
            clean_config = config.copy()
            clean_config.pop("model", None)  # Remove model from profile

            profile_data = {
                "name": profile_name,
                "description": "Custom configuration",
                "icon": "",
                "config": clean_config,  # Save config without model and environment
                "environment": env_vars,  # Save environment variables separately
                "lora_adapters": (
                    lora_modules if lora_modules else None
                ),  # Save LoRA adapter info if present
                "api_usage_info": config.get("api_usage_info"),  # Save API usage info
            }
            config_manager.save_user_profile(profile_name, profile_data)
            console.print(
                tr(
                    "server_ui.profile_saved_ok",
                    "[green]✓ Profile '{name}' saved.[/green]",
                    name=profile_name,
                )
            )
            if env_vars:
                console.print(
                    f"[green]  Including {len(env_vars)} {t('settings.env_vars', 'environment variable(s)')}[/green]"
                )

            # Now ask about creating a shortcut
            console.print()  # Add blank line for spacing
            create_shortcut = inquirer.confirm(
                t("server_control.create_shortcut", "Create a shortcut for quick launching this model+profile combination?"),
                default=False,
            )

            if create_shortcut:
                shortcut_name = input(f"{t('server_control.shortcuts', 'Shortcut')} name: ").strip()
                if shortcut_name:
                    shortcut_data = {
                        "model": model,
                        "profile": profile_name,
                        "description": f"Custom config for {model}",
                    }
                    if config_manager.save_shortcut(shortcut_name, shortcut_data):
                        console.print(
                            f"[green]✓ {t('server_control.shortcuts', 'Shortcut')} '{shortcut_name}' {t('server_control.shortcut_created', 'created')}![/green]"
                        )
                        console.print(
                            f'[dim]{tr("server_ui.quick_launch_hint", "Quick launch")}: vllm-cli serve --shortcut "{shortcut_name}"[/dim]'
                        )
                    else:
                        console.print(f"[red]{t('server_control.server_failed', 'Failed to create shortcut')}.[/red]")

    # Start server
    console.print()  # Add blank line for spacing
    confirm = inquirer.confirm(t("server_control.start_config_confirm", "Start server with this configuration?"), default=True)

    if confirm:
        config_manager = ConfigManager()
        config_manager.save_last_config(config_with_defaults)
        return start_server_with_config(config_with_defaults)

    return "continue"


def start_server_with_config(config: Dict[str, Any]) -> str:
    """
    Start vLLM server with given configuration.
    """
    from .server_monitor import monitor_server

    # Validate configuration before starting
    config_manager = ConfigManager()

    # Basic validation
    is_valid, errors = config_manager.validate_config(config)
    if not is_valid:
        console.print(
            f"[red]{tr('server_ui.config_validation_failed', 'Configuration validation failed:')}[/red]"
        )
        for error in errors:
            console.print(f"  • {error}")
        input("\n" + tr("common.press_enter", "Press Enter to continue..."))
        return "continue"

    # Compatibility validation
    is_compatible, warnings = config_manager.validate_argument_combination(config)
    if warnings:
        console.print(
            f"[yellow]{tr('server_ui.config_warnings', 'Configuration warnings:')}[/yellow]"
        )
        for warning in warnings:
            console.print(f"  • {warning}")

        # For errors (severity), ask user if they want to continue
        if not is_compatible:
            console.print(
                f"\n[red]{tr('server_ui.conflicts_detected', 'Some configuration conflicts were detected.')}[/red]"
            )
            continue_anyway = inquirer.confirm(
                tr(
                    "server_ui.continue_anyway",
                    "Continue anyway? (The server may not work as expected)",
                ),
                default=False,
            )
            if not continue_anyway:
                return "continue"

        console.print()  # Add spacing

    # Check if this is a remote model by checking if it exists locally
    model_config = config.get("model", "")

    # Extract model name from LoRA config if needed
    if isinstance(model_config, dict):
        model_name = model_config.get("model", "")
    else:
        model_name = model_config

    # Check if model exists in local cache
    from ..models import list_available_models

    local_models = list_available_models()
    local_model_names = [m.get("name", "") for m in local_models]

    # A model is remote if it has "/" (HuggingFace format) and is NOT in local cache
    is_remote_model = (
        "/" in model_name
        and not model_name.startswith("/")
        and model_name not in local_model_names
    )

    if is_remote_model:
        console.print(
            f"\n[bold cyan]{tr('server_ui.starting_remote_model', 'Starting vLLM server with remote model:')}[/bold cyan] {model_name}"
        )
        console.print(
            f"[yellow]{tr('server_ui.remote_model_note', 'Note: First-time use will download the model from HuggingFace Hub.')}[/yellow]"
        )
        console.print(
            f"[dim]{tr('server_ui.remote_model_download_time', 'Download may take 10-30 minutes depending on size and connection speed.')}[/dim]\n"
        )
    else:
        console.print(
            f"\n[bold cyan]{tr('server_ui.starting_server', 'Starting vLLM server...')}[/bold cyan]"
        )

    # Get UI preferences for configurable log lines and refresh rate
    ui_prefs = config_manager.get_ui_preferences()
    startup_refresh_rate = ui_prefs.get("startup_refresh_rate", 4.0)

    try:
        server = VLLMServer(config)

        # Start the server (this launches the process)
        server.start()

        # Give the log thread a moment to start
        time.sleep(0.2)  # Reduced delay since we fixed buffering

        # Get GPU info to calculate optimal panel size
        gpu_info = get_gpu_info()
        gpu_panel_size = calculate_gpu_panel_size(len(gpu_info) if gpu_info else 0)

        # Create a layout for showing startup progress
        layout = Layout()
        layout.split_column(
            Layout(name="status", size=3),
            Layout(name="gpu", size=gpu_panel_size),  # Dynamic size based on GPU count
            Layout(name="log_divider", size=1),
            Layout(name="logs"),  # Takes remaining space
            Layout(name="info", size=3),
            Layout(name="footer", size=1),
        )

        # Initial status
        layout["status"].update(
            create_panel(
                f"[yellow]{tr('server_ui.status_starting', '⠋ Starting vLLM server... This may take a few minutes for model loading.')}[/yellow]",
                title=tr("server_ui.panel_title_status", "Status"),
                border_style="yellow",
            )
        )

        # Initial GPU panel
        layout["gpu"].update(create_gpu_status_panel())

        # Extract model name for display
        model_display = config.get("model", "unknown")
        if isinstance(model_display, dict):
            model_display = model_display.get("model", "unknown")

        layout["info"].update(
            create_panel(
                tr(
                    "server_ui.info_panel",
                    "Port: {port} | Model: {model}",
                    port=server.port,
                    model=model_display,
                ),
                title=tr("server_ui.panel_title_info", "Info"),
                border_style="blue",
            )
        )

        # Footer with exit instructions
        layout["footer"].update(
            Align.center(
                Text(
                    tr("server_ui.footer_cancel_startup", "Press Ctrl+C to cancel startup"),
                    style="dim yellow",
                )
            )
        )

        startup_logs = []
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        frame_idx = 0
        startup_complete = False
        startup_failed = False
        no_log_count = 0  # Track how many times we've seen no logs

        # Monitor startup for up to 5 minutes
        startup_cancelled = False
        try:
            with Live(
                layout, console=console, refresh_per_second=startup_refresh_rate
            ):  # User-configurable refresh rate
                start_time = time.time()

                while not startup_complete and not startup_failed:
                    # Check if server is still running first
                    if not server.is_running():
                        startup_failed = True
                        layout["status"].update(
                            create_panel(
                                f"[red]{tr('server_ui.status_process_terminated', '✗ Server process terminated unexpectedly')}[/red]",
                                title=tr("server_ui.panel_title_status", "Status"),
                                border_style="red",
                            )
                        )
                        break

                    # Get recent logs
                    new_logs = server.get_recent_logs(50)
                    if new_logs:
                        startup_log_lines = ui_prefs.get("log_lines_startup", 50)
                        startup_logs = new_logs[
                            -startup_log_lines:
                        ]  # Keep configurable number of lines for display
                        no_log_count = 0  # Reset counter when we get logs

                        # Check for startup completion indicators
                        for log in new_logs:
                            log_lower = log.lower()
                            # vLLM ready indicators
                            if any(
                                indicator in log_lower
                                for indicator in [
                                    "uvicorn running on",
                                    "started server process",
                                    "application startup complete",
                                    "server is ready",
                                    "api server started",
                                ]
                            ):
                                startup_complete = True
                                break
                            # Check for critical errors (not warnings)
                            elif "traceback" in log or "error:" in log_lower:
                                # Look for actual error patterns, not just the word "error"
                                if any(
                                    err in log_lower
                                    for err in [
                                        "cuda out of memory",
                                        "failed to load",
                                        "no such file",
                                        "permission denied",
                                        "address already in use",
                                        "cuda error",
                                        "runtime error",
                                        "value error",
                                        "import error",
                                    ]
                                ):
                                    startup_failed = True
                                    break
                    else:
                        no_log_count += 1

                    # Update spinner
                    frame_idx = (frame_idx + 1) % len(spinner_frames)
                    spinner = spinner_frames[frame_idx]

                    # Update GPU panel periodically
                    if (
                        frame_idx % 4 == 0
                    ):  # Update every 4th frame (about once per second)
                        layout["gpu"].update(create_gpu_status_panel())

                    # Update status based on logs
                    if not startup_complete and not startup_failed:
                        elapsed = int(time.time() - start_time)
                        status_msg = tr(
                            "server_ui.status_msg_starting",
                            "{spinner} Starting vLLM server... ({elapsed}s elapsed)",
                            spinner=spinner,
                            elapsed=elapsed,
                        )

                        # Try to detect what stage we're in from logs
                        # stage_detected = False  # Not used
                        for log in reversed(startup_logs):  # Check most recent first
                            log_lower = log.lower()
                            if (
                                "loading weights" in log_lower
                                or "loading model" in log_lower
                            ):
                                status_msg = tr(
                                    "server_ui.status_msg_loading_weights",
                                    "{spinner} Loading model weights... This may take a while ({elapsed}s)",
                                    spinner=spinner,
                                    elapsed=elapsed,
                                )
                                # stage_detected = True  # Not needed
                                break
                            elif "initializing" in log_lower and "engine" in log_lower:
                                status_msg = tr(
                                    "server_ui.status_msg_initializing_engine",
                                    "{spinner} Initializing vLLM engine... ({elapsed}s)",
                                    spinner=spinner,
                                    elapsed=elapsed,
                                )
                                # stage_detected = True  # Not needed
                                break
                            elif "compiling" in log_lower or "cuda graph" in log_lower:
                                status_msg = tr(
                                    "server_ui.status_msg_compiling",
                                    "{spinner} Compiling CUDA kernels and graphs... ({elapsed}s)",
                                    spinner=spinner,
                                    elapsed=elapsed,
                                )
                                # stage_detected = True  # Not needed
                                break
                            elif "downloading" in log_lower or "fetching" in log_lower:
                                # Try to extract download progress if available
                                progress_info = ""
                                if "%" in log:
                                    # Try to extract percentage
                                    import re

                                    match = re.search(r"(\d+(?:\.\d+)?)\s*%", log)
                                    if match:
                                        progress_info = f" - {match.group(1)}%"

                                status_msg = tr(
                                    "server_ui.status_msg_downloading",
                                    "{spinner} Downloading model from HuggingFace Hub{progress}... ({elapsed}s)",
                                    spinner=spinner,
                                    progress=progress_info,
                                    elapsed=elapsed,
                                )
                                # stage_detected = True  # Not needed
                                break
                            elif (
                                "starting server" in log_lower
                                or "starting uvicorn" in log_lower
                            ):
                                status_msg = tr(
                                    "server_ui.status_msg_starting_api",
                                    "{spinner} Starting API server... Almost ready! ({elapsed}s)",
                                    spinner=spinner,
                                    elapsed=elapsed,
                                )
                                # stage_detected = True  # Not needed
                                break

                        layout["status"].update(
                            create_panel(
                                f"[yellow]{status_msg}[/yellow]",
                                title=tr("server_ui.panel_title_status", "Status"),
                                border_style="yellow",
                            )
                        )

                    # Update log divider
                    if startup_logs:
                        layout["log_divider"].update(
                            Rule(
                                tr(
                                    "server_ui.startup_logs_divider",
                                    "Startup Logs (Last {count} lines)",
                                    count=len(startup_logs),
                                ),
                                style="cyan",
                            )
                        )
                    else:
                        layout["log_divider"].update(
                            Rule(
                                tr("server_ui.startup_logs_divider_plain", "Startup Logs"),
                                style="cyan",
                            )
                        )

                    # Update logs (no panel, just text)
                    if startup_logs:
                        log_text = Text("\n".join(startup_logs), style="dim white")
                        layout["logs"].update(Padding(log_text, (0, 2)))
                    else:
                        # Show different messages based on how long we've been waiting
                        if no_log_count < 2:
                            msg = tr(
                                "server_ui.no_logs_initializing",
                                "{spinner} Initializing vLLM server...",
                                spinner=spinner,
                            )
                        elif no_log_count < 8:
                            msg = tr(
                                "server_ui.no_logs_starting_process",
                                "{spinner} Starting vLLM process...",
                                spinner=spinner,
                            )
                        else:
                            msg = tr(
                                "server_ui.no_logs_still_waiting",
                                "{spinner} Still waiting for vLLM output... Check if vLLM is installed and in your PATH.",
                                spinner=spinner,
                            )

                        log_text = Text(msg, style="dim yellow")
                        layout["logs"].update(Padding(log_text, (0, 2)))

                    time.sleep(0.25)  # Reduced sleep for faster updates
        except KeyboardInterrupt:
            startup_cancelled = True
            console.print(
                f"\n[yellow]{tr('server_ui.startup_cancelled_by_user', 'Startup cancelled by user.')}[/yellow]"
            )

        # Show final status
        if startup_cancelled:
            console.print(
                f"[yellow]{tr('server_ui.startup_was_cancelled', 'Server startup was cancelled.')}[/yellow]"
            )
            console.print("")
            console.print(
                f"[bold]{tr('server_ui.stop_server_question', 'Do you want to stop the server process?')}[/bold] [dim](Y/n):[/dim] ",
                end="",
            )
            response = input().strip().lower()
            if response != "n" and response != "no":
                console.print(
                    f"[yellow]{tr('server_ui.stopping_server', 'Stopping server...')}[/yellow]"
                )
                server.stop()
                console.print(
                    f"[green]{tr('server_ui.server_stopped', '✓ Server stopped.')}[/green]"
                )
            else:
                console.print(
                    f"[dim]{tr('server_ui.server_continues_background', 'Server process continues in background.')}[/dim]"
                )
                console.print(
                    f"[yellow]{tr('server_ui.monitor_logs_unavailable_warning', '⚠ Warning: You will not be able to monitor server logs from vLLM CLI')}[/yellow]"
                )
                console.print(
                    tr(
                        "server_ui.server_may_still_start",
                        "[dim]Note: Server may still be starting up. Check port {port}[/dim]",
                        port=server.port,
                    )
                )
        elif startup_complete:
            console.print(
                tr(
                    "server_ui.server_started_on_port",
                    "[green]✓ Server successfully started on port {port}[/green]",
                    port=server.port,
                )
            )
            console.print(
                tr(
                    "server_ui.api_endpoint",
                    "[green]API endpoint: http://localhost:{port}[/green]",
                    port=server.port,
                )
            )

            # Option to monitor - use navigation system
            post_options = [
                (
                    "monitor",
                    tr("server_ui.monitor_output_option", "Monitor server output"),
                ),
                (
                    "main_menu",
                    tr("server_ui.return_to_menu_option", "Return to main menu"),
                ),
            ]

            choice = prompt_choice(
                "post_startup",
                tr("server_ui.what_would_you_like", "What would you like to do?"),
                post_options,
                allow_back=False,
            )

            if choice == "monitor":
                return monitor_server(server)
            else:
                console.print(
                    f"[green]{tr('server_ui.server_running_background', 'Server is running in background.')}[/green]"
                )
        else:
            console.print(
                f"\n[red]{tr('server_ui.start_failed', '✗ Failed to start server')}[/red]"
            )
            if startup_logs:
                console.print(
                    f"\n[bold]{tr('server_ui.last_logs_label', 'Last logs:')}[/bold]"
                )
                for log in startup_logs[-5:]:
                    console.print(f"  {log}")
            # Offer to view logs interactively
            console.print(
                f"\n[yellow]{tr('server_ui.startup_failed_hint', 'Server startup failed. Last logs shown above.')}[/yellow]"
            )

            view_logs = (
                input(
                    "\n"
                    + tr(
                        "server_ui.view_full_logs_question",
                        "Would you like to view the full logs? (y/N): ",
                    )
                )
                .strip()
                .lower()
            )
            if view_logs in ["y", "yes"]:
                show_log_menu(server)
            else:
                console.print(
                    tr(
                        "server_ui.full_log_file",
                        "\n[dim]Full log file: {path}[/dim]",
                        path=server.log_path,
                    )
                )
                input("\n" + tr("common.press_enter", "Press Enter to continue..."))

    except Exception as e:
        logger.error(f"Error starting server: {e}")
        console.print(
            tr(
                "server_ui.error_starting_server",
                "[red]Error starting server: {error}[/red]",
                error=e,
            )
        )
        input("\n" + tr("common.press_enter", "Press Enter to continue..."))
    return "continue"
