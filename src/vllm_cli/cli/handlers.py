#!/usr/bin/env python3
"""
Command handlers for vLLM CLI commands.

Implements the actual logic for each CLI command including
serve, info, models, status, and stop operations.
"""

import argparse
import logging
from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .. import __version__
from ..config import ConfigManager
from ..i18n import tr
from ..models import list_available_models
from ..proxy import ProxyManager
from ..proxy.config import ProxyConfigManager
from ..proxy.models import ModelConfig
from ..server import (
    VLLMServer,
    find_server_by_model,
    find_server_by_port,
    get_active_servers,
    stop_all_servers,
)
from ..system import (
    check_vllm_installation,
    format_size,
    get_cuda_version,
    get_memory_info,
)
from ..ui.gpu_utils import create_gpu_status_panel
from ..ui.navigation import prompt_choice

logger = logging.getLogger(__name__)
console = Console()


def handle_serve(args: argparse.Namespace) -> bool:
    """
    Handle the 'serve' command to start a vLLM server.

    Processes the serve command arguments, sets up configuration,
    and starts a new vLLM server instance.

    Args:
        args: Parsed command line arguments

    Returns:
        True if server started successfully, False otherwise
    """
    try:
        config_manager = ConfigManager()

        # Validate that either model or shortcut is provided
        if not args.model and not (hasattr(args, "shortcut") and args.shortcut):
            console.print(
                tr(
                    "cli_msgs.err_model_or_shortcut",
                    "[red]Error: Either MODEL or --shortcut must be specified.[/red]",
                )
            )
            console.print(
                tr("cli_msgs.serve_usage", "Usage: vllm-cli serve MODEL [options]")
            )
            console.print(
                tr(
                    "cli_msgs.serve_usage_shortcut",
                    "   or: vllm-cli serve --shortcut SHORTCUT_NAME [options]",
                )
            )
            return False

        # Handle shortcut mode
        if hasattr(args, "shortcut") and args.shortcut:
            shortcut = config_manager.get_shortcut(args.shortcut)
            if not shortcut:
                console.print(
                    tr(
                        "cli_msgs.shortcut_not_found",
                        "[red]Shortcut '{name}' not found.[/red]",
                        name=args.shortcut,
                    )
                )
                console.print(
                    tr(
                        "cli_msgs.list_shortcuts_hint",
                        "Use 'vllm-cli shortcuts' to list available shortcuts.",
                    )
                )
                return False

            # Get profile configuration
            profile = config_manager.get_profile(shortcut["profile"])
            if not profile:
                console.print(
                    tr(
                        "cli_msgs.profile_ref_not_found",
                        "[red]Profile '{name}' referenced by shortcut not found.[/red]",
                        name=shortcut["profile"],
                    )
                )
                return False

            # Build config from shortcut
            config = profile.get("config", {}).copy()
            config["model"] = shortcut["model"]

            # Apply any config overrides from shortcut
            if "config_overrides" in shortcut:
                config.update(shortcut["config_overrides"])

            # Override model in args for display purposes
            args.model = shortcut["model"]

            # Update last used timestamp
            config_manager.shortcut_manager.update_last_used(args.shortcut)

            console.print(
                tr(
                    "cli_msgs.using_shortcut",
                    "[cyan]Using shortcut: {name}[/cyan]",
                    name=args.shortcut,
                )
            )
            console.print(
                tr("cli_msgs.line_model", "  Model: {value}", value=shortcut["model"])
            )
            console.print(
                tr(
                    "cli_msgs.line_profile",
                    "  Profile: {value}",
                    value=shortcut["profile"],
                )
            )
        else:
            # Build configuration from arguments
            config = _build_serve_config(args, config_manager)

        # Validate configuration
        is_valid, errors = config_manager.validate_config(config)
        if not is_valid:
            console.print(
                tr(
                    "cli_msgs.validation_failed",
                    "[red]Configuration validation failed:[/red]",
                )
            )
            for error in errors:
                console.print(f"  • {error}")
            return False

        # Check for compatibility issues
        is_compatible, warnings = config_manager.validate_argument_combination(config)
        if warnings:
            console.print(
                tr(
                    "cli_msgs.config_warnings",
                    "[yellow]Configuration warnings:[/yellow]",
                )
            )
            for warning in warnings:
                console.print(f"  • {warning}")

        # Save profile if requested
        if args.save_profile:
            profile_data = {
                "name": args.save_profile,
                "description": f"Profile for {args.model}",
                "config": config,
            }
            config_manager.save_user_profile(args.save_profile, profile_data)
            console.print(
                tr(
                    "cli_msgs.saved_profile",
                    "[green]Saved profile: {name}[/green]",
                    name=args.save_profile,
                )
            )

        # Save shortcut if requested
        if hasattr(args, "save_shortcut") and args.save_shortcut:
            # Need a profile for the shortcut
            if args.profile:
                profile_name = args.profile
            elif args.save_profile:
                profile_name = args.save_profile
            else:
                console.print(
                    tr(
                        "cli_msgs.creating_default_profile",
                        "[yellow]Creating default profile for shortcut...[/yellow]",
                    )
                )
                profile_name = f"config_{args.model.replace('/', '_')}"
                profile_data = {
                    "name": profile_name,
                    "description": f"Auto-generated profile for {args.model}",
                    "config": config,
                }
                config_manager.save_user_profile(profile_name, profile_data)

            shortcut_data = {
                "model": args.model,
                "profile": profile_name,
                "description": f"Shortcut for {args.model}",
            }
            if config_manager.save_shortcut(args.save_shortcut, shortcut_data):
                console.print(
                    tr(
                        "cli_msgs.saved_shortcut",
                        "[green]Saved shortcut: {name}[/green]",
                        name=args.save_shortcut,
                    )
                )
                console.print(
                    tr(
                        "cli_msgs.run_shortcut_hint",
                        "Use 'vllm-cli serve --shortcut \"{name}\"' to run it.",
                        name=args.save_shortcut,
                    )
                )
            else:
                console.print(
                    tr(
                        "cli_msgs.save_shortcut_failed",
                        "[red]Failed to save shortcut: {name}[/red]",
                        name=args.save_shortcut,
                    )
                )

        # Create and start server
        server = VLLMServer(config)

        # Check if this is a remote model
        is_remote_model = "/" in args.model and not args.model.startswith("/")

        if is_remote_model:
            console.print(
                tr(
                    "cli_msgs.starting_remote_model",
                    "[blue]Starting vLLM server for remote model: {model}[/blue]",
                    model=args.model,
                )
            )
            console.print(
                tr(
                    "cli_msgs.remote_model_note",
                    "[yellow]Note: Model will be downloaded from HuggingFace Hub if not cached[/yellow]",
                )
            )
        else:
            console.print(
                tr(
                    "cli_msgs.starting_model",
                    "[blue]Starting vLLM server for model: {model}[/blue]",
                    model=args.model,
                )
            )

        console.print(
            tr("cli_msgs.line_port", "Port: {value}", value=config.get("port", 8000))
        )
        console.print(
            tr(
                "cli_msgs.line_host",
                "Host: {value}",
                value=config.get("host", "localhost"),
            )
        )

        if server.start():
            console.print(
                tr(
                    "cli_msgs.server_started",
                    "[green]✓ Server started successfully[/green]",
                )
            )
            console.print(
                tr(
                    "cli_msgs.server_url",
                    "Server URL: {url}",
                    url=f"http://{config.get('host', 'localhost')}:{config.get('port', 8000)}",
                )
            )

            # Save as last used configuration
            config_manager.save_last_config(config)
            config_manager.add_recent_model(args.model)

            return True
        else:
            console.print(
                tr(
                    "cli_msgs.server_start_failed",
                    "[red]✗ Failed to start server[/red]",
                )
            )
            return False

    except Exception as e:
        logger.exception(f"Error in serve command: {e}")
        console.print(
            tr(
                "cli_msgs.serve_error",
                "[red]Error starting server: {error}[/red]",
                error=e,
            )
        )
        return False


def handle_info() -> bool:
    """
    Handle the 'info' command to show system information.

    Displays comprehensive system information including GPU status,
    memory usage, and software versions.

    Returns:
        True if information was displayed successfully
    """
    try:
        console.print(f"\n[bold cyan]{tr('system_info.title', 'System Information')}[/bold cyan]\n")

        # GPU Information
        gpu_panel = create_gpu_status_panel()
        console.print(gpu_panel)

        # System Memory
        memory_info = get_memory_info()
        memory_panel = Panel(
            tr(
                "cli_msgs.memory_usage",
                "Total: {total}\nUsed: {used} ({percent}%)\nAvailable: {available}",
                total=format_size(memory_info["total"]),
                used=format_size(memory_info["used"]),
                percent=f"{memory_info['percent']:.1f}",
                available=format_size(memory_info["available"]),
            ),
            title=tr("cli_msgs.title_system_memory", "System Memory"),
            border_style="blue",
        )
        console.print(memory_panel)

        # Software Information
        cuda_version = get_cuda_version()
        software_info = f"vLLM CLI: {__version__}\n"

        try:
            import torch

            software_info += f"PyTorch: {torch.__version__}\n"
        except ImportError:
            software_info += f"PyTorch: {tr('cli_msgs.not_installed', 'Not installed')}\n"

        if cuda_version:
            software_info += f"CUDA: {cuda_version}"
        else:
            software_info += f"CUDA: {tr('cli_msgs.not_available', 'Not available')}"

        software_panel = Panel(
            software_info,
            title=tr("cli_msgs.title_software", "Software"),
            border_style="green",
        )
        console.print(software_panel)

        # vLLM Installation Check
        if check_vllm_installation():
            console.print(
                tr(
                    "cli_msgs.vllm_installed",
                    "[green]✓ vLLM is properly installed[/green]",
                )
            )
        else:
            console.print(
                tr(
                    "cli_msgs.vllm_not_installed",
                    "[yellow]⚠ vLLM not found or not properly installed[/yellow]",
                )
            )

        return True

    except Exception as e:
        logger.exception(f"Error in info command: {e}")
        console.print(
            tr(
                "cli_msgs.info_error",
                "[red]Error getting system information: {error}[/red]",
                error=e,
            )
        )
        return False


def handle_models() -> bool:
    """
    Handle the 'models' command to list available models.

    Displays all available models that can be served with vLLM,
    including their paths and sizes.

    Returns:
        True if models were listed successfully
    """
    try:
        console.print(
            tr(
                "cli_msgs.available_models",
                "\n[bold cyan]Available Models[/bold cyan]\n",
            )
        )

        # Get available models
        models = list_available_models()

        if not models:
            console.print(
                tr("cli_msgs.no_models_found", "[yellow]No models found.[/yellow]")
            )
            console.print(
                tr(
                    "cli_msgs.download_models_hint",
                    "Use hf-model-tool to download models first.",
                )
            )
            return True

        # Create table
        table = Table(title=tr("cli_msgs.found_models_count", "Found {count} model(s)", count=len(models)))
        table.add_column(tr("cli_msgs.col_model", "Model"), style="cyan", no_wrap=True)
        table.add_column(tr("cli_msgs.col_size", "Size"), style="magenta")
        table.add_column(tr("cli_msgs.col_type", "Type"), style="green")
        table.add_column(tr("cli_msgs.col_path", "Path"), style="dim", overflow="fold")

        # Sort models by name
        models.sort(key=lambda x: x["name"])

        for model in models:
            size_str = (
                format_size(model["size"])
                if model["size"] > 0
                else tr("cli_msgs.unknown", "Unknown")
            )
            model_type = model.get("type", "model")
            path = model.get("path", tr("cli_msgs.unknown", "Unknown"))

            table.add_row(model["name"], size_str, model_type, path)

        console.print(table)
        return True

    except Exception as e:
        logger.exception(f"Error in models command: {e}")
        console.print(
            tr(
                "cli_msgs.models_error",
                "[red]Error listing models: {error}[/red]",
                error=e,
            )
        )
        return False


def handle_shortcuts(args) -> bool:
    """
    Handle the 'shortcuts' command to list and manage shortcuts.

    Args:
        args: Parsed command line arguments

    Returns:
        True if operation was successful
    """
    try:
        from pathlib import Path

        config_manager = ConfigManager()

        # Handle delete operation
        if hasattr(args, "delete") and args.delete:
            if config_manager.delete_shortcut(args.delete):
                console.print(
                    tr(
                        "cli_msgs.shortcut_deleted",
                        "[green]✓ Shortcut '{name}' deleted.[/green]",
                        name=args.delete,
                    )
                )
            else:
                console.print(
                    tr(
                        "cli_msgs.shortcut_not_found",
                        "[red]Shortcut '{name}' not found.[/red]",
                        name=args.delete,
                    )
                )
            return True

        # Handle export operation
        if hasattr(args, "export") and args.export:
            shortcut = config_manager.get_shortcut(args.export)
            if not shortcut:
                console.print(
                    tr(
                        "cli_msgs.shortcut_not_found",
                        "[red]Shortcut '{name}' not found.[/red]",
                        name=args.export,
                    )
                )
                return False

            file_path = Path(f"shortcut_{args.export}.json")
            if config_manager.shortcut_manager.export_shortcut(args.export, file_path):
                console.print(
                    tr(
                        "cli_msgs.shortcut_exported",
                        "[green]✓ Shortcut exported to {path}[/green]",
                        path=file_path,
                    )
                )
            else:
                console.print(
                    tr(
                        "cli_msgs.export_shortcut_failed",
                        "[red]Failed to export shortcut.[/red]",
                    )
                )
            return True

        # Handle import operation
        if hasattr(args, "import_file") and args.import_file:
            file_path = Path(args.import_file)
            if not file_path.exists():
                console.print(
                    tr(
                        "cli_msgs.file_not_found",
                        "[red]File not found: {path}[/red]",
                        path=args.import_file,
                    )
                )
                return False

            try:
                if config_manager.shortcut_manager.import_shortcut(file_path):
                    console.print(
                        tr(
                            "cli_msgs.shortcut_imported",
                            "[green]✓ Shortcut imported successfully![/green]",
                        )
                    )
                else:
                    console.print(
                        tr(
                            "cli_msgs.import_shortcut_failed",
                            "[red]Failed to import shortcut.[/red]",
                        )
                    )
            except Exception as e:
                console.print(
                    tr(
                        "cli_msgs.import_shortcut_error",
                        "[red]Error importing shortcut: {error}[/red]",
                        error=e,
                    )
                )
                return False
            return True

        # Default: List all shortcuts
        shortcuts = config_manager.list_shortcuts()

        if not shortcuts:
            console.print(
                tr(
                    "cli_msgs.no_shortcuts",
                    "\n[yellow]No shortcuts configured.[/yellow]",
                )
            )
            console.print(
                tr(
                    "cli_msgs.create_shortcuts_intro",
                    "\nCreate shortcuts to quickly launch frequently used configurations:",
                )
            )
            console.print(
                tr(
                    "cli_msgs.create_shortcut_cli",
                    "  • From CLI: vllm-cli serve MODEL --profile PROFILE --save-shortcut NAME",
                )
            )
            console.print(
                tr(
                    "cli_msgs.create_shortcut_ui",
                    "  • From interactive mode: Settings → Manage Shortcuts",
                )
            )
            return True

        # Create table
        table = Table(
            title=tr(
                "cli_msgs.configured_shortcuts",
                "[bold cyan]Configured Shortcuts ({count})[/bold cyan]",
                count=len(shortcuts),
            )
        )
        table.add_column(tr("cli_msgs.col_shortcut", "Shortcut"), style="cyan", no_wrap=True)
        table.add_column(tr("cli_msgs.col_model", "Model"), style="green")
        table.add_column(tr("cli_msgs.col_profile", "Profile"), style="magenta")
        table.add_column(tr("cli_msgs.col_description", "Description"), style="dim")

        for shortcut in shortcuts:
            model = shortcut["model"]
            # Truncate long model paths
            if len(model) > 40:
                model_display = "..." + model[-37:]
            else:
                model_display = model

            table.add_row(
                shortcut["name"],
                model_display,
                shortcut["profile"],
                shortcut.get("description", ""),
            )

        console.print("")
        console.print(table)
        console.print(tr("cli_msgs.usage_label", "[dim]Usage:[/dim]"))
        console.print(
            tr(
                "cli_msgs.launch_shortcut_hint",
                "  • Launch: vllm-cli serve --shortcut SHORTCUT_NAME",
            )
        )
        console.print(
            tr(
                "cli_msgs.delete_shortcut_hint",
                "  • Delete: vllm-cli shortcuts --delete SHORTCUT_NAME",
            )
        )
        console.print(
            tr(
                "cli_msgs.export_shortcut_hint",
                "  • Export: vllm-cli shortcuts --export SHORTCUT_NAME",
            )
        )

        return True

    except Exception as e:
        logger.exception(f"Error in shortcuts command: {e}")
        console.print(
            tr(
                "cli_msgs.shortcuts_error",
                "[red]Error managing shortcuts: {error}[/red]",
                error=e,
            )
        )
        return False


def handle_status() -> bool:
    """
    Handle the 'status' command to show active servers.

    Displays status information for all currently running
    vLLM servers including PIDs, ports, and uptime.

    Returns:
        True if status was displayed successfully
    """
    try:
        console.print(
            tr(
                "cli_msgs.active_servers",
                "\n[bold cyan]Active vLLM Servers[/bold cyan]\n",
            )
        )

        # Get active servers
        servers = get_active_servers()

        if not servers:
            console.print(
                tr(
                    "cli_msgs.no_active_servers",
                    "[yellow]No active servers found.[/yellow]",
                )
            )
            return True

        # Create table
        table = Table(
            title=tr("cli_msgs.active_servers_count", "{count} active server(s)", count=len(servers))
        )
        table.add_column(tr("cli_msgs.col_model", "Model"), style="cyan")
        table.add_column(tr("cli_msgs.col_port", "Port"), style="magenta")
        table.add_column(tr("cli_msgs.col_pid", "PID"), style="green")
        table.add_column(tr("cli_msgs.col_status", "Status"), style="blue")
        table.add_column(tr("cli_msgs.col_uptime", "Uptime"), style="yellow")

        for server in servers:
            status = server.get_status()

            # Determine status
            if status["running"]:
                status_str = tr("cli_msgs.status_running", "[●] Running")
            else:
                status_str = tr("cli_msgs.status_stopped", "[×] Stopped")

            # Format uptime
            uptime_str = status.get("uptime_str", tr("cli_msgs.unknown", "Unknown"))

            table.add_row(
                status["model"],
                str(status["port"]),
                str(status["pid"]) if status["pid"] else "N/A",
                status_str,
                uptime_str,
            )

        console.print(table)
        return True

    except Exception as e:
        logger.exception(f"Error in status command: {e}")
        console.print(
            tr(
                "cli_msgs.status_error",
                "[red]Error getting server status: {error}[/red]",
                error=e,
            )
        )
        return False


def handle_stop(args: argparse.Namespace) -> bool:
    """
    Handle the 'stop' command to stop vLLM servers.

    Stops one or more vLLM servers based on the provided arguments
    (specific model, port, or all servers).

    Args:
        args: Parsed command line arguments

    Returns:
        True if servers were stopped successfully
    """
    try:
        if args.all:
            # Stop all servers
            console.print(
                tr("cli_msgs.stopping_all", "[blue]Stopping all servers...[/blue]")
            )
            stopped_count = stop_all_servers()

            if stopped_count > 0:
                console.print(
                    tr(
                        "cli_msgs.stopped_count",
                        "[green]✓ Stopped {count} server(s)[/green]",
                        count=stopped_count,
                    )
                )
            else:
                console.print(
                    tr(
                        "cli_msgs.no_servers_running",
                        "[yellow]No servers were running[/yellow]",
                    )
                )

            return True

        elif args.port:
            # Stop server by port
            server = find_server_by_port(args.port)
            if server:
                console.print(
                    tr(
                        "cli_msgs.stopping_port",
                        "[blue]Stopping server on port {port}...[/blue]",
                        port=args.port,
                    )
                )
                if server.stop():
                    console.print(
                        tr(
                            "cli_msgs.stopped_port",
                            "[green]✓ Stopped server on port {port}[/green]",
                            port=args.port,
                        )
                    )
                    return True
                else:
                    console.print(
                        tr(
                            "cli_msgs.stop_port_failed",
                            "[red]✗ Failed to stop server on port {port}[/red]",
                            port=args.port,
                        )
                    )
                    return False
            else:
                console.print(
                    tr(
                        "cli_msgs.no_server_on_port",
                        "[yellow]No server found on port {port}[/yellow]",
                        port=args.port,
                    )
                )
                return False

        elif args.model:
            # Stop server by model name or try as port number
            server = None

            # First try as model name
            server = find_server_by_model(args.model)

            # If not found, try as port number
            if not server:
                try:
                    port = int(args.model)
                    server = find_server_by_port(port)
                except ValueError:
                    pass

            if server:
                console.print(
                    tr(
                        "cli_msgs.stopping_model",
                        "[blue]Stopping server for {model}...[/blue]",
                        model=args.model,
                    )
                )
                if server.stop():
                    console.print(
                        tr(
                            "cli_msgs.stopped_model",
                            "[green]✓ Stopped server for {model}[/green]",
                            model=args.model,
                        )
                    )
                    return True
                else:
                    console.print(
                        tr(
                            "cli_msgs.stop_model_failed",
                            "[red]✗ Failed to stop server for {model}[/red]",
                            model=args.model,
                        )
                    )
                    return False
            else:
                console.print(
                    tr(
                        "cli_msgs.no_server_for_model",
                        "[yellow]No server found for {model}[/yellow]",
                        model=args.model,
                    )
                )
                return False

        return True

    except Exception as e:
        logger.exception(f"Error in stop command: {e}")
        console.print(
            tr(
                "cli_msgs.stop_error",
                "[red]Error stopping server: {error}[/red]",
                error=e,
            )
        )
        return False


def _build_serve_config(
    args: argparse.Namespace, config_manager: ConfigManager
) -> Dict[str, Any]:
    """
    Build server configuration from command line arguments.

    Args:
        args: Parsed command line arguments
        config_manager: ConfigManager instance for profile handling

    Returns:
        Configuration dictionary for the server
    """
    # Handle special case where model might be a dict (for GGUF/Ollama models from UI)
    if isinstance(args.model, dict):
        # Check if this is an Ollama model with name metadata
        if args.model.get("type") == "ollama_model" and args.model.get("name"):
            # For Ollama models, create special config with served_model_name
            config = {
                "model": {
                    "model": args.model.get("path", args.model.get("model")),
                    "quantization": "gguf",
                    "served_model_name": args.model.get(
                        "name"
                    ),  # Use Ollama model name
                }
            }
            console.print(
                tr(
                    "cli_msgs.using_ollama_model",
                    "[cyan]Using Ollama model: {name}[/cyan]",
                    name=args.model.get("name"),
                )
            )
        else:
            # Extract GGUF-specific configuration (non-Ollama)
            config = {
                "model": args.model.get("model"),
                "quantization": args.model.get("quantization", "gguf"),
            }
        # Add warning about experimental support
        if args.model.get("experimental"):
            console.print(
                tr(
                    "cli_msgs.experimental_gguf",
                    "[yellow]⚠ Using experimental GGUF support[/yellow]",
                )
            )
    else:
        config = {"model": args.model}

    # Load profile if specified
    if args.profile:
        profile = config_manager.get_profile(args.profile)
        if profile and "config" in profile:
            config.update(profile["config"])
        else:
            console.print(
                tr(
                    "cli_msgs.profile_not_found",
                    "[yellow]Warning: Profile '{name}' not found[/yellow]",
                    name=args.profile,
                )
            )

    # Override with command line arguments
    if args.port:
        config["port"] = args.port
    if args.host:
        config["host"] = args.host
    if args.quantization:
        config["quantization"] = args.quantization

    # Handle HF token if provided via CLI
    if hasattr(args, "hf_token") and args.hf_token:
        console.print(
            tr(
                "cli_msgs.validating_hf_token",
                "[cyan]Validating HuggingFace token...[/cyan]",
            )
        )

        from ..validation.token import validate_hf_token

        is_valid, user_info = validate_hf_token(args.hf_token)

        if is_valid:
            # Save the token to config for this session
            config_manager.config["hf_token"] = args.hf_token
            config_manager._save_config()
            console.print(
                tr(
                    "cli_msgs.token_saved",
                    "[green]✓ Token validated and saved[/green]",
                )
            )
            if user_info:
                console.print(
                    tr(
                        "cli_msgs.authenticated_as",
                        "[dim]Authenticated as: {name}[/dim]",
                        name=user_info.get("name", "Unknown"),
                    )
                )
        else:
            console.print(
                tr(
                    "cli_msgs.token_validation_failed",
                    "[yellow]Warning: Token validation failed[/yellow]",
                )
            )
            console.print(
                tr(
                    "cli_msgs.token_maybe_invalid",
                    "[dim]The token may be invalid or expired. Continuing anyway...[/dim]",
                )
            )
            # Still save it in case it's a network issue or special token type
            config_manager.config["hf_token"] = args.hf_token
            config_manager._save_config()
    if args.tensor_parallel_size:
        config["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization != 0.9:
        config["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_model_len is not None:
        config["max_model_len"] = args.max_model_len
    if args.dtype != "auto":
        config["dtype"] = args.dtype

    # Handle GPU device selection
    if hasattr(args, "device") and args.device:
        device_str = str(args.device)
        try:
            config["device_ids"] = [
                int(v.strip()) for v in device_str.split(",") if v.strip()
            ]
        except ValueError:
            config["device_ids"] = [
                v.strip() for v in device_str.split(",") if v.strip()
            ]

    # Handle LoRA adapters
    if hasattr(args, "lora") and args.lora:
        # Enable LoRA if adapters are specified
        config["enable_lora"] = True

        # Format LoRA modules for vLLM
        lora_modules = []
        for lora_spec in args.lora:
            if "=" in lora_spec:
                # Format: name=path
                lora_modules.append(lora_spec)
            else:
                # Just path, generate a name from the path
                from pathlib import Path

                lora_path = Path(lora_spec)
                lora_name = lora_path.name.replace("-", "_").replace(" ", "_")
                lora_modules.append(f"{lora_name}={lora_spec}")

        # Join modules for command line
        config["lora_modules"] = " ".join(lora_modules)

        console.print(
            tr(
                "cli_msgs.enabling_lora",
                "[blue]Enabling LoRA with {count} adapter(s)[/blue]",
                count=len(lora_modules),
            )
        )
        for module in lora_modules:
            console.print(f"  • {module}")

    elif hasattr(args, "enable_lora") and args.enable_lora:
        config["enable_lora"] = True

    if args.extra_args:
        config["extra_args"] = args.extra_args

    return config


def handle_dirs(args: argparse.Namespace) -> bool:
    """
    Directory management is now handled by hf-model-tool.
    This command redirects users to use hf-model-tool.

    Args:
        args: Parsed command line arguments

    Returns:
        True if operation succeeded, False otherwise
    """
    import os
    import subprocess

    console.print(
        tr(
            "cli_msgs.dirs_moved",
            "[yellow]Directory management has moved to hf-model-tool[/yellow]",
        )
    )
    console.print(
        tr("cli_msgs.manage_dirs_intro", "\nYou can manage directories using:")
    )
    console.print(
        tr(
            "cli_msgs.dirs_tool_interactive",
            "  • [cyan]hf-model-tool[/cyan] - Interactive interface with Config menu",
        )
    )
    console.print(
        tr(
            "cli_msgs.dirs_tool_add",
            "  • [cyan]hf-model-tool --add-path <path>[/cyan] - Add a directory directly",
        )
    )

    if hasattr(args, "dirs_command"):
        if args.dirs_command in ["add", "remove", "list"]:
            console.print(
                tr(
                    "cli_msgs.launching_hf_tool",
                    "\n[dim]Launching hf-model-tool for directory management...[/dim]",
                )
            )

            try:
                # Launch hf-model-tool
                if args.dirs_command == "add" and hasattr(args, "path"):
                    # If adding a path, use the --add-path argument
                    subprocess.run(
                        ["hf-model-tool", "--add-path", args.path],
                        env=os.environ.copy(),
                    )
                else:
                    # Otherwise launch interactive mode
                    subprocess.run(["hf-model-tool"], env=os.environ.copy())
                return True
            except FileNotFoundError:
                console.print(
                    tr(
                        "cli_msgs.hf_tool_missing",
                        "\n[red]hf-model-tool not found. Please install it:[/red]",
                    )
                )
                console.print("  pip install hf-model-tool")
                return False
            except Exception as e:
                console.print(
                    tr(
                        "cli_msgs.dirs_tool_error",
                        "\n[red]Error launching hf-model-tool: {error}[/red]",
                        error=e,
                    )
                )
                return False

    return True


def handle_proxy(args: argparse.Namespace) -> bool:
    """
    Handle the 'proxy' command for multi-model serving.

    Args:
        args: Parsed command line arguments

    Returns:
        True if command executed successfully
    """
    import json
    from pathlib import Path

    # Initialize managers
    config_manager = ProxyConfigManager()

    # Handle proxy subcommands
    if not hasattr(args, "proxy_command") or not args.proxy_command:
        console.print(
            tr(
                "cli_msgs.specify_proxy_command",
                "[yellow]Please specify a proxy command.[/yellow]",
            )
        )
        console.print(
            tr(
                "cli_msgs.available_proxy_commands",
                "Available commands: start, stop, status, add, remove, config",
            )
        )
        return False

    if args.proxy_command == "start":
        # Start proxy server
        console.print(
            tr(
                "cli_msgs.starting_proxy",
                "[bold cyan]Starting Multi-Model Proxy Server[/bold cyan]",
            )
        )

        # Load or create configuration
        if hasattr(args, "config") and args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                console.print(
                    tr(
                        "cli_msgs.config_file_not_found",
                        "[red]Config file not found: {path}[/red]",
                        path=config_path,
                    )
                )
                return False
            proxy_config = config_manager.load_config(config_path)
        elif hasattr(args, "interactive") and args.interactive:
            # Interactive configuration
            from ..ui.proxy.control import configure_proxy_interactively

            proxy_config = configure_proxy_interactively()
            if not proxy_config:
                return False
        else:
            # Use default or saved configuration
            proxy_config = config_manager.load_config()

        # Override host/port only if explicitly specified on command line
        if hasattr(args, "host") and args.host is not None:
            proxy_config.host = args.host
        if hasattr(args, "port") and args.port is not None:
            proxy_config.port = args.port

        # Validate configuration
        errors = config_manager.validate_config(proxy_config)
        if errors:
            console.print(
                tr("cli_msgs.configuration_errors", "[red]Configuration errors:[/red]")
            )
            for error in errors:
                console.print(f"  • {error}")
            return False

        # Create proxy manager
        proxy_manager = ProxyManager(proxy_config)

        # Auto-allocate GPUs if requested
        if hasattr(args, "auto_allocate") and args.auto_allocate:
            console.print(
                tr(
                    "cli_msgs.auto_allocating_gpus",
                    "[cyan]Auto-allocating GPUs to models...[/cyan]",
                )
            )
            allocated = proxy_manager.allocate_gpus_automatically()
            proxy_config.models = allocated

        # Start all models with monitoring
        console.print(
            tr(
                "cli_msgs.launching_model_servers",
                "\n[cyan]Launching model servers...[/cyan]",
            )
        )
        launched = proxy_manager.start_all_models_no_wait()

        if launched > 0:
            # Monitor startup progress with live logs
            from ..proxy import monitor_startup_progress

            all_started = monitor_startup_progress(proxy_manager)

            if not all_started:
                console.print(
                    tr(
                        "cli_msgs.some_models_failed",
                        "[yellow]Warning: Some models failed to start.[/yellow]",
                    )
                )
                console.print(
                    tr(
                        "cli_msgs.continuing_available_models",
                        "[dim]Continuing with available models...[/dim]",
                    )
                )

        # Start proxy
        if proxy_manager.start_proxy():
            console.print(
                tr(
                    "cli_msgs.proxy_running_at",
                    "\n[green]✓ Proxy server running at {url}[/green]",
                    url=f"http://{proxy_config.host}:{proxy_config.port}",
                )
            )
            console.print(tr("cli_msgs.available_endpoints", "\nAvailable endpoints:"))
            console.print(
                tr(
                    "cli_msgs.endpoint_openai_api",
                    "  • OpenAI API: {url}",
                    url=f"http://{proxy_config.host}:{proxy_config.port}/v1/",
                )
            )
            console.print(
                tr(
                    "cli_msgs.endpoint_models",
                    "  • Models list: {url}",
                    url=f"http://{proxy_config.host}:{proxy_config.port}/v1/models",
                )
            )
            console.print(
                tr(
                    "cli_msgs.endpoint_status",
                    "  • Proxy status: {url}",
                    url=f"http://{proxy_config.host}:{proxy_config.port}/proxy/status",
                )
            )

            if proxy_config.enable_metrics:
                console.print(
                    tr(
                        "cli_msgs.endpoint_metrics",
                        "  • Metrics: {url}",
                        url=f"http://{proxy_config.host}:{proxy_config.port}/metrics",
                    )
                )

            console.print(
                tr(
                    "cli_msgs.ctrl_c_monitoring",
                    "\n[dim]Press Ctrl+C for monitoring options[/dim]",
                )
            )

            # Keep running with monitoring option
            try:
                import time

                time.sleep(2)  # Give servers a moment to fully start

                while True:
                    # Show monitoring menu
                    console.print(
                        tr(
                            "cli_msgs.proxy_server_running",
                            "\n[bold cyan]Proxy Server Running[/bold cyan]",
                        )
                    )

                    try:
                        choice = prompt_choice(
                            "proxy_running",
                            tr("cli_msgs.prompt_select_action", "Select action"),
                            [
                                (
                                    "monitor",
                                    tr("cli_msgs.opt_monitor", "Monitor proxy (all options)"),
                                ),
                                (
                                    "background",
                                    tr("cli_msgs.opt_background", "Continue running (background)"),
                                ),
                                ("stop", tr("cli_msgs.opt_stop_proxy", "Stop proxy server")),
                            ],
                            allow_back=False,
                        )

                        if choice == "monitor":
                            from ..ui.proxy.monitor import monitor_proxy

                            result = monitor_proxy(proxy_manager)
                            if result == "back":
                                continue  # Return to main proxy menu
                        elif choice == "background":
                            console.print(
                                tr(
                                    "cli_msgs.proxy_continues_background",
                                    "\n[green]Proxy continues running in background[/green]",
                                )
                            )
                            console.print(
                                tr(
                                    "cli_msgs.access_at",
                                    "Access at: {url}",
                                    url=f"http://{proxy_config.host}:{proxy_config.port}",
                                )
                            )
                            console.print(
                                tr(
                                    "cli_msgs.proxy_stops_on_exit",
                                    "[dim]The proxy will stop when you exit the program[/dim]",
                                )
                            )
                            time.sleep(2)
                        elif choice == "stop":
                            console.print(
                                tr(
                                    "cli_msgs.stopping_proxy",
                                    "\n[yellow]Stopping proxy server...[/yellow]",
                                )
                            )
                            proxy_manager.stop_proxy()
                            break

                    except KeyboardInterrupt:
                        # Nested Ctrl+C - ask if they want to stop
                        console.print(
                            tr(
                                "cli_msgs.interrupt_received",
                                "\n[yellow]Interrupt received[/yellow]",
                            )
                        )
                        if (
                            prompt_choice(
                                "confirm_stop",
                                tr("cli_msgs.prompt_stop_proxy", "Stop the proxy server?"),
                                [
                                    ("yes", tr("cli_msgs.opt_yes_stop", "Yes, stop proxy")),
                                    (
                                        "no",
                                        tr("cli_msgs.opt_no_continue", "No, continue running"),
                                    ),
                                ],
                                allow_back=False,
                            )
                            == "yes"
                        ):
                            console.print(
                                tr(
                                    "cli_msgs.stopping_proxy",
                                    "\n[yellow]Stopping proxy server...[/yellow]",
                                )
                            )
                            proxy_manager.stop_proxy()
                            break

            except KeyboardInterrupt:
                console.print(
                    tr(
                        "cli_msgs.stopping_proxy",
                        "\n[yellow]Stopping proxy server...[/yellow]",
                    )
                )
                proxy_manager.stop_proxy()

            return True
        else:
            console.print(
                tr(
                    "cli_msgs.proxy_start_failed",
                    "[red]Failed to start proxy server[/red]",
                )
            )
            return False

    elif args.proxy_command == "stop":
        # Stop proxy server
        console.print(
            tr(
                "cli_msgs.stopping_proxy_and_models",
                "[yellow]Stopping proxy server and all models...[/yellow]",
            )
        )
        # This would need a way to find and stop the running proxy
        # For now, we'll just inform the user
        console.print(
            tr(
                "cli_msgs.use_ctrl_c_to_stop",
                "[dim]Use Ctrl+C in the proxy terminal to stop it[/dim]",
            )
        )
        return True

    elif args.proxy_command == "status":
        # Show proxy status
        import httpx

        from ..proxy.runtime import get_proxy_connection

        # Get proxy connection details dynamically
        proxy_host, proxy_port = get_proxy_connection(
            cli_host=getattr(args, "proxy_host", None),
            cli_port=getattr(args, "proxy_port", None),
        )

        try:
            # Try to connect to proxy
            response = httpx.get(
                f"http://{proxy_host}:{proxy_port}/proxy/status", timeout=5
            )
            if response.status_code == 200:
                status = response.json()

                if hasattr(args, "json") and args.json:
                    console.print(json.dumps(status, indent=2))
                else:
                    # Display formatted status
                    console.print(
                        tr(
                            "cli_msgs.proxy_status_title",
                            "\n[bold cyan]Proxy Server Status[/bold cyan]",
                        )
                    )
                    console.print(
                        tr(
                            "cli_msgs.proxy_status_running",
                            "Status: [green]Running[/green]",
                        )
                    )
                    console.print(
                        tr(
                            "cli_msgs.proxy_address",
                            "Address: {url}",
                            url=f"http://{status['proxy_host']}:{status['proxy_port']}",
                        )
                    )
                    console.print(
                        tr(
                            "cli_msgs.total_requests",
                            "Total Requests: {count}",
                            count=status.get("total_requests", 0),
                        )
                    )

                    if status.get("models"):
                        console.print(
                            tr(
                                "cli_msgs.active_models",
                                "\n[bold]Active Models:[/bold]",
                            )
                        )
                        table = Table()
                        table.add_column(tr("cli_msgs.col_model", "Model"), style="cyan")
                        table.add_column(tr("cli_msgs.col_port", "Port"), style="magenta")
                        table.add_column(tr("cli_msgs.col_gpus", "GPUs"), style="green")
                        table.add_column(tr("cli_msgs.col_status", "Status"), style="yellow")
                        table.add_column(tr("cli_msgs.col_requests", "Requests"), style="dim")

                        for model in status["models"]:
                            gpu_str = ",".join(str(g) for g in model.get("gpu_ids", []))
                            table.add_row(
                                model["name"],
                                str(model["port"]),
                                gpu_str or "N/A",
                                model["status"],
                                str(model.get("request_count", 0)),
                            )

                        console.print(table)
                    else:
                        console.print(
                            tr(
                                "cli_msgs.no_models_configured",
                                "\n[yellow]No models configured[/yellow]",
                            )
                        )
            else:
                console.print(
                    tr(
                        "cli_msgs.proxy_returned_error",
                        "[red]Proxy server returned error[/red]",
                    )
                )

        except httpx.ConnectError:
            console.print(
                tr(
                    "cli_msgs.proxy_not_running_warn",
                    "[yellow]Proxy server is not running at {host}:{port}[/yellow]",
                    host=proxy_host,
                    port=proxy_port,
                )
            )
            console.print(
                tr("cli_msgs.start_proxy_hint", "Start it with: vllm-cli proxy start")
            )
            console.print(
                tr(
                    "cli_msgs.proxy_host_port_tip",
                    "[dim]Tip: Use --proxy-host and --proxy-port to specify a different proxy server[/dim]",
                )
            )
        except Exception as e:
            console.print(
                tr(
                    "cli_msgs.proxy_status_error",
                    "[red]Error checking proxy status: {error}[/red]",
                    error=e,
                )
            )

        return True

    elif args.proxy_command == "add":
        # Add model to running proxy
        if not all([hasattr(args, "name"), hasattr(args, "model_path")]):
            console.print(
                tr(
                    "cli_msgs.model_name_path_required",
                    "[red]Model name and path are required[/red]",
                )
            )
            return False

        model_config = ModelConfig(
            name=args.name,
            model_path=args.model_path,
            gpu_ids=args.gpu if hasattr(args, "gpu") and args.gpu else [],
            port=args.port if hasattr(args, "port") and args.port else 8001,
            profile=args.profile if hasattr(args, "profile") else None,
        )

        # Send request to proxy to add model
        import httpx

        from ..proxy.runtime import get_proxy_connection

        # Get proxy connection details dynamically
        proxy_host, proxy_port = get_proxy_connection(
            cli_host=getattr(args, "proxy_host", None),
            cli_port=getattr(args, "proxy_port", None),
        )

        try:
            response = httpx.post(
                f"http://{proxy_host}:{proxy_port}/proxy/add_model",
                json=model_config.dict(),
                timeout=10,
            )
            if response.status_code == 200:
                console.print(
                    tr(
                        "cli_msgs.model_added",
                        "[green]✓ Model '{name}' added successfully[/green]",
                        name=args.name,
                    )
                )
            else:
                console.print(
                    tr(
                        "cli_msgs.add_model_failed",
                        "[red]Failed to add model: {detail}[/red]",
                        detail=response.text,
                    )
                )
        except httpx.ConnectError:
            console.print(
                tr(
                    "cli_msgs.proxy_not_running_error",
                    "[red]Proxy server is not running at {host}:{port}[/red]",
                    host=proxy_host,
                    port=proxy_port,
                )
            )
        except Exception as e:
            console.print(
                tr(
                    "cli_msgs.add_model_error",
                    "[red]Error adding model: {error}[/red]",
                    error=e,
                )
            )

        return True

    elif args.proxy_command == "remove":
        # Remove model from proxy
        if not hasattr(args, "name"):
            console.print(
                tr("cli_msgs.model_name_required", "[red]Model name is required[/red]")
            )
            return False

        import httpx

        from ..proxy.runtime import get_proxy_connection

        # Get proxy connection details dynamically
        proxy_host, proxy_port = get_proxy_connection(
            cli_host=getattr(args, "proxy_host", None),
            cli_port=getattr(args, "proxy_port", None),
        )

        try:
            response = httpx.delete(
                f"http://{proxy_host}:{proxy_port}/proxy/remove_model/{args.name}",
                timeout=10,
            )
            if response.status_code == 200:
                console.print(
                    tr(
                        "cli_msgs.model_removed",
                        "[green]✓ Model '{name}' removed[/green]",
                        name=args.name,
                    )
                )
            else:
                console.print(
                    tr(
                        "cli_msgs.remove_model_failed",
                        "[red]Failed to remove model: {detail}[/red]",
                        detail=response.text,
                    )
                )
        except httpx.ConnectError:
            console.print(
                tr(
                    "cli_msgs.proxy_not_running_error",
                    "[red]Proxy server is not running at {host}:{port}[/red]",
                    host=proxy_host,
                    port=proxy_port,
                )
            )
        except Exception as e:
            console.print(
                tr(
                    "cli_msgs.remove_model_error",
                    "[red]Error removing model: {error}[/red]",
                    error=e,
                )
            )

        return True

    elif args.proxy_command == "config":
        # Manage proxy configuration
        if hasattr(args, "create") and args.create:
            # Create example configuration
            output_path = Path(args.create)
            example_config = config_manager.create_example_config()
            config_manager.save_config(example_config, output_path)
            console.print(
                tr(
                    "cli_msgs.example_config_saved",
                    "[green]✓ Example configuration saved to {path}[/green]",
                    path=output_path,
                )
            )

        elif hasattr(args, "edit") and args.edit:
            # Edit configuration interactively
            from ..ui.proxy.control import edit_proxy_config

            edit_proxy_config()

        elif hasattr(args, "export") and args.export:
            # Export current configuration
            output_path = Path(args.export)
            current_config = config_manager.load_config()
            config_manager.export_config(current_config, output_path)
            console.print(
                tr(
                    "cli_msgs.config_exported",
                    "[green]✓ Configuration exported to {path}[/green]",
                    path=output_path,
                )
            )

        else:
            # Show current configuration
            current_config = config_manager.load_config()
            console.print(
                tr(
                    "cli_msgs.current_proxy_config",
                    "\n[bold cyan]Current Proxy Configuration[/bold cyan]",
                )
            )
            console.print(
                tr("cli_msgs.line_host", "Host: {value}", value=current_config.host)
            )
            console.print(
                tr("cli_msgs.line_port", "Port: {value}", value=current_config.port)
            )
            console.print(
                tr(
                    "cli_msgs.cors_enabled",
                    "CORS Enabled: {value}",
                    value=current_config.enable_cors,
                )
            )
            console.print(
                tr(
                    "cli_msgs.metrics_enabled",
                    "Metrics Enabled: {value}",
                    value=current_config.enable_metrics,
                )
            )
            console.print(
                tr(
                    "cli_msgs.request_logging",
                    "Request Logging: {value}",
                    value=current_config.log_requests,
                )
            )

            if current_config.models:
                console.print(
                    tr(
                        "cli_msgs.configured_models_count",
                        "\nConfigured Models ({count}):",
                        count=len(current_config.models),
                    )
                )
                for model in current_config.models:
                    status = (
                        tr("cli_msgs.enabled", "[green]enabled[/green]")
                        if model.enabled
                        else tr("cli_msgs.disabled", "[dim]disabled[/dim]")
                    )
                    console.print(f"  • {model.name}: {model.model_path} ({status})")
            else:
                console.print(
                    tr(
                        "cli_msgs.no_models_configured",
                        "\n[yellow]No models configured[/yellow]",
                    )
                )

        return True

    else:
        console.print(
            tr(
                "cli_msgs.unknown_proxy_command",
                "[red]Unknown proxy command: {command}[/red]",
                command=args.proxy_command,
            )
        )
        return False


def handle_recipes(args: argparse.Namespace) -> bool:
    """
    Handle the 'recipes' command to list/sync vLLM recipes.

    Args:
        args: Parsed command line arguments

    Returns:
        True if command executed successfully
    """
    from ..config.recipes_parser import RecipesParser
    from ..ui.recipes_sync import sync_recipes

    parser = RecipesParser()

    if args.list:
        try:
            recipes = parser.fetch_recipes_list()
            if not recipes:
                console.print(
                    tr(
                        "cli_msgs.no_recipes_found",
                        "[yellow]No recipes found.[/yellow]",
                    )
                )
                return True

            table = Table(title=tr("cli_msgs.available_recipes", "Available vLLM Recipes"))
            table.add_column(tr("cli_msgs.col_id", "ID"), style="cyan")
            table.add_column(tr("cli_msgs.col_model", "Model"), style="green")
            table.add_column(tr("cli_msgs.col_hardware", "Hardware"), style="yellow")
            table.add_column(tr("cli_msgs.col_category", "Category"), style="blue")

            for recipe in recipes:
                if recipe.get("category") == "root":
                    continue
                table.add_row(
                    recipe.get("id", ""),
                    recipe.get("model_name", ""),
                    recipe.get("hardware", "default"),
                    recipe.get("category", ""),
                )

            console.print(table)
            return True

        except Exception as e:
            console.print(
                tr(
                    "cli_msgs.fetch_recipes_failed",
                    "[red]Failed to fetch recipes: {error}[/red]",
                    error=e,
                )
            )
            return False

    elif args.sync:
        sync_recipes()
        return True

    elif args.import_recipe:
        try:
            profile = parser.import_recipe(args.import_recipe)
            if profile:
                console.print(
                    tr(
                        "cli_msgs.recipe_imported",
                        "[green]Recipe imported successfully![/green]",
                    )
                )
                console.print(
                    tr(
                        "cli_msgs.line_profile_name",
                        "Profile: {name}",
                        name=profile.get("name"),
                    )
                )
                return True
            else:
                console.print(
                    tr(
                        "cli_msgs.import_recipe_failed",
                        "[red]Failed to import recipe: {name}[/red]",
                        name=args.import_recipe,
                    )
                )
                return False
        except Exception as e:
            console.print(
                tr(
                    "cli_msgs.import_recipe_error",
                    "[red]Error importing recipe: {error}[/red]",
                    error=e,
                )
            )
            return False

    elif args.sync_args:
        from ..config.cli_args_sync import sync_cli_args as do_sync_args

        apply_changes = getattr(args, "apply_args", False)
        result = do_sync_args(dry_run=not apply_changes, verbose=True)

        if result["new_count"] > 0 or result["removed_count"] > 0:
            if not apply_changes:
                console.print(
                    tr(
                        "cli_msgs.apply_args_hint",
                        "\n[yellow]Run with --apply-args to update the schema.[/yellow]",
                    )
                )
            else:
                console.print(
                    tr(
                        "cli_msgs.cli_args_updated",
                        "\n[green]✓ CLI args schema updated successfully![/green]",
                    )
                )
                if result["removed_count"] > 0:
                    console.print(
                        tr(
                            "cli_msgs.clean_deprecated_hint",
                            "\n[yellow]Run with --clean-deprecated to remove deprecated arguments from the schema.[/yellow]",
                        )
                    )
        else:
            console.print(
                tr(
                    "cli_msgs.cli_args_up_to_date",
                    "\n[green]✓ CLI args are up to date.[/green]",
                )
            )
        return True

    elif getattr(args, "clean_deprecated", False):
        from ..config.cli_args_sync import clean_deprecated_args as do_clean

        result = do_clean(verbose=True)
        if result["cleaned_count"] > 0:
            console.print(
                tr(
                    "cli_msgs.cleaned_deprecated",
                    "\n[green]✓ Cleaned {count} deprecated arguments![/green]",
                    count=result["cleaned_count"],
                )
            )
        else:
            console.print(
                tr(
                    "cli_msgs.no_deprecated_to_clean",
                    "\n[green]✓ No deprecated arguments to clean.[/green]",
                )
            )
        return True

    elif args.sync_parsers:
        from ..config.parser_sync import sync_parsers as do_sync_parsers

        apply_changes = getattr(args, "apply_parsers", False)
        result = do_sync_parsers(dry_run=not apply_changes, verbose=True)

        new_tool = len(result["tool_call_parser"]["new"])
        new_reasoning = len(result["reasoning_parser"]["new"])

        if new_tool > 0 or new_reasoning > 0:
            if not apply_changes:
                console.print(
                    tr(
                        "cli_msgs.apply_parsers_hint",
                        "\n[yellow]Run with --apply-parsers to update the schema.[/yellow]",
                    )
                )
            else:
                console.print(
                    tr(
                        "cli_msgs.parsers_updated",
                        "\n[green]✓ Parser choices updated successfully![/green]",
                    )
                )
        else:
            console.print(
                tr(
                    "cli_msgs.parsers_up_to_date",
                    "\n[green]✓ Parser choices are up to date.[/green]",
                )
            )
        return True

    else:
        console.print(
            tr(
                "cli_msgs.specify_recipe_action",
                "[yellow]Please specify an action: --list, --sync, --sync-args, --sync-parsers, or --import[/yellow]",
            )
        )
        return False


def handle_import(args: argparse.Namespace) -> bool:
    """
    Handle the 'import' command to parse a vllm serve command and create a profile.

    Args:
        args: Parsed command line arguments

    Returns:
        True if import successful, False otherwise
    """
    try:
        from ..config.command_import import CommandImporter

        # Get the command
        command = getattr(args, 'raw_command', None)
        if args.file:
            try:
                with open(args.file, "r") as f:
                    command = f.read().strip()
            except Exception as e:
                console.print(
                    tr(
                        "cli_msgs.read_file_error",
                        "[red]Error reading file: {error}[/red]",
                        error=e,
                    )
                )
                return False

        if not command:
            console.print(
                tr(
                    "cli_msgs.no_command_provided",
                    "[red]Error: No command provided.[/red]",
                )
            )
            console.print(
                tr(
                    "cli_msgs.import_usage",
                    "Usage: vllm-cli import 'vllm serve model --flag value'",
                )
            )
            console.print(
                tr(
                    "cli_msgs.import_usage_file",
                    "   or: vllm-cli import --file command.txt",
                )
            )
            console.print("")
            console.print(
                tr(
                    "cli_msgs.pipe_stdin_hint",
                    "You can also pipe a command via stdin:",
                )
            )
            console.print("   echo 'vllm serve model --flag value' | vllm-cli import --stdin")
            return False

        # Parse the command
        importer = CommandImporter()

        if args.preview:
            # Show preview
            preview = importer.preview(command)
            console.print(Panel(
                preview,
                title=tr("cli_msgs.import_preview", "[cyan]Import Preview[/cyan]"),
                border_style="cyan",
            ))
            return True

        # Convert to profile
        profile = importer.to_profile(command, name=args.name, description=args.description)

        # Display the profile
        console.print(
            tr("cli_msgs.parsed_configuration", "[green]Parsed configuration:[/green]")
        )
        console.print(
            tr(
                "cli_msgs.line_model",
                "  Model: {value}",
                value=profile["config"].get("model", profile.get("model", "N/A")),
            )
        )
        console.print(
            tr(
                "cli_msgs.line_profile_field",
                "  Profile name: {name}",
                name=profile["name"],
            )
        )
        console.print(
            tr(
                "cli_msgs.line_config_keys",
                "  Config keys: {count}",
                count=len(profile["config"]),
            )
        )
        console.print("")

        # Show config summary
        table = Table(title=tr("cli_msgs.title_configuration", "Configuration"))
        table.add_column(tr("cli_msgs.col_key", "Key"), style="cyan")
        table.add_column(tr("cli_msgs.col_value", "Value"), style="green")

        for key, value in sorted(profile["config"].items()):
            table.add_row(key, str(value))

        console.print(table)

        # Save the profile
        config_manager = ConfigManager()
        if config_manager.save_user_profile(profile["name"], profile):
            console.print(
                tr(
                    "cli_msgs.profile_saved",
                    "\n[green]✓ Profile '{name}' saved successfully![/green]",
                    name=profile["name"],
                )
            )
            console.print(
                tr(
                    "cli_msgs.use_profile_hint",
                    "  Use: vllm-cli serve --profile {name}",
                    name=profile["name"],
                )
            )
            return True
        else:
            console.print(
                tr(
                    "cli_msgs.save_profile_failed",
                    "\n[red]Failed to save profile '{name}'[/red]",
                    name=profile["name"],
                )
            )
            return False

    except Exception as e:
        console.print(
            tr(
                "cli_msgs.import_command_error",
                "[red]Error importing command: {error}[/red]",
                error=e,
            )
        )
        logger.exception("Import failed")
        return False
