#!/usr/bin/env python3
"""
Minimal integration with hf-model-tool for model management.

This module provides subprocess launchers for hf-model-tool,
delegating all model management functionality to the external tool.
"""
import logging
import os
import subprocess

from rich.console import Console
from rich.panel import Panel

from ..i18n import tr

logger = logging.getLogger(__name__)
console = Console()


def launch_hf_model_tool(args: list = None) -> None:
    """
    Launch hf-model-tool with optional arguments.

    Args:
        args: Optional list of command-line arguments for hf-model-tool
    """
    cmd = ["hf-model-tool"]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(cmd, env=os.environ.copy())
        if result.returncode != 0:
            console.print(
                tr(
                    "hf_ui.tool_exited_error",
                    "[yellow]hf-model-tool exited with an error.[/yellow]",
                )
            )
    except FileNotFoundError:
        console.print(
            tr(
                "hf_ui.tool_not_found",
                "[red]hf-model-tool not found. Please install it:[/red]",
            )
        )
        console.print("  pip install hf-model-tool")
    except Exception as e:
        console.print(
            tr(
                "hf_ui.launch_error",
                "[red]Error launching hf-model-tool: {error}[/red]",
                error=e,
            )
        )


def check_hf_model_tool_installed() -> bool:
    """
    Check if hf-model-tool is installed and available.

    Returns:
        True if hf-model-tool is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["hf-model-tool", "--version"],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except Exception:
        return False


# Legacy function names for backward compatibility (will be removed in future)
def launch_hf_model_tool_interactive() -> str:
    """Legacy function - launches hf-model-tool in interactive mode."""
    console.clear()
    console.print(
        Panel(
            f"[bold cyan]{tr('hf_ui.launching_title', 'Launching HF-Model-Tool')}[/bold cyan]\n"
            f"[dim]{tr('hf_ui.launching_subtitle', 'Full model management interface')}[/dim]",
            border_style="blue",
        )
    )
    launch_hf_model_tool()
    input("\n" + tr("common.press_enter", "Press Enter to continue..."))
    return "continue"


def launch_hf_model_tool_manage() -> str:
    """Legacy function - launches hf-model-tool in manage mode."""
    console.clear()
    console.print(
        Panel(
            f"[bold cyan]{tr('hf_ui.asset_management_title', 'Asset Management')}[/bold cyan]\n"
            f"[dim]{tr('hf_ui.asset_management_subtitle', 'Delete, deduplicate, and organize models')}[/dim]",
            border_style="blue",
        )
    )
    launch_hf_model_tool(["--manage"])
    input("\n" + tr("common.press_enter", "Press Enter to continue..."))
    return "continue"
