#!/usr/bin/env python3
"""
Model management module for vLLM CLI.

Handles model selection and delegates management to hf-model-tool.
"""

import logging
from typing import Any, Dict, Optional

import inquirer
from rich.panel import Panel

from ..i18n import tr
from ..models import list_available_models
from ..system import format_size
from .common import console
from .navigation import prompt_choice, unified_prompt

logger = logging.getLogger(__name__)


def select_shortcut_for_serving() -> Optional[Dict[str, Any]]:
    """
    Select a shortcut for serving, returning it in a special format.

    Returns:
        Dictionary with shortcut information including model and profile
    """
    from ..config import ConfigManager

    config_manager = ConfigManager()
    shortcuts = config_manager.list_shortcuts()

    if not shortcuts:
        console.print(
            tr("model_mgr.no_shortcuts", "[yellow]No shortcuts available.[/yellow]")
        )
        return None

    # Build shortcut choices with details
    shortcut_choices = []
    for shortcut in shortcuts:
        name = shortcut["name"]
        model = shortcut["model"]
        profile = shortcut["profile"]
        # Truncate long model names for display
        if len(str(model)) > 40:
            model_display = "..." + str(model)[-37:]
        else:
            model_display = str(model)
        shortcut_choices.append(f"{name}: {model_display} [{profile}]")

    # Show shortcut selection
    console.print(
        tr("model_mgr.select_shortcut", "\n[bold cyan]Select Shortcut[/bold cyan]")
    )
    selected = unified_prompt(
        "shortcut_select",
        tr("model_mgr.choose_shortcut", "Choose a shortcut to use"),
        shortcut_choices,
        allow_back=True,
    )

    if not selected or selected == "BACK":
        return None

    # Extract shortcut name from selection
    shortcut_name = selected.split(":")[0].strip()

    # Get full shortcut data
    shortcut_data = config_manager.get_shortcut(shortcut_name)
    if not shortcut_data:
        console.print(
            tr(
                "model_mgr.shortcut_not_found",
                "[red]Shortcut '{name}' not found.[/red]",
                name=shortcut_name,
            )
        )
        return None

    # Update last used timestamp
    config_manager.shortcut_manager.update_last_used(shortcut_name)

    # Return shortcut in a special format that indicates it's a shortcut
    return {
        "type": "shortcut",
        "name": shortcut_name,
        "model": shortcut_data["model"],
        "profile": shortcut_data["profile"],
        "config_overrides": shortcut_data.get("config_overrides", {}),
    }


def enter_remote_model() -> Optional[str]:
    """
    Allow user to enter a HuggingFace model ID for remote serving.

    Returns:
        Model ID string or None if cancelled
    """
    import getpass

    from ..config import ConfigManager

    console.print(
        tr("model_mgr.remote_title", "\n[bold cyan]Remote Model Selection[/bold cyan]")
    )
    console.print(
        tr(
            "model_mgr.remote_intro",
            "\nEnter a HuggingFace model ID to serve directly from the Hub.",
        )
    )
    console.print(
        tr(
            "model_mgr.remote_auto_download",
            "The model will be automatically downloaded on first use.\n",
        )
    )

    console.print(tr("model_mgr.examples_label", "[dim]Examples:[/dim]"))
    console.print("  • openai/gpt-oss-120b\n")

    console.print(
        tr(
            "model_mgr.first_download_note",
            "[yellow]Note:[/yellow] First-time download may take 10-30 minutes "
            "depending on model size.\n",
        )
    )

    while True:
        console.print(
            tr(
                "model_mgr.model_id_prompt",
                "[cyan]Model ID (or 'back' to cancel):[/cyan] ",
            ),
            end="",
        )
        model_id = input().strip()

        if model_id.lower() in ["back", "cancel", ""]:
            return None

        # Validate model ID format
        if "/" not in model_id:
            console.print(
                tr(
                    "model_mgr.invalid_model_id_format",
                    "[red]Invalid format. Model ID should be "
                    "'organization/model-name'[/red]",
                )
            )
            continue

        parts = model_id.split("/")
        if len(parts) != 2:
            console.print(
                tr(
                    "model_mgr.invalid_model_id_format",
                    "[red]Invalid format. Model ID should be "
                    "'organization/model-name'[/red]",
                )
            )
            continue

        org, model_name = parts
        if not org or not model_name:
            console.print(
                tr(
                    "model_mgr.invalid_model_id_parts",
                    "[red]Invalid model ID. Both organization and model name are "
                    "required.[/red]",
                )
            )
            continue

        # Confirm the selection
        console.print(
            tr(
                "model_mgr.selected_model",
                "\n[bold]Selected model:[/bold] {model}",
                model=model_id,
            )
        )

        # Check for HF token and offer to configure if needed
        config_manager = ConfigManager()
        has_token = bool(config_manager.config.get("hf_token"))

        if has_token:
            console.print(
                tr(
                    "model_mgr.token_configured",
                    "[green]✓ HuggingFace token is configured[/green]",
                )
            )
        else:
            console.print(
                tr(
                    "model_mgr.token_needed_note",
                    "\n[dim]Note: Some models require a HuggingFace token for "
                    "access.[/dim]",
                )
            )
            console.print(
                tr(
                    "model_mgr.token_gated_note",
                    "[dim]If this model is gated, you'll need to provide a "
                    "token.[/dim]\n",
                )
            )

            token_action = prompt_choice(
                "token_action",
                tr(
                    "model_mgr.configure_token_title",
                    "Would you like to configure a HuggingFace token?",
                ),
                [
                    (
                        "configure",
                        tr("model_mgr.configure_token", "Configure HF token now"),
                    ),
                    ("skip", tr("model_mgr.skip_token", "Continue without token")),
                ],
            )

            if token_action == "configure":
                console.print(
                    tr(
                        "model_mgr.enter_token_prompt",
                        "\n[cyan]Enter your HuggingFace token:[/cyan]",
                    )
                )
                console.print(
                    tr("model_mgr.token_link_label", "[dim]Get your token from:[/dim]")
                    + " [dim]https://huggingface.co/settings/tokens[/dim]"
                )
                console.print(
                    tr(
                        "model_mgr.token_hidden_note",
                        "[dim]The token will be hidden as you type.[/dim]\n",
                    )
                )

                token = getpass.getpass(
                    tr("model_mgr.token_input_prompt", "Token: ")
                ).strip()
                if token:
                    console.print(
                        tr(
                            "model_mgr.validating_token",
                            "\n[cyan]Validating token...[/cyan]",
                        )
                    )

                    from ..validation.token import validate_hf_token

                    is_valid, user_info = validate_hf_token(token)

                    if is_valid:
                        config_manager.config["hf_token"] = token
                        config_manager._save_config()
                        console.print(
                            tr(
                                "model_mgr.token_saved",
                                "[green]✓ Token validated and saved successfully[/green]",
                            )
                        )
                        if user_info:
                            console.print(
                                tr(
                                    "model_mgr.authenticated_as",
                                    "[dim]Authenticated as: {name}[/dim]\n",
                                    name=user_info.get("name", "Unknown"),
                                )
                            )
                    else:
                        console.print(
                            tr(
                                "model_mgr.token_validation_failed",
                                "[red]✗ Token validation failed[/red]",
                            )
                        )
                        console.print(
                            tr(
                                "model_mgr.token_invalid_note",
                                "[dim]The token may be invalid or expired.[/dim]",
                            )
                        )

                        # Ask if they want to continue anyway
                        confirm = (
                            input(
                                tr(
                                    "model_mgr.save_token_anyway_prompt",
                                    "\nContinue with this token anyway? (y/N): ",
                                )
                            )
                            .strip()
                            .lower()
                        )
                        if confirm == "y":
                            config_manager.config["hf_token"] = token
                            config_manager._save_config()
                            console.print(
                                tr(
                                    "model_mgr.token_saved_unverified",
                                    "[yellow]Token saved (but may not work)[/yellow]\n",
                                )
                            )
                        else:
                            console.print(
                                tr(
                                    "model_mgr.continuing_without_token",
                                    "[yellow]Continuing without token[/yellow]\n",
                                )
                            )
                else:
                    console.print(
                        tr(
                            "model_mgr.no_token_provided",
                            "[yellow]No token provided, continuing without "
                            "token[/yellow]\n",
                        )
                    )

        console.print(
            tr(
                "model_mgr.download_warning",
                "\n[yellow]Warning:[/yellow] This model will be downloaded from "
                "HuggingFace Hub.",
            )
        )
        console.print(
            tr(
                "model_mgr.download_size_note",
                "Download size can range from a few GB to 100+ GB depending on the "
                "model.\n",
            )
        )

        console.print(
            tr(
                "model_mgr.proceed_confirm",
                "[cyan]Proceed with this model? (Y/n):[/cyan] ",
            ),
            end="",
        )
        confirm = input().strip().lower()

        if confirm in ["", "y", "yes"]:
            logger.info(f"User selected remote model: {model_id}")
            return model_id
        elif confirm in ["n", "no"]:
            console.print(
                tr(
                    "model_mgr.model_selection_cancelled",
                    "[yellow]Model selection cancelled.[/yellow]",
                )
            )
            continue
        else:
            console.print(
                tr(
                    "model_mgr.invalid_yn_response",
                    "[red]Invalid response. Please enter 'y' or 'n'.[/red]",
                )
            )
            continue


def select_model() -> Optional[Any]:
    """
    Select a model from available models with provider categorization.
    Can return either a string (model name) or a dict (model with LoRA config).
    """
    console.print(
        tr(
            "model_mgr.model_selection_title",
            "\n[bold cyan]Model Selection[/bold cyan]",
        )
    )

    try:
        # First, ask if user wants to use local or remote model
        source_choice = prompt_choice(
            "model_source",
            tr(
                "model_mgr.how_select_model",
                "How would you like to select a model?",
            ),
            [
                ("local", tr("model_mgr.source_local", "Select from local models")),
                (
                    "lora",
                    tr("model_mgr.source_lora", "Serve model with LoRA adapters"),
                ),
                (
                    "remote",
                    tr(
                        "model_mgr.source_remote",
                        "Use a model from HuggingFace Hub (auto-download)",
                    ),
                ),
            ],
            allow_back=True,
        )

        if not source_choice or source_choice == "BACK":
            return None

        if source_choice == "remote":
            return enter_remote_model()

        if source_choice == "lora":
            return select_model_with_lora()

        # Continue with local model selection
        console.print(
            tr(
                "model_mgr.fetching_models",
                "\n[bold cyan]Fetching available models...[/bold cyan]",
            )
        )
        models = list_available_models()

        if not models:
            console.print(
                tr(
                    "model_mgr.no_local_models",
                    "[yellow]No local models found.[/yellow]",
                )
            )
            console.print(tr("model_mgr.no_models_options", "\nYou can either:"))
            console.print(
                tr(
                    "model_mgr.no_models_hint_download",
                    "  1. Download models using HuggingFace tools",
                )
            )
            console.print(
                tr(
                    "model_mgr.no_models_hint_remote",
                    "  2. Go back and select 'Use a model from HuggingFace Hub'",
                )
            )
            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            return None

        # Group models by provider
        providers_dict = {}
        for model in models:
            # Special handling for Ollama models
            if model.get("type") == "ollama_model":
                provider = "ollama"
            else:
                provider = model.get("publisher", "unknown")
                if provider == "unknown" or not provider:
                    # Try to extract provider from model name
                    if "/" in model["name"]:
                        provider = model["name"].split("/")[0]
                    else:
                        provider = "local"

            if provider not in providers_dict:
                providers_dict[provider] = []
            providers_dict[provider].append(model)

        # Separate local provider from others
        local_provider = None
        other_providers = []

        for provider in providers_dict.keys():
            if provider == "local":
                local_provider = provider
            else:
                other_providers.append(provider)

        # Sort other providers alphabetically
        other_providers.sort()

        # Build provider choices with separation
        provider_choices = []

        # Add other providers first
        for provider in other_providers:
            count = len(providers_dict[provider])
            provider_choices.append(
                f"{provider} ({count} model{'s' if count > 1 else ''})"
            )

        # Add separator and local provider at the bottom if it exists
        if local_provider:
            # Add separator if there are other providers
            if other_providers:
                provider_choices.append("─" * 30)
            count = len(providers_dict[local_provider])
            provider_choices.append(
                f"{local_provider} ({count} model{'s' if count > 1 else ''})"
            )

        # Count total providers (excluding separator)
        total_providers = len(providers_dict)

        selected_provider = unified_prompt(
            "provider",
            tr(
                "model_mgr.select_provider",
                "Select Provider ({count} available)",
                count=total_providers,
            ),
            provider_choices,
            allow_back=True,
        )

        if not selected_provider or selected_provider == "BACK":
            return None

        # Check if separator was selected (shouldn't happen, but handle it)
        if selected_provider.startswith("─"):
            # Retry selection
            return select_model()

        # Extract provider name
        provider_name = selected_provider.split(" (")[0]

        # Now show models for selected provider
        provider_models = providers_dict[provider_name]

        # Create model choices for selected provider
        model_choices = []
        for model in provider_models:
            size_str = format_size(model.get("size", 0))
            # Show only the model name without provider if it's already in the name
            display_name = model["name"]
            if display_name.startswith(f"{provider_name}/"):
                display_name = display_name[len(provider_name) + 1 :]  # noqa: E203
            model_choices.append(f"{display_name} ({size_str})")

        # Show model selection for the provider
        selected = unified_prompt(
            "model",
            tr(
                "model_mgr.select_provider_model",
                "Select {provider} Model ({count} available)",
                provider=provider_name,
                count=len(provider_models),
            ),
            model_choices,
            allow_back=True,
        )

        if not selected or selected == "BACK":
            # Go back to provider selection
            return select_model()

        # Extract model name and reconstruct full name if needed
        model_display_name = selected.split(" (")[0]

        # Find the full model and return appropriate identifier
        for model in provider_models:
            check_name = model["name"]
            if check_name.startswith(f"{provider_name}/"):
                check_name = check_name[len(provider_name) + 1 :]  # noqa: E203
            if check_name == model_display_name or model["name"] == model_display_name:
                # Special handling for Ollama models
                if model.get("type") == "ollama_model":
                    console.print(
                        tr(
                            "model_mgr.gguf_warning",
                            "\n[yellow]⚠ Warning: Ollama GGUF Model[/yellow]",
                        )
                    )
                    console.print(
                        tr(
                            "model_mgr.gguf_experimental",
                            "GGUF support in vLLM is experimental and varies by model "
                            "architecture.",
                        )
                    )
                    console.print(
                        tr(
                            "model_mgr.gguf_important",
                            "\n[cyan]Important:[/cyan] Not all GGUF models are supported.",
                        )
                    )
                    console.print(
                        tr(
                            "model_mgr.gguf_compat_info",
                            "\nFor compatibility information, see:",
                        )
                    )
                    console.print(
                        f"{tr('model_mgr.guide_link_label', '  • vLLM-CLI Guide:')} "
                        "[cyan]https://github.com/Chen-zexi/vllm-cli/blob/main/docs/ollama-integration.md[/cyan]"
                    )
                    console.print(
                        f"{tr('model_mgr.vllm_docs_link_label', '  • vLLM Docs:')} "
                        "[cyan]https://docs.vllm.ai/en/latest/models/supported_models.html[/cyan]"
                    )

                    console.print(
                        tr(
                            "model_mgr.continue_model_confirm",
                            "\n[cyan]Continue with this model? (Y/n):[/cyan] ",
                        ),
                        end="",
                    )
                    confirm = input().strip().lower()
                    if confirm not in ["", "y", "yes"]:
                        return select_model()  # Go back to selection

                    # Return the GGUF file path with all metadata including name
                    return {
                        "model": model["path"],
                        "path": model["path"],  # Include path
                        "served_model_name": model[
                            "name"
                        ],  # Use correct field name for vLLM
                        "type": "ollama_model",  # Preserve type
                        "quantization": "gguf",
                        "experimental": True,
                    }

                # For custom models, always use path regardless of publisher
                # Custom models are identified by their type
                if model.get("type") == "custom_model" and model.get("path"):
                    return model["path"]

                # For non-HF pattern models (local), also use path
                model_name = model["name"]
                if model.get("path") and (
                    "/" not in model_name  # No HF org/model pattern
                    or model_name.startswith("/")  # Absolute path
                    or model.get("publisher")
                    in ["local", "unknown", None]  # Local publisher
                ):
                    return model["path"]
                return model["name"]

        # Fallback
        return model_display_name

    except Exception as e:
        logger.error(f"Error selecting model: {e}")
        console.print(
            tr(
                "model_mgr.error_selecting_model",
                "[red]Error selecting model: {error}[/red]",
                error=e,
            )
        )
        return None


def handle_model_management(i18n_manager=None) -> str:
    """
    Handle model management operations by delegating to hf-model-tool.
    All model management is handled by hf-model-tool for consistency.

    Args:
        i18n_manager: Optional I18nManager for translations
    """

    # Helper function for translation
    def t(key: str, default: str = None, **kwargs) -> str:
        if i18n_manager:
            result = i18n_manager.t(key, **kwargs)
            return result if result != key else (default if default else key)
        return default if default else key

    import os
    import subprocess

    while True:  # Loop to stay in model management menu
        # Simplified menu - delegate everything to hf-model-tool
        management_options = [
            t("menu.model_management.open_tool", "Open Model Management Tool"),
            t("menu.model_management.list_models", "List All Models"),
            t("menu.model_management.manage_assets", "Manage Assets"),
            t("menu.model_management.view_details", "View Model Details"),
            t("menu.model_management.refresh_cache", "Refresh Model Cache"),
        ]

        # Stable action keys keep the comparison language-independent while the
        # labels below are what the user actually sees.
        management_actions = [
            "open_tool",
            "list_models",
            "manage_assets",
            "view_details",
            "refresh_cache",
        ]

        action = prompt_choice(
            "model_management",
            t("menu.model_management.title", "Model Management"),
            list(zip(management_actions, management_options)),
            allow_back=True,
        )

        if not action or action == "BACK":
            return "continue"  # Return to main menu

        if action == "open_tool":
            # Launch full hf-model-tool interface
            console.print(
                Panel(
                    tr(
                        "model_mgr.panel_launch_tool",
                        "[bold cyan]Launching HF-Model-Tool[/bold cyan]\n"
                        "[dim]Full model management interface[/dim]",
                    ),
                    border_style="blue",
                )
            )
            try:
                subprocess.run(["hf-model-tool"], env=os.environ.copy())
            except FileNotFoundError:
                console.print(
                    tr(
                        "model_mgr.tool_not_found",
                        "[red]hf-model-tool not found. Please install it:[/red]",
                    )
                )
                console.print("  pip install hf-model-tool")
            except Exception as e:
                console.print(
                    tr(
                        "model_mgr.error_launching_tool",
                        "[red]Error launching hf-model-tool: {error}[/red]",
                        error=e,
                    )
                )
            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            # Continue loop - stay in model management menu

        elif action == "list_models":
            # Launch hf-model-tool in list mode
            console.print(
                Panel(
                    tr(
                        "model_mgr.panel_model_list",
                        "[bold cyan]Model List[/bold cyan]\n"
                        "[dim]Displaying all discovered models[/dim]",
                    ),
                    border_style="blue",
                )
            )
            try:
                subprocess.run(["hf-model-tool", "--list"], env=os.environ.copy())
            except FileNotFoundError:
                console.print(
                    tr(
                        "model_mgr.tool_not_found",
                        "[red]hf-model-tool not found. Please install it:[/red]",
                    )
                )
                console.print("  pip install hf-model-tool")
            except Exception as e:
                console.print(
                    tr(
                        "model_mgr.error_generic",
                        "[red]Error: {error}[/red]",
                        error=e,
                    )
                )
            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            # Continue loop - stay in model management menu

        elif action == "manage_assets":
            # Launch hf-model-tool in manage mode
            console.print(
                Panel(
                    tr(
                        "model_mgr.panel_asset_management",
                        "[bold cyan]Asset Management[/bold cyan]\n"
                        "[dim]Delete, deduplicate, and organize models[/dim]",
                    ),
                    border_style="blue",
                )
            )
            try:
                subprocess.run(["hf-model-tool", "--manage"], env=os.environ.copy())
            except FileNotFoundError:
                console.print(
                    tr(
                        "model_mgr.tool_not_found",
                        "[red]hf-model-tool not found. Please install it:[/red]",
                    )
                )
                console.print("  pip install hf-model-tool")
            except Exception as e:
                console.print(
                    tr(
                        "model_mgr.error_generic",
                        "[red]Error: {error}[/red]",
                        error=e,
                    )
                )
            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            # Continue loop - stay in model management menu

        elif action == "view_details":
            # Launch hf-model-tool in details mode
            console.print(
                Panel(
                    tr(
                        "model_mgr.panel_model_details",
                        "[bold cyan]Model Details[/bold cyan]\n"
                        "[dim]View detailed information about models[/dim]",
                    ),
                    border_style="blue",
                )
            )
            try:
                subprocess.run(["hf-model-tool", "--details"], env=os.environ.copy())
            except FileNotFoundError:
                console.print(
                    tr(
                        "model_mgr.tool_not_found",
                        "[red]hf-model-tool not found. Please install it:[/red]",
                    )
                )
                console.print("  pip install hf-model-tool")
            except Exception as e:
                console.print(
                    tr(
                        "model_mgr.error_generic",
                        "[red]Error: {error}[/red]",
                        error=e,
                    )
                )
            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            # Continue loop - stay in model management menu

        elif action == "refresh_cache":
            # Refresh the model cache used by serving menu
            console.print(
                Panel(
                    tr(
                        "model_mgr.panel_refresh_cache",
                        "[bold cyan]Refreshing Model Cache[/bold cyan]\n"
                        "[dim]Scanning all model directories for updates...[/dim]",
                    ),
                    border_style="blue",
                )
            )

            try:
                import time

                from ..models import get_model_manager

                # Get the model manager
                model_manager = get_model_manager()

                # Get old cache stats for comparison
                old_stats = model_manager.get_cache_stats()
                old_count = old_stats.get("cached_models_count", 0)

                # Show scanning progress
                console.print(
                    tr(
                        "model_mgr.refresh_step_registry",
                        "\n[cyan]Step 1/3:[/cyan] Refreshing model registry...",
                    )
                )

                # Refresh the cache
                model_manager.refresh_cache()

                console.print(
                    tr(
                        "model_mgr.refresh_step_clear",
                        "[cyan]Step 2/3:[/cyan] Clearing old cache...",
                    )
                )
                time.sleep(0.1)  # Brief pause for visual feedback

                console.print(
                    tr(
                        "model_mgr.refresh_step_load",
                        "[cyan]Step 3/3:[/cyan] Loading fresh model data...",
                    )
                )
                time.sleep(0.1)  # Brief pause for visual feedback

                # Get new cache stats to show user
                new_stats = model_manager.get_cache_stats()
                new_count = new_stats.get("cached_models_count", 0)

                # Show success message with details
                console.print(
                    tr(
                        "model_mgr.cache_refreshed",
                        "\n[green]✓ Model cache refreshed successfully![/green]",
                    )
                )
                console.print(
                    tr(
                        "model_mgr.cache_refreshed_note",
                        "[dim]The serving menu will now show all current models.[/dim]",
                    )
                )

                # Show model count changes
                if old_count != new_count:
                    diff = new_count - old_count
                    if diff > 0:
                        console.print(
                            tr(
                                "model_mgr.cache_count_added",
                                "\n[cyan]Models in cache: {count} (+{added} new)[/cyan]",
                                count=new_count,
                                added=diff,
                            )
                        )
                    else:
                        console.print(
                            tr(
                                "model_mgr.cache_count_removed",
                                "\n[cyan]Models in cache: {count} ({removed} removed)[/cyan]",
                                count=new_count,
                                removed=diff,
                            )
                        )
                else:
                    console.print(
                        tr(
                            "model_mgr.cache_count_unchanged",
                            "\n[cyan]Models in cache: {count} (no changes)[/cyan]",
                            count=new_count,
                        )
                    )

                # Show cache freshness
                console.print(
                    tr(
                        "model_mgr.cache_ttl",
                        "[dim]Cache TTL: {seconds} seconds[/dim]",
                        seconds=new_stats.get("ttl_seconds", 30),
                    )
                )

            except Exception as e:
                console.print(
                    tr(
                        "model_mgr.error_refreshing_cache",
                        "[red]Error refreshing cache: {error}[/red]",
                        error=e,
                    )
                )
                logger.error(f"Cache refresh error: {e}")

            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            # Continue loop - stay in model management menu


def select_model_with_lora() -> Optional[Dict[str, Any]]:
    """
    Select a base model and LoRA adapters to serve together.

    Returns:
        Dictionary with model and LoRA configuration for serving
    """
    console.print(
        tr("model_mgr.lora_title", "\n[bold cyan]Model + LoRA Selection[/bold cyan]")
    )
    console.print(
        tr(
            "model_mgr.lora_intro",
            "[dim]Select a base model and LoRA adapters to serve together[/dim]\n",
        )
    )

    # First scan for LoRA adapters to determine which models have adapters
    console.print(
        tr("model_mgr.scanning_lora", "[cyan]Scanning for LoRA adapters...[/cyan]")
    )
    try:
        from ..models.discovery import scan_for_lora_adapters

        lora_adapters = scan_for_lora_adapters()

        if not lora_adapters:
            console.print(
                tr(
                    "model_mgr.no_lora_found",
                    "[yellow]No LoRA adapters found.[/yellow]",
                )
            )
            console.print(
                tr(
                    "model_mgr.lora_dir_hint",
                    "\nLoRA adapters should be placed in directories with:",
                )
            )
            console.print("  • adapter_config.json")
            console.print("  • adapter_model.safetensors or adapter_model.bin")
            console.print(
                tr(
                    "model_mgr.lora_manage_hint",
                    "\nYou can manage LoRA adapters using hf-model-tool",
                )
            )
            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            return None

        console.print(
            tr(
                "model_mgr.lora_found",
                "[green]Found {count} LoRA adapter(s)[/green]\n",
                count=len(lora_adapters),
            )
        )

        # Extract base models from LoRA metadata
        base_models_with_lora = set()
        lora_by_base = {}

        for lora in lora_adapters:
            # First try metadata which should have the exact base model
            metadata = lora.get("metadata", {})
            base_model = metadata.get("base_model", "")

            # If not in metadata, try config
            if not base_model:
                config = lora.get("config", {})
                if isinstance(config, dict):
                    base_model = config.get("base_model_name_or_path", "")

            # Store the mapping if we found a base model
            if base_model:
                base_models_with_lora.add(base_model)
                if base_model not in lora_by_base:
                    lora_by_base[base_model] = []
                lora_by_base[base_model].append(lora)

        logger.debug(
            f"Found LoRA adapters for these base models: {base_models_with_lora}"
        )

        # Get all available models
        models = list_available_models()

        if not models:
            console.print(
                tr(
                    "model_mgr.no_local_models",
                    "[yellow]No local models found.[/yellow]",
                )
            )
            console.print(
                tr(
                    "model_mgr.download_models_hint",
                    "Please download models first using hf-model-tool",
                )
            )
            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            return None

        # Filter models to only show those with LoRA adapters
        models_with_lora = []
        for model in models:
            model_name = model["name"]
            # Check if this model has LoRA adapters
            has_lora = False

            # First try exact match
            if model_name in base_models_with_lora:
                has_lora = True

            # If no exact match, don't do fuzzy matching to avoid false positives
            # Only show models that have exact LoRA matches

            if has_lora:
                models_with_lora.append(model)

        # If no models with LoRA found, show all and let user know
        if not models_with_lora:
            console.print(
                tr(
                    "model_mgr.no_matching_lora_models",
                    "[yellow]No models found with matching LoRA adapters.[/yellow]",
                )
            )
            console.print(
                tr(
                    "model_mgr.showing_all_models_note",
                    "Showing all models. LoRA compatibility will be checked after "
                    "selection.\n",
                )
            )
            models_with_lora = models

        # Select the base model
        console.print(
            tr("model_mgr.step1_base_model", "[cyan]Step 1: Select Base Model[/cyan]")
        )
        console.print(
            tr(
                "model_mgr.showing_lora_models",
                "[dim]Showing {count} model(s) with LoRA adapters[/dim]\n",
                count=len(models_with_lora),
            )
        )

        model_choices = []
        for model in models_with_lora:
            size_str = format_size(model.get("size", 0))
            # Check how many LoRAs this model has
            model_name = model["name"]
            lora_count = 0

            # Use exact matching for counting
            if model_name in lora_by_base:
                lora_count = len(lora_by_base[model_name])

            if lora_count > 0:
                model_choices.append(
                    f"{model_name} ({size_str}) [{lora_count} LoRA(s)]"
                )
            else:
                model_choices.append(f"{model_name} ({size_str})")

        selected_model = unified_prompt(
            "base_model",
            tr("model_mgr.select_base_model", "Select Base Model"),
            model_choices,
            allow_back=True,
        )

        if not selected_model or selected_model == "BACK":
            return None

        # Extract model name
        base_model_name = selected_model.split(" (")[0]

        # Find the actual model object to get its path if it's a local/custom model
        base_model_path_or_name = base_model_name
        for model in models_with_lora:
            if model["name"] == base_model_name:
                # For custom models, always use path regardless of publisher
                if model.get("type") == "custom_model" and model.get("path"):
                    base_model_path_or_name = model["path"]
                # For non-HF pattern models, use the path instead of name
                elif model.get("path") and (
                    "/" not in base_model_name  # No HF org/model pattern
                    or base_model_name.startswith("/")  # Absolute path
                    or model.get("publisher")
                    in ["local", "unknown", None]  # Local publisher
                ):
                    base_model_path_or_name = model["path"]
                break

        # Now select LoRA adapters for this model
        console.print(
            tr("model_mgr.step2_lora", "\n[cyan]Step 2: Select LoRA Adapters[/cyan]")
        )
        console.print(
            tr(
                "model_mgr.base_model_line",
                "[dim]Base model: {model}[/dim]\n",
                model=base_model_name,
            )
        )

        # Filter compatible LoRAs for the selected model
        compatible_loras = []
        incompatible_loras = []

        for lora in lora_adapters:
            metadata = lora.get("metadata", {})
            lora_base = metadata.get("base_model", "unknown")

            # Simple compatibility check
            if (
                lora_base == "unknown"
                or lora_base in base_model_name
                or base_model_name in lora_base
            ):
                compatible_loras.append(lora)
            else:
                incompatible_loras.append(lora)

        if not compatible_loras and incompatible_loras:
            console.print(
                tr(
                    "model_mgr.no_compatible_lora",
                    "[yellow]No clearly compatible LoRA adapters found for this "
                    "model.[/yellow]",
                )
            )
            if inquirer.confirm(
                tr(
                    "model_mgr.show_all_lora_confirm",
                    "Show all LoRA adapters anyway?",
                ),
                default=True,
            ):
                compatible_loras = incompatible_loras
            else:
                return base_model_name

        # Create LoRA choices
        lora_choices = []
        for lora in compatible_loras:
            name = lora.get("name", lora.get("display_name", "Unknown"))
            rank = lora.get("rank", lora.get("metadata", {}).get("rank", "N/A"))
            size = lora.get("size", 0)
            size_str = f"{size / (1024**2):.1f}MB" if size > 0 else "N/A"
            lora_choices.append(f"{name} (rank={rank}, {size_str})")

        # Allow multiple LoRA selection
        console.print(
            tr(
                "model_mgr.multiple_lora_hint",
                "[dim]You can select multiple LoRA adapters (space to select, enter "
                "to confirm)[/dim]",
            )
        )

        questions = [
            inquirer.Checkbox(
                "loras",
                message=tr("model_mgr.select_lora_adapters", "Select LoRA adapters"),
                choices=lora_choices,
            )
        ]

        answers = inquirer.prompt(questions)
        if not answers or not answers["loras"]:
            if inquirer.confirm(
                tr(
                    "model_mgr.continue_without_lora_confirm",
                    "Continue without LoRA adapters?",
                ),
                default=False,
            ):
                return base_model_name
            return None

        # Build the configuration
        selected_lora_configs = []
        for lora_choice in answers["loras"]:
            # Find the matching LoRA
            for lora in compatible_loras:
                name = lora.get("name", lora.get("display_name", "Unknown"))
                if name in lora_choice:
                    selected_lora_configs.append(
                        {
                            "name": name,
                            "path": lora.get("path", ""),
                            "rank": lora.get("rank", 16),
                        }
                    )
                    break

        # Return configuration for serving
        return {"model": base_model_path_or_name, "lora_modules": selected_lora_configs}

    except Exception as e:
        console.print(
            tr(
                "model_mgr.error_selecting_lora",
                "[red]Error selecting LoRA adapters: {error}[/red]",
                error=e,
            )
        )
        logger.error(f"LoRA selection error: {e}")

        if inquirer.confirm(
            tr("model_mgr.continue_base_only", "Continue with base model only?"),
            default=True,
        ):
            return base_model_path_or_name
        return None


# Removed view_available_models and show_model_details functions
# These are now handled by hf-model-tool --list and --details
