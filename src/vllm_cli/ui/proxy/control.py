#!/usr/bin/env python3
"""
UI components for proxy server control and configuration.
"""
import logging
from typing import List, Optional

from rich.table import Table

from ...i18n import tr
from ...proxy.config import ProxyConfigManager
from ...proxy.models import ModelConfig, ProxyConfig
from ..common import console
from ..custom_config import select_gpus
from ..model_manager import select_model
from ..navigation import prompt_choice, unified_prompt

logger = logging.getLogger(__name__)


def _status_label(enabled: bool) -> str:
    """Return the displayed 'Enabled'/'Disabled' label for a boolean setting."""
    if enabled:
        return tr("proxy_wizard.status_enabled", "Enabled")
    return tr("proxy_wizard.status_disabled", "Disabled")


def _status_label_lowercase(enabled: bool) -> str:
    """Return the displayed lowercase 'enabled'/'disabled' label."""
    if enabled:
        return tr("proxy_wizard.tgl_enabled", "enabled")
    return tr("proxy_wizard.tgl_disabled", "disabled")


def display_configured_models(models: List[ModelConfig]):
    """
    Display a summary table of configured models.

    Args:
        models: List of configured ModelConfig instances
    """
    if not models:
        return

    table = Table(
        title=tr(
            "proxy_wizard.configured_models_title",
            "Configured Models ({count})",
            count=len(models),
        )
    )
    table.add_column("#", style="dim", width=3)
    table.add_column(tr("proxy_wizard.col_model", "Model"), style="cyan")
    table.add_column(tr("proxy_wizard.col_port", "Port"), style="magenta")
    table.add_column(tr("proxy_wizard.col_gpus", "GPU(s)"), style="yellow")
    table.add_column(tr("proxy_wizard.col_profile", "Profile"), style="blue")
    table.add_column(
        tr("proxy_wizard.col_priority", "Priority"), style="green", width=8
    )

    for idx, model in enumerate(models, 1):
        gpu_str = (
            ",".join(str(g) for g in model.gpu_ids)
            if model.gpu_ids
            else tr("proxy_wizard.auto", "Auto")
        )
        profile_str = model.profile or tr("proxy_wizard.none", "None")
        priority_str = str(model.loading_priority) if model.loading_priority else "-"
        table.add_row(
            str(idx),
            model.name[:30] + "..." if len(model.name) > 30 else model.name,
            str(model.port),
            gpu_str,
            profile_str,
            priority_str,
        )

    console.print("\n")
    console.print(table)
    console.print()


def configure_proxy_interactively() -> Optional[ProxyConfig]:
    """
    Configure proxy server interactively.

    Returns:
        ProxyConfig instance or None if cancelled
    """
    console.print(
        tr(
            "proxy_wizard.configure_proxy_title",
            "\n[bold cyan]Configure Multi-Model Proxy Server[/bold cyan]",
        )
    )

    # Proxy settings
    console.print(
        tr(
            "proxy_wizard.proxy_settings_header",
            "\n[bold]Proxy Server Settings[/bold]",
        )
    )

    host = (
        input(
            tr("proxy_wizard.host_prompt_default", "Host (default: 0.0.0.0): ")
        ).strip()
        or "0.0.0.0"
    )  # nosec B104
    port_str = (
        input(
            tr("proxy_wizard.port_prompt_default", "Port (default: 8000): ")
        ).strip()
        or "8000"
    )

    try:
        port = int(port_str)
    except ValueError:
        console.print(
            tr("proxy_wizard.invalid_port_number", "[red]Invalid port number[/red]")
        )
        return None

    # CORS and metrics
    yes_no = [
        ("yes", tr("messages.yes", "Yes")),
        ("no", tr("messages.no", "No")),
    ]

    enable_cors = (
        prompt_choice(
            "enable_cors",
            tr("proxy_wizard.enable_cors", "Enable CORS?"),
            yes_no,
            allow_back=False,
        )
        == "yes"
    )

    enable_metrics = (
        prompt_choice(
            "enable_metrics",
            tr("proxy_wizard.enable_metrics", "Enable metrics endpoint?"),
            yes_no,
            allow_back=False,
        )
        == "yes"
    )

    log_requests = (
        prompt_choice(
            "log_requests",
            tr("proxy_wizard.log_requests", "Log all requests?"),
            yes_no,
            allow_back=False,
        )
        == "yes"
    )

    # Configure models
    console.print(
        tr("proxy_wizard.configure_models_header", "\n[bold]Configure Models[/bold]")
    )
    models = []

    while True:
        # Display currently configured models if any
        if models:
            display_configured_models(models)

        # Update prompt based on number of configured models
        prompt_msg = tr(
            "proxy_wizard.model_configuration", "Model configuration"
        )
        if models:
            prompt_msg = tr(
                "proxy_wizard.model_configuration_count",
                "Model configuration ({count} model(s) configured)",
                count=len(models),
            )

        action = prompt_choice(
            "model_action",
            prompt_msg,
            [
                ("add", tr("proxy_wizard.action_add_model", "Add a model")),
                (
                    "done",
                    tr(
                        "proxy_wizard.action_done_models",
                        "✓ Done configuring models",
                    ),
                ),
            ],
            allow_back=False,
        )

        if action == "done":
            break

        # Add a model - pass existing models for conflict checking
        model_config = configure_model_for_proxy(
            len(models), existing_models=models, is_running_proxy=False
        )
        if model_config:
            models.append(model_config)
            console.print(
                "\n"
                + tr(
                    "proxy_wizard.model_added",
                    "[green]✓ Added model: {name}[/green]",
                    name=model_config.name,
                )
            )

    if not models:
        console.print(
            tr(
                "proxy_wizard.no_models_for_proxy",
                "[yellow]No models configured. Proxy needs at least one model.[/yellow]",
            )
        )
        return None

    return ProxyConfig(
        host=host,
        port=port,
        models=models,
        enable_cors=enable_cors,
        enable_metrics=enable_metrics,
        log_requests=log_requests,
    )


def _check_parallel_settings_conflict(
    profile: str, gpu_ids: List[int], config_manager
) -> None:
    """
    Check and warn about parallel settings conflicts with GPU selection.

    Args:
        profile: Profile name
        gpu_ids: List of selected GPU IDs
        config_manager: ConfigManager instance
    """
    if profile and len(gpu_ids) == 1:
        all_profiles = config_manager.get_all_profiles()
        profile_config = all_profiles.get(profile, {}).get("config", {})
        parallel_settings = []

        if profile_config.get("tensor_parallel_size"):
            parallel_settings.append(
                f"tensor_parallel_size={profile_config['tensor_parallel_size']}"
            )
        if profile_config.get("pipeline_parallel_size"):
            parallel_settings.append(
                f"pipeline_parallel_size={profile_config['pipeline_parallel_size']}"
            )
        if profile_config.get("enable_expert_parallel"):
            parallel_settings.append("enable_expert_parallel=True")

        if parallel_settings:
            console.print(
                tr(
                    "proxy_wizard.warning_label",
                    "\n[yellow]⚠ Warning:[/yellow]",
                )
            )
            console.print(
                tr(
                    "proxy_wizard.profile_parallel_settings",
                    "Profile '{profile}' contains parallel settings: {settings}",
                    profile=profile,
                    settings=", ".join(parallel_settings),
                )
            )
            console.print(
                tr(
                    "proxy_wizard.parallel_disabled_note",
                    "These settings will be automatically disabled for single-GPU deployment.",
                )
            )
            console.print(
                tr(
                    "proxy_wizard.single_gpu_note",
                    "[dim]The model will run on a single GPU as configured.[/dim]\n",
                )
            )


def configure_model_for_proxy(
    index: int,
    existing_models: Optional[List[ModelConfig]] = None,
    is_running_proxy: bool = False,
    proxy_manager=None,
) -> Optional[ModelConfig]:
    """
    Unified function to configure a model for proxy serving.

    This function provides a consistent UI experience for both:
    - Initial proxy configuration (multiple models)
    - Adding models to a running proxy

    Args:
        index: Index of this model (for default port calculation)
        existing_models: List of already configured models (for conflict checking)
        is_running_proxy: Whether configuring for a running proxy (affects prompts)

    Returns:
        ModelConfig instance or None if cancelled
    """
    if existing_models is None:
        existing_models = []

    # Display appropriate header
    if is_running_proxy:
        console.print(
            tr(
                "proxy_wizard.add_new_model_title",
                "\n[bold cyan]Add New Model[/bold cyan]\n",
            )
        )
    else:
        console.print(
            tr(
                "proxy_wizard.configure_model_number",
                "\n[cyan]Configure Model #{number}[/cyan]",
                number=index + 1,
            )
        )

    # Import here to avoid circular dependency
    from ...config import ConfigManager
    from ..model_manager import select_shortcut_for_serving
    config_manager = ConfigManager()
    shortcuts = config_manager.list_shortcuts()

    # First ask if user wants to use shortcut or select model
    model_source_choices = []
    if shortcuts:
        model_source_choices.append(
            (
                "shortcut",
                tr("proxy_wizard.use_saved_shortcut", "Use saved shortcut"),
            )
        )
    model_source_choices.extend(
        [
            (
                "models",
                tr("proxy_wizard.select_from_models", "Select from models"),
            )
        ]
    )

    source_choice = prompt_choice(
        "model_source_proxy",
        tr(
            "proxy_wizard.how_configure_model",
            "How would you like to configure this model?",
        ),
        model_source_choices,
        allow_back=True,
    )

    if not source_choice or source_choice == "BACK":
        return None

    model_selection = None
    if source_choice == "shortcut":
        # Handle shortcut selection directly
        model_selection = select_shortcut_for_serving()
    else:
        # Select model using existing UI
        model_selection = select_model()

    if not model_selection:
        return None

    # Process model selection (handles shortcuts, ollama, lora, etc.)
    model_path = None
    model_name = None
    shortcut_profile = None
    model_config_overrides = {}
    is_shortcut = False

    if isinstance(model_selection, dict):
        if model_selection.get("type") == "shortcut":
            # Shortcut selected
            is_shortcut = True
            model_path = model_selection["model"]
            model_name = model_path
            shortcut_profile = model_selection["profile"]
            shortcut_name = model_selection["name"]
            model_config_overrides = model_selection.get("config_overrides", {})

            console.print(
                tr(
                    "proxy_wizard.using_shortcut",
                    "\n[bold cyan]Using Shortcut: {name}[/bold cyan]",
                    name=shortcut_name,
                )
            )
            console.print(
                tr(
                    "proxy_wizard.model_label",
                    "[green]Model: {name}[/green]",
                    name=model_name,
                )
            )
            console.print(
                tr(
                    "proxy_wizard.profile_label",
                    "[blue]Profile: {name}[/blue]",
                    name=shortcut_profile,
                )
            )

        elif model_selection.get("type") == "ollama_model":
            # Ollama/GGUF model
            model_path = model_selection.get("model", model_selection.get("path"))
            model_name = model_selection.get("name", model_path)
            if model_selection.get("served_model_name"):
                model_config_overrides["served_model_name"] = model_selection[
                    "served_model_name"
                ]
            model_config_overrides["quantization"] = "gguf"

        elif "lora_modules" in model_selection:
            # LoRA configuration
            model_path = model_selection["model"]
            lora_modules = model_selection["lora_modules"]
            model_name = model_path.split("/")[-1] if "/" in model_path else model_path
            model_config_overrides["enable_lora"] = True
            model_config_overrides["lora_modules"] = lora_modules

        else:
            # Other dict format
            model_path = model_selection.get("model", model_selection.get("path"))
            model_name = model_selection.get(
                "name", model_path.split("/")[-1] if "/" in model_path else model_path
            )
    else:
        # Simple string model name/path
        model_path = model_selection
        model_name = model_path.split("/")[-1] if "/" in model_path else model_path

    # Display the model (only if not a shortcut, as shortcuts already display it)
    if not is_shortcut:
        console.print(
            "\n"
            + tr(
                "proxy_wizard.model_label",
                "[green]Model: {name}[/green]",
                name=model_name,
            )
        )

    # Optional: Allow user to provide an alias
    alias = None
    use_alias = (
        prompt_choice(
            "use_alias",
            tr(
                "proxy_wizard.add_alias_question",
                "Would you like to add an alias for this model?",
            ),
            [
                (
                    "no",
                    tr(
                        "proxy_wizard.alias_no",
                        "No, use model path/name",
                    ),
                ),
                ("yes", tr("proxy_wizard.alias_yes", "Yes, add an alias")),
            ],
            allow_back=False,
        )
        == "yes"
    )

    if use_alias:
        alias_input = (
            input(
                tr(
                    "proxy_wizard.enter_alias_prompt",
                    "Enter alias (optional, press Enter to skip): ",
                )
            )
            .strip()
        )
        if alias_input:
            alias = alias_input
            console.print(
                tr(
                    "proxy_wizard.alias_routing_note",
                    "[dim]Note: Both '{name}' and '{alias}' will route to this model[/dim]",
                    name=model_name,
                    alias=alias,
                )
            )

    # GPU assignment
    console.print(
        tr("proxy_wizard.gpu_assignment_header", "\n[bold]GPU Assignment[/bold]")
    )

    # Get gpu_memory_utilization from profile if available
    required_utilization = None
    if is_shortcut and shortcut_profile:
        # For shortcuts, we know the profile
        from ...config import ConfigManager

        config_manager = ConfigManager()
        profile_data = config_manager.get_profile(shortcut_profile)
        if profile_data and "config" in profile_data:
            required_utilization = profile_data["config"].get(
                "gpu_memory_utilization", 0.9
            )
            console.print(
                tr(
                    "proxy_wizard.profile_memory_use",
                    "[dim]Profile '{profile}' uses {percent}% GPU memory[/dim]",
                    profile=shortcut_profile,
                    percent=f"{required_utilization * 100:.0f}",
                )
            )

    # Show warning if using shortcut with parallel settings
    if is_shortcut:
        console.print(
            tr(
                "proxy_wizard.gpu_override_warning",
                "[yellow]⚠ Warning:[/yellow] GPU selection may override profile settings",
            )
        )
        console.print(
            tr(
                "proxy_wizard.profile_parallel_possible",
                "[dim]Profile '{profile}' may have tensor_parallel_size or pipeline_parallel_size[/dim]",
                profile=shortcut_profile,
            )
        )
        console.print(
            tr(
                "proxy_wizard.gpu_determines_parallelism",
                "[dim]Your GPU selection will determine the actual parallelism used[/dim]\n",
            )
        )

    gpu_str = select_gpus(
        current_selection=None,
        proxy_manager=proxy_manager,
        required_utilization=required_utilization,
    )
    gpu_ids = []
    if gpu_str:
        try:
            gpu_ids = [int(g.strip()) for g in gpu_str.split(",")]
        except ValueError:
            console.print(
                tr(
                    "proxy_wizard.invalid_gpu_ids",
                    "[yellow]Invalid GPU IDs, using automatic assignment[/yellow]",
                )
            )
    else:
        console.print(
            tr(
                "proxy_wizard.no_gpus_selected",
                "[dim]No GPUs selected, will use automatic assignment[/dim]",
            )
        )

    # Port selection with conflict checking
    console.print(
        tr("proxy_wizard.port_selection_header", "\n[bold]Port Selection[/bold]")
    )

    # Build port usage map from existing models with running status if proxy is running
    used_ports = {}
    for model in existing_models:
        used_ports[model.port] = {"name": model.name, "model": model}

    if used_ports:
        console.print(
            "\n"
            + tr(
                "proxy_wizard.configured_ports_label",
                "[dim]Currently configured ports:[/dim]",
            )
        )
        for port, info in sorted(used_ports.items()):
            # For running proxy, check if model is actually running
            if is_running_proxy:
                # For running proxy, just show the port is in use
                # The actual running status will be checked by the caller
                console.print(
                    tr(
                        "proxy_wizard.port_used_line",
                        "  Port {port}: {name}",
                        port=port,
                        name=info["name"],
                    )
                )
            else:
                console.print(
                    tr(
                        "proxy_wizard.port_used_line",
                        "  Port {port}: {name}",
                        port=port,
                        name=info["name"],
                    )
                )

    # Suggest next available port
    default_port = 8001 + index
    # Find next available port if default is taken
    while default_port in used_ports:
        default_port += 1

    port_str = (
        input(
            tr(
                "proxy_wizard.port_prompt",
                "Port (default: {port}): ",
                port=default_port,
            )
        )
        .strip()
        or str(default_port)
    )
    try:
        port = int(port_str)
        # Check for port conflicts
        if port in used_ports:
            if is_running_proxy:
                # For running proxy, the caller will handle port reuse for stopped models
                console.print(
                    tr(
                        "proxy_wizard.port_configured_for",
                        "[yellow]Port {port} is configured for '{name}'[/yellow]",
                        port=port,
                        name=used_ports[port]["name"],
                    )
                )
                console.print(
                    tr(
                        "proxy_wizard.port_replacement_note",
                        "[dim]Note: If this model is stopped, it will be replaced.[/dim]",
                    )
                )
            else:
                # For initial config, don't allow duplicate ports
                console.print(
                    tr(
                        "proxy_wizard.port_in_use",
                        "[red]Port {port} is already in use by '{name}'[/red]",
                        port=port,
                        name=used_ports[port]["name"],
                    )
                )
                console.print(
                    tr(
                        "proxy_wizard.choose_other_port",
                        "[yellow]Please choose a different port[/yellow]",
                    )
                )
                return None
    except ValueError:
        console.print(
            tr(
                "proxy_wizard.invalid_port_default",
                "[yellow]Invalid port, using default: {port}[/yellow]",
                port=default_port,
            )
        )
        port = default_port

    # Profile selection
    profile = None
    if is_shortcut:
        # Use the profile from the shortcut
        profile = shortcut_profile
        console.print(
            tr(
                "proxy_wizard.using_profile_from_shortcut",
                "\n[dim]Using profile from shortcut: {profile}[/dim]",
                profile=profile,
            )
        )
    else:
        # Normal profile selection for non-shortcut models
        from ...config import ConfigManager

        config_manager = ConfigManager()
        all_profiles = config_manager.get_all_profiles()

        if all_profiles:
            profile_choices: list = [
                (name, name) for name in all_profiles.keys()
            ]
            profile_choices.append(
                ("none", tr("proxy_wizard.no_profile", "No profile"))
            )

            selected_profile = prompt_choice(
                "profile_selection",
                tr("proxy_wizard.select_profile", "Select profile"),
                profile_choices,
                allow_back=False,
            )

            profile = None if selected_profile == "none" else selected_profile

            # If profile selected and no GPUs selected yet, update required_utilization
            if profile and not gpu_ids:
                profile_data = config_manager.get_profile(profile)
                if profile_data and "config" in profile_data:
                    profile_util = profile_data["config"].get(
                        "gpu_memory_utilization", 0.9
                    )
                    console.print(
                        tr(
                            "proxy_wizard.selected_profile_memory",
                            "\n[yellow]Note: Selected profile uses {percent}% GPU memory[/yellow]",
                            percent=f"{profile_util * 100:.0f}",
                        )
                    )
                    console.print(
                        tr(
                            "proxy_wizard.reconsider_gpu_note",
                            "[dim]You may want to reconsider GPU selection based on this requirement[/dim]",
                        )
                    )
        else:
            profile = None

    # Check for parallel settings conflicts
    if profile:
        from ...config import ConfigManager

        config_manager = ConfigManager()
        _check_parallel_settings_conflict(profile, gpu_ids, config_manager)

    # Loading priority (optional)
    loading_priority = None
    if len(existing_models) > 0 or not is_running_proxy:
        # Only ask about loading priority when configuring multiple models
        console.print(
            tr(
                "proxy_wizard.loading_priority_header",
                "\n[bold]Loading Priority (Optional)[/bold]",
            )
        )
        console.print(
            tr(
                "proxy_wizard.loading_priority_order",
                "[dim]Set loading order for sequential startup. "
                "Lower numbers load first (e.g., 1, 2, 3).[/dim]",
            )
        )
        console.print(
            tr(
                "proxy_wizard.loading_priority_kv_note",
                "[dim]Useful when models share GPUs to control KV cache allocation.[/dim]",
            )
        )
        console.print(
            tr(
                "proxy_wizard.loading_priority_parallel_note",
                "[dim]Leave empty for parallel loading or if this is the only model.[/dim]",
            )
        )

        priority_input = (
            input(
                tr(
                    "proxy_wizard.loading_priority_prompt",
                    "Loading priority (press Enter to skip): ",
                )
            )
            .strip()
        )
        if priority_input:
            try:
                loading_priority = int(priority_input)
                if loading_priority < 1:
                    console.print(
                        tr(
                            "proxy_wizard.priority_must_be_one",
                            "[yellow]Priority must be >= 1, using default (no priority)[/yellow]",
                        )
                    )
                    loading_priority = None
                else:
                    console.print(
                        tr(
                            "proxy_wizard.priority_set",
                            "[green]✓ Loading priority set to {priority}[/green]",
                            priority=loading_priority,
                        )
                    )
            except ValueError:
                console.print(
                    tr(
                        "proxy_wizard.invalid_priority",
                        "[yellow]Invalid priority number, using default (no priority)[/yellow]",
                    )
                )
                loading_priority = None

    # Create model configuration
    config = ModelConfig(
        name=model_name,
        model_path=model_path,
        gpu_ids=gpu_ids,
        port=port,
        profile=profile,
        enabled=True,
        config_overrides=model_config_overrides,
        loading_priority=loading_priority,
    )

    # Add alias to config_overrides if provided
    if alias:
        config.config_overrides["aliases"] = [alias]

    return config


def configure_model_interactively(index: int) -> Optional[ModelConfig]:
    """
    Configure a single model interactively.

    This is a wrapper around configure_model_for_proxy for backward compatibility
    and initial proxy configuration context.

    Args:
        index: Index of this model (for default port calculation)

    Returns:
        ModelConfig instance or None if cancelled
    """
    # During initial proxy configuration, we don't have existing models yet
    # The configure_proxy_interactively function will accumulate them
    return configure_model_for_proxy(index, existing_models=[], is_running_proxy=False)


# Note: manage_proxy_configs has been moved to settings.py for consistency
# with other configuration management (profiles, shortcuts)


def edit_proxy_config():
    """
    Edit proxy configuration interactively.

    Loads the current configuration and provides an interactive
    editing interface.

    Returns:
        Modified ProxyConfig or None if cancelled
    """
    config_manager = ProxyConfigManager()
    current_config = config_manager.load_config()
    return edit_proxy_config_interactive(current_config)


def edit_proxy_config_interactive(current_config: ProxyConfig) -> Optional[ProxyConfig]:
    """
    Edit a proxy configuration interactively.

    Args:
        current_config: The configuration to edit

    Returns:
        Modified ProxyConfig or None if cancelled
    """
    # Note: ProxyConfigManager used for validation if needed

    console.print(
        tr(
            "proxy_wizard.edit_proxy_config_title",
            "\n[bold cyan]Edit Proxy Configuration[/bold cyan]",
        )
    )

    while True:
        # Show current configuration
        display_proxy_config(current_config)

        # Menu options
        options = [
            (
                "edit_settings",
                tr("proxy_wizard.action_edit_settings", "Edit proxy settings"),
            ),
            ("add_model", tr("proxy_wizard.action_add_model_menu", "Add model")),
            ("remove_model", tr("proxy_wizard.action_remove_model", "Remove model")),
            (
                "toggle_model",
                tr("proxy_wizard.action_toggle_model", "Enable/disable model"),
            ),
            ("save_exit", tr("proxy_wizard.action_save_exit", "Save and exit")),
            (
                "exit_no_save",
                tr("proxy_wizard.action_exit_no_save", "Exit without saving"),
            ),
        ]

        action = prompt_choice(
            "edit_action",
            tr("proxy_wizard.select_action", "Select action"),
            options,
            allow_back=False,
        )

        if action == "save_exit":
            return current_config
        elif action == "exit_no_save":
            return None
        elif action == "edit_settings":
            edit_proxy_settings(current_config)
        elif action == "add_model":
            model = configure_model_for_proxy(
                len(current_config.models),
                existing_models=current_config.models,
                is_running_proxy=False,
            )
            if model:
                current_config.models.append(model)
                console.print(
                    tr(
                        "proxy_wizard.model_added",
                        "[green]✓ Added model: {name}[/green]",
                        name=model.name,
                    )
                )
        elif action == "remove_model":
            if current_config.models:
                model_names = [m.name for m in current_config.models]
                selected = unified_prompt(
                    "remove_model",
                    tr(
                        "proxy_wizard.select_model_to_remove",
                        "Select model to remove",
                    ),
                    model_names,
                    allow_back=True,
                )
                if selected != "BACK":
                    current_config.models = [
                        m for m in current_config.models if m.name != selected
                    ]
                    console.print(
                        tr(
                            "proxy_wizard.model_removed",
                            "[green]✓ Removed model: {name}[/green]",
                            name=selected,
                        )
                    )
        elif action == "toggle_model":
            if current_config.models:
                toggle_choices = []
                for m in current_config.models:
                    status_label = _status_label_lowercase(m.enabled)
                    toggle_choices.append((m.name, f"{m.name} ({status_label})"))
                selected = prompt_choice(
                    "toggle_model",
                    tr(
                        "proxy_wizard.select_model_to_toggle", "Select model to toggle"
                    ),
                    toggle_choices,
                    allow_back=True,
                )
                if selected != "BACK" and selected:
                    model_name = selected
                    for model in current_config.models:
                        if model.name == model_name:
                            model.enabled = not model.enabled
                            if model.enabled:
                                console.print(
                                    tr(
                                        "proxy_wizard.model_enabled",
                                        "[green]✓ Model {name} enabled[/green]",
                                        name=model_name,
                                    )
                                )
                            else:
                                console.print(
                                    tr(
                                        "proxy_wizard.model_disabled",
                                        "[green]✓ Model {name} disabled[/green]",
                                        name=model_name,
                                    )
                                )
                            break

    return None  # If we exit the loop without saving


def edit_proxy_settings(config: ProxyConfig):
    """Edit proxy server settings."""
    console.print(
        tr(
            "proxy_wizard.edit_proxy_settings_header",
            "\n[bold]Edit Proxy Settings[/bold]",
        )
    )
    console.print(
        tr("proxy_wizard.current_host", "Current host: {host}", host=config.host)
    )
    console.print(
        tr("proxy_wizard.current_port", "Current port: {port}", port=config.port)
    )

    new_host = (
        input(
            tr(
                "proxy_wizard.new_host_prompt",
                "New host (press Enter to keep current): ",
            )
        )
        .strip()
    )
    if new_host:
        config.host = new_host

    new_port = (
        input(
            tr(
                "proxy_wizard.new_port_prompt",
                "New port (press Enter to keep current): ",
            )
        )
        .strip()
    )
    if new_port:
        try:
            config.port = int(new_port)
        except ValueError:
            console.print(
                tr(
                    "proxy_wizard.invalid_port_number",
                    "[red]Invalid port number[/red]",
                )
            )

    # Toggle settings
    cors_choice = prompt_choice(
        "cors_setting",
        tr(
            "proxy_wizard.cors_current",
            "CORS (currently {status})",
            status=_status_label_lowercase(config.enable_cors),
        ),
        [
            ("enable", tr("proxy_wizard.action_enable", "Enable")),
            ("disable", tr("proxy_wizard.action_disable", "Disable")),
            ("keep", tr("proxy_wizard.action_keep_current", "Keep current")),
        ],
        allow_back=False,
    )
    if cors_choice == "enable":
        config.enable_cors = True
    elif cors_choice == "disable":
        config.enable_cors = False
    # "Keep current" - no change needed

    console.print(
        tr("proxy_wizard.settings_updated", "[green]✓ Settings updated[/green]")
    )


def display_proxy_config(config: ProxyConfig):
    """Display proxy configuration."""
    console.print(
        tr(
            "proxy_wizard.current_configuration_header",
            "\n[bold]Current Configuration[/bold]",
        )
    )
    console.print(tr("proxy_wizard.host_label", "Host: {host}", host=config.host))
    console.print(tr("proxy_wizard.port_label", "Port: {port}", port=config.port))
    console.print(
        tr(
            "proxy_wizard.cors_label",
            "CORS: {status}",
            status=_status_label(config.enable_cors),
        )
    )
    console.print(
        tr(
            "proxy_wizard.metrics_label",
            "Metrics: {status}",
            status=_status_label(config.enable_metrics),
        )
    )
    console.print(
        tr(
            "proxy_wizard.request_logging_label",
            "Request Logging: {status}",
            status=_status_label(config.log_requests),
        )
    )

    if config.models:
        console.print(
            tr(
                "proxy_wizard.models_count_header",
                "\nModels ({count}):",
                count=len(config.models),
            )
        )
        table = Table()
        table.add_column(tr("proxy_wizard.col_name", "Name"), style="cyan")
        table.add_column(tr("proxy_wizard.col_model_path", "Model Path"), style="green")
        table.add_column(tr("proxy_wizard.col_gpus_list", "GPUs"), style="magenta")
        table.add_column(tr("proxy_wizard.col_port", "Port"), style="yellow")
        table.add_column(tr("proxy_wizard.col_profile", "Profile"), style="blue")
        table.add_column(
            tr("proxy_wizard.col_priority", "Priority"), style="green", width=8
        )
        table.add_column(tr("proxy_wizard.col_status", "Status"), style="dim")

        for model in config.models:
            gpu_str = (
                ",".join(str(g) for g in model.gpu_ids)
                if model.gpu_ids
                else tr("proxy_wizard.auto", "Auto")
            )
            status = _status_label(model.enabled)
            priority_str = (
                str(model.loading_priority) if model.loading_priority else "-"
            )
            table.add_row(
                model.name,
                (
                    model.model_path[:40] + "..."
                    if len(model.model_path) > 40
                    else model.model_path
                ),
                gpu_str,
                str(model.port),
                model.profile or tr("proxy_wizard.none", "None"),
                priority_str,
                status,
            )

        console.print(table)
    else:
        console.print(
            tr(
                "proxy_wizard.no_models_configured",
                "\n[yellow]No models configured[/yellow]",
            )
        )
