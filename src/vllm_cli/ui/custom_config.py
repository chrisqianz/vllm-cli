#!/usr/bin/env python3
"""
Enhanced category-based custom configuration for vLLM CLI with simplified numerical inputs.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import inquirer

from ..config import ConfigManager
from ..i18n import tr
from ..system import gpu
from .common import console
from .navigation import prompt_choice

logger = logging.getLogger(__name__)


def parse_model_length(value: str):
    """Parse model length value - supports 1M, 100K format."""
    if value is None or value == "":
        return None
    if value.lower() == "none":
        return None
    pattern = r"^(\d+)([MKmk])?$"
    match = re.match(pattern, value)
    if match:
        number = int(match.group(1))
        suffix = match.group(2)
        if suffix:
            suffix = suffix.upper()
            if suffix == "M":
                return number * 1000000
            elif suffix == "K":
                return number * 1000
        return number
    try:
        return int(value)
    except ValueError:
        return None


def configure_by_categories(
    base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Configure vLLM server arguments using a category-based approach.

    Args:
        base_config: Optional base configuration to start from

    Returns:
        Configured dictionary of arguments
    """
    config_manager = ConfigManager()
    config = base_config.copy() if base_config else {}

    console.print(
        f"\n[bold cyan]{tr('custom_cfg.category_based_title', 'Category-Based Configuration')}[/bold cyan]"
    )
    console.print(
        tr(
            "custom_cfg.category_based_subtitle",
            "Configure only what you need - press Enter to use defaults\n",
        )
    )

    # Get ordered categories
    categories = config_manager.get_ordered_categories()

    # First handle default categories (essential and performance)
    for category_id, category_info in categories:
        show_by_default = category_info.get("show_by_default", False)

        # Only process default categories in this loop
        if not show_by_default:
            continue

        category_name = category_info["name"]
        category_desc = category_info["description"]
        # icon = category_info.get("icon", "")  # Reserved for future use

        # Show category header
        console.print(f"\n[bold]{category_name}[/bold]")
        console.print(f"[dim]{category_desc}[/dim]")

        # Get arguments for this category
        args = config_manager.get_arguments_by_category(category_id)

        # Exclude model field from profiles (it's selected separately when serving)
        args = [a for a in args if a["name"] != "model"]

        # Filter to show only high/critical importance by default
        important_args = [
            a for a in args if a.get("importance") in ["critical", "high"]
        ]
        other_args = [
            a for a in args if a.get("importance") not in ["critical", "high"]
        ]

        # Configure important arguments
        for arg_info in important_args:
            config = configure_argument(arg_info, config, config_manager)

        # Ask if they want to see additional options
        if other_args:
            show_additional = inquirer.confirm(
                tr(
                    "custom_cfg.configure_additional_options",
                    "Configure additional {name} options?",
                    name=category_name.lower(),
                ),
                default=False,
            )
            if show_additional:
                for arg_info in other_args:
                    config = configure_argument(arg_info, config, config_manager)

    # Ask about GPU selection
    console.print()  # Add blank line for spacing
    configure_gpus = inquirer.confirm(
        tr("custom_cfg.configure_gpu_selection", "Configure GPU selection?"),
        default=False,
    )

    if configure_gpus:
        current_device_ids = config.get("device_ids")
        # Convert list to comma-separated string for select_gpus
        current_str = (
            ",".join(str(v) for v in current_device_ids)
            if isinstance(current_device_ids, list)
            else None
        )
        new_device = select_gpus(current_str)
        if new_device is not None:
            device_list = [int(v.strip()) for v in new_device.split(",")]
            config["device_ids"] = device_list
            console.print(
                f"[green]{tr('custom_cfg.gpu_selection_set', 'GPU selection')}: {device_list}[/green]"
            )

            # Check tensor_parallel_size compatibility if already configured
            if config.get("tensor_parallel_size"):
                num_selected = len(device_list)
                tp_size = config["tensor_parallel_size"]

                if tp_size > num_selected:
                    console.print(
                        tr(
                            "custom_cfg.tp_size_exceeds_gpus",
                            "[yellow]Warning: tensor_parallel_size ({tp_size}) > selected GPUs ({gpu_count})[/yellow]",
                            tp_size=tp_size,
                            gpu_count=num_selected,
                        )
                    )
                    if inquirer.confirm(
                        tr(
                            "custom_cfg.adjust_tp_size",
                            "Adjust tensor_parallel_size to {gpu_count}?",
                            gpu_count=num_selected,
                        ),
                        default=True,
                    ):
                        config["tensor_parallel_size"] = num_selected
                        console.print(
                            tr(
                                "custom_cfg.tp_size_adjusted",
                                "[green]tensor_parallel_size adjusted to {gpu_count}[/green]",
                                gpu_count=num_selected,
                            )
                        )
                elif num_selected % tp_size != 0:
                    console.print(
                        tr(
                            "custom_cfg.tp_size_efficiency_note",
                            "[yellow]Note: {gpu_count} GPUs with tensor_parallel_size={tp_size} "
                            "may not utilize all GPUs efficiently[/yellow]",
                            gpu_count=num_selected,
                            tp_size=tp_size,
                        )
                    )

        elif new_device is None and current_device_ids:
            # User cleared the selection
            del config["device_ids"]
            console.print(
                tr(
                    "custom_cfg.gpu_selection_cleared",
                    "[yellow]GPU selection cleared[/yellow]",
                )
            )

    # Now ask about advanced options with hierarchical menu
    console.print()  # Add blank line for spacing
    configure_advanced = inquirer.confirm(
        tr("custom_cfg.configure_advanced", "Configure advanced options?"),
        default=False,
    )

    if configure_advanced:
        config = configure_advanced_hierarchical(config, config_manager)

    # Ask about environment variables - available for all configurations
    console.print()  # Add blank line for spacing
    configure_env = inquirer.confirm(
        tr("custom_cfg.configure_env_vars", "Configure environment variables?"),
        default=False,
    )

    if configure_env:
        # For direct server configuration (has model), store as session_environment
        # For profile creation (no model), store as environment
        if base_config and "model" in base_config:
            # This is for Custom Configuration from main menu
            config["session_environment"] = configure_environment_variables(
                config.get("session_environment", {})
            )
        else:
            # This is for profile creation
            config["environment"] = configure_environment_variables(
                config.get("environment", {})
            )

    return config


def configure_advanced_hierarchical(
    config: Dict[str, Any], config_manager: ConfigManager
) -> Dict[str, Any]:
    """
    Configure advanced options using a hierarchical category menu.

    Args:
        config: Current configuration
        config_manager: ConfigManager instance

    Returns:
        Updated configuration
    """
    # Get all categories for complete configuration
    categories = config_manager.get_ordered_categories()
    advanced_categories = categories  # Include all categories

    while True:
        # Build category menu
        category_choices = []

        # Add GPU selection as the first option
        gpu_status = ""
        if "device_ids" in config:
            gpu_status = tr(
                "custom_cfg.gpu_status_suffix",
                " [GPUs: {device_ids}]",
                device_ids=config["device_ids"],
            )
        category_choices.append(
            (
                "gpu",
                tr("custom_cfg.gpu_selection_option", "GPU Selection") + gpu_status,
            )
        )

        for cat_id, cat_info in advanced_categories:
            # icon = cat_info.get("icon", "")  # Reserved for future use
            name = cat_info["name"]

            # Count configured arguments in this category
            args = config_manager.get_arguments_by_category(cat_id)
            configured_count = sum(1 for arg in args if arg["name"] in config)
            total_count = len(args)

            if configured_count > 0:
                status = tr(
                    "custom_cfg.category_configured_status",
                    " [{configured_count}/{total_count} configured]",
                    configured_count=configured_count,
                    total_count=total_count,
                )
            else:
                status = tr(
                    "custom_cfg.category_options_status",
                    " [{total_count} options]",
                    total_count=total_count,
                )

            category_choices.append((cat_id, f"{name}{status}"))

        category_choices.append(
            ("done", tr("custom_cfg.done_configuring", "✓ Done configuring"))
        )

        # Show category selection menu
        console.print(
            f"\n[bold cyan]{tr('custom_cfg.advanced_select_category', 'Advanced Options - Select Category')}[/bold cyan]"
        )
        selected = prompt_choice(
            "category_select",
            tr("custom_cfg.choose_category", "Choose category to configure"),
            category_choices,
            allow_back=False,
        )

        if selected == "done" or not selected:
            break

        # Handle GPU selection
        if selected == "gpu":
            current_device_ids = config.get("device_ids")
            current_str = (
                ",".join(str(v) for v in current_device_ids)
                if isinstance(current_device_ids, list)
                else None
            )
            new_device = select_gpus(current_str)
            if new_device is not None:
                device_list = [int(v.strip()) for v in new_device.split(",")]
                config["device_ids"] = device_list
                console.print(
                    f"[green]{tr('custom_cfg.gpu_selection_updated', 'GPU selection updated')}: {device_list}[/green]"
                )

                # Check tensor_parallel_size compatibility
                if config.get("tensor_parallel_size"):
                    num_selected = len(device_list)
                    tp_size = config["tensor_parallel_size"]

                    if tp_size > num_selected:
                        console.print(
                            tr(
                                "custom_cfg.tp_size_exceeds_gpus",
                                "[yellow]Warning: tensor_parallel_size ({tp_size}) > selected GPUs ({gpu_count})[/yellow]",
                                tp_size=tp_size,
                                gpu_count=num_selected,
                            )
                        )
                        if inquirer.confirm(
                            tr(
                                "custom_cfg.adjust_tp_size",
                                "Adjust tensor_parallel_size to {gpu_count}?",
                                gpu_count=num_selected,
                            ),
                            default=True,
                        ):
                            config["tensor_parallel_size"] = num_selected
                            console.print(
                                tr(
                                    "custom_cfg.tp_size_adjusted",
                                    "[green]tensor_parallel_size adjusted to {gpu_count}[/green]",
                                    gpu_count=num_selected,
                                )
                            )
                    elif num_selected % tp_size != 0:
                        console.print(
                            tr(
                                "custom_cfg.tp_size_efficiency_note",
                                "[yellow]Note: {gpu_count} GPUs with tensor_parallel_size={tp_size} "
                                "may not utilize all GPUs efficiently[/yellow]",
                                gpu_count=num_selected,
                                tp_size=tp_size,
                            )
                        )

            elif new_device is None and current_device_ids:
                # User cleared the selection
                del config["device_ids"]
                console.print(
                    tr(
                        "custom_cfg.gpu_selection_cleared",
                        "[yellow]GPU selection cleared[/yellow]",
                    )
                )
            continue

        # Find the selected category
        for cat_id, cat_info in advanced_categories:
            if cat_id == selected:
                # Configure this category
                config = configure_category_arguments(
                    cat_id, cat_info, config, config_manager
                )
                break

    return config


def configure_category_arguments(
    category_id: str,
    category_info: Dict[str, Any],
    config: Dict[str, Any],
    config_manager: ConfigManager,
) -> Dict[str, Any]:
    """
    Configure arguments within a specific category using list selection.

    Args:
        category_id: Category identifier
        category_info: Category metadata
        config: Current configuration
        config_manager: ConfigManager instance

    Returns:
        Updated configuration
    """
    category_name = category_info["name"]
    # icon = category_info.get("icon", "")  # Reserved for future use

    # Get arguments for this category
    args = config_manager.get_arguments_by_category(category_id)

    # Exclude model field from profiles (it's selected separately when serving)
    args = [a for a in args if a["name"] != "model"]

    while True:
        # Build argument list with current values
        console.print(f"\n[bold cyan]{category_name}[/bold cyan]")
        console.print(f"[dim]{category_info.get('description', '')}[/dim]\n")

        arg_choices = []

        for arg_info in args:
            arg_name = arg_info["name"]
            current_value = config.get(arg_name, arg_info.get("default"))
            importance = arg_info.get("importance", "low")
            description = arg_info.get("description", "")

            # Format display string WITHOUT Rich markup for inquirer
            if current_value is not None and arg_name in config:
                # Configured value (user set)
                value_str = str(current_value)
                status_icon = "●"
            elif current_value is not None:
                # Default value
                value_str = f"({current_value})"
                status_icon = "○"
            else:
                # Not set
                value_str = ""
                status_icon = "○"

            # Add importance indicator
            if importance in ["high", "critical"]:
                importance_icon = "!"
            else:
                importance_icon = " "

            # Create display string with description (plain text for inquirer)
            if value_str:
                display = f"{status_icon} {importance_icon} {arg_name}: {value_str}"
            else:
                display = f"{status_icon} {importance_icon} {arg_name}"

            if description and len(description) < 40:
                display += f" - {description[:40]}"

            arg_choices.append((arg_name, display))

        # Add navigation options
        arg_choices.append(
            ("back", tr("custom_cfg.back_to_categories", "← Back to categories"))
        )

        # Show argument selection
        selected = prompt_choice(
            "arg_select",
            tr("custom_cfg.select_argument", "Select argument to configure"),
            arg_choices,
            allow_back=False,
        )

        if selected == "back" or not selected:
            break

        # Configure selected argument
        arg_info = next((a for a in args if a["name"] == selected), None)
        if arg_info:
            config = configure_argument(arg_info, config, config_manager)

    return config


def configure_advanced_list(
    args: List[Dict],
    config: Dict[str, Any],
    config_manager: ConfigManager,
    category_name: str,
) -> Dict[str, Any]:
    """
    Configure advanced arguments using a list-based selection approach.
    This is kept for backward compatibility but enhanced with better formatting.

    Args:
        args: List of argument information
        config: Current configuration
        config_manager: ConfigManager instance
        category_name: Name of the category

    Returns:
        Updated configuration
    """
    while True:
        # Build list of arguments with current values
        console.print(
            f"\n[bold cyan]{tr('custom_cfg.category_configuration_title', '{category} Configuration', category=category_name)}[/bold cyan]"
        )

        arg_choices = []

        for arg_info in args:
            arg_name = arg_info["name"]
            current_value = config.get(arg_name, arg_info.get("default"))
            importance = arg_info.get("importance", "low")
            # description = arg_info.get("description", "")  # Reserved for future use

            # Format display string WITHOUT Rich markup for inquirer
            if current_value is not None and arg_name in config:
                # User configured
                value_str = str(current_value)
                status_icon = "●"
            elif current_value is not None:
                # Default value
                value_str = f"({current_value})"
                status_icon = "○"
            else:
                # Not set
                value_str = ""
                status_icon = "○"

            # Add importance indicator
            if importance in ["high", "critical"]:
                importance_icon = "!"
            else:
                importance_icon = " "

            if value_str:
                display = f"{status_icon} {importance_icon} {arg_name}: {value_str}"
            else:
                display = f"{status_icon} {importance_icon} {arg_name}"

            arg_choices.append((arg_name, display))

        # Add navigation options
        arg_choices.append(("back", tr("messages.back", "← Back")))

        selected = prompt_choice(
            "arg_select",
            tr("custom_cfg.select_argument", "Select argument to configure"),
            arg_choices,
            allow_back=False,
        )

        if selected == "back" or not selected:
            break

        # Configure selected argument
        arg_info = next((a for a in args if a["name"] == selected), None)
        if arg_info:
            config = configure_argument(arg_info, config, config_manager)

    return config


def configure_argument(
    arg_info: Dict[str, Any], config: Dict[str, Any], config_manager: ConfigManager
) -> Dict[str, Any]:
    """
    Configure a single argument with simplified numerical inputs.

    Args:
        arg_info: Argument schema information
        config: Current configuration dictionary
        config_manager: ConfigManager instance

    Returns:
        Updated configuration dictionary
    """
    arg_name = arg_info["name"]
    arg_type = arg_info.get("type")
    description = arg_info.get("description", "")
    default = arg_info.get("default")
    hint = arg_info.get("hint", "")

    # Check dependencies
    if "depends_on" in arg_info:
        dependency = arg_info["depends_on"]
        # Check if dependency exists in config (even if value is None/null)
        # Only skip if dependency key is completely absent
        if dependency not in config:
            return config  # Skip if dependency not met

    # Build description
    current_value = config.get(arg_name, default)

    console.print(f"\n[bold]{arg_name}[/bold]")
    console.print(f"[dim]{description}[/dim]")
    if hint:
        console.print(f"[yellow dim]{hint}[/yellow dim]")

    # Handle different argument types
    if arg_type == "boolean":
        # Use inquirer for boolean choices
        current_str = current_value if current_value is not None else default
        console.print()  # Add blank line for spacing
        result = inquirer.confirm(
            tr("custom_cfg.enable_argument", "Enable {name}?", name=arg_name),
            default=current_str if isinstance(current_str, bool) else False,
        )
        config[arg_name] = result

    elif arg_type == "choice":
        choices = arg_info.get("choices", [])

        # Special handling for quantization
        if arg_name == "quantization":
            choice_options = [
                (
                    "none",
                    tr(
                        "custom_cfg.quant_none",
                        "None - No quantization (full precision)",
                    ),
                ),
                (
                    "awq",
                    "awq - "
                    + tr(
                        "custom_cfg.quant_awq",
                        "AutoAWQ (Activation-aware Weight Quantization)",
                    ),
                ),
                (
                    "awq_marlin",
                    "awq_marlin - "
                    + tr(
                        "custom_cfg.quant_awq_marlin",
                        "AutoAWQ with Marlin kernel",
                    ),
                ),
                (
                    "bitsandbytes",
                    "bitsandbytes - "
                    + tr("custom_cfg.quant_bitsandbytes", "8-bit/4-bit quantization"),
                ),
                (
                    "gptq",
                    "gptq - " + tr("custom_cfg.quant_gptq", "GPT Quantization"),
                ),
                (
                    "fp8",
                    "fp8 - " + tr("custom_cfg.quant_fp8", "8-bit floating point"),
                ),
                (
                    "gguf",
                    "gguf - " + tr("custom_cfg.quant_gguf", "GGML universal format"),
                ),
                (
                    "compressed-tensors",
                    "compressed-tensors - "
                    + tr("custom_cfg.quant_compressed_tensors", "INT4/INT8 compressed"),
                ),
                (
                    "modelopt_fp4",
                    "modelopt_fp4 - "
                    + tr(
                        "custom_cfg.quant_modelopt_fp4",
                        "ModelOpt NVFP4 (FP4 weight quantization)",
                    ),
                ),
                (
                    "modelopt_mxfp8",
                    "modelopt_mxfp8 - "
                    + tr(
                        "custom_cfg.quant_modelopt_mxfp8",
                        "ModelOpt MXFP8 (Mixed-precision FP8)",
                    ),
                ),
                ("skip", tr("custom_cfg.skip_use_default", "Skip (use default)")),
            ]

            selected = prompt_choice(
                arg_name,
                tr("custom_cfg.select_quantization", "Select quantization method"),
                choice_options,
                allow_back=False,
            )

            if selected and selected != "skip":
                if selected == "none":
                    config[arg_name] = None
                else:
                    # Stable value is the quantization method itself
                    config[arg_name] = selected
        else:
            # Regular choice field
            choice_options = []
            for choice in choices:
                if choice is None:
                    choice_options.append(
                        (
                            "none",
                            tr(
                                "custom_cfg.choice_none_default",
                                "None (use vLLM default)",
                            ),
                        )
                    )
                else:
                    choice_options.append((str(choice), str(choice)))

            choice_options.append(
                ("skip", tr("custom_cfg.skip_use_default", "Skip (use default)"))
            )

            if current_value is not None:
                console.print(
                    f"{tr('custom_cfg.current_label', 'Current')}: {current_value}"
                )

            selected = prompt_choice(
                arg_name,
                tr("custom_cfg.select_value_for", "Select {name}", name=arg_name),
                choice_options,
                allow_back=False,
            )

            if selected and selected != "skip":
                if selected == "none":
                    config[arg_name] = None
                else:
                    for c in choices:
                        if str(c) == selected:
                            config[arg_name] = c
                            break

    elif arg_type == "integer":
        validation = arg_info.get("validation", {})
        min_val = validation.get("min")
        max_val = validation.get("max")

        # Special handling for tensor_parallel_size
        if arg_name == "tensor_parallel_size":
            from ..system.gpu import get_gpu_info

            # Check if specific GPUs are selected
            if config.get("device_ids"):
                device_ids = config["device_ids"]
                num_gpus = len(device_ids) if isinstance(device_ids, list) else 1
                console.print(
                    tr(
                        "custom_cfg.using_selected_gpus",
                        "[dim]Using {gpu_count} selected GPU(s): {device_ids}[/dim]",
                        gpu_count=num_gpus,
                        device_ids=device_ids,
                    )
                )
            else:
                try:
                    gpus = get_gpu_info()
                    num_gpus = len(gpus) if gpus else 1
                    console.print(
                        tr(
                            "custom_cfg.detected_gpus",
                            "[dim]Detected {gpu_count} GPU(s) in system[/dim]",
                            gpu_count=num_gpus,
                        )
                    )
                except Exception:
                    num_gpus = 1

            if num_gpus == 1:
                console.print(
                    tr(
                        "custom_cfg.single_gpu_note",
                        "[yellow]Single GPU detected, vLLM will use 1 GPU by default[/yellow]",
                    )
                )
                response = input(
                    tr(
                        "custom_cfg.enter_tensor_parallel_size",
                        "Enter tensor parallel size (1) or press Enter to use default: ",
                    )
                ).strip()
                if response:
                    try:
                        config[arg_name] = int(response)
                    except ValueError:
                        console.print(
                            tr(
                                "custom_cfg.invalid_number_using_default",
                                "[red]Invalid number, using default[/red]",
                            )
                        )
                # Don't set tensor_parallel_size for single GPU (let vLLM default)
                return config
            else:
                # For multi-GPU, suggest using all GPUs but let user choose
                console.print(
                    tr(
                        "custom_cfg.multi_gpu_note",
                        "[green]Multi-GPU system detected, tensor parallelism recommended[/green]",
                    )
                )
                # Suggest common divisors for better GPU utilization
                if num_gpus > 1:
                    divisors = [i for i in range(1, num_gpus + 1) if num_gpus % i == 0]
                    console.print(
                        tr(
                            "custom_cfg.recommended_tp_values",
                            "[dim]Recommended values for {gpu_count} GPUs: {values}[/dim]",
                            gpu_count=num_gpus,
                            values=", ".join(map(str, divisors)),
                        )
                    )
                # Set default to detected GPU count
                default = num_gpus

        # Special handling for max_model_len
        if arg_name == "max_model_len":
            console.print(
                tr(
                    "custom_cfg.max_model_len_native_note",
                    "[dim]Leave empty to use model's native maximum context length[/dim]",
                )
            )
            response = input(
                tr(
                    "custom_cfg.enter_max_model_len",
                    "Enter max model length (e.g., 1M, 100K, 1000) or press Enter for native max: ",
                )
            ).strip()
            if response:
                parsed = parse_model_length(response)
                if parsed is not None:
                    config[arg_name] = parsed
                else:
                    console.print(
                        tr(
                            "custom_cfg.invalid_format_skipping",
                            "[red]Invalid format, skipping max_model_len[/red]",
                        )
                    )
            # If empty, don't set max_model_len (let vLLM use model's native max)
            return config

        # Simple numerical input
        range_str = ""
        if min_val is not None or max_val is not None:
            if min_val is not None and max_val is not None:
                range_str = tr(
                    "custom_cfg.range_min_max",
                    " (range: {min}-{max})",
                    min=min_val,
                    max=max_val,
                )
            elif min_val is not None:
                range_str = tr("custom_cfg.range_min", " (min: {min})", min=min_val)
            elif max_val is not None:
                range_str = tr("custom_cfg.range_max", " (max: {max})", max=max_val)

        if default is not None:
            prompt_text = tr(
                "custom_cfg.enter_value_default",
                "Enter value{range} or press Enter for default ({default}): ",
                range=range_str,
                default=default,
            )
        else:
            prompt_text = tr(
                "custom_cfg.enter_value_skip",
                "Enter value{range} or press Enter to skip: ",
                range=range_str,
            )

        response = input(prompt_text).strip()

        if response:
            try:
                value = int(response)
                if min_val is not None and value < min_val:
                    console.print(
                        tr(
                            "custom_cfg.value_below_min",
                            "[yellow]Value must be at least {min_val}, using {min_val}[/yellow]",
                            min_val=min_val,
                        )
                    )
                    config[arg_name] = min_val
                elif max_val is not None and value > max_val:
                    console.print(
                        tr(
                            "custom_cfg.value_above_max",
                            "[yellow]Value must be at most {max_val}, using {max_val}[/yellow]",
                            max_val=max_val,
                        )
                    )
                    config[arg_name] = max_val
                else:
                    config[arg_name] = value
            except ValueError:
                console.print(
                    tr(
                        "custom_cfg.invalid_integer",
                        "[yellow]Invalid integer value, skipping[/yellow]",
                    )
                )
        # If no response and user pressed Enter, don't add to config (use default)

    elif arg_type == "float":
        validation = arg_info.get("validation", {})
        min_val = validation.get("min")
        max_val = validation.get("max")

        # Simple numerical input for all float fields
        range_str = ""
        if min_val is not None or max_val is not None:
            if min_val is not None and max_val is not None:
                range_str = tr(
                    "custom_cfg.range_min_max",
                    " (range: {min}-{max})",
                    min=min_val,
                    max=max_val,
                )
            elif min_val is not None:
                range_str = tr("custom_cfg.range_min", " (min: {min})", min=min_val)
            elif max_val is not None:
                range_str = tr("custom_cfg.range_max", " (max: {max})", max=max_val)

        # Special prompt for gpu_memory_utilization
        if arg_name == "gpu_memory_utilization":
            console.print(
                tr(
                    "custom_cfg.gpu_memory_common_values",
                    "[dim]Common values: 0.5 (50%), 0.7 (70%), 0.9 (90%), 0.95 (95%)[/dim]",
                )
            )

        if default is not None:
            prompt_text = tr(
                "custom_cfg.enter_value_default",
                "Enter value{range} or press Enter for default ({default}): ",
                range=range_str,
                default=default,
            )
        else:
            prompt_text = tr(
                "custom_cfg.enter_value_skip",
                "Enter value{range} or press Enter to skip: ",
                range=range_str,
            )

        response = input(prompt_text).strip()

        if response:
            try:
                value = float(response)
                if min_val is not None and value < min_val:
                    console.print(
                        tr(
                            "custom_cfg.value_below_min",
                            "[yellow]Value must be at least {min_val}, using {min_val}[/yellow]",
                            min_val=min_val,
                        )
                    )
                    config[arg_name] = min_val
                elif max_val is not None and value > max_val:
                    console.print(
                        tr(
                            "custom_cfg.value_above_max",
                            "[yellow]Value must be at most {max_val}, using {max_val}[/yellow]",
                            max_val=max_val,
                        )
                    )
                    config[arg_name] = max_val
                else:
                    config[arg_name] = value
            except ValueError:
                console.print(
                    tr(
                        "custom_cfg.invalid_float",
                        "[yellow]Invalid float value, skipping[/yellow]",
                    )
                )
        # If no response and user pressed Enter, don't add to config (use default)

    elif arg_type == "string":
        sensitive = arg_info.get("sensitive", False)

        # Special handling for chat_template - offer file browsing
        if arg_name == "chat_template":
            import os
            from pathlib import Path

            console.print(
                tr(
                    "custom_cfg.chat_template_jinja_note",
                    "\n[dim]Chat templates are Jinja2 files that define how to format conversations.[/dim]",
                )
            )
            console.print(
                tr(
                    "custom_cfg.chat_template_locations_note",
                    "[dim]Common locations: ./examples/, ~/.cache/huggingface/, or custom paths[/dim]",
                )
            )

            # Offer options
            options = [
                (
                    "manual",
                    tr(
                        "custom_cfg.enter_file_path_manually",
                        "Enter file path manually",
                    ),
                ),
                (
                    "browse",
                    tr("custom_cfg.browse_jinja_files", "Browse for .jinja files"),
                ),
                ("skip", tr("custom_cfg.skip", "Skip")),
            ]
            if current_value:
                options.insert(
                    0,
                    (
                        "keep",
                        tr(
                            "custom_cfg.keep_current_value",
                            "Keep current: {value}",
                            value=current_value,
                        ),
                    ),
                )

            choice = prompt_choice(
                "chat_template_option",
                tr(
                    "custom_cfg.chat_template_how_question",
                    "How would you like to specify the chat template?",
                ),
                options,
                allow_back=False,
            )

            if not choice or choice in ("skip", "keep"):
                pass  # Don't set chat_template / keep existing value
            elif choice == "browse":
                # Search for .jinja files in common locations
                search_paths = [
                    Path.cwd() / "examples",
                    Path.cwd(),
                    Path.home() / ".cache" / "huggingface",
                ]

                jinja_files = []
                for search_path in search_paths:
                    if search_path.exists():
                        # Search recursively but limit depth
                        for jinja_file in search_path.rglob("*.jinja"):
                            if jinja_file.is_file():
                                # Make path relative if it's under cwd
                                try:
                                    rel_path = jinja_file.relative_to(Path.cwd())
                                    jinja_files.append(str(rel_path))
                                except ValueError:
                                    jinja_files.append(str(jinja_file))

                if jinja_files:
                    jinja_files = sorted(set(jinja_files))[:20]  # Limit to 20 files
                    file_options = [(path, path) for path in jinja_files]
                    file_options.append(
                        (
                            "manual",
                            tr(
                                "custom_cfg.enter_path_manually",
                                "← Enter path manually",
                            ),
                        )
                    )

                    selected_file = prompt_choice(
                        "select_jinja",
                        tr(
                            "custom_cfg.select_chat_template_file",
                            "Select chat template file",
                        ),
                        file_options,
                        allow_back=False,
                    )

                    if selected_file and selected_file != "manual":
                        config[arg_name] = selected_file
                        console.print(
                            tr(
                                "custom_cfg.chat_template_set",
                                "[green]Chat template set to: {path}[/green]",
                                path=selected_file,
                            )
                        )
                    elif selected_file == "manual":
                        response = (
                            input(
                                tr(
                                    "custom_cfg.enter_chat_template_path",
                                    "Enter chat template file path: ",
                                )
                            )
                            .strip()
                        )
                        if response:
                            if os.path.exists(response):
                                config[arg_name] = response
                                console.print(
                                    tr(
                                        "custom_cfg.chat_template_set",
                                        "[green]Chat template set to: {path}[/green]",
                                        path=response,
                                    )
                                )
                            else:
                                console.print(
                                    tr(
                                        "custom_cfg.file_not_found_warning",
                                        "[yellow]Warning: File not found: {path}[/yellow]",
                                        path=response,
                                    )
                                )
                                confirm = inquirer.confirm(
                                    tr(
                                        "custom_cfg.use_path_anyway",
                                        "Use this path anyway?",
                                    ),
                                    default=False,
                                )
                                if confirm:
                                    config[arg_name] = response
                else:
                    console.print(
                        tr(
                            "custom_cfg.no_jinja_files_found",
                            "[yellow]No .jinja files found in common locations.[/yellow]",
                        )
                    )
                    response = (
                        input(
                            tr(
                                "custom_cfg.enter_chat_template_path_or_skip",
                                "Enter chat template file path or press Enter to skip: ",
                            )
                        )
                        .strip()
                    )
                    if response:
                        config[arg_name] = response
            elif choice == "manual":
                if current_value:
                    prompt_text = tr(
                        "custom_cfg.enter_file_path_current",
                        "Enter file path (current: {value}) or press Enter to skip: ",
                        value=current_value,
                    )
                else:
                    prompt_text = tr(
                        "custom_cfg.enter_file_path_or_skip",
                        "Enter file path or press Enter to skip: ",
                    )

                response = input(prompt_text).strip()
                if response:
                    # Check if file exists
                    if os.path.exists(response):
                        config[arg_name] = response
                        console.print(
                            tr(
                                "custom_cfg.chat_template_set",
                                "[green]Chat template set to: {path}[/green]",
                                path=response,
                            )
                        )
                    else:
                        console.print(
                            tr(
                                "custom_cfg.file_not_found_warning",
                                "[yellow]Warning: File not found: {path}[/yellow]",
                                path=response,
                            )
                        )
                        confirm = inquirer.confirm(
                            tr("custom_cfg.use_path_anyway", "Use this path anyway?"),
                            default=False,
                        )
                        if confirm:
                            config[arg_name] = response
        else:
            # Regular string input for other fields
            if sensitive:
                prompt_text = tr(
                    "custom_cfg.enter_value_hidden",
                    "Enter value (hidden) or press Enter to skip: ",
                )
            else:
                if current_value:
                    prompt_text = tr(
                        "custom_cfg.enter_value_current",
                        "Enter value (current: {value}) or press Enter to skip: ",
                        value=current_value,
                    )
                else:
                    prompt_text = tr(
                        "custom_cfg.enter_value_plain",
                        "Enter value or press Enter to skip: ",
                    )

            response = input(prompt_text).strip()
            if response:
                config[arg_name] = response

    elif arg_type == "list":
        # Generic list input - comma-separated values
        # Note: device_ids is handled by GPU Selection, not exposed here
        if arg_name == "device_ids":
            console.print(
                tr(
                    "custom_cfg.device_ids_via_gpu_selection",
                    "[yellow]Device IDs are managed via GPU Selection. Skipping.[/yellow]",
                )
            )
        elif isinstance(current_value, list) and current_value:
            current_display = ", ".join(str(v) for v in current_value)
            prompt_text = tr(
                "custom_cfg.enter_values_current",
                "Enter values (comma-separated, current: {value}) or press Enter to skip: ",
                value=current_display,
            )
            response = input(prompt_text).strip()
            if response:
                config[arg_name] = [
                    v.strip() for v in response.split(",") if v.strip()
                ]
        else:
            prompt_text = tr(
                "custom_cfg.enter_values",
                "Enter values (comma-separated) or press Enter to skip: ",
            )
            response = input(prompt_text).strip()
            if response:
                config[arg_name] = [
                    v.strip() for v in response.split(",") if v.strip()
                ]

    return config


def select_gpus(
    current_selection: Optional[str] = None,
    proxy_manager=None,
    required_utilization: float = None,
) -> Optional[str]:
    """
    Interactive GPU selection with checkbox UI and memory usage display.

    Args:
        current_selection: Current GPU selection as comma-separated string
        proxy_manager: ProxyManager instance (unused, kept for compatibility)
        required_utilization: Unused, kept for compatibility

    Returns:
        Comma-separated GPU indices (e.g., "0,1,2") or None
    """
    # Get available GPUs
    gpu_info = gpu.get_gpu_info()

    if not gpu_info:
        console.print(
            tr(
                "custom_cfg.no_gpus_detected",
                "[yellow]No GPUs detected[/yellow]",
            )
        )
        return None

    # Get user's preferred progress bar style
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()
    ui_prefs.get("progress_bar_style", "blocks")

    # Get current GPU memory usage
    from .gpu_utils import get_gpu_memory_dict

    gpu_memory = get_gpu_memory_dict()

    # Parse current selection
    selected_indices = []
    if current_selection:
        try:
            selected_indices = [
                int(idx.strip()) for idx in current_selection.split(",")
            ]
        except (ValueError, AttributeError):
            selected_indices = []

    # Build choices for checkbox with memory usage info
    gpu_choices = []
    for gpu_data in gpu_info:
        idx = gpu_data["index"]
        name = gpu_data["name"]
        memory_gb = gpu_data["memory_total"] / (1024**3)  # Convert bytes to GB

        # Build base label
        choice_label = f"GPU {idx}: {name} ({memory_gb:.1f}GB)"

        # Add memory usage info if available
        if idx in gpu_memory:
            mem_info = gpu_memory[idx]
            usage_percent = mem_info["usage_percent"]

            # Show memory usage percentage
            choice_label += " - " + tr(
                "custom_cfg.gpu_percent_used",
                "{percent}% used",
                percent=f"{usage_percent:.0f}",
            )

            # Add warning if memory is already in use
            if usage_percent > 10:
                choice_label += " ⚠️"

        gpu_choices.append((choice_label, idx, idx in selected_indices))

    # Show current selection if exists
    if current_selection:
        console.print(
            tr(
                "custom_cfg.current_gpu_selection",
                "\n[dim]Current selection: {selection}[/dim]",
                selection=current_selection,
            )
        )

    # Show warning about GPU memory
    has_gpu_usage = any(
        gpu_memory.get(i, {}).get("usage_percent", 0) > 10 for i in range(len(gpu_info))
    )
    if has_gpu_usage:
        console.print(
            tr(
                "custom_cfg.gpu_memory_in_use_warning",
                "\n[yellow]⚠️  Warning: Some GPUs have memory in use. Loading models may cause OOM errors.[/yellow]",
            )
        )

    console.print(
        tr(
            "custom_cfg.gpu_select_instructions",
            "\n[dim]Select GPUs to use (space to select/deselect, enter to confirm)[/dim]",
        )
    )

    # Create checkbox question
    questions = [
        inquirer.Checkbox(
            "gpus",
            message=tr("custom_cfg.select_gpus_message", "Select GPUs to use"),
            choices=[(label, idx) for label, idx, _ in gpu_choices],
            default=[idx for _, idx, selected in gpu_choices if selected],
        )
    ]

    answers = inquirer.prompt(questions)

    if not answers:
        return current_selection  # User cancelled

    selected_gpus = answers.get("gpus", [])

    if not selected_gpus:
        # Ask for confirmation if no GPUs selected
        if current_selection:
            if inquirer.confirm(
                tr("custom_cfg.clear_gpu_selection", "Clear GPU selection?"),
                default=False,
            ):
                return None
        else:
            console.print(
                tr(
                    "custom_cfg.no_gpus_selected",
                    "[yellow]No GPUs selected, using default[/yellow]",
                )
            )
        return current_selection

    # Return comma-separated string
    return ",".join(str(idx) for idx in sorted(selected_gpus))


def configure_environment_variables(
    existing_env: Dict[str, str] = None,
) -> Dict[str, str]:
    """
    Configure environment variables for the profile.

    Args:
        existing_env: Existing environment variables

    Returns:
        Dictionary of environment variables
    """
    env = existing_env.copy() if existing_env else {}

    # Define environment variable presets by category
    # NO DEFAULT VALUES - let vLLM use its own defaults unless user explicitly sets
    env_presets = {
        "GPU/Hardware Optimization": {
            "VLLM_USE_TRITON_FLASH_ATTN": tr(
                "custom_cfg.env_desc_use_triton_flash_attn",
                "Enable Triton flash attention (usually 1)",
            ),
            "VLLM_ATTENTION_BACKEND": tr(
                "custom_cfg.env_desc_attention_backend",
                "Attention backend (e.g., TRITON_ATTN_VLLM_V1 for A100)",
            ),
            "VLLM_USE_TRTLLM_ATTENTION": tr(
                "custom_cfg.env_desc_use_trtllm_attention",
                "TensorRT-LLM attention for Blackwell GPUs (usually 1)",
            ),
            "VLLM_USE_TRTLLM_DECODE_ATTENTION": tr(
                "custom_cfg.env_desc_use_trtllm_decode_attention",
                "Use TensorRT-LLM for decode attention (usually 1)",
            ),
            "VLLM_USE_TRTLLM_CONTEXT_ATTENTION": tr(
                "custom_cfg.env_desc_use_trtllm_context_attention",
                "Use TensorRT-LLM for context attention (usually 1)",
            ),
            "VLLM_FLASHINFER_FORCE_TENSOR_CORES": tr(
                "custom_cfg.env_desc_force_tensor_cores",
                "Force tensor core usage (usually 1)",
            ),
        },
        "MoE Model Optimization": {
            "VLLM_USE_FLASHINFER_MXFP4_MOE": tr(
                "custom_cfg.env_desc_use_flashinfer_mxfp4_moe",
                "Enable FlashInfer MXFP4 for MoE (recommended for GPT-OSS, usually 1)",
            ),
            "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16": tr(
                "custom_cfg.env_desc_flashinfer_moe_mxfp4_bf16",
                "MoE with BF16 precision for GPT-OSS (usually 1)",
            ),
            "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8": tr(
                "custom_cfg.env_desc_flashinfer_moe_mxfp4_mxfp8",
                "MoE with FP8 precision (usually 1)",
            ),
            "VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING": tr(
                "custom_cfg.env_desc_enable_fused_moe_activation_chunking",
                "Enable MoE activation chunking (usually 1)",
            ),
            "VLLM_FUSED_MOE_CHUNK_SIZE": tr(
                "custom_cfg.env_desc_fused_moe_chunk_size",
                "MoE chunk size (e.g., 65536)",
            ),
        },
        "AMD/ROCm Optimization": {
            "VLLM_ROCM_USE_AITER": tr(
                "custom_cfg.env_desc_rocm_use_aiter",
                "Enable AMD async iterator (usually 1)",
            ),
            "VLLM_USE_AITER_UNIFIED_ATTENTION": tr(
                "custom_cfg.env_desc_use_aiter_unified_attention",
                "Unified attention for AMD GPUs (usually 1)",
            ),
            "VLLM_ROCM_USE_SKINNY_GEMM": tr(
                "custom_cfg.env_desc_rocm_use_skinny_gemm",
                "Enable skinny GEMM for ROCm (usually 1)",
            ),
            "VLLM_ROCM_FP8_PADDING": tr(
                "custom_cfg.env_desc_rocm_fp8_padding",
                "FP8 padding for ROCm (usually 1)",
            ),
        },
        "Logging & Monitoring": {
            "VLLM_LOGGING_LEVEL": tr(
                "custom_cfg.env_desc_logging_level",
                "Log level (DEBUG/INFO/WARNING/ERROR)",
            ),
            "VLLM_LOG_STATS_INTERVAL": tr(
                "custom_cfg.env_desc_log_stats_interval",
                "Stats logging interval in seconds",
            ),
            "VLLM_TRACE_FUNCTION": tr(
                "custom_cfg.env_desc_trace_function",
                "Enable function tracing (usually 1)",
            ),
            "VLLM_CONFIGURE_LOGGING": tr(
                "custom_cfg.env_desc_configure_logging",
                "Enable vLLM logging configuration (usually 1)",
            ),
        },
        "Memory & Performance": {
            "VLLM_CPU_KVCACHE_SPACE": tr(
                "custom_cfg.env_desc_cpu_kvcache_space",
                "CPU KV cache space in GiB",
            ),
            "VLLM_CPU_OMP_THREADS_BIND": tr(
                "custom_cfg.env_desc_cpu_omp_threads_bind",
                "OpenMP thread binding",
            ),
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": tr(
                "custom_cfg.env_desc_allow_long_max_model_len",
                "Allow long context lengths (usually 1)",
            ),
            "VLLM_ENABLE_V1_MULTIPROCESSING": tr(
                "custom_cfg.env_desc_enable_v1_multiprocessing",
                "Enable V1 multiprocessing (usually 1)",
            ),
        },
        "CUDA Configuration": {
            "CUDA_HOME": tr("custom_cfg.env_desc_cuda_home", "Path to CUDA toolkit"),
            "CUDA_VISIBLE_DEVICES": tr(
                "custom_cfg.env_desc_cuda_visible_devices",
                "Visible GPU devices (e.g., 0,1,2)",
            ),
            "TORCH_CUDA_ARCH_LIST": tr(
                "custom_cfg.env_desc_torch_cuda_arch_list",
                "CUDA architectures (e.g., 9.0)",
            ),
            "PYTHONUNBUFFERED": tr(
                "custom_cfg.env_desc_pythonunbuffered",
                "Unbuffered Python output (usually 1)",
            ),
        },
    }

    # Display labels for preset categories (dict keys stay as-is)
    env_category_labels = {
        "GPU/Hardware Optimization": tr(
            "custom_cfg.env_cat_gpu_hardware", "GPU/Hardware Optimization"
        ),
        "MoE Model Optimization": tr(
            "custom_cfg.env_cat_moe", "MoE Model Optimization"
        ),
        "AMD/ROCm Optimization": tr(
            "custom_cfg.env_cat_amd_rocm", "AMD/ROCm Optimization"
        ),
        "Logging & Monitoring": tr(
            "custom_cfg.env_cat_logging_monitoring", "Logging & Monitoring"
        ),
        "Memory & Performance": tr(
            "custom_cfg.env_cat_memory_performance", "Memory & Performance"
        ),
        "CUDA Configuration": tr("custom_cfg.env_cat_cuda", "CUDA Configuration"),
    }

    while True:
        console.print(
            f"\n[bold cyan]{tr('custom_cfg.env_config_title', 'Environment Variable Configuration')}[/bold cyan]"
        )

        # Show current environment variables
        if env:
            console.print(
                f"\n[bold]{tr('custom_cfg.current_env_vars_title', 'Current Environment Variables:')}[/bold]"
            )
            for key, value in env.items():
                if "KEY" in key.upper() or "TOKEN" in key.upper():
                    console.print(
                        f"  • {key}: "
                        + tr("custom_cfg.hidden_value_placeholder", "<hidden>")
                    )
                else:
                    console.print(f"  • {key}: {value}")

        # Build menu options
        category_map = {}
        menu_options = []
        for index, category in enumerate(env_presets.keys()):
            option_value = f"category_{index}"
            category_label = env_category_labels.get(category, category)
            category_map[option_value] = (category, category_label)
            menu_options.append(
                (
                    option_value,
                    tr(
                        "custom_cfg.add_from_category",
                        "Add from {category}",
                        category=category_label,
                    ),
                )
            )
        menu_options.extend(
            [
                (
                    "custom",
                    tr("custom_cfg.add_custom_variable", "Add custom variable"),
                ),
                ("remove", tr("custom_cfg.remove_variable", "Remove variable")),
                ("clear", tr("custom_cfg.clear_all_variables", "Clear all variables")),
                ("done", tr("custom_cfg.done_configuring", "✓ Done configuring")),
            ]
        )

        choice = prompt_choice(
            "env_config",
            tr("custom_cfg.select_action", "Select action"),
            menu_options,
            allow_back=False,
        )

        if choice == "done" or not choice:
            break
        elif choice == "clear":
            if env:
                confirm = inquirer.confirm(
                    tr(
                        "custom_cfg.clear_all_env_vars_confirm",
                        "Clear all environment variables?",
                    ),
                    default=False,
                )
                if confirm:
                    env.clear()
                    console.print(
                        tr(
                            "custom_cfg.env_vars_cleared",
                            "[green]All environment variables cleared.[/green]",
                        )
                    )
            else:
                console.print(
                    tr(
                        "custom_cfg.no_env_vars_to_clear",
                        "[yellow]No environment variables to clear.[/yellow]",
                    )
                )
        elif choice == "remove":
            if env:
                var_options = [(var_key, var_key) for var_key in env.keys()]
                var_options.append(("back", tr("messages.back", "← Back")))
                var_to_remove = prompt_choice(
                    "remove_env",
                    tr(
                        "custom_cfg.select_variable_to_remove",
                        "Select variable to remove",
                    ),
                    var_options,
                    allow_back=False,
                )
                if var_to_remove and var_to_remove != "back":
                    del env[var_to_remove]
                    console.print(
                        tr(
                            "custom_cfg.env_var_removed",
                            "[green]Removed {name}[/green]",
                            name=var_to_remove,
                        )
                    )
            else:
                console.print(
                    tr(
                        "custom_cfg.no_env_vars_to_remove",
                        "[yellow]No environment variables to remove.[/yellow]",
                    )
                )
        elif choice == "custom":
            console.print(
                f"\n[bold]{tr('custom_cfg.add_custom_env_var_title', 'Add Custom Environment Variable')}[/bold]"
            )
            var_name = input(
                tr(
                    "custom_cfg.env_var_name_prompt",
                    "Variable name (e.g., VLLM_LOGGING_LEVEL): ",
                )
            ).strip()
            if var_name:
                var_value = input(
                    tr(
                        "custom_cfg.env_var_value_prompt",
                        "Value for {name}: ",
                        name=var_name,
                    )
                ).strip()
                if var_value:
                    env[var_name] = var_value
                    console.print(
                        tr(
                            "custom_cfg.env_var_added",
                            "[green]Added {name}={value}[/green]",
                            name=var_name,
                            value=var_value,
                        )
                    )
        elif choice in category_map:
            # Resolve the selected category
            category, category_label = category_map[choice]
            category_vars = env_presets[category]

            # Build list of variables in this category
            var_options = []
            for var_name, description in category_vars.items():
                if var_name in env:
                    status = tr(
                        "custom_cfg.env_var_status_set",
                        " [green]✓ Set to: {value}[/green]",
                        value=env[var_name],
                    )
                else:
                    status = ""
                var_options.append((var_name, f"{var_name}: {description}{status}"))
            var_options.append(("back", tr("messages.back", "← Back")))

            selected_var = prompt_choice(
                "select_env_var",
                tr(
                    "custom_cfg.select_variable_from_category",
                    "Select variable from {category}",  # nosec B608
                    category=category_label,
                ),
                var_options,
                allow_back=False,
            )

            if selected_var and selected_var != "back":
                var_name = selected_var
                if var_name in category_vars:
                    description = category_vars[var_name]

                    console.print(f"\n[bold]{var_name}[/bold]")
                    console.print(f"[dim]{description}[/dim]")

                    current = env.get(var_name, "")
                    if current:
                        prompt_text = tr(
                            "custom_cfg.enter_env_value_current",
                            "Enter value (current: {value}): ",
                            value=current,
                        )
                    else:
                        prompt_text = tr(
                            "custom_cfg.enter_env_value_default",
                            "Enter value (leave empty to use vLLM default): ",
                        )

                    new_value = input(prompt_text).strip()
                    if new_value:
                        env[var_name] = new_value
                        console.print(
                            tr(
                                "custom_cfg.env_var_set",
                                "[green]Set {name}={value}[/green]",
                                name=var_name,
                                value=new_value,
                            )
                        )
                    elif current:
                        # Keep current value
                        console.print(
                            tr(
                                "custom_cfg.keeping_current_value",
                                "[dim]Keeping current value: {value}[/dim]",
                                value=current,
                            )
                        )
                    else:
                        console.print(
                            tr(
                                "custom_cfg.using_vllm_default",
                                "[dim]Will use vLLM default (not set)[/dim]",
                            )
                        )

    return env


def create_custom_profile_interactive() -> Optional[str]:
    """
    Create a custom profile using the category-based configuration.

    Returns:
        Profile name if created successfully, None otherwise
    """
    console.print(
        f"\n[bold cyan]{tr('custom_cfg.create_custom_profile_title', 'Create Custom Profile')}[/bold cyan]"
    )

    # Get profile name
    name = input(tr("custom_cfg.profile_name_prompt", "Profile name: ")).strip()
    if not name:
        console.print(
            tr(
                "custom_cfg.profile_name_required",
                "[yellow]Profile name required.[/yellow]",
            )
        )
        return None

    description = input(
        tr("custom_cfg.profile_description_prompt", "Profile description (optional): ")
    ).strip()

    # Choose starting point
    config_manager = ConfigManager()
    all_profiles = config_manager.get_all_profiles()

    start_options = [
        ("scratch", tr("custom_cfg.start_from_scratch", "Start from scratch"))
    ] + [(pname, pname) for pname in all_profiles]
    start_choice = prompt_choice(
        "start",
        tr("custom_cfg.starting_configuration", "Starting configuration"),
        start_options,
        allow_back=True,
    )

    if not start_choice or start_choice == "BACK":
        return None

    # Get base configuration
    if start_choice == "scratch":
        base_config = {}
    else:
        profile = config_manager.get_profile(start_choice)
        base_config = profile.get("config", {}).copy() if profile else {}

    # Configure using categories
    config = configure_by_categories(base_config)

    # Show summary
    console.print(
        f"\n[bold cyan]{tr('custom_cfg.profile_summary_title', 'Profile Summary:')}[/bold cyan]"
    )
    if config:
        for key, value in config.items():
            if value is not None:
                console.print(f"  {key}: {value}")
    else:
        console.print(
            tr(
                "custom_cfg.no_custom_configuration",
                "[dim]No custom configuration - will use all defaults[/dim]",
            )
        )

    # Confirm save
    console.print()  # Add blank line for spacing
    save = inquirer.confirm(
        tr("custom_cfg.save_profile_confirm", "Save this profile?"), default=True
    )
    if not save:
        return None

    # Save profile
    profile_data = {
        "name": name,
        "description": description or "Custom profile",
        "icon": "",
        "config": config,
    }

    if config_manager.save_user_profile(name, profile_data):
        console.print(
            tr(
                "custom_cfg.profile_saved",
                "[green]Profile '{name}' saved successfully.[/green]",
                name=name,
            )
        )
        return name
    else:
        console.print(
            tr("custom_cfg.profile_save_failed", "[red]Failed to save profile.[/red]")
        )
        return None
