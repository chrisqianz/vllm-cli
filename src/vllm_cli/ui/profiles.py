#!/usr/bin/env python3
"""
Profile management module for vLLM CLI.

Handles creation, editing, and deletion of configuration profiles.
"""

import logging
from typing import Any, Dict, Optional
from .custom_config import parse_model_length

import inquirer

from ..config import ConfigManager
from ..i18n import tr
from .common import console
from .display import display_config
from .navigation import prompt_choice, unified_prompt

logger = logging.getLogger(__name__)


def manage_profiles(i18n_manager=None) -> str:
    """
    Manage configuration profiles.

    Args:
        i18n_manager: Optional I18nManager for translations
    """

    # Helper function for translation
    def t(key: str, default: str = None, **kwargs) -> str:
        if i18n_manager:
            result = i18n_manager.t(key, **kwargs)
            return result if result != key else (default if default else key)
        # Fallback to English
        fallback_map = {
            "menu.profiles.title": "Configuration Profiles",
            "menu.profiles.builtin": "Built-in Profiles:",
            "menu.profiles.user": "User Profiles:",
            "menu.profiles.customized": "(customized)",
            "messages.success": "Operation completed successfully",
            "messages.back": "← Back",
        }
        text = fallback_map.get(key, default if default else key)
        if kwargs:
            return text.format(**kwargs)
        return text

    config_manager = ConfigManager()
    # all_profiles = config_manager.get_all_profiles()  # Not used directly

    # Show existing profiles
    console.print(f"\n[bold cyan]{t('menu.profiles.title')}[/bold cyan]")

    # Separate default and user profiles
    default_profiles = config_manager.default_profiles
    user_profiles = config_manager.user_profiles

    # Built-in profiles
    console.print(
        tr("profiles_ui.builtin_profiles_header", "\n[bold]Built-in Profiles:[/bold]")
    )
    customized_tag = tr("profiles_ui.customized_suffix", "(customized)")
    for name, profile in default_profiles.items():
        icon = profile.get("icon", "")
        desc = profile.get("description", "")
        # Check if this built-in profile has been customized
        if config_manager.profile_manager.has_user_override(name):
            console.print(f"  {icon} {name} - {desc} [yellow]{customized_tag}[/yellow]")
        else:
            console.print(f"  {icon} {name} - {desc}")

    # User profiles (excluding overrides of built-in profiles)
    user_only_profiles = {
        name: profile
        for name, profile in user_profiles.items()
        if name not in default_profiles
    }

    if user_only_profiles:
        console.print(
            tr("profiles_ui.user_profiles_header", "\n[bold]User Profiles:[/bold]")
        )
        for name, profile in user_only_profiles.items():
            icon = profile.get("icon", "")
            desc = profile.get("description", "Custom profile")
            console.print(f"  {icon} {name} - {desc}")
    else:
        console.print(
            tr("profiles_ui.no_user_profiles", "\n[dim]No user-created profiles[/dim]")
        )

    # Profile actions - streamlined menu
    actions = [
        ("manage", tr("profiles_ui.action_manage", "Manage Profiles")),
        ("create", tr("profiles_ui.action_create", "Create New Profile")),
        ("import", tr("profiles_ui.action_import", "Import Profile")),
        ("sync_recipes", tr("profiles_ui.action_sync_recipes", "Sync Recipes")),
        ("sync_cli_args", tr("profiles_ui.action_sync_cli_args", "Sync CLI Args")),
    ]
    action = prompt_choice(
        "profile_action",
        tr("profiles_ui.profile_management", "Profile Management"),
        actions,
        allow_back=True,
    )

    if action == "manage":
        manage_selected_profile()
    elif action == "create":
        create_custom_profile()
    elif action == "import":
        import_profile()
    elif action == "sync_recipes":
        from .recipes_sync import sync_recipes

        sync_recipes()
    elif action == "sync_cli_args":
        from ..config.cli_args_sync import sync_cli_args as do_sync_args

        console.print(
            tr(
                "profiles_ui.sync_args_title",
                "\n[bold cyan]Sync vLLM CLI Arguments[/bold cyan]",
            )
        )
        console.print(
            tr(
                "profiles_ui.fetching_args",
                "[dim]Fetching latest arguments from vLLM GitHub...[/dim]\n",
            )
        )

        result = do_sync_args(dry_run=True, verbose=True)

        if result["new_count"] > 0 or result["removed_count"] > 0:
            console.print(
                tr(
                    "profiles_ui.apply_args_hint",
                    "\n[yellow]Run with --apply-args to update the schema, or use CLI:[/yellow]",
                )
            )
            console.print("  [cyan]vllm-cli recipes --sync-args --apply-args[/cyan]")
        else:
            console.print(
                tr(
                    "profiles_ui.cli_args_up_to_date",
                    "\n[green]✓ CLI args are up to date.[/green]",
                )
            )

        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")

    return "continue"


def manage_selected_profile() -> None:
    """
    Manage a selected profile - view details and perform actions.
    """
    config_manager = ConfigManager()
    all_profiles = config_manager.get_all_profiles()

    if not all_profiles:
        console.print(
            tr(
                "profiles_ui.no_profiles_available",
                "[yellow]No profiles available to view.[/yellow]",
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return

    # Prepare profile list with type indicators (plain text for menu)
    profile_choices = []
    profile_map = {}  # Map display name to actual profile name

    # Add built-in profiles
    for name in config_manager.default_profiles.keys():
        if config_manager.profile_manager.has_user_override(name):
            display_name = f"{name} (customized)"
        else:
            display_name = f"{name} (built-in)"
        profile_choices.append(display_name)
        profile_map[display_name] = name

    # Add user-only profiles
    for name in config_manager.user_profiles.keys():
        if name not in config_manager.default_profiles:
            display_name = f"{name} (user-created)"
            profile_choices.append(display_name)
            profile_map[display_name] = name

    # Select profile to manage
    console.print(
        tr(
            "profiles_ui.manage_profiles_title",
            "\n[bold cyan]Manage Profiles[/bold cyan]",
        )
    )
    selected = unified_prompt(
        "profile",
        tr("profiles_ui.select_profile_to_manage", "Select a profile to manage"),
        profile_choices,
        allow_back=True,
    )

    if not selected or selected == "BACK":
        return

    # Get actual profile name
    profile_name = profile_map.get(selected, selected)

    # Clear screen for better view
    console.print("\n" * 2)

    # Display profile header
    console.rule(
        tr(
            "profiles_ui.profile_header",
            "[bold cyan]Profile: {name}[/bold cyan]",
            name=profile_name,
        )
    )

    # Get profile data
    profile = config_manager.get_profile(profile_name)
    if not profile:
        console.print(
            tr(
                "profiles_ui.profile_not_found",
                "[red]Profile '{name}' not found.[/red]",
                name=profile_name,
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return

    # Display profile metadata
    console.print(
        tr("profiles_ui.profile_information", "\n[bold]Profile Information:[/bold]")
    )
    console.print(tr("profiles_ui.field_name", "  Name: {name}", name=profile_name))
    console.print(
        tr(
            "profiles_ui.field_description",
            "  Description: {description}",
            description=profile.get(
                "description", tr("profiles_ui.no_description", "No description")
            ),
        )
    )
    if profile.get("icon"):
        console.print(
            tr("profiles_ui.field_icon", "  Icon: {icon}", icon=profile.get("icon"))
        )

    # Determine and display profile type
    if config_manager.profile_manager.has_user_override(profile_name):
        console.print(
            tr(
                "profiles_ui.type_customized_builtin",
                "  Type: [yellow]Customized Built-in Profile[/yellow]",
            )
        )
    elif config_manager.profile_manager.is_user_profile(profile_name):
        if profile_name in config_manager.default_profiles:
            console.print(
                tr(
                    "profiles_ui.type_user_override",
                    "  Type: [yellow]User Override of Built-in[/yellow]",
                )
            )
        else:
            console.print(
                tr(
                    "profiles_ui.type_user_created",
                    "  Type: [cyan]User-Created Profile[/cyan]",
                )
            )
    else:
        console.print(
            tr(
                "profiles_ui.type_builtin_default",
                "  Type: [green]Built-in Default Profile[/green]",
            )
        )

    # Display current configuration
    console.print(
        tr(
            "profiles_ui.configuration_settings",
            "\n[bold]Configuration Settings:[/bold]",
        )
    )
    config = profile.get("config", {})

    if config:
        # Apply dynamic defaults for display
        config_with_defaults = config_manager.profile_manager.apply_dynamic_defaults(
            config
        )
        display_config(config_with_defaults)
    else:
        console.print(
            tr(
                "profiles_ui.no_custom_config",
                "[dim]  No custom configuration (uses all vLLM defaults)[/dim]",
            )
        )

    # Display environment variables
    environment = profile.get("environment", {})
    if environment:
        console.print(
            tr(
                "profiles_ui.environment_variables",
                "\n[bold]Environment Variables:[/bold]",
            )
        )
        for key, value in environment.items():
            if "KEY" in key.upper() or "TOKEN" in key.upper():
                console.print(
                    tr("profiles_ui.env_var_hidden", "  • {key}: <hidden>", key=key)
                )
            else:
                console.print(f"  • {key}: {value}")
    else:
        console.print(
            tr(
                "profiles_ui.environment_variables",
                "\n[bold]Environment Variables:[/bold]",
            )
        )
        console.print(
            tr(
                "profiles_ui.no_environment_configured",
                "[dim]  No environment variables configured[/dim]",
            )
        )

    # If this is a customized built-in profile, offer to show original
    if config_manager.profile_manager.has_user_override(profile_name):
        console.print(
            tr(
                "profiles_ui.customized_builtin_notice",
                "\n[yellow]This is a customized built-in profile.[/yellow]",
            )
        )

        show_original = (
            input(
                tr(
                    "profiles_ui.show_original_prompt",
                    "\nShow original default configuration? (y/N): ",
                )
            )
            .strip()
            .lower()
        )
        if show_original in ["y", "yes"]:
            original = config_manager.profile_manager.get_original_default_profile(
                profile_name
            )
            if original:
                console.print(
                    tr(
                        "profiles_ui.original_default_configuration",
                        "\n[bold]Original Default Configuration:[/bold]",
                    )
                )
                original_config = original.get("config", {})
                if original_config:
                    original_with_defaults = (
                        config_manager.profile_manager.apply_dynamic_defaults(
                            original_config
                        )
                    )
                    display_config(original_with_defaults)
                else:
                    console.print(
                        tr(
                            "profiles_ui.no_custom_config",
                            "[dim]  No custom configuration (uses all vLLM defaults)[/dim]",
                        )
                    )

    # Offer quick actions using unified navigation
    actions = [
        ("edit", tr("profiles_ui.action_edit", "Edit this profile")),
        ("export", tr("profiles_ui.action_export", "Export this profile")),
        ("rename", tr("profiles_ui.action_rename", "Rename this profile")),
    ]

    # Add delete option for user profiles (not for unmodified built-in)
    if config_manager.profile_manager.is_user_profile(profile_name):
        if profile_name not in config_manager.default_profiles:
            # Only user-created profiles can be deleted
            actions.append(
                ("delete", tr("profiles_ui.action_delete", "Delete this profile"))
            )

    # Add reset option if this is a customized built-in profile
    if config_manager.profile_manager.has_user_override(profile_name):
        actions.append(("reset", tr("profiles_ui.action_reset", "Reset to default")))

    # Use unified prompt for consistent navigation
    action = prompt_choice(
        "profile_detail_action",
        tr("profiles_ui.what_would_you_like_to_do", "What would you like to do?"),
        actions,
        allow_back=True,
    )

    if action == "edit":
        # Edit the profile
        edit_specific_profile(profile_name)
        # After editing, ask if they want to view the updated profile
        view_again = inquirer.confirm(
            tr("profiles_ui.view_updated_profile", "View the updated profile?"),
            default=True,
        )
        if view_again:
            # Recursive call to view the updated profile
            manage_selected_profile()
    elif action == "export":
        # Export the profile
        export_specific_profile(profile_name)
    elif action == "rename":
        new_name = inquirer.text(
            message=tr("profiles_ui.enter_new_profile_name", "Enter new profile name:"),
            default=profile_name,
        ).strip()
        if new_name and new_name != profile_name:
            if config_manager.profile_manager.rename_user_profile(
                profile_name, new_name
            ):
                console.print(
                    tr(
                        "profiles_ui.profile_renamed",
                        "[green]✓ Profile renamed to '{name}'.[/green]",
                        name=new_name,
                    )
                )
                input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            else:
                console.print(
                    tr(
                        "profiles_ui.rename_failed",
                        "[red]Failed to rename profile. Name may already exist.[/red]",
                    )
                )
                input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        elif not new_name:
            console.print(
                tr(
                    "profiles_ui.name_cannot_be_empty",
                    "[yellow]Name cannot be empty.[/red]",
                )
            )
            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
    elif action == "delete":
        # Delete the profile
        confirm = inquirer.confirm(
            tr(
                "profiles_ui.delete_profile_confirm",
                "Delete profile '{name}'? This cannot be undone.",
                name=profile_name,
            ),
            default=False,
        )
        if confirm:
            if config_manager.delete_user_profile(profile_name):
                console.print(
                    tr(
                        "profiles_ui.profile_deleted",
                        "[green]✓ Profile '{name}' deleted.[/green]",
                        name=profile_name,
                    )
                )
                input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
                return  # Exit to main menu after deletion
            else:
                console.print(
                    tr(
                        "profiles_ui.profile_delete_failed",
                        "[red]Failed to delete profile '{name}'.[/red]",
                        name=profile_name,
                    )
                )
                input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
    elif action == "reset":
        # Reset to default
        confirm = inquirer.confirm(
            tr(
                "profiles_ui.reset_to_default_confirm",
                "Reset '{name}' to its default configuration?",
                name=profile_name,
            ),
            default=False,
        )
        if confirm:
            if config_manager.profile_manager.reset_to_default(profile_name):
                console.print(
                    tr(
                        "profiles_ui.profile_reset",
                        "[green]✓ Profile '{name}' reset to default.[/green]",
                        name=profile_name,
                    )
                )
                # Ask if they want to view the reset profile
                view_again = inquirer.confirm(
                    tr("profiles_ui.view_reset_profile", "View the reset profile?"),
                    default=True,
                )
                if view_again:
                    manage_selected_profile()
                else:
                    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            else:
                console.print(
                    tr(
                        "profiles_ui.profile_reset_failed",
                        "[red]Failed to reset profile '{name}'.[/red]",
                        name=profile_name,
                    )
                )
                input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
    # If "BACK" or nothing selected, just return


def build_profile_configuration(
    existing_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Build a profile configuration through interactive prompts.
    This shared function is used by both profile creation workflows.

    Uses the comprehensive category-based configuration from custom_config module
    to ensure consistency across all profile creation flows.

    Args:
        existing_config: Optional existing configuration to use as defaults

    Returns:
        Configuration dictionary
    """
    # Use the comprehensive category-based configuration
    # Note: configure_by_categories automatically excludes the model field
    from .custom_config import configure_by_categories

    return configure_by_categories(existing_config)


# Note: The advanced configuration functions have been removed since we now use
# the comprehensive category-based configuration from custom_config module


def create_custom_profile() -> None:
    """
    Create a new custom profile using the comprehensive category-based configuration.
    This ensures consistency with the Custom Configuration flow from the main menu.
    """
    from .custom_config import configure_by_categories

    console.print(
        tr(
            "profiles_ui.create_custom_profile_title",
            "\n[bold cyan]Create Custom Profile[/bold cyan]",
        )
    )

    name = input(tr("profiles_ui.profile_name_prompt", "Profile name: ")).strip()
    if not name:
        console.print(
            tr(
                "profiles_ui.profile_name_required",
                "[yellow]Profile name required.[/yellow]",
            )
        )
        return

    description = input(
        tr(
            "profiles_ui.profile_description_prompt",
            "Profile description (optional): ",
        )
    ).strip()
    if not description:
        description = "Custom configuration"

    # Use the comprehensive category-based configuration directly
    # This ensures the exact same configuration experience as Custom Configuration flow
    config = configure_by_categories()

    # Extract environment variables if present (configure_by_categories returns them)
    environment = config.pop("environment", {})

    # Save profile
    config_manager = ConfigManager()
    profile_data = {
        "name": name,
        "description": description,
        "config": config,
        "environment": environment,
    }

    config_manager.save_user_profile(name, profile_data)
    console.print(
        tr(
            "profiles_ui.profile_created",
            "\n[green]✓ Profile '{name}' created successfully.[/green]",
            name=name,
        )
    )

    # Display the created profile
    config_with_defaults = config_manager.profile_manager.apply_dynamic_defaults(config)
    display_config(
        config_with_defaults,
        title=tr("profiles_ui.profile_configuration", "Profile Configuration"),
    )

    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")


def edit_specific_profile(profile_name: str) -> None:
    """
    Edit a specific profile directly.
    Helper function for quick editing from view details.
    """
    from .custom_config import (
        configure_advanced_hierarchical,
        configure_environment_variables,
    )

    config_manager = ConfigManager()

    # Get the profile
    profile = config_manager.get_profile(profile_name)
    if not profile:
        console.print(
            tr(
                "profiles_ui.profile_not_found",
                "[red]Profile '{name}' not found.[/red]",
                name=profile_name,
            )
        )
        return

    # Get configuration
    config = profile.get("config", {}).copy()
    environment = profile.get("environment", {}).copy()

    # If editing a built-in profile that hasn't been customized yet, inform the user
    if (
        profile_name in config_manager.default_profiles
        and not config_manager.profile_manager.has_user_override(profile_name)
    ):
        console.print(
            tr(
                "profiles_ui.editing_builtin_note",
                "\n[yellow]Note: Editing built-in profile '{name}'.[/yellow]",
                name=profile_name,
            )
        )
        console.print(
            tr(
                "profiles_ui.editing_builtin_changes_note",
                "[yellow]Your changes will create a customized version that "
                "overrides the default.[/yellow]",
            )
        )
        console.print(
            tr(
                "profiles_ui.editing_builtin_reset_note",
                "[dim]You can reset to default later if needed.[/dim]\n",
            )
        )

    console.print(
        tr(
            "profiles_ui.editing_profile_title",
            "\n[bold cyan]Editing Profile: {name}[/bold cyan]",
            name=profile_name,
        )
    )

    # Offer edit options
    edit_options = [
        (
            "modify_existing",
            tr("profiles_ui.edit_option_modify", "Modify existing values only"),
        ),
        (
            "full_config",
            tr(
                "profiles_ui.edit_option_full",
                "Full configuration (add/remove/modify)",
            ),
        ),
        (
            "edit_env",
            tr("profiles_ui.edit_option_env", "Edit environment variables"),
        ),
        ("cancel", tr("profiles_ui.edit_option_cancel", "Cancel")),
    ]

    edit_choice = prompt_choice(
        "edit_choice",
        tr("profiles_ui.what_to_edit", "What would you like to edit?"),
        edit_options,
        allow_back=False,
    )

    if edit_choice in ("cancel", None):
        return

    # Modify existing values only
    if edit_choice == "modify_existing":
        if not config:
            console.print(
                tr(
                    "profiles_ui.no_configuration_values",
                    "\n[yellow]No configuration values to modify.[/yellow]",
                )
            )
            console.print(
                tr(
                    "profiles_ui.use_full_configuration_hint",
                    "[dim]Use 'Full configuration' option to add arguments first.[/dim]",
                )
            )
            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        else:
            console.print(
                tr(
                    "profiles_ui.current_configuration",
                    "\n[bold]Current configuration:[/bold]",
                )
            )
            display_config(config)

            console.print(
                tr(
                    "profiles_ui.enter_new_values",
                    "\nEnter new values (press Enter to keep current):",
                )
            )
            console.print(
                tr(
                    "profiles_ui.type_aware_editing_hint",
                    "[dim]For type-aware editing, use 'Full configuration' "
                    "option[/dim]\n",
                )
            )

            for key in list(config.keys()):
                current_value = config[key]
                value_prompt = tr(
                    "profiles_ui.edit_value_prompt",
                    "{key} [{value}]: ",
                    key=key,
                    value=current_value,
                )
                bool_prompt = tr(
                    "profiles_ui.edit_bool_prompt",
                    "{key} [{value}] (true/false): ",
                    key=key,
                    value=current_value,
                )

                # Simple value editing - try to preserve type
                if isinstance(current_value, bool):
                    new_value = (
                        input(bool_prompt).strip().lower()
                    )
                    if new_value:
                        config[key] = new_value in ["true", "yes", "1", "y"]
                elif isinstance(current_value, int):
                    new_value = input(value_prompt).strip()
                    if new_value:
                        # Special handling for max_model_len to support "1M", "100K" formats
                        if key == "max_model_len":
                            parsed = parse_model_length(new_value)
                            if parsed is not None:
                                config[key] = parsed
                            else:
                                console.print(
                                    tr(
                                        "profiles_ui.invalid_format_for_key",
                                        "[red]Invalid format for {key}, keeping current "
                                        "value[/red]",
                                        key=key,
                                    )
                                )
                        else:
                            try:
                                config[key] = int(new_value)
                            except ValueError:
                                console.print(
                                    tr(
                                        "profiles_ui.invalid_int_for_key",
                                        "[red]Invalid integer value for {key}, keeping "
                                        "current value[/red]",
                                        key=key,
                                    )
                                )
                elif isinstance(current_value, float):
                    new_value = input(value_prompt).strip()
                    if new_value:
                        try:
                            config[key] = float(new_value)
                        except ValueError:
                            console.print(
                                tr(
                                    "profiles_ui.invalid_number_for_key",
                                    "[red]Invalid number value for {key}, keeping current "
                                    "value[/red]",
                                    key=key,
                                )
                            )
                else:  # string or other types
                    new_value = input(value_prompt).strip()
                    if new_value:
                        config[key] = new_value

    # Full configuration (add/remove/modify)
    elif edit_choice == "full_config":
        console.print(
            tr(
                "profiles_ui.full_configuration_mode",
                "\n[bold]Full Configuration Mode[/bold]",
            )
        )
        console.print(
            tr(
                "profiles_ui.full_configuration_hint",
                "[dim]Navigate categories to add, modify, or remove arguments[/dim]",
            )
        )

        # Use the existing hierarchical configuration system
        config = configure_advanced_hierarchical(config, config_manager)

        console.print(
            tr(
                "profiles_ui.configuration_updated",
                "\n[green]Configuration updated[/green]",
            )
        )

    # Edit environment variables
    elif edit_choice == "edit_env":
        console.print(
            tr(
                "profiles_ui.environment_variables",
                "\n[bold]Environment Variables:[/bold]",
            )
        )
        if environment:
            for key, value in environment.items():
                if "KEY" in key.upper() or "TOKEN" in key.upper():
                    console.print(
                        tr("profiles_ui.env_var_hidden", "  • {key}: <hidden>", key=key)
                    )
                else:
                    console.print(f"  • {key}: {value}")
        else:
            console.print(
                tr(
                    "profiles_ui.no_environment_variables",
                    "[dim]No environment variables configured[/dim]",
                )
            )

        console.print("")
        environment = configure_environment_variables(environment)

    # Prepare profile data
    profile_data = {
        "name": profile_name,
        "description": profile.get("description", "Custom profile"),
        "icon": profile.get("icon", ""),
        "config": config,
        "environment": environment,
    }

    # Save updated profile (this will create a user override if editing a built-in profile)
    config_manager.save_user_profile(profile_name, profile_data)

    if profile_name in config_manager.default_profiles:
        console.print(
            tr(
                "profiles_ui.profile_customized",
                "[green]Profile '{name}' customized successfully.[/green]",
                name=profile_name,
            )
        )
        console.print(
            tr(
                "profiles_ui.builtin_version_preserved",
                "[dim]The built-in version is preserved and can be restored later.[/dim]",
            )
        )
    else:
        console.print(
            tr(
                "profiles_ui.profile_updated",
                "[green]Profile '{name}' updated.[/green]",
                name=profile_name,
            )
        )

    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")


def export_specific_profile(profile_name: str) -> None:
    """
    Export a specific profile directly.
    Helper function for quick export from view details.
    """
    config_manager = ConfigManager()

    console.print(
        tr(
            "profiles_ui.export_profile_title",
            "\n[bold cyan]Export Profile: {name}[/bold cyan]",
            name=profile_name,
        )
    )

    # Get export path
    filepath = input(
        tr(
            "profiles_ui.export_path_prompt",
            "Enter export path (e.g., profile.json): ",
        )
    ).strip()
    if not filepath:
        console.print(
            tr(
                "profiles_ui.no_file_path_provided",
                "[yellow]No file path provided.[/yellow]",
            )
        )
        return

    from pathlib import Path

    file_path = Path(filepath)

    # Add .json extension if not present
    if not file_path.suffix:
        file_path = file_path.with_suffix(".json")

    if config_manager.export_profile(profile_name, file_path):
        console.print(
            tr(
                "profiles_ui.profile_exported",
                "[green]Profile exported to {path}[/green]",
                path=file_path,
            )
        )
    else:
        console.print(
            tr("profiles_ui.export_failed", "[red]Failed to export profile.[/red]")
        )

    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")


def import_profile() -> None:
    """Import a profile from a JSON file."""
    console.print(
        tr(
            "profiles_ui.import_profile_title",
            "\n[bold cyan]Import Profile[/bold cyan]",
        )
    )

    filepath = input(
        tr("profiles_ui.import_path_prompt", "Enter path to profile JSON file: ")
    ).strip()
    if not filepath:
        console.print(
            tr(
                "profiles_ui.no_file_path_provided",
                "[yellow]No file path provided.[/yellow]",
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return

    from pathlib import Path

    file_path = Path(filepath)

    if not file_path.exists():
        console.print(
            tr(
                "profiles_ui.file_not_found",
                "[red]File not found: {path}[/red]",
                path=filepath,
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return

    # Ask for a name for the imported profile
    name = input(
        tr(
            "profiles_ui.import_name_prompt",
            "Profile name (leave empty to use file name): ",
        )
    ).strip()

    config_manager = ConfigManager()
    if config_manager.import_profile(file_path, name if name else None):
        console.print(
            tr(
                "profiles_ui.profile_imported",
                "[green]Profile imported successfully.[/green]",
            )
        )
    else:
        console.print(
            tr("profiles_ui.import_failed", "[red]Failed to import profile.[/red]")
        )

    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
