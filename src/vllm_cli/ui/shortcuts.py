#!/usr/bin/env python3
"""
Shortcut management UI for vLLM CLI.

Handles creation, editing, and deletion of shortcuts which are
saved combinations of model + profile.
"""

import logging
from pathlib import Path

import inquirer

from ..config import ConfigManager
from ..i18n import tr
from .common import console
from .display import display_config
from .model_manager import select_model
from .navigation import prompt_choice, unified_prompt
from .server_control import select_profile

logger = logging.getLogger(__name__)


def manage_shortcuts(i18n_manager=None) -> str:
    """
    Main shortcut management interface.

    Args:
        i18n_manager: Optional I18nManager for translations

    Returns:
        "continue" to return to main menu
    """

    # Helper function for translation
    def t(key: str, default: str = None, **kwargs) -> str:
        if i18n_manager:
            result = i18n_manager.t(key, **kwargs)
            return result if result != key else (default if default else key)
        return default if default else key

    config_manager = ConfigManager()

    while True:
        # Display existing shortcuts
        console.print(
            f"\n[bold cyan]{t('menu.shortcuts.title', 'Shortcut Management')}[/bold cyan]"
        )
        console.print(
            f"[dim]{t('menu.shortcuts.subtitle', 'Shortcuts are saved combinations of model + profile for quick launching')}[/dim]\n"
        )

        shortcuts = config_manager.list_shortcuts()

        if shortcuts:
            console.print(
                tr("shortcuts_ui.your_shortcuts", "[bold]Your Shortcuts:[/bold]")
            )
            for shortcut in shortcuts:
                name = shortcut["name"]
                model = shortcut["model"]
                profile = shortcut["profile"]
                desc = shortcut.get("description", "")

                # Truncate long model paths for display
                if len(model) > 40:
                    model_display = "..." + model[-37:]
                else:
                    model_display = model

                console.print(f"  → {name}")
                console.print(
                    tr(
                        "shortcuts_ui.shortcut_list_line",
                        "    Model: {model} | Profile: {profile}",
                        model=model_display,
                        profile=profile,
                    )
                )
                if desc:
                    console.print(f"    [dim]{desc}[/dim]")
        else:
            console.print(
                tr(
                    "shortcuts_ui.no_shortcuts_yet",
                    "[dim]No shortcuts configured yet[/dim]",
                )
            )

        # Action menu
        actions = [
            ("create", tr("shortcuts_ui.action_create", "Create New Shortcut")),
            (
                "view_edit",
                tr("shortcuts_ui.action_view_edit", "View/Edit Shortcut"),
            ),
            ("delete", tr("shortcuts_ui.action_delete", "Delete Shortcut")),
            ("import", tr("shortcuts_ui.action_import", "Import Shortcut")),
            ("export", tr("shortcuts_ui.action_export", "Export Shortcut")),
        ]

        action = prompt_choice(
            "shortcut_action",
            tr("shortcuts_ui.select_action", "What would you like to do?"),
            actions,
            allow_back=True,
        )

        if action == "BACK" or not action:
            return "continue"
        elif action == "create":
            create_shortcut()
        elif action == "view_edit":
            view_edit_shortcut()
        elif action == "delete":
            delete_shortcut()
        elif action == "import":
            import_shortcut()
        elif action == "export":
            export_shortcut()


def create_shortcut() -> None:
    """Create a new shortcut."""
    console.print(
        tr("shortcuts_ui.create_title", "\n[bold cyan]Create New Shortcut[/bold cyan]")
    )

    # Step 1: Select model
    console.print(
        tr(
            "shortcuts_ui.step1_select_model",
            "\n[bold]Step 1: Select Model[/bold]",
        )
    )
    model_selection = select_model()
    if not model_selection:
        console.print(
            tr(
                "shortcuts_ui.no_model_cancelled",
                "[yellow]No model selected. Shortcut creation cancelled.[/yellow]",
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return

    # Handle different model formats
    if isinstance(model_selection, dict):
        model = model_selection.get("model", model_selection)
        model_name = model_selection.get("name", model)
    else:
        model = model_selection
        model_name = model

    # Step 2: Select profile
    console.print(
        tr("shortcuts_ui.step2_select_profile", "\n[bold]Step 2: Select Profile[/bold]")
    )
    profile_name = select_profile()
    if not profile_name:
        console.print(
            tr(
                "shortcuts_ui.no_profile_cancelled",
                "[yellow]No profile selected. Shortcut creation cancelled.[/yellow]",
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return

    # Step 3: Name the shortcut
    console.print(
        tr("shortcuts_ui.step3_name", "\n[bold]Step 3: Name Your Shortcut[/bold]")
    )

    # Suggest a default name
    if "/" in str(model_name):
        model_short = str(model_name).split("/")[-1]
    else:
        model_short = str(model_name)
    suggested_name = f"{model_short}-{profile_name}"

    name = input(
        tr(
            "shortcuts_ui.name_prompt",
            "Shortcut name [{suggested}]: ",
            suggested=suggested_name,
        )
    ).strip()
    if not name:
        name = suggested_name

    # Step 4: Optional description
    description = input(
        tr("shortcuts_ui.description_prompt", "Description (optional): ")
    ).strip()
    if not description:
        description = f"{model_name} with {profile_name} profile"

    # Save the shortcut
    config_manager = ConfigManager()
    shortcut_data = {
        "model": model,
        "profile": profile_name,
        "description": description,
    }

    try:
        if config_manager.save_shortcut(name, shortcut_data):
            console.print(
                tr(
                    "shortcuts_ui.created",
                    "\n[green]✓ Shortcut '{name}' created successfully![/green]",
                    name=name,
                )
            )
            console.print(
                tr(
                    "shortcuts_ui.use_shortcut_from",
                    "\nYou can now use this shortcut from:",
                )
            )
            console.print(
                tr(
                    "shortcuts_ui.use_quick_serve",
                    "  • Quick Serve menu in interactive mode",
                )
            )
            console.print(
                f"{tr('shortcuts_ui.use_command_line', '  • Command line:')} "
                f'vllm-cli serve --shortcut "{name}"'
            )
        else:
            console.print(
                tr(
                    "shortcuts_ui.create_failed",
                    "[red]Failed to create shortcut '{name}'.[/red]",
                    name=name,
                )
            )
    except Exception as e:
        console.print(
            tr(
                "shortcuts_ui.create_error",
                "[red]Error creating shortcut: {error}[/red]",
                error=e,
            )
        )

    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")


def view_edit_shortcut() -> None:
    """View and optionally edit a shortcut."""
    config_manager = ConfigManager()
    shortcuts = config_manager.list_shortcuts()

    if not shortcuts:
        console.print(
            tr(
                "shortcuts_ui.none_to_view",
                "[yellow]No shortcuts available to view.[/yellow]",
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return

    # Select shortcut to view
    shortcut_names = [s["name"] for s in shortcuts]
    selected_name = unified_prompt(
        "select_shortcut",
        tr("shortcuts_ui.select_to_view", "Select a shortcut to view/edit"),
        shortcut_names,
        allow_back=True,
    )

    if selected_name == "BACK" or not selected_name:
        return

    # Get full shortcut data
    shortcut = config_manager.get_shortcut(selected_name)
    if not shortcut:
        console.print(
            tr(
                "shortcuts_ui.not_found",
                "[red]Shortcut '{name}' not found.[/red]",
                name=selected_name,
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return

    # Display shortcut details
    console.print(
        tr(
            "shortcuts_ui.detail_title",
            "\n[bold cyan]Shortcut: {name}[/bold cyan]",
            name=selected_name,
        )
    )
    console.print(tr("shortcuts_ui.configuration", "\n[bold]Configuration:[/bold]"))
    console.print(
        tr("shortcuts_ui.detail_model", "  Model: {model}", model=shortcut["model"])
    )
    console.print(
        tr(
            "shortcuts_ui.detail_profile",
            "  Profile: {profile}",
            profile=shortcut["profile"],
        )
    )
    console.print(
        tr(
            "shortcuts_ui.detail_description",
            "  Description: {description}",
            description=shortcut.get(
                "description",
                tr("shortcuts_ui.no_description", "No description"),
            ),
        )
    )

    if shortcut.get("created_at"):
        console.print(
            tr(
                "shortcuts_ui.detail_created",
                "  Created: {value}",
                value=shortcut["created_at"],
            )
        )
    if shortcut.get("last_used"):
        console.print(
            tr(
                "shortcuts_ui.detail_last_used",
                "  Last Used: {value}",
                value=shortcut["last_used"],
            )
        )

    # Show profile configuration
    profile = config_manager.get_profile(shortcut["profile"])
    if profile and profile.get("config"):
        display_config(
            profile["config"],
            title=(
                f"{tr('shortcuts_ui.profile_settings', 'Profile Settings')} "
                f"({shortcut['profile']})"
            ),
        )
        console.print()  # Add spacing after table to prevent display issues

    # Edit options
    edit_actions = [
        (
            "edit_description",
            tr("shortcuts_ui.action_edit_description", "Edit Description"),
        ),
        (
            "change_model",
            tr("shortcuts_ui.action_change_model", "Change Model"),
        ),
        (
            "change_profile",
            tr("shortcuts_ui.action_change_profile", "Change Profile"),
        ),
        ("rename", tr("shortcuts_ui.action_rename", "Rename Shortcut")),
    ]

    action = prompt_choice(
        "edit_shortcut_action",
        tr("shortcuts_ui.select_action", "What would you like to do?"),
        edit_actions,
        allow_back=True,
    )

    if action == "BACK" or not action:
        return

    modified = False

    if action == "edit_description":
        new_desc = input(
            tr(
                "shortcuts_ui.new_description_prompt",
                "New description [{current}]: ",
                current=shortcut.get("description", ""),
            )
        ).strip()
        if new_desc:
            shortcut["description"] = new_desc
            modified = True

    elif action == "change_model":
        console.print(
            tr(
                "shortcuts_ui.select_new_model",
                "\n[bold]Select new model:[/bold]",
            )
        )
        new_model = select_model()
        if new_model:
            if isinstance(new_model, dict):
                shortcut["model"] = new_model.get("model", new_model)
            else:
                shortcut["model"] = new_model
            modified = True

    elif action == "change_profile":
        console.print(
            tr("shortcuts_ui.select_new_profile", "\n[bold]Select new profile:[/bold]")
        )
        new_profile = select_profile()
        if new_profile:
            shortcut["profile"] = new_profile
            modified = True

    elif action == "rename":
        new_name = input(
            tr(
                "shortcuts_ui.new_name_prompt",
                "New name [{current}]: ",
                current=selected_name,
            )
        ).strip()
        if new_name and new_name != selected_name:
            # Check if new name already exists
            if config_manager.get_shortcut(new_name):
                console.print(
                    tr(
                        "shortcuts_ui.name_exists",
                        "[red]Shortcut '{name}' already exists.[/red]",
                        name=new_name,
                    )
                )
            else:
                # Rename by saving with new name and deleting old
                shortcut["name"] = new_name
                if config_manager.save_shortcut(new_name, shortcut):
                    config_manager.delete_shortcut(selected_name)
                    console.print(
                        tr(
                            "shortcuts_ui.renamed",
                            "[green]✓ Shortcut renamed to '{name}'[/green]",
                            name=new_name,
                        )
                    )
                    selected_name = new_name
                else:
                    console.print(
                        tr(
                            "shortcuts_ui.rename_failed",
                            "[red]Failed to rename shortcut.[/red]",
                        )
                    )

    # Save modifications if any
    if modified:
        if config_manager.save_shortcut(selected_name, shortcut):
            console.print(
                tr(
                    "shortcuts_ui.updated",
                    "[green]✓ Shortcut '{name}' updated successfully![/green]",
                    name=selected_name,
                )
            )
        else:
            console.print(
                tr(
                    "shortcuts_ui.update_failed",
                    "[red]Failed to update shortcut '{name}'.[/red]",
                    name=selected_name,
                )
            )

    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")


def delete_shortcut() -> None:
    """Delete a shortcut."""
    config_manager = ConfigManager()
    shortcuts = config_manager.list_shortcuts()

    if not shortcuts:
        console.print(
            tr(
                "shortcuts_ui.none_to_delete",
                "[yellow]No shortcuts available to delete.[/yellow]",
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return

    # Select shortcut to delete
    shortcut_names = [s["name"] for s in shortcuts]
    selected_name = unified_prompt(
        "delete_shortcut",
        tr("shortcuts_ui.select_to_delete", "Select a shortcut to delete"),
        shortcut_names,
        allow_back=True,
    )

    if selected_name == "BACK" or not selected_name:
        return

    # Confirm deletion
    confirm = inquirer.confirm(
        tr(
            "shortcuts_ui.delete_confirm",
            "Delete shortcut '{name}'? This cannot be undone.",
            name=selected_name,
        ),
        default=False,
    )

    if confirm:
        if config_manager.delete_shortcut(selected_name):
            console.print(
                tr(
                    "shortcuts_ui.deleted",
                    "[green]✓ Shortcut '{name}' deleted.[/green]",
                    name=selected_name,
                )
            )
        else:
            console.print(
                tr(
                    "shortcuts_ui.delete_failed",
                    "[red]Failed to delete shortcut '{name}'.[/red]",
                    name=selected_name,
                )
            )
    else:
        console.print(
            tr(
                "shortcuts_ui.deletion_cancelled",
                "[yellow]Deletion cancelled.[/yellow]",
            )
        )

    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")


def export_shortcut() -> None:
    """Export a shortcut to a file."""
    config_manager = ConfigManager()
    shortcuts = config_manager.list_shortcuts()

    if not shortcuts:
        console.print(
            tr(
                "shortcuts_ui.none_to_export",
                "[yellow]No shortcuts available to export.[/yellow]",
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return

    # Select shortcut to export
    shortcut_names = [s["name"] for s in shortcuts]
    selected_name = unified_prompt(
        "export_shortcut",
        tr("shortcuts_ui.select_to_export", "Select a shortcut to export"),
        shortcut_names,
        allow_back=True,
    )

    if selected_name == "BACK" or not selected_name:
        return

    # Get export path
    filepath = input(
        tr(
            "shortcuts_ui.export_path_prompt",
            "Export path [{suggested}]: ",
            suggested=f"shortcut_{selected_name}.json",
        )
    ).strip()
    if not filepath:
        filepath = f"shortcut_{selected_name}.json"

    file_path = Path(filepath)

    # Add .json extension if not present
    if not file_path.suffix:
        file_path = file_path.with_suffix(".json")

    if config_manager.shortcut_manager.export_shortcut(selected_name, file_path):
        console.print(
            tr(
                "shortcuts_ui.exported",
                "[green]✓ Shortcut exported to {path}[/green]",
                path=file_path,
            )
        )
    else:
        console.print(
            tr("shortcuts_ui.export_failed", "[red]Failed to export shortcut.[/red]")
        )

    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")


def import_shortcut() -> None:
    """Import a shortcut from a file."""
    console.print(
        tr("shortcuts_ui.import_title", "\n[bold cyan]Import Shortcut[/bold cyan]")
    )

    filepath = input(
        tr(
            "shortcuts_ui.import_path_prompt",
            "Enter path to shortcut JSON file: ",
        )
    ).strip()
    if not filepath:
        console.print(
            tr(
                "shortcuts_ui.no_path_provided",
                "[yellow]No file path provided.[/yellow]",
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return

    file_path = Path(filepath)

    if not file_path.exists():
        console.print(
            tr(
                "shortcuts_ui.file_not_found",
                "[red]File not found: {path}[/red]",
                path=filepath,
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return

    # Ask for a name for the imported shortcut
    name = input(
        tr(
            "shortcuts_ui.import_name_prompt",
            "Shortcut name (leave empty to use file name): ",
        )
    ).strip()

    config_manager = ConfigManager()
    try:
        if config_manager.shortcut_manager.import_shortcut(
            file_path, name if name else None
        ):
            console.print(
                tr(
                    "shortcuts_ui.imported",
                    "[green]✓ Shortcut imported successfully![/green]",
                )
            )
        else:
            console.print(
                tr(
                    "shortcuts_ui.import_failed",
                    "[red]Failed to import shortcut.[/red]",
                )
            )
    except Exception as e:
        console.print(
            tr(
                "shortcuts_ui.import_error",
                "[red]Error importing shortcut: {error}[/red]",
                error=e,
            )
        )

    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")


def serve_with_shortcut(shortcut_name: str) -> str:
    """
    Start a server using a shortcut configuration.

    Args:
        shortcut_name: Name of the shortcut to use

    Returns:
        Status string for menu navigation
    """
    from .server_control import start_server_with_config

    config_manager = ConfigManager()

    # Get the shortcut
    shortcut = config_manager.get_shortcut(shortcut_name)
    if not shortcut:
        console.print(
            tr(
                "shortcuts_ui.not_found",
                "[red]Shortcut '{name}' not found.[/red]",
                name=shortcut_name,
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return "continue"

    # Get the profile configuration
    profile = config_manager.get_profile(shortcut["profile"])
    if not profile:
        console.print(
            tr(
                "shortcuts_ui.profile_not_found",
                "[red]Profile '{name}' not found.[/red]",
                name=shortcut["profile"],
            )
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return "continue"

    # Build the configuration
    config = profile.get("config", {}).copy()
    config["model"] = shortcut["model"]

    # Apply any config overrides from the shortcut
    if "config_overrides" in shortcut:
        config.update(shortcut["config_overrides"])

    # Apply dynamic defaults
    config_with_defaults = config_manager.profile_manager.apply_dynamic_defaults(config)

    # Display configuration
    console.print(
        tr(
            "shortcuts_ui.starting_with",
            "\n[bold cyan]Starting with Shortcut: {name}[/bold cyan]",
            name=shortcut_name,
        )
    )
    console.print(
        tr("shortcuts_ui.line_model", "Model: {model}", model=shortcut["model"])
    )
    console.print(
        tr(
            "shortcuts_ui.line_profile",
            "Profile: {profile}",
            profile=shortcut["profile"],
        )
    )
    if shortcut.get("description"):
        console.print(
            tr(
                "shortcuts_ui.line_description",
                "Description: {description}",
                description=shortcut["description"],
            )
        )

    display_config(
        config_with_defaults,
        title=tr("shortcuts_ui.config_title", "Shortcut Configuration"),
    )

    # Confirm and start
    confirm = inquirer.confirm(
        tr(
            "shortcuts_ui.start_confirm",
            "Start server with this configuration?",
        ),
        default=True,
    )

    if confirm:
        # Update last used timestamp
        config_manager.shortcut_manager.update_last_used(shortcut_name)

        # Save as last config
        config_manager.save_last_config(config_with_defaults)

        return start_server_with_config(config_with_defaults)

    return "continue"
