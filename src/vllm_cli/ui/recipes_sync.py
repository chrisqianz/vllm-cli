#!/usr/bin/env python3
"""
Recipes sync UI for vLLM CLI.

Handles importing official vLLM recipes as profiles.
"""

import logging
from typing import Any, Dict, List, Optional

import inquirer

from ..config import ConfigManager
from ..i18n import tr
from .common import console
from .navigation import prompt_choice

logger = logging.getLogger(__name__)


def sync_recipes() -> str:
    """
    Sync and import official vLLM recipes as profiles.

    Returns:
        Navigation string
    """
    from ..config.recipes_parser import RecipesParser

    console.print(
        f"\n[bold cyan]{tr('recipes_ui.sync_title', 'Sync Official vLLM Recipes')}[/bold cyan]"
    )
    console.print(
        f"[dim]{tr('recipes_ui.sync_subtitle', 'Import official vLLM recipe configurations as profiles')}[/dim]\n"
    )

    parser = RecipesParser()

    try:
        recipes = parser.fetch_recipes_list()
    except Exception as e:
        console.print(
            tr(
                "recipes_ui.fetch_failed",
                "[red]Failed to fetch recipes: {error}[/red]",
                error=e,
            )
        )
        input("\n" + tr("common.press_enter", "Press Enter to continue..."))
        return "continue"

    if not recipes:
        console.print(
            f"[yellow]{tr('recipes_ui.no_recipes_found', 'No recipes found.')}[/yellow]"
        )
        input("\n" + tr("common.press_enter", "Press Enter to continue..."))
        return "continue"

    # Group recipes by category
    categories: Dict[str, List[Dict[str, Any]]] = {}
    for recipe in recipes:
        category = recipe.get("category", "Other")
        if category not in categories:
            categories[category] = []
        categories[category].append(recipe)

    # Build choices for selection
    choices = []
    recipe_map = {}

    for category, category_recipes in sorted(categories.items()):
        if category == "root":
            continue
        for recipe in category_recipes:
            display_name = f"{recipe['model_name']} ({recipe['hardware']})"
            recipe_id = recipe["id"]
            choices.append(f"[{category}] {display_name}")
            recipe_map[f"[{category}] {display_name}"] = recipe_id

    if not choices:
        console.print(
            f"[yellow]{tr('recipes_ui.no_recipes_available', 'No recipes available for import.')}[/yellow]"
        )
        input("\n" + tr("common.press_enter", "Press Enter to continue..."))
        return "continue"

    # Let user select recipes
    # Recipe labels come from the remote recipe catalogue (dynamic data) and are
    # matched by the same expression via recipe_map, so they stay untranslated.
    questions = [
        inquirer.Checkbox(
            "recipes",
            message=tr(
                "recipes_ui.select_recipes",
                "Select recipes to import (press Space to select, Enter to confirm):",
            ),
            choices=choices,
        )
    ]

    answers = inquirer.prompt(questions)
    if not answers or not answers.get("recipes"):
        console.print(
            f"[dim]{tr('recipes_ui.no_recipes_selected', 'No recipes selected.')}[/dim]"
        )
        return "continue"

    selected_ids = [recipe_map[r] for r in answers["recipes"]]

    # Import selected recipes
    config_manager = ConfigManager()
    imported = []
    failed = []

    for recipe_id in selected_ids:
        try:
            details = parser.fetch_profile_details(recipe_id)
            if not details or not details.get("commands"):
                failed.append((recipe_id, "No commands found"))
                continue

            # Use the first command
            command = details["commands"][0]
            profile = parser.command_to_profile(
                command=command,
                model_name=details["model_name"],
                hardware=details["hardware"],
            )

            profile_name = parser._generate_profile_name(
                details["model_name"], details["hardware"]
            )

            # Check if already exists
            if config_manager.profile_manager.profile_exists(profile_name):
                # Generate new version name
                existing = config_manager.profile_manager.get_profile(profile_name)
                if existing and existing.get("name", "").endswith("v1)"):
                    profile_name = profile_name.replace("official_", "official_v2_")

            # Save profile
            success = config_manager.profile_manager.save_user_profile(
                profile_name, profile
            )

            if success:
                imported.append(profile_name)
                console.print(
                    tr(
                        "recipes_ui.imported_profile",
                        "[green]✓[/green] Imported: {name}",
                        name=profile_name,
                    )
                )
            else:
                failed.append((recipe_id, "Failed to save"))

        except Exception as e:
            logger.error(f"Failed to import {recipe_id}: {e}")
            failed.append((recipe_id, str(e)))

    # Summary
    console.print(
        f"\n[bold]{tr('recipes_ui.import_summary', 'Import Summary:')}[/bold]"
    )
    console.print(
        tr(
            "recipes_ui.imported_count",
            "  [green]Imported: {count}[/green]",
            count=len(imported),
        )
    )
    if failed:
        console.print(
            tr(
                "recipes_ui.failed_count",
                "  [red]Failed: {count}[/red]",
                count=len(failed),
            )
        )

    input("\n" + tr("common.press_enter", "Press Enter to continue..."))
    return "continue"


def prompt_save_as_official(
    profile_name: str, current_config: Dict[str, Any]
) -> Optional[str]:
    """
    Prompt user to save modifications to an official profile as a new profile.

    Args:
        profile_name: Current profile name
        current_config: Current configuration

    Returns:
        New profile name if user chooses to save, None otherwise
    """
    from ..config.recipes_parser import is_official_profile

    if not is_official_profile(profile_name):
        return None

    warning = tr(
        "recipes_ui.modifying_official_warning",
        "Warning: You are modifying an official profile: {name}",
        name=profile_name,
    )
    console.print(f"\n[yellow]{warning}[/yellow]")

    action = prompt_choice(
        "recipe_action",
        tr("recipes_ui.what_would_you_like", "What would you like to do?"),
        [
            ("save_new", tr("recipes_ui.save_as_new_profile", "Save as new profile")),
            (
                "save_current",
                tr("recipes_ui.save_to_current_profile", "Save to current profile"),
            ),
            ("cancel_changes", tr("recipes_ui.cancel_changes", "Cancel changes")),
        ],
        allow_back=False,
    )

    if not action:
        return None

    if action == "cancel_changes":
        console.print(
            f"[dim]{tr('recipes_ui.changes_discarded', 'Changes discarded.')}[/dim]"
        )
        return None

    if action == "save_current":
        return profile_name

    # Save as new profile
    new_name_question = [
        inquirer.Text(
            "new_name",
            message=tr(
                "recipes_ui.enter_new_profile_name", "Enter new profile name:"
            ),
            default=profile_name.replace("official_", "custom_"),
        )
    ]

    new_answers = inquirer.prompt(new_name_question)
    if not new_answers:
        return None

    new_name = new_answers["new_name"].strip()
    if not new_name:
        return None

    return new_name
