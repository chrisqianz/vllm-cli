#!/usr/bin/env python3
"""
Recipes sync UI for vLLM CLI.

Handles importing official vLLM recipes as profiles.
"""

import logging
from typing import Any, Dict, List, Optional

import inquirer

from ..config import ConfigManager
from .common import console

logger = logging.getLogger(__name__)


def sync_recipes() -> str:
    """
    Sync and import official vLLM recipes as profiles.

    Returns:
        Navigation string
    """
    from ..config.recipes_parser import RecipesParser

    console.print("\n[bold cyan]Sync Official vLLM Recipes[/bold cyan]")
    console.print("[dim]Import official vLLM recipe configurations as profiles[/dim]\n")

    parser = RecipesParser()

    try:
        recipes = parser.fetch_recipes_list()
    except Exception as e:
        console.print(f"[red]Failed to fetch recipes: {e}[/red]")
        input("\nPress Enter to continue...")
        return "continue"

    if not recipes:
        console.print("[yellow]No recipes found.[/yellow]")
        input("\nPress Enter to continue...")
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
        console.print("[yellow]No recipes available for import.[/yellow]")
        input("\nPress Enter to continue...")
        return "continue"

    # Let user select recipes
    questions = [
        inquirer.Checkbox(
            "recipes",
            message="Select recipes to import (press Space to select, Enter to confirm):",
            choices=choices,
        )
    ]

    answers = inquirer.prompt(questions)
    if not answers or not answers.get("recipes"):
        console.print("[dim]No recipes selected.[/dim]")
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
                console.print(f"[green]✓[/green] Imported: {profile_name}")
            else:
                failed.append((recipe_id, "Failed to save"))

        except Exception as e:
            logger.error(f"Failed to import {recipe_id}: {e}")
            failed.append((recipe_id, str(e)))

    # Summary
    console.print("\n[bold]Import Summary:[/bold]")
    console.print(f"  [green]Imported: {len(imported)}[/green]")
    if failed:
        console.print(f"  [red]Failed: {len(failed)}[/red]")

    input("\nPress Enter to continue...")
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

    console.print(
        f"\n[yellow]Warning: You are modifying an official profile: {profile_name}[/yellow]"
    )

    questions = [
        inquirer.List(
            "action",
            message="What would you like to do?",
            choices=[
                "Save as new profile",
                "Save to current profile",
                "Cancel changes",
            ],
        )
    ]

    answers = inquirer.prompt(questions)
    if not answers:
        return None

    action = answers["action"]

    if action == "Cancel changes":
        console.print("[dim]Changes discarded.[/dim]")
        return None

    if action == "Save to current profile":
        return profile_name

    # Save as new profile
    new_name_question = [
        inquirer.Text(
            "new_name",
            message="Enter new profile name:",
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
