#!/usr/bin/env python3
"""
Unified navigation system for vLLM CLI.

Provides consistent menu navigation and user interaction.
"""
import logging
from typing import List, Optional

import inquirer
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


def unified_prompt(
    key: str, message: str, choices: List[str], allow_back: bool = True
) -> Optional[str]:
    """
    Display a unified prompt for user selection.

    Args:
        key: Unique key for this prompt
        message: Prompt message to display
        choices: List of choices
        allow_back: Whether to include a Back option

    Returns:
        Selected choice or None if cancelled
    """
    try:
        # Add navigation options
        prompt_choices = choices.copy()
        back_label = "← Back"
        try:
            from ..i18n import tr as _tr
            back_label = _tr("messages.back", "← Back")
        except Exception:
            pass
        if allow_back:
            prompt_choices.append(back_label)

        # Create inquirer prompt
        questions = [
            inquirer.List(
                key,
                message=message,
                choices=prompt_choices,
            )
        ]

        # Get user selection
        answers = inquirer.prompt(questions)

        if not answers:
            return None

        answer = answers[key]

        # Handle navigation
        if answer == back_label:
            return "BACK"

        return answer

    except KeyboardInterrupt:
        logger.info("User interrupted prompt")
        return None
    except Exception as e:
        logger.error(f"Error in unified prompt: {e}")
        return None


def prompt_choice(key, message, choices, allow_back=True):
    """Prompt with translated labels while returning a stable action key.

    Args:
        key: Unique key for this prompt
        message: Prompt message (already translated by caller, e.g. via tr)
        choices: List of ``(value, label)`` tuples; ``label`` is displayed,
            ``value`` is returned. Values must be stable ASCII identifiers.
        allow_back: Whether to include a Back option (returns "BACK")

    Returns:
        The selected ``value``, "BACK", or None if cancelled/interrupted.
    """
    labels = [label for _value, label in choices]
    selected = unified_prompt(key, message, labels, allow_back=allow_back)
    if selected is None or selected == "BACK":
        return selected
    for value, label in choices:
        if label == selected:
            return value
    # Displayed text did not match any label (should not happen): treat as
    # cancel rather than silently selecting the wrong action.
    logger.warning(f"prompt_choice[{key}]: unmatched selection {selected!r}")
    return None
