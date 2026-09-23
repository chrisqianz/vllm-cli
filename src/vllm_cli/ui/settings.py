#!/usr/bin/env python3
"""
Settings module for vLLM CLI.

Handles configuration settings and application preferences.
"""

import logging

from rich.table import Table

from ..config import ConfigManager
from ..i18n import tr
from ..ui.progress_styles import get_progress_bar, list_available_styles
from .common import console
from .navigation import prompt_choice, unified_prompt
from .profiles import manage_profiles
from .shortcuts import manage_shortcuts

logger = logging.getLogger(__name__)


def configure_language(i18n_manager) -> str:
    """
    Configure language preference.

    Args:
        i18n_manager: I18nManager instance
    """
    if not i18n_manager:
        console.print(
            f"[red]{tr('settings_ui.language_config_unavailable', 'Language configuration not available')}[/red]"
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return "continue"

    from ..i18n import LanguageSelector

    console.print(
        f"\n[bold cyan]{tr('settings_ui.language_settings_title', 'Language Settings')}[/bold cyan]\n"
    )

    # Show current language
    current_lang = i18n_manager.get_current_language()
    lang_names = {"en": "English", "zh": "中文"}
    current_name = lang_names.get(current_lang, current_lang)

    console.print(
        f"{tr('settings_ui.current_language', 'Current language')}: [yellow]{current_name}[/yellow]\n"
    )

    # Get available languages
    languages = i18n_manager.get_available_languages()

    # Create language selection table
    from rich.table import Table

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(tr("settings_ui.col_number", "Number"), style="cyan", width=4)
    table.add_column(tr("menu.settings.language", "Language"), style="yellow")

    for idx, lang in enumerate(languages, 1):
        marker = " ✓" if lang["code"] == current_lang else ""
        table.add_row(f"{idx}.", f"{lang['native']} ({lang['name']}){marker}")

    console.print(table)
    console.print("")

    # Prompt for selection
    console.print(
        tr(
            "settings_ui.select_language_prompt",
            "Select a language (1-{count}), or press Enter to cancel",
            count=len(languages),
        )
    )
    choice = input().strip()

    if not choice:
        console.print(
            f"[yellow]{tr('settings_ui.language_change_cancelled', 'Language change cancelled')}[/yellow]"
        )
        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
        return "continue"

    if choice.isdigit():
        choice_num = int(choice)
        if 1 <= choice_num <= len(languages):
            new_lang = languages[choice_num - 1]["code"]

            if new_lang == current_lang:
                console.print(
                    f"[yellow]{tr('settings_ui.language_unchanged', 'Language unchanged')}[/yellow]"
                )
            else:
                # Change language
                old_lang = current_lang
                if i18n_manager.set_language(new_lang, persist=True):
                    # Show confirmation
                    selector = LanguageSelector(i18n_manager)
                    selector.show_language_change_confirmation(old_lang, new_lang)
                    console.print(
                        f"[green]{tr('settings_ui.language_preference_saved', 'Language preference saved')}[/green]"
                    )
                else:
                    console.print(
                        f"[red]{tr('settings_ui.language_change_failed', 'Failed to change language')}[/red]"
                    )

            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            return "continue"

    console.print(f"[red]{tr('common.invalid_choice', 'Invalid choice')}[/red]")
    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
    return "continue"


def handle_settings(i18n_manager=None) -> str:
    """
    Handle settings and configuration.

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
            "menu.settings.title": "Settings",
            "menu.settings.language": "Language",
            "menu.settings.shortcuts": "Manage Shortcuts",
            "menu.settings.profiles": "Manage Profiles",
            "menu.settings.proxy_configs": "Manage Proxy Configurations",
            "menu.settings.env_vars": "Universal Environment Variables",
            "menu.settings.model_dirs": "Model Directories",
            "menu.settings.server_defaults": "Server Defaults",
            "menu.settings.hf_token": "HuggingFace Token",
            "menu.settings.ui_prefs": "UI Preferences",
            "menu.settings.clear_cache": "Clear Cache",
            "messages.back": "← Back",
            "messages.success": "Operation completed successfully",
        }
        text = fallback_map.get(key, default if default else key)
        if kwargs:
            return text.format(**kwargs)
        return text

    while True:
        settings_options = []
        option_map = {}

        # Build menu options with translations
        options_config = [
            ("menu.settings.language", "language"),
            ("menu.settings.shortcuts", "shortcuts"),
            ("menu.settings.profiles", "profiles"),
            ("menu.settings.proxy_configs", "proxy_configs"),
            ("menu.settings.env_vars", "env_vars"),
            ("menu.settings.model_dirs", "model_dirs"),
            ("menu.settings.server_defaults", "server_defaults"),
            ("menu.settings.hf_token", "hf_token"),
            ("menu.settings.ui_prefs", "ui_prefs"),
            ("menu.settings.clear_cache", "clear_cache"),
        ]

        for key, action in options_config:
            option_text = t(key)
            settings_options.append(option_text)
            option_map[option_text] = action

        action = unified_prompt(
            "settings", t("menu.settings.title"), settings_options, allow_back=True
        )

        if action == t("messages.back") or action == "BACK" or not action:
            return "continue"

        # Get the action key from the translated text
        action_key = option_map.get(action, "")

        if action_key == "language":
            configure_language(i18n_manager)
        elif action_key == "shortcuts":
            manage_shortcuts(i18n_manager)
        elif action_key == "profiles":
            manage_profiles(i18n_manager)
        elif action_key == "proxy_configs":
            manage_proxy_configurations(i18n_manager)
        elif action_key == "env_vars":
            configure_universal_environment(i18n_manager)
        elif action_key == "model_dirs":
            manage_model_directories(i18n_manager)
        elif action_key == "server_defaults":
            configure_server_defaults(i18n_manager)
        elif action_key == "hf_token":
            configure_hf_token(i18n_manager)
        elif action_key == "ui_prefs":
            configure_ui_preferences(i18n_manager)
        elif action_key == "clear_cache":
            config_manager = ConfigManager()
            config_manager.clear_cache()
            console.print(f"[green]{t('messages.success')}[/green]")
            input(f"\n{t('common.press_enter')}")

    return "continue"


def manage_proxy_configurations(i18n_manager=None) -> str:
    """
    Manage saved proxy configurations.

    Allows viewing, editing, deleting, and exporting saved proxy configurations.
    This function is now part of Settings for consistency with profile and shortcut management.

    Args:
        i18n_manager: Optional I18nManager for translations
    """
    def t(key: str, default: str = None, **kwargs) -> str:
        if i18n_manager:
            result = i18n_manager.t(key, **kwargs)
            return result if result != key else (default if default else key)
        return default if default else key

    from ..proxy.config import ProxyConfigManager
    from .proxy.control import display_proxy_config, edit_proxy_config_interactive

    config_manager = ProxyConfigManager()

    while True:
        saved_configs = config_manager.list_saved_configs()

        console.print(f"\n[bold cyan]{t('settings.manage_proxy_title', 'Manage Proxy Configurations')}[/bold cyan]")

        if not saved_configs:
            console.print(f"\n[yellow]{t('settings.no_saved_configs', 'No saved proxy configurations found.')}[/yellow]")
            console.print(t("settings.create_from_main_menu", "Use 'Multi-Model Proxy' from the main menu to create one."))
            input(f"\n{t('common.press_enter', 'Press Enter to continue...')}")
            return "continue"

        # Display saved configurations
        from rich.table import Table

        table = Table(title=t("settings.saved_configs_title", "Saved Proxy Configurations"))
        table.add_column(t("settings.name", "Name"), style="cyan")
        table.add_column(t("settings.port", "Port"), style="magenta")
        table.add_column(t("settings.models", "Models"), style="yellow")
        table.add_column(t("settings.preview", "Preview"), style="dim")

        for name, info in saved_configs.items():
            preview = ", ".join(info["model_names"][:2])
            if len(info["model_names"]) > 2:
                preview += ", ..."
            table.add_row(name, str(info["port"]), str(info["models"]), preview)

        console.print(table)

        # Menu options
        options = [
            ("view", t("settings.view_details", "View configuration details")),
            ("edit", t("settings.edit_config", "Edit configuration")),
            ("delete", t("settings.delete_config", "Delete configuration")),
            ("export", t("settings.export_config", "Export configuration")),
        ]
        option_labels = dict(options)

        action = prompt_choice(
            "manage_proxy_action",
            t("settings.select_action", "Select action"),
            options,
            allow_back=True,
        )

        if action == "BACK" or not action:
            return "continue"

        # Select configuration for action
        config_names = list(saved_configs.keys())
        cancel_label = tr("messages.cancel", "Cancel")
        config_names.append(cancel_label)

        selected_name = unified_prompt(
            "select_proxy_config",
            tr(
                "settings_ui.select_config_for_action",
                "Select configuration for action: {action}",
                action=option_labels.get(action, action),
            ),
            config_names,
            allow_back=False,
        )

        if selected_name == cancel_label:
            continue

        if action == "view":
            # Load and display full configuration
            config = config_manager.load_named_config(selected_name)
            if config:
                console.print(
                    f"\n[bold]{tr('settings.configuration', 'Configuration')}: {selected_name}[/bold]"
                )
                display_proxy_config(config)
                input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")

        elif action == "edit":
            # Load configuration for editing
            config = config_manager.load_named_config(selected_name)
            if config:
                edited_config = edit_proxy_config_interactive(config)
                if edited_config:
                    # Ask if user wants to save changes
                    if (
                        prompt_choice(
                            "save_proxy_changes",
                            tr("settings.save_changes", "Save Changes?"),
                            [
                                ("save", tr("settings.yes_save", "Yes, Save Changes")),
                                (
                                    "discard",
                                    tr("settings.no_discard", "No, Discard Changes"),
                                ),
                            ],
                            allow_back=False,
                        )
                        == "save"
                    ):
                        config_manager.save_named_config(edited_config, selected_name)
                        updated = tr(
                            "settings_ui.config_updated_named",
                            "Configuration '{name}' updated",
                            name=selected_name,
                        )
                        console.print(f"[green]✓ {updated}[/green]")

        elif action == "delete":
            # Confirm deletion
            delete_warning = tr(
                "settings_ui.delete_config_warning",
                "Warning: This will delete configuration '{name}'",
                name=selected_name,
            )
            console.print(f"\n[yellow]{delete_warning}[/yellow]")
            if (
                prompt_choice(
                    "confirm_proxy_delete",
                    tr("messages.confirm", "Are you sure?"),
                    [
                        ("confirm", tr("settings.yes_delete", "Yes, Delete")),
                        ("cancel", tr("settings.no_cancel", "No, Cancel")),
                    ],
                    allow_back=False,
                )
                == "confirm"
            ):
                if config_manager.delete_named_config(selected_name):
                    deleted = tr(
                        "settings_ui.config_deleted_named",
                        "Configuration '{name}' deleted",
                        name=selected_name,
                    )
                    console.print(f"[green]✓ {deleted}[/green]")
                else:
                    console.print(
                        f"[red]{tr('settings_ui.delete_config_failed', 'Failed to delete configuration')}[/red]"
                    )

        elif action == "export":
            # Export to custom location
            config = config_manager.load_named_config(selected_name)
            if config:
                console.print(
                    f"\n{tr('settings_ui.export_path_header', 'Enter path to export configuration to:')}"
                )
                export_path = input(
                    tr("settings_ui.export_path_input", "Path (e.g., ./my-proxy.yaml): ")
                ).strip()
                if export_path:
                    try:
                        from pathlib import Path

                        export_file = Path(export_path)
                        config_manager.save_config(config, export_file)
                        exported = tr(
                            "settings_ui.config_exported_to",
                            "Configuration exported to {path}",
                            path=export_file,
                        )
                        console.print(f"[green]✓ {exported}[/green]")
                    except Exception as e:
                        failed = tr(
                            "settings_ui.export_failed_error",
                            "Export failed: {error}",
                            error=e,
                        )
                        console.print(f"[red]{failed}[/red]")
                input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")


def configure_universal_environment(i18n_manager=None) -> str:
    """
    Configure environment variables that apply universally to all servers.

    Args:
        i18n_manager: Optional I18nManager for translations
    """
    from .custom_config import configure_environment_variables

    config_manager = ConfigManager()

    console.print(
        f"\n[bold cyan]{tr('settings.universal_env_title', 'Universal Environment Variables')}[/bold cyan]"
    )
    console.print(
        f"\n{tr('settings_ui.universal_env_desc', 'These environment variables will be applied to ALL servers.')}"
    )
    console.print(
        f"[dim]{tr('settings_ui.profile_override_note', 'Profile-specific variables can override these settings.')}[/dim]\n"
    )

    # Get current universal environment
    universal_env = config_manager.config.get("universal_environment", {})

    if universal_env:
        console.print(
            f"[bold]{tr('settings.current_vars', 'Current Universal Environment Variables')}:[/bold]"
        )
        for key, value in universal_env.items():
            if "KEY" in key.upper() or "TOKEN" in key.upper():
                console.print(f"  • {key}: <hidden>")
            else:
                console.print(f"  • {key}: {value}")
        console.print("")
    else:
        console.print(
            f"[dim]{tr('settings_ui.no_universal_env', 'No universal environment variables configured.')}\n[/dim]"
        )

    # Configure environment variables using the same UI
    updated_env = configure_environment_variables(universal_env)

    # Save to config
    config_manager.config["universal_environment"] = updated_env
    config_manager._save_config()

    if updated_env:
        saved = tr(
            "settings_ui.universal_env_saved",
            "Saved {count} universal environment variable(s).",
            count=len(updated_env),
        )
        console.print(f"\n[green]✓ {saved}[/green]")
    else:
        console.print(
            f"\n[green]✓ {tr('settings.vars_cleared', 'Universal Environment Variables Cleared')}[/green]"
        )

    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
    return "continue"


def manage_model_directories(i18n_manager=None) -> str:
    """
    Manage model directories using integrated hf-model-tool API.

    Args:
        i18n_manager: Optional I18nManager for translations

    This function uses the hf-model-tool API directly to provide
    a seamless directory management experience within vLLM CLI.
    """
    from .model_directories import manage_model_directories as manage_dirs

    return manage_dirs()


def configure_hf_token(i18n_manager=None) -> str:
    """
    Configure HuggingFace authentication token for accessing gated/private models.

    Args:
        i18n_manager: Optional I18nManager for translations
    """
    import getpass

    config_manager = ConfigManager()

    console.print(
        f"\n[bold cyan]{tr('settings.hf_token_title', 'HuggingFace Token Configuration')}[/bold cyan]"
    )
    console.print(
        f"\n{tr('settings_ui.hf_token_desc', 'Configure your HuggingFace token for accessing gated or private models.')}"
    )
    console.print(
        f"[dim]{tr('settings_ui.hf_token_stored', 'Your token will be stored securely in your user config.')}[/dim]\n"
    )

    # Check if token already exists
    current_token = config_manager.config.get("hf_token", "")
    if current_token:
        console.print(
            f"[green]✓[/green] {tr('settings.token_configured', 'HuggingFace Token Is Currently Configured')}"
        )
        console.print(
            f"[dim]Token: {current_token[:8]}...{current_token[-4:] if len(current_token) > 12 else ''}[/dim]\n"
        )
    else:
        console.print(
            f"[yellow]⚠[/yellow] {tr('settings.no_token', 'No HuggingFace Token Configured')}\n"
        )

    # Options
    options = [
        ("set", tr("settings.set_update_token", "Set/Update Token")),
        ("remove", tr("settings.remove_token", "Remove Token")),
        ("test", tr("settings.test_token", "Test Token")),
        ("view", tr("settings.view_token_info", "View Token Info")),
    ]

    action = prompt_choice(
        "hf_token_action",
        tr("settings.select_action", "Select Action"),
        options,
        allow_back=True,
    )

    if not action or action == "BACK":
        return "continue"

    if action == "set":
        console.print(
            f"\n[cyan]{tr('settings.enter_token', 'Enter Your HuggingFace Token')}:[/cyan]"
        )
        console.print(
            f"[dim]{tr('settings.get_token_from', 'Get Your Token from')}: https://huggingface.co/settings/tokens[/dim]"
        )
        console.print(
            f"[dim]{tr('settings.token_hidden', 'The Token Will Be Hidden as You Type')}\n[/dim]"
        )

        # Use getpass for secure input
        token = getpass.getpass(f"{tr('settings_ui.token_input', 'Token')} ").strip()

        if token:
            # Validate token with HuggingFace API
            console.print(
                f"\n[cyan]{tr('settings.validating_token', 'Validating Token...')}[/cyan]"
            )

            from ..validation.token import validate_hf_token

            is_valid, user_info = validate_hf_token(token)

            if is_valid:
                # Save token to config
                config_manager.config["hf_token"] = token
                config_manager._save_config()

                console.print(
                    f"[green]✓ {tr('settings.token_validated', 'Token Validated and Saved Successfully')}[/green]"
                )
                unknown = tr("settings_ui.unknown", "Unknown")
                if user_info:
                    console.print(
                        f"[dim]{tr('settings.authenticated_as', 'Authenticated as')}: {user_info.get('name', unknown)}[/dim]"
                    )
                    if user_info.get("email"):
                        console.print(
                            f"[dim]{tr('settings.email', 'Email')}: {user_info.get('email')}[/dim]"
                        )
                console.print(
                    f"[dim]{tr('settings_ui.token_used_automatically', 'The token will be used automatically when accessing gated models.')}[/dim]"
                )
            else:
                console.print(
                    f"[red]✗ {tr('settings.token_invalid', 'Token Validation Failed')}[/red]"
                )
                console.print(
                    f"[dim]{tr('settings_ui.token_invalid_detail', 'The token appears to be invalid or expired.')}[/dim]"
                )
                console.print(
                    f"[dim]{tr('settings_ui.check_token_again', 'Please check your token and try again')}[/dim]"
                )

                # Ask if they want to save it anyway
                confirm = input(
                    f"\n{tr('settings.save_anyway', 'Save the Token Anyway?')} (y/N): "
                ).strip().lower()
                if confirm == "y":
                    config_manager.config["hf_token"] = token
                    config_manager._save_config()
                    console.print(
                        f"[yellow]{tr('settings_ui.token_saved_maybe', 'Token saved (but may not work properly)')}[/yellow]"
                    )
        else:
            console.print(
                f"[yellow]{tr('settings.no_token_provided', 'No Token Provided')}[/yellow]"
            )

    elif action == "remove":
        if current_token:
            confirm = (
                input(
                    f"{tr('settings.remove_confirm', 'Are You Sure You Want to Remove the Token?')} (y/N): "
                )
                .strip()
                .lower()
            )
            if confirm == "y":
                config_manager.config.pop("hf_token", None)
                config_manager._save_config()
                console.print(
                    f"[green]{tr('settings.token_removed', 'Token Removed Successfully')}[/green]"
                )
        else:
            console.print(
                f"[yellow]{tr('settings.no_token_to_remove', 'No Token to Remove')}[/yellow]"
            )

    elif action == "test":
        if not current_token:
            console.print(
                f"[red]{tr('settings_ui.no_token_configured', 'No token configured')}[/red]"
            )
        else:
            console.print(
                f"\n[cyan]{tr('settings.testing_token', 'Testing HuggingFace Token...')}[/cyan]"
            )
            try:
                import requests

                # Use the HuggingFace whoami-v2 API endpoint
                response = requests.get(
                    "https://huggingface.co/api/whoami-v2",
                    headers={"Authorization": f"Bearer {current_token}"},
                    timeout=10,
                )

                unknown = tr("settings_ui.unknown", "Unknown")
                if response.status_code == 200:
                    user_info = response.json()
                    console.print(
                        f"[green]✓ {tr('settings.token_valid', 'Token is Valid')}[/green]"
                    )
                    console.print(
                        f"[dim]{tr('settings.authenticated_as', 'Authenticated as')}: {user_info.get('name', unknown)}[/dim]"
                    )
                    console.print(
                        f"[dim]{tr('settings.email', 'Email')}: {user_info.get('email', tr('settings_ui.not_available', 'Not available'))}[/dim]"
                    )
                    if user_info.get("orgs"):
                        org_names = [
                            org.get("name", unknown)
                            for org in user_info.get("orgs", [])
                        ]
                        console.print(
                            f"[dim]{tr('settings.organizations', 'Organizations')}: {', '.join(org_names)}[/dim]"
                        )
                elif response.status_code == 401:
                    console.print(
                        f"[red]✗ {tr('settings.token_expired', 'Token is Invalid or Expired')}[/red]"
                    )
                    console.print(
                        f"[dim]{tr('settings_ui.check_token_again', 'Please check your token and try again')}[/dim]"
                    )
                else:
                    console.print(
                        f"[red]✗ {tr('settings.token_invalid', 'Token Validation Failed')} (HTTP {response.status_code})[/red]"
                    )
                    body = tr(
                        "settings_ui.response_body",
                        "Response: {response}",
                        response=response.text,
                    )
                    console.print(f"[dim]{body}[/dim]")
            except requests.exceptions.Timeout:
                console.print(
                    f"[red]{tr('settings.token_test_timeout', 'Token Test Timed Out')}[/red]"
                )
                console.print(
                    f"[dim]{tr('settings_ui.check_internet', 'Check your internet connection')}[/dim]"
                )
            except requests.exceptions.ConnectionError:
                console.print(
                    f"[red]{tr('settings.connection_failed', 'Failed to Connect to HuggingFace API')}[/red]"
                )
                console.print(
                    f"[dim]{tr('settings_ui.check_internet', 'Check your internet connection')}[/dim]"
                )
            except Exception as e:
                error_msg = tr(
                    "settings_ui.token_test_error",
                    "Error testing token: {error}",
                    error=e,
                )
                console.print(f"[red]{error_msg}[/red]")

    elif action == "view":
        if not current_token:
            console.print(
                f"[red]{tr('settings_ui.no_token_configured', 'No token configured')}[/red]"
            )
        else:
            console.print(
                f"\n[bold]{tr('settings.token_info', 'Token Information')}:[/bold]"
            )
            console.print(
                f"{tr('settings.token_prefix', 'Token Prefix')}: {current_token[:8]}..."
            )
            console.print(
                f"{tr('settings.token_suffix', 'Token Suffix')}: ...{current_token[-4:]}"
            )
            console.print(
                f"{tr('settings.token_length', 'Token Length')}: {len(current_token)} {tr('settings.token_length_chars', 'Characters')}"
            )
            info_hint = tr(
                "settings_ui.use_test_token_hint",
                "To get more information, use 'Test Token' option",
            )
            console.print(f"\n[dim]{info_hint}[/dim]")

    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
    return "continue"


def configure_server_defaults(i18n_manager=None) -> str:
    """
    Configure default server settings.

    Args:
        i18n_manager: Optional I18nManager for translations
    """
    config_manager = ConfigManager()
    defaults = config_manager.get_server_defaults()

    console.print(
        f"\n[bold cyan]{tr('settings.server_defaults_title', 'Server Defaults')}[/bold cyan]"
    )
    console.print(
        f"{tr('settings.configure_defaults', 'Configure Default Settings for All Servers')}:"
    )

    # Edit defaults
    defaults["default_port"] = int(
        input(
            f"{tr('settings.default_port', 'Default Port')} [{defaults.get('default_port', 8000)}]: "
        ).strip()
        or defaults.get("default_port", 8000)
    )
    defaults["auto_restart"] = input(
        f"{tr('settings.auto_restart', 'Auto-Restart on Failure')} (yes/no) [{defaults.get('auto_restart', False)}]: "
    ).strip().lower() in ["yes", "true", "1"]
    defaults["log_level"] = input(
        f"{tr('settings.log_level_options', 'Log Level (info/debug/warning/error)')} [{defaults.get('log_level', 'info')}]: "
    ).strip() or defaults.get("log_level", "info")

    # Add cleanup_on_exit setting
    current_cleanup = defaults.get("cleanup_on_exit", True)
    cleanup_str = "yes" if current_cleanup else "no"
    console.print(
        f"\n[yellow]{tr('settings_ui.server_cleanup_title', 'Server Cleanup on Exit')}:[/yellow]"
    )
    console.print(
        f"[dim]{tr('settings.cleanup_enabled', 'When Enabled, All Servers Will Be Stopped When CLI Exits')}[/dim]"
    )
    console.print(
        f"[dim]{tr('settings.cleanup_disabled', 'When Disabled, Servers Will Continue Running in Background')}[/dim]"
    )

    cleanup_input = (
        input(
            f"{tr('settings.stop_on_exit', 'Stop All Servers on CLI Exit?')} (yes/no) [{cleanup_str}]: "
        )
        .strip()
        .lower()
    )

    if cleanup_input in ["yes", "y", "true", "1"]:
        defaults["cleanup_on_exit"] = True
    elif cleanup_input in ["no", "n", "false", "0"]:
        defaults["cleanup_on_exit"] = False
        console.print(
            f"\n[yellow]⚠ {tr('settings_ui.warning_label', 'Warning')}:[/yellow]"
        )
        console.print(
            f"[dim]{tr('settings_ui.servers_keep_running_note', 'Servers will continue running after CLI exits.')}[/dim]"
        )
        status_hint = tr(
            "settings.use_status_command",
            "Use 'vllm-cli status' to View Active Servers",
        )
        stop_hint = tr(
            "settings.use_stop_command",
            "Use 'vllm-cli stop --port PORT' to Stop Servers Manually",
        )
        console.print(f"[dim]{status_hint}[/dim]")
        console.print(f"[dim]{stop_hint}[/dim]")
    # else keep current value

    config_manager.save_server_defaults(defaults)
    console.print(
        f"[green]{tr('settings.defaults_updated', 'Server Defaults Updated')}[/green]"
    )
    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")

    return "continue"


def configure_ui_preferences(i18n_manager=None) -> str:
    """
    Configure UI preferences including progress bar style.

    Args:
        i18n_manager: Optional I18nManager for translations
    """
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()

    console.print(
        f"\n[bold cyan]{tr('settings.ui_prefs_title', 'UI Preferences')}[/bold cyan]"
    )

    # Show current settings
    current_style = ui_prefs.get("progress_bar_style", "blocks")
    console.print(
        f"\n{tr('settings.current_style', 'Current Progress Bar Style')}: [yellow]{current_style}[/yellow]"
    )

    # Create preview table
    preview_table = Table(
        title=f"[bold]{tr('settings.style_preview', 'Progress Bar Style Preview')}[/bold]",
        show_header=True,
        header_style="bold cyan",
    )
    preview_table.add_column("#", style="cyan", width=3)
    preview_table.add_column(tr("settings.style", "Style"), style="yellow", width=12)
    preview_table.add_column("25%", style="white")
    preview_table.add_column("50%", style="white")
    preview_table.add_column("75%", style="white")
    preview_table.add_column("100%", style="white")

    styles = list_available_styles()
    for i, style_name in enumerate(styles, 1):
        # style_obj = PROGRESS_STYLES[style_name]  # Not used in the preview
        preview_table.add_row(
            str(i),
            style_name,
            get_progress_bar(25, style_name, 10),
            get_progress_bar(50, style_name, 10),
            get_progress_bar(75, style_name, 10),
            get_progress_bar(100, style_name, 10),
        )

    console.print(preview_table)

    # Select new style
    console.print(f"\n{tr('settings.select_style', 'Select a Progress Bar Style')}:")
    for i, style in enumerate(styles, 1):
        console.print(f"  {i}. {style}")

    choice = input(
        tr(
            "settings_ui.enter_style_choice",
            "Enter choice (1-{count}) [{current}]:",
            count=len(styles),
            current=styles.index(current_style) + 1,
        )
        + " "
    ).strip()

    if choice.isdigit() and 1 <= int(choice) <= len(styles):
        new_style = styles[int(choice) - 1]
        ui_prefs["progress_bar_style"] = new_style
        style_set = tr("settings.style_set", "Progress Bar Style Set to")
        console.print(f"\n[green]{style_set}: {new_style}[/green]")
    else:
        console.print(
            f"[yellow]{tr('settings.no_change', 'No Change Made to Progress Bar Style')}[/yellow]"
        )

    # Configure GPU monitoring
    console.print(
        f"\n[bold]{tr('settings.gpu_monitoring', 'GPU Monitoring Settings')}[/bold]"
    )
    show_gpu = ui_prefs.get("show_gpu_in_monitor", True)
    gpu_choice = (
        input(
            f"{tr('settings.show_gpu_panel', 'Show GPU Panel in Server Monitor?')} (yes/no) [{'yes' if show_gpu else 'no'}]: "
        )
        .strip()
        .lower()
    )

    if gpu_choice in ["yes", "y", "true", "1"]:
        ui_prefs["show_gpu_in_monitor"] = True
        console.print(
            f"[green]{tr('settings.gpu_panel_shown', 'GPU Panel Will Be Shown in Server Monitor')}[/green]"
        )
    elif gpu_choice in ["no", "n", "false", "0"]:
        ui_prefs["show_gpu_in_monitor"] = False
        console.print(
            f"[yellow]{tr('settings.gpu_panel_hidden', 'GPU Panel Will Be Hidden in Server Monitor')}[/yellow]"
        )

    # Configure log display settings
    console.print(f"\n[bold]{tr('settings.log_display', 'Log Display Settings')}[/bold]")

    # Startup log lines
    current_startup_lines = ui_prefs.get("log_lines_startup", 50)
    console.print(
        f"{tr('settings.startup_log_lines', 'Startup Log Lines')}: [yellow]{current_startup_lines}[/yellow]"
    )
    startup_choice = input(
        tr(
            "settings_ui.startup_lines_prompt",
            "Number of log lines during startup (5-50) [{current}]:",
            current=current_startup_lines,
        )
        + " "
    ).strip()

    if startup_choice.isdigit() and 5 <= int(startup_choice) <= 50:
        ui_prefs["log_lines_startup"] = int(startup_choice)
        lines_set = tr("settings.startup_lines_set", "Startup Log Lines Set to")
        console.print(f"[green]{lines_set}: {startup_choice}[/green]")
    elif startup_choice:
        console.print(
            f"[yellow]{tr('settings_ui.invalid_startup_lines', 'Invalid input. Startup log lines unchanged.')}[/yellow]"
        )

    # Monitor log lines
    current_monitor_lines = ui_prefs.get("log_lines_monitor", 50)
    console.print(
        f"{tr('settings.monitor_log_lines', 'Monitor Log Lines')}: [yellow]{current_monitor_lines}[/yellow]"
    )
    monitor_choice = input(
        tr(
            "settings_ui.monitor_lines_prompt",
            "Number of log lines in server monitor (10-100) [{current}]:",
            current=current_monitor_lines,
        )
        + " "
    ).strip()

    if monitor_choice.isdigit() and 10 <= int(monitor_choice) <= 100:
        ui_prefs["log_lines_monitor"] = int(monitor_choice)
        lines_set = tr("settings.monitor_lines_set", "Monitor Log Lines Set to")
        console.print(f"[green]{lines_set}: {monitor_choice}[/green]")
    elif monitor_choice:
        console.print(
            f"[yellow]{tr('settings_ui.invalid_monitor_lines', 'Invalid input. Monitor log lines unchanged.')}[/yellow]"
        )

    # Configure refresh rates
    console.print(
        f"\n[bold]{tr('settings.refresh_rate', 'Log Refresh Rate Settings')}[/bold]"
    )
    console.print(
        f"[dim]{tr('settings.higher_rate_note', 'Higher Refresh Rates Provide More Responsive Logs but Use More CPU')}[/dim]"
    )

    # Startup refresh rate
    current_startup_rate = ui_prefs.get("startup_refresh_rate", 4.0)
    console.print(
        f"{tr('settings.startup_refresh_rate', 'Startup Log Refresh Rate')}: [yellow]{current_startup_rate} Hz[/yellow]"
    )
    startup_rate_choice = input(
        f"{tr('settings.startup_refresh_rate', 'Startup Log Refresh Rate')} (1-10 Hz) [{current_startup_rate}]: "
    ).strip()

    if startup_rate_choice:
        try:
            rate = float(startup_rate_choice)
            if 1.0 <= rate <= 10.0:
                ui_prefs["startup_refresh_rate"] = rate
                rate_set = tr("settings.startup_rate_set", "Startup Refresh Rate Set to")
                console.print(f"[green]{rate_set}: {rate} Hz[/green]")
            else:
                console.print(
                    f"[yellow]{tr('settings_ui.invalid_startup_rate_range', 'Invalid range. Startup refresh rate unchanged.')}[/yellow]"
                )
        except ValueError:
            console.print(
                f"[yellow]{tr('settings_ui.invalid_startup_rate_input', 'Invalid input. Startup refresh rate unchanged.')}[/yellow]"
            )

    # Monitor refresh rate
    current_monitor_rate = ui_prefs.get("monitor_refresh_rate", 1.0)
    console.print(
        f"{tr('settings.monitor_refresh_rate', 'Monitor Log Refresh Rate')}: [yellow]{current_monitor_rate} Hz[/yellow]"
    )
    monitor_rate_choice = input(
        f"{tr('settings.monitor_refresh_rate', 'Monitor Log Refresh Rate')} (0.5-5 Hz) [{current_monitor_rate}]: "
    ).strip()

    if monitor_rate_choice:
        try:
            rate = float(monitor_rate_choice)
            if 0.5 <= rate <= 5.0:
                ui_prefs["monitor_refresh_rate"] = rate
                rate_set = tr("settings.monitor_rate_set", "Monitor Refresh Rate Set to")
                console.print(f"[green]{rate_set}: {rate} Hz[/green]")
            else:
                console.print(
                    f"[yellow]{tr('settings_ui.invalid_monitor_rate_range', 'Invalid range. Monitor refresh rate unchanged.')}[/yellow]"
                )
        except ValueError:
            console.print(
                f"[yellow]{tr('settings_ui.invalid_monitor_rate_input', 'Invalid input. Monitor refresh rate unchanged.')}[/yellow]"
            )

    # Save preferences
    config_manager.save_ui_preferences(ui_prefs)
    console.print(f"\n[green]{tr('settings.prefs_saved', 'UI Preferences Saved')}[/green]")
    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")

    return "continue"
