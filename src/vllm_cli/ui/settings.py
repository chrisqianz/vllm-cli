#!/usr/bin/env python3
"""
Settings module for vLLM CLI.

Handles configuration settings and application preferences.
"""

import logging

from rich.table import Table

from ..config import ConfigManager
from ..ui.progress_styles import get_progress_bar, list_available_styles
from .common import console
from .navigation import unified_prompt
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
        console.print("[red]Language configuration not available[/red]")
        input("\nPress Enter to continue...")
        return "continue"

    from ..i18n import LanguageSelector

    console.print("\n[bold cyan]Language Settings / 语言设置[/bold cyan]\n")

    # Show current language
    current_lang = i18n_manager.get_current_language()
    lang_names = {"en": "English", "zh": "中文"}
    current_name = lang_names.get(current_lang, current_lang)

    console.print(f"Current language / 当前语言: [yellow]{current_name}[/yellow]\n")

    # Get available languages
    languages = i18n_manager.get_available_languages()

    # Create language selection table
    from rich.table import Table

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Number", style="cyan", width=4)
    table.add_column("Language", style="yellow")

    for idx, lang in enumerate(languages, 1):
        marker = " ✓" if lang["code"] == current_lang else ""
        table.add_row(f"{idx}.", f"{lang['native']} ({lang['name']}){marker}")

    console.print(table)
    console.print("")

    # Prompt for selection
    console.print(
        "Select a language / 选择语言 (1-2), or press Enter to cancel / 或按回车取消:"
    )
    choice = input().strip()

    if not choice:
        console.print("[yellow]Language change cancelled / 语言更改已取消[/yellow]")
        input("\nPress Enter to continue / 按回车继续...")
        return "continue"

    if choice.isdigit():
        choice_num = int(choice)
        if 1 <= choice_num <= len(languages):
            new_lang = languages[choice_num - 1]["code"]

            if new_lang == current_lang:
                console.print("[yellow]Language unchanged / 语言未更改[/yellow]")
            else:
                # Change language
                old_lang = current_lang
                if i18n_manager.set_language(new_lang, persist=True):
                    # Show confirmation
                    selector = LanguageSelector(i18n_manager)
                    selector.show_language_change_confirmation(old_lang, new_lang)
                    console.print(
                        f"[green]Language preference saved / 语言偏好已保存[/green]"
                    )
                else:
                    console.print("[red]Failed to change language / 语言更改失败[/red]")

            input("\nPress Enter to continue / 按回车继续...")
            return "continue"

    console.print("[red]Invalid choice / 无效选择[/red]")
    input("\nPress Enter to continue / 按回车继续...")
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
    from .proxy_control import display_proxy_config, edit_proxy_config_interactive

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
            t("settings.view_details", "View configuration details"),
            t("settings.edit_config", "Edit configuration"),
            t("settings.delete_config", "Delete configuration"),
            t("settings.export_config", "Export configuration"),
        ]

        action = unified_prompt(
            "manage_proxy_action", t("settings.select_action", "Select action"), options, allow_back=True
        )

        if action == "BACK":
            return "continue"

        # Select configuration for action
        config_names = list(saved_configs.keys())
        config_names.append("Cancel")

        selected_name = unified_prompt(
            "select_proxy_config",
            f"Select configuration to {action.lower()}",
            config_names,
            allow_back=False,
        )

        if selected_name == "Cancel":
            continue

        if action == "View configuration details":
            # Load and display full configuration
            config = config_manager.load_named_config(selected_name)
            if config:
                console.print(f"\n[bold]Configuration: {selected_name}[/bold]")
                display_proxy_config(config)
                input("\nPress Enter to continue...")

        elif action == "Edit configuration":
            # Load configuration for editing
            config = config_manager.load_named_config(selected_name)
            if config:
                edited_config = edit_proxy_config_interactive(config)
                if edited_config:
                    # Ask if user wants to save changes
                    if (
                        unified_prompt(
                            "save_proxy_changes",
                            "Save changes?",
                            ["Yes, save changes", "No, discard changes"],
                            allow_back=False,
                        )
                        == "Yes, save changes"
                    ):
                        config_manager.save_named_config(edited_config, selected_name)
                        console.print(
                            f"[green]✓ Configuration '{selected_name}' updated[/green]"
                        )

        elif action == "Delete configuration":
            # Confirm deletion
            console.print(
                f"\n[yellow]Warning: This will delete configuration '{selected_name}'[/yellow]"
            )
            if (
                unified_prompt(
                    "confirm_proxy_delete",
                    "Are you sure?",
                    ["Yes, delete", "No, cancel"],
                    allow_back=False,
                )
                == "Yes, delete"
            ):
                if config_manager.delete_named_config(selected_name):
                    console.print(
                        f"[green]✓ Configuration '{selected_name}' deleted[/green]"
                    )
                else:
                    console.print("[red]Failed to delete configuration[/red]")

        elif action == "Export configuration":
            # Export to custom location
            config = config_manager.load_named_config(selected_name)
            if config:
                console.print("\nEnter path to export configuration to:")
                export_path = input("Path (e.g., ./my-proxy.yaml): ").strip()
                if export_path:
                    try:
                        from pathlib import Path

                        export_file = Path(export_path)
                        config_manager.save_config(config, export_file)
                        console.print(
                            f"[green]✓ Configuration exported to {export_file}[/green]"
                        )
                    except Exception as e:
                        console.print(f"[red]Export failed: {e}[/red]")
                input("\nPress Enter to continue...")


def configure_universal_environment(i18n_manager=None) -> str:
    """
    Configure environment variables that apply universally to all servers.

    Args:
        i18n_manager: Optional I18nManager for translations
    """
    from .custom_config import configure_environment_variables

    config_manager = ConfigManager()

    console.print("\n[bold cyan]Universal Environment Variables[/bold cyan]")
    console.print("\nThese environment variables will be applied to ALL servers.")
    console.print(
        "[dim]Profile-specific variables can override these settings.[/dim]\n"
    )

    # Get current universal environment
    universal_env = config_manager.config.get("universal_environment", {})

    if universal_env:
        console.print("[bold]Current Universal Environment Variables:[/bold]")
        for key, value in universal_env.items():
            if "KEY" in key.upper() or "TOKEN" in key.upper():
                console.print(f"  • {key}: <hidden>")
            else:
                console.print(f"  • {key}: {value}")
        console.print("")
    else:
        console.print("[dim]No universal environment variables configured.\n[/dim]")

    # Configure environment variables using the same UI
    updated_env = configure_environment_variables(universal_env)

    # Save to config
    config_manager.config["universal_environment"] = updated_env
    config_manager._save_config()

    if updated_env:
        console.print(
            f"\n[green]✓ Saved {len(updated_env)} universal environment variable(s).[/green]"
        )
    else:
        console.print("\n[green]✓ Universal environment variables cleared.[/green]")

    input("\nPress Enter to continue...")
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

    console.print("\n[bold cyan]HuggingFace Token Configuration[/bold cyan]")
    console.print(
        "\nConfigure your HuggingFace token for accessing gated or private models."
    )
    console.print(
        "[dim]Your token will be stored securely in your user config.[/dim]\n"
    )

    # Check if token already exists
    current_token = config_manager.config.get("hf_token", "")
    if current_token:
        console.print("[green]✓[/green] HuggingFace token is currently configured")
        console.print(
            f"[dim]Token: {current_token[:8]}...{current_token[-4:] if len(current_token) > 12 else ''}[/dim]\n"
        )
    else:
        console.print("[yellow]⚠[/yellow] No HuggingFace token configured\n")

    # Options
    options = [
        "Set/Update Token",
        "Remove Token",
        "Test Token",
        "View Token Info",
    ]

    action = unified_prompt(
        "hf_token_action", "Select Action", options, allow_back=True
    )

    if not action or action == "BACK":
        return "continue"

    if action == "Set/Update Token":
        console.print("\n[cyan]Enter your HuggingFace token:[/cyan]")
        console.print(
            "[dim]Get your token from: https://huggingface.co/settings/tokens[/dim]"
        )
        console.print("[dim]The token will be hidden as you type.[/dim]\n")

        # Use getpass for secure input
        token = getpass.getpass("Token: ").strip()

        if token:
            # Validate token with HuggingFace API
            console.print("\n[cyan]Validating token...[/cyan]")

            from ..validation.token import validate_hf_token

            is_valid, user_info = validate_hf_token(token)

            if is_valid:
                # Save token to config
                config_manager.config["hf_token"] = token
                config_manager._save_config()

                console.print("[green]✓ Token validated and saved successfully[/green]")
                if user_info:
                    console.print(
                        f"[dim]Authenticated as: {user_info.get('name', 'Unknown')}[/dim]"
                    )
                    if user_info.get("email"):
                        console.print(f"[dim]Email: {user_info.get('email')}[/dim]")
                console.print(
                    "[dim]The token will be used automatically when accessing gated models.[/dim]"
                )
            else:
                console.print("[red]✗ Token validation failed[/red]")
                console.print("[dim]The token appears to be invalid or expired.[/dim]")
                console.print("[dim]Please check your token and try again.[/dim]")

                # Ask if they want to save it anyway
                confirm = input("\nSave the token anyway? (y/N): ").strip().lower()
                if confirm == "y":
                    config_manager.config["hf_token"] = token
                    config_manager._save_config()
                    console.print(
                        "[yellow]Token saved (but may not work properly)[/yellow]"
                    )
        else:
            console.print("[yellow]No token provided[/yellow]")

    elif action == "Remove Token":
        if current_token:
            confirm = (
                input("Are you sure you want to remove the token? (y/N): ")
                .strip()
                .lower()
            )
            if confirm == "y":
                config_manager.config.pop("hf_token", None)
                config_manager._save_config()
                console.print("[green]Token removed successfully[/green]")
        else:
            console.print("[yellow]No token to remove[/yellow]")

    elif action == "Test Token":
        if not current_token:
            console.print("[red]No token configured[/red]")
        else:
            console.print("\n[cyan]Testing HuggingFace token...[/cyan]")
            try:
                import requests

                # Use the HuggingFace whoami-v2 API endpoint
                response = requests.get(
                    "https://huggingface.co/api/whoami-v2",
                    headers={"Authorization": f"Bearer {current_token}"},
                    timeout=10,
                )

                if response.status_code == 200:
                    user_info = response.json()
                    console.print("[green]✓ Token is valid[/green]")
                    console.print(
                        f"[dim]Authenticated as: {user_info.get('name', 'Unknown')}[/dim]"
                    )
                    console.print(
                        f"[dim]Email: {user_info.get('email', 'Not available')}[/dim]"
                    )
                    if user_info.get("orgs"):
                        org_names = [
                            org.get("name", "Unknown")
                            for org in user_info.get("orgs", [])
                        ]
                        console.print(
                            f"[dim]Organizations: {', '.join(org_names)}[/dim]"
                        )
                elif response.status_code == 401:
                    console.print("[red]✗ Token is invalid or expired[/red]")
                    console.print("[dim]Please check your token and try again[/dim]")
                else:
                    console.print(
                        f"[red]✗ Token validation failed (HTTP {response.status_code})[/red]"
                    )
                    console.print(f"[dim]Response: {response.text}[/dim]")
            except requests.exceptions.Timeout:
                console.print("[red]Token test timed out[/red]")
                console.print("[dim]Check your internet connection[/dim]")
            except requests.exceptions.ConnectionError:
                console.print("[red]Failed to connect to HuggingFace API[/red]")
                console.print("[dim]Check your internet connection[/dim]")
            except Exception as e:
                console.print(f"[red]Error testing token: {e}[/red]")

    elif action == "View Token Info":
        if not current_token:
            console.print("[red]No token configured[/red]")
        else:
            console.print("\n[bold]Token Information:[/bold]")
            console.print(f"Token prefix: {current_token[:8]}...")
            console.print(f"Token suffix: ...{current_token[-4:]}")
            console.print(f"Token length: {len(current_token)} characters")
            console.print(
                "\n[dim]To get more information, use 'Test Token' option[/dim]"
            )

    input("\nPress Enter to continue...")
    return "continue"


def configure_server_defaults(i18n_manager=None) -> str:
    """
    Configure default server settings.

    Args:
        i18n_manager: Optional I18nManager for translations
    """
    config_manager = ConfigManager()
    defaults = config_manager.get_server_defaults()

    console.print("\n[bold cyan]Server Defaults[/bold cyan]")
    console.print("Configure default settings for all servers:")

    # Edit defaults
    defaults["default_port"] = int(
        input(f"Default port [{defaults.get('default_port', 8000)}]: ").strip()
        or defaults.get("default_port", 8000)
    )
    defaults["auto_restart"] = input(
        f"Auto-restart on failure (yes/no) [{defaults.get('auto_restart', False)}]: "
    ).strip().lower() in ["yes", "true", "1"]
    defaults["log_level"] = input(
        f"Log level (info/debug/warning/error) [{defaults.get('log_level', 'info')}]: "
    ).strip() or defaults.get("log_level", "info")

    # Add cleanup_on_exit setting
    current_cleanup = defaults.get("cleanup_on_exit", True)
    cleanup_str = "yes" if current_cleanup else "no"
    console.print("\n[yellow]Server Cleanup on Exit:[/yellow]")
    console.print(
        "[dim]When enabled, all servers will be stopped when the CLI exits.[/dim]"
    )
    console.print(
        "[dim]When disabled, servers will continue running in the background.[/dim]"
    )

    cleanup_input = (
        input(f"Stop all servers on CLI exit (yes/no) [{cleanup_str}]: ")
        .strip()
        .lower()
    )

    if cleanup_input in ["yes", "y", "true", "1"]:
        defaults["cleanup_on_exit"] = True
    elif cleanup_input in ["no", "n", "false", "0"]:
        defaults["cleanup_on_exit"] = False
        console.print("\n[yellow]⚠ Warning:[/yellow]")
        console.print("[dim]Servers will continue running after CLI exits.[/dim]")
        console.print("[dim]Use 'vllm-cli status' to view active servers.[/dim]")
        console.print(
            "[dim]Use 'vllm-cli stop --port PORT' to stop servers manually.[/dim]"
        )
    # else keep current value

    config_manager.save_server_defaults(defaults)
    console.print("[green]Server defaults updated.[/green]")
    input("\nPress Enter to continue...")

    return "continue"


def configure_ui_preferences(i18n_manager=None) -> str:
    """
    Configure UI preferences including progress bar style.

    Args:
        i18n_manager: Optional I18nManager for translations
    """
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()

    console.print("\n[bold cyan]UI Preferences[/bold cyan]")

    # Show current settings
    current_style = ui_prefs.get("progress_bar_style", "blocks")
    console.print(f"\nCurrent progress bar style: [yellow]{current_style}[/yellow]")

    # Create preview table
    preview_table = Table(
        title="[bold]Progress Bar Style Preview[/bold]",
        show_header=True,
        header_style="bold cyan",
    )
    preview_table.add_column("#", style="cyan", width=3)
    preview_table.add_column("Style", style="yellow", width=12)
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
    console.print("\nSelect a progress bar style:")
    for i, style in enumerate(styles, 1):
        console.print(f"  {i}. {style}")

    choice = input(
        f"\nEnter choice (1-{len(styles)}) [{styles.index(current_style) + 1}]: "
    ).strip()

    if choice.isdigit() and 1 <= int(choice) <= len(styles):
        new_style = styles[int(choice) - 1]
        ui_prefs["progress_bar_style"] = new_style
        console.print(f"\n[green]Progress bar style set to: {new_style}[/green]")
    else:
        console.print("[yellow]No change made to progress bar style[/yellow]")

    # Configure GPU monitoring
    console.print("\n[bold]GPU Monitoring Settings[/bold]")
    show_gpu = ui_prefs.get("show_gpu_in_monitor", True)
    gpu_choice = (
        input(
            f"Show GPU panel in server monitor? (yes/no) [{'yes' if show_gpu else 'no'}]: "
        )
        .strip()
        .lower()
    )

    if gpu_choice in ["yes", "y", "true", "1"]:
        ui_prefs["show_gpu_in_monitor"] = True
        console.print("[green]GPU panel will be shown in server monitor[/green]")
    elif gpu_choice in ["no", "n", "false", "0"]:
        ui_prefs["show_gpu_in_monitor"] = False
        console.print("[yellow]GPU panel will be hidden in server monitor[/yellow]")

    # Configure log display settings
    console.print("\n[bold]Log Display Settings[/bold]")

    # Startup log lines
    current_startup_lines = ui_prefs.get("log_lines_startup", 50)
    console.print(
        f"Current startup log lines: [yellow]{current_startup_lines}[/yellow]"
    )
    startup_choice = input(
        f"Number of log lines during startup (5-50) [{current_startup_lines}]: "
    ).strip()

    if startup_choice.isdigit() and 5 <= int(startup_choice) <= 50:
        ui_prefs["log_lines_startup"] = int(startup_choice)
        console.print(f"[green]Startup log lines set to: {startup_choice}[/green]")
    elif startup_choice:
        console.print("[yellow]Invalid input. Startup log lines unchanged.[/yellow]")

    # Monitor log lines
    current_monitor_lines = ui_prefs.get("log_lines_monitor", 50)
    console.print(
        f"Current monitor log lines: [yellow]{current_monitor_lines}[/yellow]"
    )
    monitor_choice = input(
        f"Number of log lines in server monitor (10-100) [{current_monitor_lines}]: "
    ).strip()

    if monitor_choice.isdigit() and 10 <= int(monitor_choice) <= 100:
        ui_prefs["log_lines_monitor"] = int(monitor_choice)
        console.print(f"[green]Monitor log lines set to: {monitor_choice}[/green]")
    elif monitor_choice:
        console.print("[yellow]Invalid input. Monitor log lines unchanged.[/yellow]")

    # Configure refresh rates
    console.print("\n[bold]Log Refresh Rate Settings[/bold]")
    console.print(
        "[dim]Higher refresh rates provide more responsive logs but use more CPU[/dim]"
    )

    # Startup refresh rate
    current_startup_rate = ui_prefs.get("startup_refresh_rate", 4.0)
    console.print(
        f"Current startup refresh rate: [yellow]{current_startup_rate} Hz[/yellow]"
    )
    startup_rate_choice = input(
        f"Startup log refresh rate (1-10 Hz) [{current_startup_rate}]: "
    ).strip()

    if startup_rate_choice:
        try:
            rate = float(startup_rate_choice)
            if 1.0 <= rate <= 10.0:
                ui_prefs["startup_refresh_rate"] = rate
                console.print(f"[green]Startup refresh rate set to: {rate} Hz[/green]")
            else:
                console.print(
                    "[yellow]Invalid range. Startup refresh rate unchanged.[/yellow]"
                )
        except ValueError:
            console.print(
                "[yellow]Invalid input. Startup refresh rate unchanged.[/yellow]"
            )

    # Monitor refresh rate
    current_monitor_rate = ui_prefs.get("monitor_refresh_rate", 1.0)
    console.print(
        f"Current monitor refresh rate: [yellow]{current_monitor_rate} Hz[/yellow]"
    )
    monitor_rate_choice = input(
        f"Monitor log refresh rate (0.5-5 Hz) [{current_monitor_rate}]: "
    ).strip()

    if monitor_rate_choice:
        try:
            rate = float(monitor_rate_choice)
            if 0.5 <= rate <= 5.0:
                ui_prefs["monitor_refresh_rate"] = rate
                console.print(f"[green]Monitor refresh rate set to: {rate} Hz[/green]")
            else:
                console.print(
                    "[yellow]Invalid range. Monitor refresh rate unchanged.[/yellow]"
                )
        except ValueError:
            console.print(
                "[yellow]Invalid input. Monitor refresh rate unchanged.[/yellow]"
            )

    # Save preferences
    config_manager.save_ui_preferences(ui_prefs)
    console.print("\n[green]UI preferences saved.[/green]")
    input("\nPress Enter to continue...")

    return "continue"
