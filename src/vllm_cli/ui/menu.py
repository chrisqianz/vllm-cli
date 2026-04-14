#!/usr/bin/env python3
"""
Main menu module for vLLM CLI.

Handles main menu display and navigation routing.
"""

import logging

from ..server import get_active_servers
from .navigation import unified_prompt

logger = logging.getLogger(__name__)


def show_main_menu(i18n_manager=None) -> str:
    """
    Display the main menu and return the selected action.

    Args:
        i18n_manager: Optional I18nManager for translations
    """
    # Import here to avoid circular dependencies
    from .model_manager import handle_model_management
    from .proxy.menu import (
        get_active_proxy,
        handle_multi_model_proxy,
        manage_running_proxy,
    )
    from .server_control import (
        handle_custom_config,
        handle_quick_serve,
        handle_serve_with_profile,
    )
    from .server_monitor import monitor_active_servers
    from .settings import handle_settings
    from .system_info import show_system_info

    # Helper function for translation
    def t(key: str, default: str = None, **kwargs) -> str:
        if i18n_manager:
            result = i18n_manager.t(key, **kwargs)
            return result if result != key else (default if default else key)
        # Fallback to English
        fallback_map = {
            "menu.main.title": "Main Menu",
            "menu.main.return_to_proxy": "Return to Proxy Monitoring",
            "menu.main.monitor_servers": "Monitor Active Servers ({count})",
            "menu.main.quick_serve": "Quick Serve",
            "menu.main.serve_with_profile": "Serve with Profile",
            "menu.main.serve_custom": "Serve with Custom Config",
            "menu.main.multi_model_proxy": "Multi-Model Proxy (Exp)",
            "menu.main.model_management": "Model Management",
            "menu.main.system_info": "System Information",
            "menu.main.settings": "Settings",
            "menu.main.quit": "Quit",
        }
        text = fallback_map.get(key, default if default else key)
        if kwargs:
            return text.format(**kwargs)
        return text

    # Check for active proxy and servers
    active_proxy_manager, active_proxy_config = get_active_proxy()
    active_servers = get_active_servers()

    menu_options = []
    option_map = {}  # Map translated text to action keys

    # Add proxy monitoring option if proxy is running
    if active_proxy_manager and active_proxy_config:
        try:
            # Check if proxy is actually still running
            if active_proxy_manager.proxy_process:
                option_text = t("menu.main.return_to_proxy")
                menu_options.append(option_text)
                option_map[option_text] = "return_to_proxy"
        except Exception:
            # If there's any error checking, ignore
            pass

    if active_servers:
        option_text = t("menu.main.monitor_servers", count=len(active_servers))
        menu_options.append(option_text)
        option_map[option_text] = "monitor_servers"

    # Add main menu options
    options_config = [
        ("menu.main.quick_serve", "quick_serve"),
        ("menu.main.serve_with_profile", "serve_with_profile"),
        ("menu.main.serve_custom", "serve_custom"),
        ("menu.main.multi_model_proxy", "multi_model_proxy"),
        ("menu.main.model_management", "model_management"),
        ("menu.main.system_info", "system_info"),
        ("menu.main.settings", "settings"),
        ("menu.main.quit", "quit"),
    ]

    for key, action in options_config:
        option_text = t(key)
        menu_options.append(option_text)
        option_map[option_text] = action

    action = unified_prompt(
        "action", t("menu.main.title"), menu_options, allow_back=False
    )

    if not action:
        return "quit"

    # Get the action key from the translated text
    action_key = option_map.get(action, "")

    if action_key == "quit":
        return "quit"

    # Handle menu selections
    if action_key == "return_to_proxy":
        # Return to proxy monitoring
        if active_proxy_manager and active_proxy_config:
            manage_running_proxy(active_proxy_manager, active_proxy_config)
        return "continue"
    elif action_key == "quick_serve":
        return handle_quick_serve(i18n_manager)
    elif action_key == "serve_with_profile":
        return handle_serve_with_profile(i18n_manager)
    elif action_key == "serve_custom":
        return handle_custom_config(i18n_manager)
    elif action_key == "multi_model_proxy":
        return handle_multi_model_proxy(i18n_manager)
    elif action_key == "model_management":
        return handle_model_management(i18n_manager)
    elif action_key == "system_info":
        return show_system_info(i18n_manager)
    elif action_key == "settings":
        return handle_settings(i18n_manager)
    elif action_key == "monitor_servers":
        return monitor_active_servers()

    return "continue"
