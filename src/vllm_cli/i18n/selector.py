#!/usr/bin/env python3
"""
Language Selector for vLLM CLI.

Displays language selection interface on first launch.
"""
import logging
from typing import Optional

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)


class LanguageSelector:
    """
    Language selector for first-time setup.
    
    Displays a user-friendly interface for selecting the preferred language.
    """
    
    def __init__(self, i18n_manager):
        """
        Initialize the language selector.
        
        Args:
            i18n_manager: I18nManager instance
        """
        self.i18n_manager = i18n_manager
        self.console = Console()
    
    def should_show_selector(self) -> bool:
        """
        Determine if the language selector should be shown.
        
        Returns:
            True if this is first launch or language not set, False otherwise
        """
        return not self.i18n_manager.is_language_set()
    
    def show_language_selection(self) -> str:
        """
        Display the language selection interface.
        
        Returns:
            Selected language code ('en' or 'zh')
        """
        self.console.clear()
        
        # Create title
        title = Text()
        title.append("vLLM CLI - Language Selection\n", style="bold cyan")
        title.append("vLLM CLI - 语言选择", style="bold cyan")
        
        # Create language options table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Number", style="cyan", width=4)
        table.add_column("Language", style="yellow")
        
        languages = self.i18n_manager.get_available_languages()
        for idx, lang in enumerate(languages, 1):
            table.add_row(f"{idx}.", f"{lang['native']} ({lang['name']})")
        
        # Create prompt text
        prompt_text = Text()
        prompt_text.append("\nPlease select your preferred language\n", style="white")
        prompt_text.append("请选择您的首选语言\n", style="white")
        
        # Display the selection panel
        panel_content = Align.center(title)
        self.console.print(Panel(panel_content, border_style="bright_blue", padding=(1, 2)))
        self.console.print("")
        self.console.print(Align.center(table))
        self.console.print(Align.center(prompt_text))
        
        # Get user selection
        while True:
            try:
                choice_text = Text()
                choice_text.append("Enter your choice (1-2) / 请输入您的选择 (1-2): ", style="bold white")
                self.console.print(choice_text, end="")
                
                choice = input().strip()
                
                if choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(languages):
                        selected_lang = languages[choice_num - 1]["code"]
                        logger.info(f"User selected language: {selected_lang}")
                        return selected_lang
                
                # Invalid choice
                error_text = Text()
                error_text.append("\n❌ Invalid choice. Please enter 1 or 2.\n", style="red")
                error_text.append("❌ 无效选择。请输入 1 或 2。\n", style="red")
                self.console.print(error_text)
                
            except (KeyboardInterrupt, EOFError):
                # Default to English on interrupt
                logger.info("Language selection interrupted, defaulting to English")
                return "en"
            except Exception as e:
                logger.error(f"Error during language selection: {e}")
                self.console.print(f"\n[red]Error: {e}[/red]\n")
                # Default to English on error
                return "en"
    
    def show_language_change_confirmation(self, old_lang: str, new_lang: str) -> None:
        """
        Show confirmation message after language change.
        
        Args:
            old_lang: Previous language code
            new_lang: New language code
        """
        lang_names = {
            "en": "English",
            "zh": "中文"
        }
        
        old_name = lang_names.get(old_lang, old_lang)
        new_name = lang_names.get(new_lang, new_lang)
        
        message = Text()
        message.append(f"✓ Language changed from {old_name} to {new_name}\n", style="green")
        message.append(f"✓ 语言已从 {old_name} 更改为 {new_name}", style="green")
        
        self.console.print("")
        self.console.print(Panel(Align.center(message), border_style="green"))
        self.console.print("")
