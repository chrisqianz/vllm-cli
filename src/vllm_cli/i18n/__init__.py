#!/usr/bin/env python3
"""
Internationalization (i18n) module for vLLM CLI.

Provides multi-language support for the application interface.
"""

from .manager import I18nManager
from .selector import LanguageSelector

__all__ = ["I18nManager", "LanguageSelector"]
