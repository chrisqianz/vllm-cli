#!/usr/bin/env python3
"""
I18n Manager for vLLM CLI.

Manages language switching, translation loading, and text translation.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class I18nManager:
    """
    Internationalization manager for vLLM CLI.
    
    Handles loading translations, switching languages, and providing
    translated text throughout the application.
    """
    
    # Supported languages
    SUPPORTED_LANGUAGES = {
        "en": {"name": "English", "native": "English"},
        "zh": {"name": "Chinese", "native": "中文"}
    }
    
    def __init__(self, config_manager=None):
        """
        Initialize the I18n manager.
        
        Args:
            config_manager: Optional ConfigManager instance for persisting language preference
        """
        self.config_manager = config_manager
        self._translation_cache: Dict[str, Dict[str, Any]] = {}
        self._current_language: Optional[str] = None
        self._current_translations: Dict[str, Any] = {}
        
        # Get translations directory
        self.translations_dir = Path(__file__).parent / "translations"
        
        # Load initial language
        self._load_initial_language()
    
    def _load_initial_language(self) -> None:
        """Load the initial language from config or default to English."""
        if self.config_manager:
            saved_lang = self.config_manager.config.get("language")
            if saved_lang and saved_lang in self.SUPPORTED_LANGUAGES:
                self.set_language(saved_lang, persist=False)
                return
        
        # Default to English
        self.set_language("en", persist=False)
    
    def get_current_language(self) -> str:
        """
        Get the current language code.
        
        Returns:
            Current language code ('en' or 'zh')
        """
        return self._current_language or "en"
    
    def set_language(self, lang_code: str, persist: bool = True) -> bool:
        """
        Set the current language and optionally persist to config.
        
        Args:
            lang_code: Language code to set ('en' or 'zh')
            persist: Whether to save the language preference to config
            
        Returns:
            True if language was set successfully, False otherwise
        """
        if lang_code not in self.SUPPORTED_LANGUAGES:
            logger.error(f"Unsupported language: {lang_code}")
            return False
        
        try:
            # Load translations for the new language
            translations = self.load_translations(lang_code)
            
            # Update current language and translations
            self._current_language = lang_code
            self._current_translations = translations
            
            # Persist to config if requested
            if persist and self.config_manager:
                self.config_manager.config["language"] = lang_code
                self.config_manager.config["language_set"] = True
                self.config_manager._save_config()
                logger.info(f"Language set to {lang_code} and saved to config")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set language to {lang_code}: {e}")
            return False
    
    def load_translations(self, lang_code: str) -> Dict[str, Any]:
        """
        Load translations for a specific language.
        
        Args:
            lang_code: Language code to load
            
        Returns:
            Dictionary containing translations
            
        Raises:
            FileNotFoundError: If translation file doesn't exist
            json.JSONDecodeError: If translation file is invalid
        """
        # Check cache first
        if lang_code in self._translation_cache:
            logger.debug(f"Loading translations for {lang_code} from cache")
            return self._translation_cache[lang_code]
        
        # Load from file
        translation_file = self.translations_dir / f"{lang_code}.json"
        
        if not translation_file.exists():
            raise FileNotFoundError(f"Translation file not found: {translation_file}")
        
        logger.info(f"Loading translations from {translation_file}")
        
        with open(translation_file, "r", encoding="utf-8") as f:
            translations = json.load(f)
        
        # Cache the translations
        self._translation_cache[lang_code] = translations
        
        return translations
    
    def t(self, key: str, **kwargs) -> str:
        """
        Translate a key to the current language.
        
        Supports nested keys using dot notation (e.g., 'menu.main.quit')
        and parameter interpolation using keyword arguments.
        
        Args:
            key: Translation key (supports dot notation)
            **kwargs: Parameters for string formatting
            
        Returns:
            Translated text, or the key itself if translation not found
            
        Examples:
            >>> i18n.t('menu.main.quit')
            'Quit'
            >>> i18n.t('messages.error', error='Connection failed')
            'An error occurred: Connection failed'
        """
        # Navigate through nested dictionary using dot notation
        keys = key.split(".")
        value = self._current_translations
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # Translation not found, return key as fallback
                logger.debug(f"Translation not found for key: {key}")
                return key
        
        # If value is not a string, return key as fallback
        if not isinstance(value, str):
            logger.warning(f"Translation value is not a string for key: {key}")
            return key
        
        # Apply parameter interpolation if kwargs provided
        if kwargs:
            try:
                return value.format(**kwargs)
            except KeyError as e:
                logger.warning(f"Missing parameter for translation key {key}: {e}")
                return value
        
        return value
    
    def get_available_languages(self) -> List[Dict[str, str]]:
        """
        Get list of available languages.
        
        Returns:
            List of language dictionaries with 'code', 'name', and 'native' keys
        """
        return [
            {
                "code": code,
                "name": info["name"],
                "native": info["native"]
            }
            for code, info in self.SUPPORTED_LANGUAGES.items()
        ]
    
    def is_language_set(self) -> bool:
        """
        Check if user has explicitly set a language preference.
        
        Returns:
            True if language has been set by user, False otherwise
        """
        if self.config_manager:
            return self.config_manager.config.get("language_set", False)
        return False
    
    def clear_cache(self) -> None:
        """Clear the translation cache."""
        self._translation_cache.clear()
        logger.info("Translation cache cleared")
