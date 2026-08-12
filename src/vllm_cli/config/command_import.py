#!/usr/bin/env python3
"""
Command Import Module

Parses raw `vllm serve` command lines and converts them into profile configurations.
Handles standard CLI flags, dot-notation config keys (e.g., --speculative_config.method),
boolean flags, and positional model arguments.
"""

import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CommandImporter:
    """
    Parses raw vllm serve commands and converts them to profile configurations.

    Supports:
    - Standard CLI flags (--flag value, --boolean-flag)
    - Dot-notation config keys (--config_key.sub_key value)
    - Positional model argument
    - Multi-line commands with backslash continuation
    - Quoted values
    """

    # Known boolean flags (no value needed)
    BOOLEAN_FLAGS = {
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--enable-lora",
        "--enable-auto-tool-choice",
        "--enable-expert-parallel",
        "--enable-dbo",
        "--enable-eplb",
        "--enable-elastic-ep",
        "--enable-sleep-mode",
        "--enable-prompt-embeds",
        "--enable-log-requests",
        "--enable-return-routed-experts",
        "--enable-flashinfer-autotune",
        "--enable-ep-weight-filter",
        "--async-scheduling",
        "--enforce-eager",
        "--disable-log-stats",
        "--disable-sliding-window",
        "--disable-async-output-proc",
        "--disable-frontend-multiprocessing",
        "--disable-custom-all-reduce",
        "--disable-cascade-attn",
        "--trust-remote-code",
        "--headless",
        "--skip-tokenizer-init",
        "--numa-bind",
        "--calculate-kv-scales",
        "--enable-mamba-cache-stochastic-rounding",
    }

    # CLI flag to config key mapping (partial; full mapping from schema)
    # Dot-notation args that should be nested in config dicts
    DOT_NOTATION_PATTERNS = {
        "speculative_config": "speculative_config",
        "attention_config": "attention_config",
        "compilation_config": "compilation_config",
        "generation_config": "generation_config",
        "kv_transfer_config": "kv_transfer_config",
        "kv_events_config": "kv_events_config",
        "ec_transfer_config": "ec_transfer_config",
        "reasoning_config": "reasoning_config",
        "structured_outputs_config": "structured_outputs_config",
        "kernel_config": "kernel_config",
        "profiler_config": "profiler_config",
        "weight_transfer_config": "weight_transfer_config",
        "pooler_config": "pooler_config",
        "mamba_config": "mamba_config",
        "quantization_config": "quantization_config",
        "additional_config": "additional_config",
        "eplb_config": "eplb_config",
    }

    def __init__(self, schema_path: Optional[Path] = None):
        if schema_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            schema_path = base_dir / "schemas" / "argument_schema.json"
        self.schema_path = Path(schema_path)
        self._schema = None
        self._flag_to_key = None

    @property
    def schema(self) -> dict:
        if self._schema is None:
            with open(self.schema_path) as f:
                self._schema = json.load(f)
        return self._schema

    @property
    def flag_to_key(self) -> Dict[str, str]:
        """Build reverse mapping: cli_flag -> config_key."""
        if self._flag_to_key is None:
            self._flag_to_key = {}
            for key, val in self.schema.get("arguments", {}).items():
                flag = val.get("cli_flag")
                if flag:
                    self._flag_to_key[flag] = key
        return self._flag_to_key

    def parse_command(self, command: str) -> Dict[str, Any]:
        """
        Parse a raw vllm serve command into a structured configuration.

        Args:
            command: Raw command string, e.g., "vllm serve model --flag value"

        Returns:
            Dictionary with 'model', 'config', and 'environment' keys.
        """
        # Clean up the command
        command = self._clean_command(command)

        # Tokenize
        tokens = self._tokenize(command)

        # Parse tokens
        result = self._parse_tokens(tokens)

        return result

    def _clean_command(self, command: str) -> str:
        """Clean up multi-line commands with backslash continuation."""
        # Remove leading 'vllm serve' or 'vllm serve '
        command = re.sub(r"^vllm\s+serve\s*", "", command.strip())

        # Remove backslash line continuations
        command = command.replace("\\\n", " ").replace("\\\r\n", " ")

        # Remove leading/trailing whitespace
        return command.strip()

    def _tokenize(self, command: str) -> List[str]:
        """Tokenize command while preserving quoted strings."""
        try:
            return shlex.split(command)
        except ValueError:
            # Fallback: simple split
            return command.split()

    def _parse_tokens(self, tokens: List[str]) -> Dict[str, Any]:
        """Parse token list into structured configuration."""
        config = {}
        model = None
        i = 0

        while i < len(tokens):
            token = tokens[i]

            if token.startswith("--"):
                # It's a flag
                if token in self.BOOLEAN_FLAGS:
                    # Boolean flag
                    config_key = self._resolve_flag(token)
                    if config_key:
                        config[config_key] = True
                elif "." in token:
                    # Dot-notation config (e.g., --speculative_config.method)
                    dot_key = token[2:]  # Remove --
                    if i + 1 < len(tokens):
                        i += 1
                        value = self._parse_value(tokens[i], dot_key)
                        self._set_nested_config(config, dot_key, value)
                    else:
                        logger.warning(f"Missing value for {token}")
                else:
                    # Standard flag with value
                    if i + 1 < len(tokens):
                        i += 1
                        config_key = self._resolve_flag(token)
                        value = self._parse_value(tokens[i], config_key)
                        if config_key:
                            config[config_key] = value
                    else:
                        logger.warning(f"Missing value for {token}")
            else:
                # Positional argument (model name)
                if model is None:
                    model = token

            i += 1

        # Convert types based on schema
        config = self._convert_config_types(config)

        # Build result
        result = {
            "model": model or "",
            "config": config,
        }

        return result

    def _resolve_flag(self, flag: str) -> Optional[str]:
        """Resolve a CLI flag to a config key."""
        return self.flag_to_key.get(flag)

    def _parse_value(self, value: str, config_key: str) -> Any:
        """Parse a string value to the appropriate Python type."""
        # Boolean strings
        if value.lower() in ("true", "yes"):
            return True
        if value.lower() in ("false", "no"):
            return False

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # JSON (for config dicts)
        if value.startswith("{") or value.startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # String
        return value

    def _convert_config_types(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert config values to proper types based on schema definitions.

        This ensures that values like '3' become 3 (int) when the schema
        expects an integer type.
        """
        arguments = self.schema.get("arguments", {})
        converted = {}

        for key, value in config.items():
            # Handle nested dicts (e.g., speculative_config)
            if isinstance(value, dict):
                converted[key] = self._convert_nested_types(key, value, arguments)
                continue

            # Look up schema type
            arg_info = arguments.get(key)
            if not arg_info:
                converted[key] = value
                continue

            arg_type = arg_info.get("type", "string")

            if arg_type == "integer" and isinstance(value, str):
                try:
                    converted[key] = int(value)
                except ValueError:
                    converted[key] = value
            elif arg_type == "float" and isinstance(value, str):
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
            elif arg_type == "boolean":
                if isinstance(value, str):
                    converted[key] = value.lower() in ("true", "yes", "1")
                else:
                    converted[key] = bool(value)
            else:
                converted[key] = value

        return converted

    def _convert_nested_types(
        self,
        parent_key: str,
        nested: Dict[str, Any],
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert types for nested config values."""
        converted = {}
        for key, value in nested.items():
            # Try to find the full key in arguments
            full_key = f"{parent_key}.{key}"
            arg_info = arguments.get(full_key)

            # Also check common patterns
            if not arg_info:
                # For speculative_config, check num_speculative_tokens
                if key == "num_speculative_tokens":
                    try:
                        converted[key] = int(value)
                    except (ValueError, TypeError):
                        converted[key] = value
                elif key == "method":
                    converted[key] = value
                elif key == "model":
                    converted[key] = value
                else:
                    converted[key] = value
            else:
                arg_type = arg_info.get("type", "string")
                if arg_type == "integer" and isinstance(value, str):
                    try:
                        converted[key] = int(value)
                    except ValueError:
                        converted[key] = value
                else:
                    converted[key] = value

        return converted

    def _set_nested_config(self, config: Dict[str, Any], dot_key: str, value: Any) -> None:
        """
        Set a nested config value using dot notation.

        e.g., 'speculative_config.method' -> config['speculative_config']['method']
        """
        parts = dot_key.split(".", 1)
        parent_key = parts[0]
        child_key = parts[1] if len(parts) > 1 else ""

        if not child_key:
            config[parent_key] = value
            return

        # Initialize parent dict if needed
        if parent_key not in config:
            config[parent_key] = {}

        if isinstance(config[parent_key], dict):
            config[parent_key][child_key] = value
        else:
            # If parent is already a scalar, replace with dict
            config[parent_key] = {child_key: value}

    def to_profile(
        self,
        command: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert a raw command to a profile dictionary.

        Args:
            command: Raw vllm serve command
            name: Profile name (auto-generated from model if not provided)
            description: Profile description

        Returns:
            Profile dictionary ready for save_user_profile()
        """
        parsed = self.parse_command(command)

        # Auto-generate name from model
        if not name and parsed["model"]:
            name = self._generate_profile_name(parsed["model"])

        if not description:
            description = f"Imported from vllm serve command: {parsed['model']}"

        profile = {
            "name": name,
            "description": description,
            "config": parsed["config"],
            "environment": {},
        }

        return profile

    def _generate_profile_name(self, model: str) -> str:
        """Generate a profile name from a model path."""
        # Extract last component of model path
        parts = model.rstrip("/").split("/")
        name = parts[-1].lower()

        # Replace special characters with underscores
        name = re.sub(r"[^a-z0-9]+", "_", name)
        name = name.strip("_")

        return name

    def preview(self, command: str) -> str:
        """
        Preview what a command would look like as a profile.

        Returns:
            Formatted preview string
        """
        parsed = self.parse_command(command)

        lines = [
            f"Model: {parsed['model']}",
            f"Config keys: {len(parsed['config'])}",
            "",
        ]

        for key, value in sorted(parsed["config"].items()):
            lines.append(f"  {key} = {value}")

        return "\n".join(lines)
