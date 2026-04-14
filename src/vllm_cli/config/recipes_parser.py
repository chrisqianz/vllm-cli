#!/usr/bin/env python3
"""
Recipes parser for vLLM CLI.

Handles fetching and parsing of official vLLM recipes from:
https://github.com/vllm-project/recipes

Provides functionality to:
- Fetch list of available recipes
- Parse vllm serve commands
- Convert recipes to profile format
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# GitHub API and raw content URLs
GITHUB_API_URL = "https://api.github.com/repos/vllm-project/recipes/contents"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/vllm-project/recipes/main"

# Cache for recipes list (memory only)
_recipes_cache: Optional[List[Dict[str, Any]]] = None


@dataclass
class RecipeProfile:
    """Represents a recipe profile from vLLM recipes."""

    id: str
    model_name: str
    hardware: str
    description: str
    config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    source_file: str = ""
    version: str = "v1"


def is_official_profile(name: str) -> bool:
    """
    Check if a profile name is an official vLLM recipe profile.

    Args:
        name: Profile name to check

    Returns:
        True if the profile is an official recipe profile
    """
    return name.startswith("official_")


class RecipesParser:
    """
    Parser for vLLM official recipes.

    Fetches and parses recipes from:
    https://github.com/vllm-project/recipes
    """

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize the recipes parser.

        Args:
            cache_ttl: Cache time-to-live in seconds (default 5 minutes)
        """
        self.cache_ttl = cache_ttl
        self._recipes_list = None

    def fetch_recipes_list(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Fetch list of available recipes from GitHub.

        Args:
            force_refresh: Force refresh cache

        Returns:
            List of recipe metadata dictionaries

        Raises:
            requests.RequestException: If network request fails
        """
        global _recipes_cache

        if _recipes_cache is not None and not force_refresh:
            logger.debug("Using cached recipes list")
            return _recipes_cache

        try:
            logger.info("Fetching recipes list from GitHub...")
            response = requests.get(GITHUB_API_URL, timeout=30)
            response.raise_for_status()
            contents = response.json()

            recipes = []
            directories = []

            for item in contents:
                if item.get("type") == "dir":
                    directories.append(item["name"])
                elif item.get("type") == "file" and item["name"].endswith(".md"):
                    model_name, hardware = self._parse_filename(item["name"][:-3])
                    recipes.append(
                        {
                            "id": f"{item['name'][:-3].lower().replace('-', '_')}",
                            "filename": item["name"],
                            "model_name": model_name,
                            "hardware": hardware,
                            "source_url": item.get("download_url", ""),
                            "category": "root",
                        }
                    )

            for dir_name in directories:
                try:
                    dir_url = f"{GITHUB_API_URL}/{dir_name}"
                    dir_response = requests.get(dir_url, timeout=30)
                    dir_response.raise_for_status()
                    dir_contents = dir_response.json()

                    for item in dir_contents:
                        if item.get("type") == "file" and item["name"].endswith(".md"):
                            filename = item["name"][:-3]
                            model_name, hardware = self._parse_filename(filename)
                            recipes.append(
                                {
                                    "id": f"{dir_name.lower()}_{filename.lower().replace('-', '_')}",
                                    "filename": f"{dir_name}/{item['name']}",
                                    "model_name": model_name,
                                    "hardware": hardware,
                                    "source_url": item.get("download_url", ""),
                                    "category": dir_name,
                                }
                            )
                except requests.RequestException as e:
                    logger.warning(f"Failed to fetch directory {dir_name}: {e}")

            _recipes_cache = recipes
            logger.info(f"Found {len(recipes)} recipes")
            return recipes

        except requests.RequestException as e:
            logger.error(f"Failed to fetch recipes: {e}")
            raise

    def fetch_profile_details(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch details for a specific recipe.

        Args:
            recipe_id: Recipe identifier

        Returns:
            Recipe details including commands, or None if not found
        """
        recipes = self.fetch_recipes_list()
        recipe = None

        for r in recipes:
            if r["id"] == recipe_id:
                recipe = r
                break

        if not recipe:
            logger.warning(f"Recipe not found: {recipe_id}")
            return None

        try:
            url = recipe["source_url"]
            if not url:
                url = f"{GITHUB_RAW_URL}/{recipe['filename']}"

            response = requests.get(url, timeout=30)
            response.raise_for_status()
            content = response.text

            # Parse commands from markdown content
            commands = self._extract_commands(content)

            return {
                "id": recipe["id"],
                "model_name": recipe["model_name"],
                "hardware": recipe["hardware"],
                "filename": recipe["filename"],
                "commands": commands,
                "raw_content": content,
            }

        except requests.RequestException as e:
            logger.error(f"Failed to fetch recipe details: {e}")
            return None

    def _parse_filename(self, filename: str) -> tuple:
        """
        Parse filename to extract model name and hardware.

        Examples:
            Qwen3-235B-H200 -> ("Qwen3-235B", "H200")
            DeepSeek-V3 -> ("DeepSeek-V3", "default")
            Llama-3.1-8B-A100 -> ("Llama-3.1-8B", "A100")
        """
        # Common hardware suffixes
        hardware_suffixes = [
            "H200",
            "H100",
            "A100",
            "A800",
            "A10",
            "B200",
            "B100",
            "B50",
            "MI300x",
            "MI325",
            "MI355",
            "L40S",
            "L40",
            "RTX4090",
            "RTX3090",
        ]

        for hw in hardware_suffixes:
            if filename.endswith(f"-{hw}"):
                model_name = filename[: -(len(hw) + 1)]  # -1 for dash
                return model_name, hw

        # Default: no hardware specified
        return filename, "default"

    def _extract_commands(self, content: str) -> List[str]:
        """
        Extract vllm serve commands from markdown content.

        Args:
            content: Markdown content

        Returns:
            List of command strings
        """
        commands = []

        # Pattern to match code blocks with vllm serve commands
        # Matches: ```bash or ```shell or ``` (with optional language)
        code_block_pattern = r"```(?:bash|shell)?\s*\n(.*?)```"

        for match in re.finditer(code_block_pattern, content, re.DOTALL):
            block_content = match.group(1).strip()

            # Look for vllm serve commands
            for line in block_content.split("\n"):
                line = line.strip()
                # Match vllm serve command (with optional environment variables prefix)
                if re.match(r"(?:[\w\-\.]+=[\w\-\.]*\s+)*vllm\s+serve\s+", line):
                    commands.append(line)

        return commands

    def parse_command(self, command: str) -> Dict[str, Any]:
        """
        Parse a vllm serve command into config and environment variables.

        Args:
            command: Command string (may include env vars prefix)

        Returns:
            Dictionary with 'config', 'environment', 'model' keys
        """
        result = {
            "config": {},
            "environment": {},
            "model": None,
        }

        # Extract environment variables from the beginning
        # Pattern: VAR=value VAR2=value2 vllm serve ...
        env_pattern = r"^([A-Za-z_][A-Za-z0-9_]*)=(\S+)\s+"
        env_vars = {}

        remaining = command
        while True:
            match = re.match(env_pattern, remaining)
            if match:
                var_name = match.group(1)
                var_value = match.group(2)
                env_vars[var_name] = var_value
                remaining = remaining[match.end() :]
            else:
                break

        result["environment"] = env_vars

        # Now parse vllm serve arguments
        # Remove leading/trailing whitespace
        remaining = remaining.strip()

        # Check it starts with vllm serve
        if not remaining.startswith("vllm serve "):
            logger.warning(f"Invalid command format: {command[:50]}...")
            return result

        # Get the model (first argument after vllm serve)
        rest = remaining[11:].strip()  # Remove "vllm serve "

        # Extract model (first word, could have org/model format)
        model_match = re.match(r"([^\s/]+/[^\s]+|[^\s]+)", rest)
        if model_match:
            result["model"] = model_match.group(1)
            rest = rest[model_match.end() :].strip()

        # Parse remaining arguments
        config = self._parse_arguments(rest)
        result["config"] = config

        return result

    def _parse_arguments(self, args_str: str) -> Dict[str, Any]:
        """
        Parse vllm serve command arguments.

        Args:
            args_str: Argument string after the model

        Returns:
            Dictionary of config options
        """
        config = {}

        # Tokenize arguments (handle quoted strings)
        tokens = []
        current = ""
        in_quote = False
        quote_char = None

        i = 0
        while i < len(args_str):
            char = args_str[i]

            if char in ('"', "'") and not in_quote:
                in_quote = True
                quote_char = char
            elif char == quote_char and in_quote:
                in_quote = False
                quote_char = None
            elif char == " " and not in_quote:
                if current:
                    tokens.append(current)
                    current = ""
            else:
                current += char

            i += 1

        if current:
            tokens.append(current)

        # Parse tokens into key-value pairs
        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Skip non-flag tokens (model was already extracted)
            if not token.startswith("-"):
                i += 1
                continue

            # Handle --no-xxx flags (boolean negation)
            if token.startswith("--no-"):
                key = self._convert_flag_name(token[5:])
                config[key] = False
                i += 1
                continue

            # Handle --flag (boolean true)
            if token.startswith("--") and i + 1 >= len(tokens):
                key = self._convert_flag_name(token[2:])
                config[key] = True
                i += 1
                continue

            # Regular --key value or -k value
            if token.startswith("--"):
                key = self._convert_flag_name(token[2:])
            elif token.startswith("-"):
                key = self._convert_flag_name(token[1:])
            else:
                i += 1
                continue

            # Get value
            if i + 1 < len(tokens):
                next_token = tokens[i + 1]
                # Check if next token is a value (not a flag)
                if not next_token.startswith("-"):
                    value = self._parse_value(next_token)
                    config[key] = value
                    i += 2
                    continue

            # Boolean flag without value
            config[key] = True
            i += 1

        return config

    def _convert_flag_name(self, flag: str) -> str:
        """
        Convert CLI flag name to config key.

        Examples:
            gpu-memory-utilization -> gpu_memory_utilization
            trust-remote-code -> trust_remote_code
            tp -> tensor_parallel_size (common abbreviations)
        """
        # Common abbreviations
        abbrev_map = {
            "tp": "tensor_parallel_size",
            "pp": "pipeline_parallel_size",
            "dp": "data_parallel_size",
            "gpu-memory-utilization": "gpu_memory_utilization",
            "max-model-len": "max_model_len",
            "trust-remote-code": "trust_remote_code",
            "enable-chunked-prefill": "enable_chunked_prefill",
            "enable-prefix-caching": "enable_prefix_caching",
            "enable-expert-parallel": "enable_expert_parallel",
            "max-num-batched-tokens": "max_num_batched_tokens",
            "max-num-seqs": "max_num_seqs",
            "enforce-eager": "enforce_eager",
            "enable-lora": "enable_lora",
            "lora-modules": "lora_modules",
            "compilation-config": "compilation_config",
            "speculative-config": "speculative_config",
            "kv-cache-dtype": "kv_cache_dtype",
            "chat-template": "chat_template",
            "chat-template-content-format": "chat_template_content_format",
            "tool-call-parser": "tool_call_parser",
            "reasoning-parser": "reasoning_parser",
            "enable-auto-tool-choice": "enable_auto_tool_choice",
        }

        if flag in abbrev_map:
            return abbrev_map[flag]

        # Default: convert kebab-case to snake_case
        return flag.replace("-", "_")

    def _parse_value(self, value: str) -> Any:
        """
        Parse a value string into appropriate Python type.

        Args:
            value: String value to parse

        Returns:
            Parsed value (int, float, bool, or string)
        """
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try boolean
        if value.lower() in ("true", "yes", "on", "1"):
            return True
        if value.lower() in ("false", "no", "off", "0"):
            return False

        # Keep as string
        return value

    def command_to_profile(
        self,
        command: str,
        model_name: str,
        hardware: str = "default",
    ) -> Dict[str, Any]:
        """
        Convert a vllm serve command to profile format.

        Args:
            command: vllm serve command string
            model_name: Model name (e.g., "Qwen3-235B")
            hardware: Hardware type (e.g., "H200")

        Returns:
            Profile dictionary ready to save
        """
        # Parse the command
        parsed = self.parse_command(command)

        # Generate profile name with official_ prefix
        profile_name = self._generate_profile_name(model_name, hardware)

        # Build description
        description = f"Official vLLM recipe for {model_name}"
        if hardware != "default":
            description += f" on {hardware}"

        profile = {
            "name": profile_name.replace("_", " ").title(),
            "description": description,
            "config": parsed.get("config", {}),
            "environment": parsed.get("environment", {}),
        }

        return profile

    def _generate_profile_name(self, model_name: str, hardware: str) -> str:
        """
        Generate a profile name from model and hardware.

        Args:
            model_name: Model name (e.g., "Qwen3-235B")
            hardware: Hardware type (e.g., "H200")

        Returns:
            Profile name (e.g., "official_qwen3_235b_h200")
        """
        # Convert to lowercase and sanitize
        model_part = model_name.lower().replace(" ", "_")
        # Remove special characters
        model_part = re.sub(r"[^a-z0-9_]", "", model_part)

        if hardware != "default":
            hw_part = hardware.lower().replace(" ", "_")
            return f"official_{model_part}_{hw_part}"
        else:
            return f"official_{model_part}"

    def get_next_version(self, base_name: str) -> str:
        """
        Get the next version suffix for a profile.

        Args:
            base_name: Base profile name (e.g., "official_qwen3_235b")

        Returns:
            Version suffix (e.g., "_v2")
        """
        # This would check existing profiles and find the next version
        # For now, return v1 as default
        return "_v1"

    def fetch_vllm_cli_args(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch vLLM CLI arguments from vllm serve --help.

        This method attempts to get CLI arguments by running:
        1. `vllm serve --help` (if vllm is installed)

        Returns:
            Dictionary of argument definitions
        """
        try:
            result = subprocess.run(
                ["vllm", "serve", "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.warning(f"vllm serve --help failed: {result.stderr}")
                return self._get_fallback_args()

            return self._parse_vllm_help(result.stdout)

        except FileNotFoundError:
            logger.warning("vllm command not found - using fallback args")
            return self._get_fallback_args()
        except Exception as e:
            logger.error(f"Failed to fetch vLLM CLI args: {e}")
            return self._get_fallback_args()

    def _get_fallback_args(self) -> Dict[str, Dict[str, Any]]:
        """
        Get fallback CLI args from existing argument_schema.json.

        Returns:
            Dictionary of argument definitions from schema
        """
        try:
            schema_path = (
                Path(__file__).parent.parent / "schemas" / "argument_schema.json"
            )
            if schema_path.exists():
                with open(schema_path) as f:
                    data = json.load(f)
                    return data.get("arguments", {})
        except Exception:
            pass
        return {}

    def _parse_vllm_help(self, help_text: str) -> Dict[str, Dict[str, Any]]:
        """
        Parse vllm serve --help output to extract argument definitions.

        Args:
            help_text: Output from vllm serve --help

        Returns:
            Dictionary of argument definitions
        """
        args = {}

        # Pattern to match: --argument or -a
        # Format: --argument description (default: value)
        patterns = [
            r"--?([a-z][a-z0-9-]*)\s*[\s\n]+([^\n-]+?)(?:\(default:`?([^`)`\n]+)`?\))?",
        ]

        for line in help_text.split("\n"):
            # Match --argument description
            match = re.match(r"\s*(--?[a-z][a-z0-9-]*)\s+(.+)", line)
            if match:
                arg_name = match.group(1).lstrip("-")
                description = match.group(2).strip()

                # Extract default if present
                default = None
                if "default:" in description.lower():
                    default_match = re.search(r"default:`?([^`)\n]+)`?", description)
                    if default_match:
                        default = default_match.group(1).strip()
                        description = re.sub(
                            r"\(default:`?[^`)+`?\)", "", description
                        ).strip()

                # Determine type
                arg_type = "string"
                if default:
                    if default.lower() in ("true", "false"):
                        arg_type = "boolean"
                    elif default.isdigit():
                        arg_type = "integer"
                    elif "." in default and default.replace(".", "").isdigit():
                        arg_type = "float"

                args[arg_name] = {
                    "type": arg_type,
                    "description": description[:200],  # Truncate long descriptions
                    "default": default,
                }

        logger.info(f"Parsed {len(args)} arguments from vllm serve --help")
        return args


def list_recipes(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Convenience function to list all available recipes.

    Args:
        force_refresh: Force refresh cache

    Returns:
        List of recipe metadata
    """
    parser = RecipesParser()
    return parser.fetch_recipes_list(force_refresh=force_refresh)


def import_recipe(recipe_id: str) -> Optional[Dict[str, Any]]:
    """
    Import a recipe and convert to profile format.

    Args:
        recipe_id: Recipe identifier

    Returns:
        Profile dictionary, or None if import fails
    """
    parser = RecipesParser()
    details = parser.fetch_profile_details(recipe_id)

    if not details or not details.get("commands"):
        logger.warning(f"No commands found for recipe: {recipe_id}")
        return None

    # Use the first command for now
    # TODO: Handle multiple commands (different configs)
    command = details["commands"][0]

    profile = parser.command_to_profile(
        command=command,
        model_name=details["model_name"],
        hardware=details["hardware"],
    )

    return profile
