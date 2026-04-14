#!/usr/bin/env python3
"""
CLI Args Sync Module

Fetches the latest vLLM CLI arguments from GitHub source code and compares
with the local schema to identify new, deprecated, or modified arguments.
"""

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# GitHub URLs for vLLM source files
VLLM_CLI_ARGS_URL = "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/entrypoints/openai/cli_args.py"
VLLM_ARG_UTILS_URL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/engine/arg_utils.py"
)


@dataclass
class ArgChange:
    """Represents a change in CLI arguments."""

    name: str
    change_type: str  # 'added', 'removed', 'modified'
    details: str = ""


class CLIArgsSync:
    """Syncs CLI arguments from vLLM GitHub repository."""

    def __init__(self, schema_path: Optional[Path] = None):
        if schema_path is None:
            # Default to schemas directory relative to project root
            # This file is in src/vllm_cli/config/, so go up 2 levels
            base_dir = Path(__file__).resolve().parent.parent
            schema_path = base_dir / "schemas" / "argument_schema.json"
        self.schema_path = schema_path
        self._schema: Optional[dict] = None

    @property
    def schema(self) -> dict:
        """Load local schema."""
        if self._schema is None:
            if self.schema_path.exists():
                with open(self.schema_path) as f:
                    self._schema = json.load(f)
            else:
                self._schema = {"arguments": {}}
        return self._schema

    def fetch_remote_args(self) -> set[str]:
        """Fetch CLI arguments from GitHub using curl/urllib."""
        all_args = set()

        try:
            # Fetch cli_args.py
            cli_args_content = self._fetch_url(VLLM_CLI_ARGS_URL)
            # Extract from @config classes (BaseFrontendArgs, FrontendArgs)
            cli_params = re.findall(
                r"^\s+([a-z_][a-z0-9_]*)\s*:\s*[\w\[\]|,\s\.\-\>]+(?:\s*=|\|)",
                cli_args_content,
                re.MULTILINE,
            )
            # Filter internal variables
            cli_params = [
                p
                for p in cli_params
                if not p.startswith("_")
                and p
                not in [
                    "frontend_kwargs",
                    "parser",
                    "https",
                    "try",
                    "else",
                    "lora_list",
                    "values",
                    "option_string",
                ]
            ]
            all_args.update(cli_params)

            # Fetch arg_utils.py
            arg_utils_content = self._fetch_url(VLLM_ARG_UTILS_URL)
            # Extract from add_argument calls
            arg_params = re.findall(
                r'\.add_argument\("(--[a-z\-]+)"', arg_utils_content
            )
            # Convert --arg-name to arg_name
            arg_params = [p.lstrip("-").replace("-", "_") for p in arg_params]
            all_args.update(arg_params)

        except Exception as e:
            raise RuntimeError(f"Failed to fetch CLI args from GitHub: {e}")

        return all_args

    def _fetch_url(self, url: str) -> str:
        """Fetch content from URL using urllib (or curl if available)."""
        try:
            # Try urllib first (no external dependencies)
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read().decode("utf-8")
        except Exception:
            # Fallback to subprocess curl
            import subprocess

            result = subprocess.run(
                ["curl", "-s", "--fail", "--silent", "--location", url],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"curl failed: {result.stderr}")
            return result.stdout

    def get_local_args(self) -> set[str]:
        """Get arguments from local schema."""
        return set(self.schema.get("arguments", {}).keys())

    def compare(self) -> tuple[list[ArgChange], list[ArgChange]]:
        """
        Compare local and remote arguments.

        Returns:
            Tuple of (new_args, removed_args)
        """
        remote_args = self.fetch_remote_args()
        local_args = self.get_local_args()

        # Find new arguments (in remote but not in local)
        new_args = [
            ArgChange(
                name=arg, change_type="added", details="New argument in vLLM source"
            )
            for arg in sorted(remote_args - local_args)
        ]

        # Find removed arguments (in local but not in remote)
        removed_args = [
            ArgChange(
                name=arg,
                change_type="removed",
                details="Argument no longer in vLLM source",
            )
            for arg in sorted(local_args - remote_args)
        ]

        return new_args, removed_args

    def sync(self, dry_run: bool = True) -> dict:
        """
        Sync local schema with remote.

        Args:
            dry_run: If True, only report changes without updating

        Returns:
            Dict with sync results
        """
        new_args, removed_args = self.compare()

        result = {
            "success": True,
            "dry_run": dry_run,
            "new_args": [a.name for a in new_args],
            "removed_args": [a.name for a in removed_args],
            "new_count": len(new_args),
            "removed_count": len(removed_args),
            "total_remote": len(self.fetch_remote_args()),
            "total_local": len(self.get_local_args()),
        }

        if not dry_run and (new_args or removed_args):
            # Update schema with new arguments
            for arg in new_args:
                self.schema["arguments"][arg.name] = {
                    "type": "string",
                    "description": f"Synced from vLLM GitHub: {arg.details}",
                    "cli_flag": f"--{arg.name.replace('_', '-')}",
                    "importance": "medium",
                    "synced": True,
                }

            # Note: We don't remove arguments to maintain backward compatibility
            # Mark removed args as deprecated
            for arg in removed_args:
                if arg.name in self.schema["arguments"]:
                    self.schema["arguments"][arg.name]["deprecated"] = True
                    self.schema["arguments"][arg.name]["deprecation_note"] = arg.details

            # Update schema version
            self.schema["version"] = self.schema.get("version", "1.0") + ".1"

            # Save updated schema
            with open(self.schema_path, "w") as f:
                json.dump(self.schema, f, indent=2)

            result["updated"] = True

        return result


def sync_cli_args(dry_run: bool = True, verbose: bool = False) -> dict:
    """
    Convenience function to sync CLI args.

    Args:
        dry_run: If True, only report changes
        verbose: Print detailed output

    Returns:
        Sync result dict
    """
    sync = CLIArgsSync()

    if verbose:
        print("Fetching latest CLI arguments from vLLM GitHub...")
        print(f"  Source: {VLLM_CLI_ARGS_URL}")
        print(f"  Source: {VLLM_ARG_UTILS_URL}")

    result = sync.sync(dry_run=dry_run)

    if verbose:
        print(f"\nResults:")
        print(f"  Remote args: {result['total_remote']}")
        print(f"  Local args:  {result['total_local']}")
        print(f"  New:         {result['new_count']}")
        print(f"  Removed:     {result['removed_count']}")

        if result["new_args"]:
            print(f"\nNew arguments: {', '.join(result['new_args'][:10])}")
            if len(result["new_args"]) > 10:
                print(f"  ... and {len(result['new_args']) - 10} more")

        if result["removed_args"]:
            print(f"\nRemoved arguments: {', '.join(result['removed_args'][:10])}")
            if len(result["removed_args"]) > 10:
                print(f"  ... and {len(result['removed_args']) - 10} more")

        if not dry_run and result.get("updated"):
            print(f"\n✓ Schema updated: {sync.schema_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync vLLM CLI arguments from GitHub")
    parser.add_argument(
        "--apply", action="store_true", help="Apply changes to local schema"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    result = sync_cli_args(dry_run=not args.apply, verbose=args.verbose)

    if result["new_count"] > 0 or result["removed_count"] > 0:
        print(f"\nRun with --apply to update the schema.")
