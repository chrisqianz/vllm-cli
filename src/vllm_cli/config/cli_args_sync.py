#!/usr/bin/env python3
"""
CLI Args Sync Module

Fetches the latest vLLM CLI arguments from GitHub source code and compares
with the local schema to identify new, deprecated, or modified arguments.
Supports fetching from specific version tags or the main branch.

In vLLM 0.22+, CLI arguments are defined as dataclass fields in EngineArgs
and exposed through add_cli_args(). This module handles both the legacy
add_argument pattern and the newer dataclass-based approach.
"""

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# GitHub URLs for vLLM source files
VLLM_GITHUB_BASE = "https://raw.githubusercontent.com/vllm-project/vllm"
VLLM_DEFAULT_BRANCH = "main"

# Supported vLLM versions for targeted sync
SUPPORTED_VLLM_VERSIONS = [
    "v0.22.1",
    "v0.22.0",
    "v0.21.0",
    "v0.20.2",
    "v0.20.1",
    "v0.20.0",
]

# Noise words to filter out from regex matches
_NOISE_WORDS = frozenset({
    "m", "n", "T", "K", "else", "try", "NOTE", "parser",
    "https", "lora_list", "values", "option_string",
    "frontend_kwargs", "model_config", "parallel_config",
    "usage_context", "target_model_config", "target_parallel_config",
})


def _get_url(path: str, branch_or_tag: str = VLLM_DEFAULT_BRANCH) -> str:
    """Build a GitHub raw URL for a given path and branch/tag."""
    return f"{VLLM_GITHUB_BASE}/{branch_or_tag}/{path}"


@dataclass
class ArgChange:
    """Represents a change in CLI arguments."""

    name: str
    change_type: str  # 'added', 'removed', 'modified'
    details: str = ""


class CLIArgsSync:
    """Syncs CLI arguments from vLLM GitHub repository."""

    def __init__(
        self,
        schema_path: Optional[Path] = None,
        branch_or_tag: str = VLLM_DEFAULT_BRANCH,
    ):
        if schema_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            schema_path = base_dir / "schemas" / "argument_schema.json"
        self.schema_path = schema_path
        self.branch_or_tag = branch_or_tag
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

    def _get_source_urls(self) -> tuple[str, str]:
        """Get the source URLs for the current branch/tag."""
        cli_args_url = _get_url(
            "vllm/entrypoints/openai/cli_args.py", self.branch_or_tag
        )
        arg_utils_url = _get_url("vllm/engine/arg_utils.py", self.branch_or_tag)
        return cli_args_url, arg_utils_url

    def _extract_cli_args_from_arg_utils(self, content: str) -> set[str]:
        """
        Extract CLI argument names from arg_utils.py.

        Handles both legacy explicit add_argument calls and v0.22+ dataclass
        field references with kwargs unpacking.
        """
        args = set()

        # Pattern 1: Explicit add_argument("--name", ...)
        for m in re.finditer(r'add_argument\("(--[a-z\-]+)"', content):
            arg = m.group(1).lstrip("-").replace("-", "_")
            args.add(arg)

        # Pattern 2: kwargs unpacking: **xxx_kwargs["name"]
        for m in re.finditer(r'\*\*(\w+)_kwargs\[([\"\'])+(\w+)\3', content):
            field_name = m.group(3)
            if field_name not in _NOISE_WORDS:
                args.add(field_name)

        # Pattern 3: get_field(XxxConfig, "field_name")
        for m in re.finditer(r'get_field\s*\(\s*\w+\s*,\s*[\"\']([a-z_]+)[\"\']', content, re.DOTALL):
            args.add(m.group(1))

        return args

    def _extract_frontend_args(self, content: str) -> set[str]:
        """
        Extract frontend/API argument names from cli_args.py.

        Parses @dataclass field definitions from config classes.
        """
        args = set()
        for m in re.findall(
            r"^\s+([a-z_][a-z0-9_]*)\s*:\s*[\w\[\]|,\s\.\-\>]+(?:\s*=|\|)",
            content,
            re.MULTILINE,
        ):
            if m not in _NOISE_WORDS and not m.startswith("_"):
                args.add(m)
        return args

    def fetch_remote_args(self) -> set[str]:
        """
        Fetch CLI arguments from GitHub.

        Combines arguments from both cli_args.py (frontend/API args) and
        arg_utils.py (engine args). In vLLM 0.22+, engine args are defined
        as dataclass fields and exposed through kwargs unpacking.
        """
        all_args = set()
        cli_args_url, arg_utils_url = self._get_source_urls()

        try:
            # Fetch cli_args.py (frontend arguments)
            cli_args_content = self._fetch_url(cli_args_url)
            frontend_args = self._extract_frontend_args(cli_args_content)
            all_args.update(frontend_args)

            # Fetch arg_utils.py (engine arguments)
            arg_utils_content = self._fetch_url(arg_utils_url)
            engine_args = self._extract_cli_args_from_arg_utils(arg_utils_content)
            all_args.update(engine_args)

        except Exception as e:
            raise RuntimeError(f"Failed to fetch CLI args from GitHub: {e}")

        return all_args

    def _fetch_url(self, url: str) -> str:
        """Fetch content from URL using urllib (or curl if available)."""
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read().decode("utf-8")
        except Exception:
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

    @staticmethod
    def _bump_version(version: str) -> str:
        """Bump the patch part of a semantic version string.

        Handles versions like '1.0.0', '2.0.0', '0.9', '1.0', or plain floats.
        Falls back to a simple float-based bump for non-semver strings.
        """
        parts = version.split(".")
        try:
            # Try to bump the last numeric component
            *major, patch = parts
            patch_int = int(patch)
            parts = major + [str(patch_int + 1)]
            return ".".join(parts)
        except (ValueError, IndexError):
            pass
        # Fallback: treat as simple float (e.g. '1.0')
        try:
            return str(float(version) + 0.1)
        except ValueError:
            # Last resort: append .1
            return f"{version}.1"

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

        new_args = [
            ArgChange(
                name=arg, change_type="added", details="New argument in vLLM source"
            )
            for arg in sorted(remote_args - local_args)
        ]

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
            "branch_or_tag": self.branch_or_tag,
        }

        if not dry_run and (new_args or removed_args):
            for arg in new_args:
                self.schema["arguments"][arg.name] = {
                    "type": "string",
                    "description": f"Synced from vLLM GitHub: {arg.details}",
                    "cli_flag": f"--{arg.name.replace('_', '-')}",
                    "importance": "medium",
                    "synced": True,
                }

            # Mark removed args as deprecated (don't remove for backward compat)
            for arg in removed_args:
                if arg.name in self.schema["arguments"]:
                    self.schema["arguments"][arg.name]["deprecated"] = True
                    self.schema["arguments"][arg.name]["deprecation_note"] = (
                        arg.details
                    )

            self.schema["version"] = self._bump_version(
                self.schema.get("version", "1.0.0")
            )

            with open(self.schema_path, "w") as f:
                json.dump(self.schema, f, indent=2)

            result["updated"] = True

        return result

    def clean_deprecated(self) -> dict:
        """Remove all deprecated arguments from the schema.

        Returns:
            Dict with cleanup results
        """
        deprecated_names = [
            name
            for name, info in self.schema.get("arguments", {}).items()
            if info.get("deprecated")
        ]

        for name in deprecated_names:
            del self.schema["arguments"][name]

        if deprecated_names:
            self.schema["version"] = self._bump_version(
                self.schema.get("version", "1.0.0")
            )
            with open(self.schema_path, "w") as f:
                json.dump(self.schema, f, indent=2)

        return {
            "success": True,
            "cleaned_count": len(deprecated_names),
            "cleaned_args": deprecated_names,
            "remaining_args": len(self.schema.get("arguments", {})),
        }


def sync_cli_args(
    dry_run: bool = True,
    verbose: bool = False,
    branch_or_tag: str = VLLM_DEFAULT_BRANCH,
) -> dict:
    """
    Convenience function to sync CLI args.

    Args:
        dry_run: If True, only report changes
        verbose: Print detailed output
        branch_or_tag: Git branch or tag to fetch from (default: main)

    Returns:
        Sync result dict
    """
    sync = CLIArgsSync(branch_or_tag=branch_or_tag)

    if verbose:
        cli_args_url, arg_utils_url = sync._get_source_urls()
        print("Fetching latest CLI arguments from vLLM GitHub...")
        print(f"  Branch/Tag: {branch_or_tag}")
        print(f"  Source: {cli_args_url}")
        print(f"  Source: {arg_utils_url}")

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
            print(f"\nSchema updated: {sync.schema_path}")

    return result


def clean_deprecated_args(verbose: bool = False) -> dict:
    """
    Convenience function to clean deprecated CLI args.

    Args:
        verbose: Print detailed output

    Returns:
        Cleanup result dict
    """
    sync = CLIArgsSync()
    result = sync.clean_deprecated()

    if verbose:
        print(f"Cleaning deprecated arguments...")
        print(f"  Cleaned: {result['cleaned_count']} arguments")
        print(f"  Remaining: {result['remaining_args']} arguments")
        if result["cleaned_args"]:
            print(f"  Removed: {', '.join(result['cleaned_args'][:10])}")
            if len(result["cleaned_args"]) > 10:
                print(f"    ... and {len(result['cleaned_args']) - 10} more")
        print(f"  Schema updated: {sync.schema_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync vLLM CLI arguments from GitHub")
    parser.add_argument(
        "--apply", action="store_true", help="Apply changes to local schema"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--tag",
        type=str,
        default=VLLM_DEFAULT_BRANCH,
        help=f"Git branch or tag to fetch from (default: {VLLM_DEFAULT_BRANCH}). "
        f"Available tags: {', '.join(SUPPORTED_VLLM_VERSIONS)}",
    )

    args = parser.parse_args()

    result = sync_cli_args(
        dry_run=not args.apply, verbose=args.verbose, branch_or_tag=args.tag
    )

    if result["new_count"] > 0 or result["removed_count"] > 0:
        print(f"\nRun with --apply to update the schema.")
