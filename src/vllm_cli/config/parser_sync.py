#!/usr/bin/env python3
"""
Parser Sync Module

Fetches the latest tool_call_parser and reasoning_parser choices from vLLM
GitHub source code and updates the local schema.
"""

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# GitHub URLs for vLLM source directories
VLLM_TOOL_PARSERS_URL = (
    "https://api.github.com/repos/vllm-project/vllm/contents/vllm/tool_parsers"
)
VLLM_REASONING_URL = (
    "https://api.github.com/repos/vllm-project/vllm/contents/vllm/reasoning"
)
VLLM_RAW_BASE = "https://raw.githubusercontent.com/vllm-project/vllm/main"

TOOL_PARSER_NAME_MAP = {
    "deepseekv3": "deepseek_v3",
    "deepseekv31": "deepseek_v31",
    "deepseekv32": "deepseek_v32",
    "granite_20b_fc": "granite-20b-fc",
    "glm4_moe": "glm45",
    "glm47_moe": "glm45",
    "llama4_pythonic": "llama3_json",
    "minimax": "minimax_m2",
}

REASONING_PARSER_NAME_MAP = {
    "gptoss": "GptOss",
    "seedoss": "GptOss",
    "step3p5": "step3",
}


@dataclass
class ParserChange:
    """Represents a change in parser choices."""

    parser_type: str  # 'tool_call_parser' or 'reasoning_parser'
    name: str
    change_type: str  # 'added', 'removed'
    details: str = ""


class ParserSync:
    """Syncs parser choices from vLLM GitHub repository."""

    def __init__(self, schema_path: Optional[Path] = None):
        if schema_path is None:
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

    def _fetch_url(self, url: str) -> str:
        """Fetch content from URL."""
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

    def _extract_parser_names_from_files(
        self, file_list: list, parser_type: str
    ) -> list[str]:
        """Extract parser names from GitHub API file list.

        Args:
            file_list: List of file dicts from GitHub API
            parser_type: 'tool_call_parser' or 'reasoning_parser'

        Returns:
            List of parser names (e.g., 'gemma4', 'deepseek_r1')
        """
        parsers = []

        for file_info in file_list:
            name = file_info.get("name", "")

            # Skip non-Python files and base classes
            if not name.endswith(".py"):
                continue
            if name in (
                "__init__.py",
                "abstract_tool_parser.py",
                "abs_reasoning_parsers.py",
                "basic_parsers.py",
                "utils.py",
                "gemma4_utils.py",
            ):
                continue

            # Extract parser name from filename
            # Format: {model}_tool_parser.py or {model}_reasoning_parser.py
            suffix = f"_{parser_type}.py"
            if name.endswith(suffix):
                parser_name = name[: -len(suffix)]
            else:
                continue

            # Skip certain patterns
            if parser_name in ("llama", "longcat", "internlm2"):
                continue  # Not registered as CLI parsers

            parsers.append(parser_name)

        return sorted(parsers)

    def _convert_parser_names(self, parsers: list[str], parser_type: str) -> list[str]:
        """Convert remote parser names to local format."""
        name_map = (
            TOOL_PARSER_NAME_MAP if parser_type == "tool" else REASONING_PARSER_NAME_MAP
        )
        converted = []
        for p in parsers:
            converted.append(name_map.get(p, p))
        return converted

    def fetch_remote_parsers(self) -> tuple[list[str], list[str]]:
        """Fetch available parsers from GitHub.

        Returns:
            Tuple of (tool_call_parser choices, reasoning_parser choices)
        """
        import json

        # Fetch tool_parsers directory
        tool_content = self._fetch_url(VLLM_TOOL_PARSERS_URL)
        tool_files = json.loads(tool_content)

        # Fetch reasoning directory
        reasoning_content = self._fetch_url(VLLM_REASONING_URL)
        reasoning_files = json.loads(reasoning_content)

        # Extract parser names
        tool_parsers = self._extract_parser_names_from_files(tool_files, "tool_parser")
        reasoning_parsers = self._extract_parser_names_from_files(
            reasoning_files, "reasoning_parser"
        )

        # Convert to local naming format, deduplicate preserving order
        tool_parsers = list(
            dict.fromkeys(self._convert_parser_names(tool_parsers, "tool"))
        )
        reasoning_parsers = list(
            dict.fromkeys(self._convert_parser_names(reasoning_parsers, "reasoning"))
        )

        return tool_parsers, reasoning_parsers

    def get_local_parsers(self) -> tuple[list[str], list[str]]:
        """Get local parser choices from schema."""
        args = self.schema.get("arguments", {})

        tool_call_parser = args.get("tool_call_parser", {})
        tool_parsers = [x for x in tool_call_parser.get("choices", []) if x is not None]

        reasoning_parser = args.get("reasoning_parser", {})
        reasoning_parsers = [
            x for x in reasoning_parser.get("choices", []) if x is not None
        ]

        return tool_parsers, reasoning_parsers

    def compare(self) -> tuple[list[ParserChange], list[ParserChange]]:
        """Compare local and remote parsers.

        Returns:
            Tuple of (tool_parser_changes, reasoning_parser_changes)
        """
        remote_tool, remote_reasoning = self.fetch_remote_parsers()
        local_tool, local_reasoning = self.get_local_parsers()

        # Find new tool parsers
        remote_tool_set = set(remote_tool)
        local_tool_set = set(local_tool)

        tool_changes = []
        for parser in sorted(remote_tool_set - local_tool_set):
            tool_changes.append(
                ParserChange(
                    parser_type="tool_call_parser",
                    name=parser,
                    change_type="added",
                    details=f"New tool parser in vLLM: {parser}",
                )
            )

        # Find removed tool parsers
        for parser in sorted(local_tool_set - remote_tool_set):
            tool_changes.append(
                ParserChange(
                    parser_type="tool_call_parser",
                    name=parser,
                    change_type="removed",
                    details=f"Tool parser removed from vLLM: {parser}",
                )
            )

        # Find new reasoning parsers
        remote_reasoning_set = set(remote_reasoning)
        local_reasoning_set = set(local_reasoning)

        reasoning_changes = []
        for parser in sorted(remote_reasoning_set - local_reasoning_set):
            reasoning_changes.append(
                ParserChange(
                    parser_type="reasoning_parser",
                    name=parser,
                    change_type="added",
                    details=f"New reasoning parser in vLLM: {parser}",
                )
            )

        # Find removed reasoning parsers
        for parser in sorted(local_reasoning_set - remote_reasoning_set):
            reasoning_changes.append(
                ParserChange(
                    parser_type="reasoning_parser",
                    name=parser,
                    change_type="removed",
                    details=f"Reasoning parser removed from vLLM: {parser}",
                )
            )

        return tool_changes, reasoning_changes

    def sync(self, dry_run: bool = True) -> dict:
        """Sync local schema with remote parsers.

        Args:
            dry_run: If True, only report changes without updating

        Returns:
            Dict with sync results
        """
        tool_changes, reasoning_changes = self.compare()
        remote_tool, remote_reasoning = self.fetch_remote_parsers()
        local_tool, local_reasoning = self.get_local_parsers()

        result = {
            "success": True,
            "dry_run": dry_run,
            "tool_call_parser": {
                "remote": remote_tool,
                "local": list(local_tool),
                "new": [c.name for c in tool_changes if c.change_type == "added"],
                "removed": [c.name for c in tool_changes if c.change_type == "removed"],
            },
            "reasoning_parser": {
                "remote": remote_reasoning,
                "local": list(local_reasoning),
                "new": [c.name for c in reasoning_changes if c.change_type == "added"],
                "removed": [
                    c.name for c in reasoning_changes if c.change_type == "removed"
                ],
            },
        }

        if not dry_run and (tool_changes or reasoning_changes):
            # Update tool_call_parser choices
            if "tool_call_parser" in self.schema["arguments"]:
                self.schema["arguments"]["tool_call_parser"]["choices"] = remote_tool

            # Update reasoning_parser choices
            if "reasoning_parser" in self.schema["arguments"]:
                self.schema["arguments"]["reasoning_parser"]["choices"] = (
                    remote_reasoning
                )

            # Update schema version
            self.schema["version"] = self.schema.get("version", "1.0") + ".1"

            # Save updated schema
            with open(self.schema_path, "w") as f:
                json.dump(self.schema, f, indent=2)

            result["updated"] = True

        return result


def sync_parsers(dry_run: bool = True, verbose: bool = False) -> dict:
    """Convenience function to sync parsers.

    Args:
        dry_run: If True, only report changes
        verbose: Print detailed output

    Returns:
        Sync result dict
    """
    sync = ParserSync()

    if verbose:
        print("Fetching latest parsers from vLLM GitHub...")
        print(f"  Tool parsers: {VLLM_TOOL_PARSERS_URL}")
        print(f"  Reasoning parsers: {VLLM_REASONING_URL}")

    result = sync.sync(dry_run=dry_run)

    if verbose:
        print(f"\n=== Tool Call Parser ===")
        print(f"  Remote: {len(result['tool_call_parser']['remote'])}")
        print(f"  Local:  {len(result['tool_call_parser']['local'])}")
        if result["tool_call_parser"]["new"]:
            print(f"  New:    {', '.join(result['tool_call_parser']['new'])}")

        print(f"\n=== Reasoning Parser ===")
        print(f"  Remote: {len(result['reasoning_parser']['remote'])}")
        print(f"  Local:  {len(result['reasoning_parser']['local'])}")
        if result["reasoning_parser"]["new"]:
            print(f"  New:    {', '.join(result['reasoning_parser']['new'])}")

        if not dry_run and result.get("updated"):
            print(f"\n✓ Schema updated: {sync.schema_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync vLLM parsers from GitHub")
    parser.add_argument(
        "--apply", action="store_true", help="Apply changes to local schema"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    result = sync_parsers(dry_run=not args.apply, verbose=args.verbose)

    new_tool = len(result["tool_call_parser"]["new"])
    new_reasoning = len(result["reasoning_parser"]["new"])

    if new_tool > 0 or new_reasoning > 0:
        print(f"\nRun with --apply to update the schema.")
