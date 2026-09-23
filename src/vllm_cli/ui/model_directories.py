#!/usr/bin/env python3
"""
Model directories management UI for vLLM CLI.

Provides an interface for managing model directories using hf-model-tool API.
"""
import logging
from pathlib import Path

from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..i18n import tr
from .common import console
from .navigation import prompt_choice, unified_prompt

logger = logging.getLogger(__name__)


class ModelDirectoriesUI:
    """UI component for managing model directories."""

    def __init__(self):
        """Initialize the Model Directories UI."""
        self.api = None
        self._init_api()

    def _init_api(self):
        """Initialize the hf-model-tool API."""
        try:
            import sys

            # Add hf-model-tool to path if needed
            hf_tool_path = Path("/home/chen/hf-model-tool")
            if hf_tool_path.exists() and str(hf_tool_path) not in sys.path:
                sys.path.insert(0, str(hf_tool_path))

            from hf_model_tool.api import HFModelAPI

            self.api = HFModelAPI()
            logger.info("Successfully initialized hf-model-tool API")
        except ImportError as e:
            logger.error(f"Failed to import hf-model-tool API: {e}")
            self.api = None
        except Exception as e:
            logger.error(f"Error initializing hf-model-tool API: {e}")
            self.api = None

    def show(self) -> str:
        """
        Show the model directories management interface.

        Returns:
            Action to take after directory management
        """
        if not self.api:
            console.print(
                Panel.fit(
                    tr(
                        "model_dirs.api_unavailable",
                        "[bold red]Error[/bold red]\n"
                        "[yellow]hf-model-tool is not available.[/yellow]\n\n"
                        "Please ensure hf-model-tool is installed or updated:\n"
                        "  [cyan]pip install --upgrade hf-model-tool[/cyan]",
                    ),
                    border_style="red",
                )
            )
            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            return "continue"

        while True:
            # Show header
            console.print(
                Panel.fit(
                    tr(
                        "model_dirs.header",
                        "[bold cyan]Model Directory Management[/bold cyan]\n"
                        "[dim]Configure directories for model discovery[/dim]",
                    ),
                    border_style="blue",
                )
            )

            # Display current directories
            self._display_directories()

            # Show menu options
            options = [
                ("add", tr("model_dirs.action_add", "Add Directory")),
                ("remove", tr("model_dirs.action_remove", "Remove Directory")),
                (
                    "toggle_ollama",
                    tr("model_dirs.action_toggle_ollama", "Toggle Ollama Scanning"),
                ),
                (
                    "scan_all",
                    tr("model_dirs.action_scan_all", "Scan All Directories"),
                ),
                (
                    "view_stats",
                    tr("model_dirs.action_view_stats", "View Directory Statistics"),
                ),
            ]

            action = prompt_choice(
                "model_directories",
                tr("model_dirs.directory_management", "Directory Management"),
                options,
                allow_back=True,
            )

            if action == "BACK" or not action:
                return "continue"
            elif action == "add":
                self._add_directory()
            elif action == "remove":
                self._remove_directory()
            elif action == "toggle_ollama":
                self._toggle_ollama_scanning()
            elif action == "scan_all":
                self._scan_directories()
            elif action == "view_stats":
                self._show_statistics()

    def _display_directories(self):
        """Display currently configured directories."""
        try:
            directories = self.api.list_directories()

            # Get Ollama status
            ollama_status = self.api.get_ollama_status()
            ollama_enabled = ollama_status.get("scan_enabled", False)

            # Display Ollama scanning status
            if ollama_enabled:
                console.print(
                    tr(
                        "model_dirs.ollama_scanning_enabled",
                        "\n[bold]Ollama Scanning:[/bold] [green]✓ Enabled[/green]",
                    )
                )
            else:
                console.print(
                    tr(
                        "model_dirs.ollama_scanning_disabled",
                        "\n[bold]Ollama Scanning:[/bold] [red]✗ Disabled[/red]",
                    )
                )

            if not directories:
                console.print(
                    tr(
                        "model_dirs.no_directories_configured",
                        "\n[yellow]No directories configured.[/yellow]",
                    )
                )
                console.print(
                    tr(
                        "model_dirs.using_default_locations",
                        "[dim]Using default locations only.[/dim]\n",
                    )
                )
                return

            # Create table for directories
            table = Table(
                title=tr(
                    "model_dirs.configured_directories", "[bold]Configured Directories[/bold]"
                ),
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("#", style="cyan", width=3)
            table.add_column(
                tr("model_dirs.column_path", "Path"), style="white"
            )
            table.add_column(
                tr("model_dirs.column_type", "Type"), style="yellow", width=15
            )
            table.add_column(
                tr("model_dirs.column_source", "Source"), style="magenta", width=15
            )
            table.add_column(
                tr("model_dirs.column_status", "Status"), style="green", width=10
            )

            for idx, dir_info in enumerate(directories, 1):
                path = dir_info.get("path", "")
                dir_type = dir_info.get("type", "custom")

                dir_source = dir_info.get("source", "unknown")

                # Format source for display
                if dir_source == "default_cache":
                    source_display = tr("model_dirs.source_default_hf", "Default HF")
                elif dir_source == "default_ollama":
                    source_display = tr(
                        "model_dirs.source_default_ollama", "Default Ollama"
                    )
                elif dir_source == "custom_ollama":
                    source_display = tr(
                        "model_dirs.source_custom_ollama", "Custom Ollama"
                    )
                elif dir_source.startswith("custom"):
                    source_display = tr("model_dirs.source_custom", "Custom")
                else:
                    source_display = dir_source.capitalize()

                # Check if directory exists
                path_obj = Path(path)
                status = (
                    tr("model_dirs.status_valid", "[green]✓ Valid[/green]")
                    if path_obj.exists()
                    else tr("model_dirs.status_missing", "[red]✗ Missing[/red]")
                )

                table.add_row(
                    str(idx), path, dir_type.capitalize(), source_display, status
                )

            console.print(table)
            console.print()

        except Exception as e:
            logger.error(f"Error displaying directories: {e}")
            console.print(
                tr(
                    "model_dirs.error_loading_directories",
                    "[red]Error loading directories: {error}[/red]\n",
                    error=e,
                )
            )

    def _add_directory(self):
        """Add a new directory for model scanning."""
        console.print(
            tr(
                "model_dirs.add_model_directory",
                "\n[bold cyan]Add Model Directory[/bold cyan]",
            )
        )

        # Get directory path
        console.print(
            tr(
                "model_dirs.enter_directory_path",
                "\nEnter the full path to the directory containing models:",
            )
        )
        console.print(
            tr(
                "model_dirs.path_example",
                "[dim]Example: /home/user/models or ~/my_models[/dim]",
            )
        )
        path = input(tr("model_dirs.path_prompt", "Path: ")).strip()

        if not path:
            console.print(
                tr(
                    "model_dirs.no_path_provided",
                    "[yellow]No path provided, cancelling.[/yellow]",
                )
            )
            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            return

        # Expand user path
        path = str(Path(path).expanduser().resolve())

        # Validate directory exists
        path_obj = Path(path)
        if not path_obj.exists():
            # Use unified prompt for confirmation
            create_choices = [
                ("yes", tr("model_dirs.option_create_directory", "Yes, create it")),
                ("no", tr("model_dirs.option_no_cancel", "No, cancel")),
            ]
            create_choice = prompt_choice(
                "create_dir",
                tr(
                    "model_dirs.create_directory_confirm",
                    "Directory '{path}' does not exist. Create it?",
                    path=path,
                ),
                create_choices,
                allow_back=False,
            )

            if create_choice == "yes":
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                    console.print(
                        tr(
                            "model_dirs.created_directory",
                            "\n[green]Created directory: {path}[/green]",
                            path=path,
                        )
                    )
                except Exception as e:
                    console.print(
                        tr(
                            "model_dirs.failed_create_directory",
                            "\n[red]Failed to create directory: {error}[/red]",
                            error=e,
                        )
                    )
                    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
                    return
            else:
                console.print(
                    tr(
                        "model_dirs.directory_must_exist",
                        "\n[yellow]Directory must exist. Cancelling.[/yellow]",
                    )
                )
                input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
                return

        # Ask for directory type using unified prompt
        type_choices = [
            ("auto", tr("model_dirs.type_auto", "Auto-detect (recommended)")),
            (
                "huggingface",
                tr("model_dirs.type_huggingface", "HuggingFace cache"),
            ),
            ("custom", tr("model_dirs.type_custom", "Custom models")),
            ("lora", tr("model_dirs.type_lora", "LoRA adapters")),
            ("ollama", tr("model_dirs.type_ollama", "Ollama models")),
        ]

        type_selection = prompt_choice(
            "dir_type",
            tr("model_dirs.select_directory_type", "Select directory type"),
            type_choices,
            allow_back=False,
        )

        # prompt_choice returns the stable API value used by add_directory()
        dir_type = type_selection or "auto"

        # Special handling for Ollama directories
        if dir_type == "ollama":
            # Check for Ollama structure
            has_manifests = (Path(path) / "manifests").exists()
            has_blobs = (Path(path) / "blobs").exists()

            if not (has_manifests and has_blobs):
                console.print(
                    tr(
                        "model_dirs.ollama_structure_warning",
                        "\n[yellow]Warning: Directory doesn't have standard Ollama "
                        "structure[/yellow]",
                    )
                )
                console.print(
                    tr(
                        "model_dirs.manifests_directory",
                        "  Manifests directory: {status}",
                        status="✓" if has_manifests else "✗",
                    )
                )
                console.print(
                    tr(
                        "model_dirs.blobs_directory",
                        "  Blobs directory: {status}",
                        status="✓" if has_blobs else "✗",
                    )
                )

                # Ask if they want to add it anyway
                confirm_choices = [
                    (
                        "yes",
                        tr("model_dirs.option_add_anyway", "Yes, add anyway"),
                    ),
                    ("no", tr("model_dirs.option_no_cancel", "No, cancel")),
                ]
                confirm_choice = prompt_choice(
                    "confirm_ollama",
                    tr(
                        "model_dirs.add_ollama_anyway_confirm",
                        "Add as Ollama directory anyway?",
                    ),
                    confirm_choices,
                    allow_back=False,
                )

                if confirm_choice != "yes":
                    console.print(
                        tr("model_dirs.cancelled", "\n[yellow]Cancelled[/yellow]")
                    )
                    input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
                    return

            # Add as Ollama directory
            try:
                success = self.api.add_ollama_directory(path)
                if success:
                    console.print(
                        tr(
                            "model_dirs.ollama_directory_added",
                            "\n[green]✓ Successfully added Ollama directory:[/green] {path}",
                            path=path,
                        )
                    )

                    # Enable Ollama scanning if not already enabled
                    ollama_status = self.api.get_ollama_status()
                    if not ollama_status.get("scan_enabled", False):
                        console.print(
                            tr(
                                "model_dirs.ollama_scanning_currently_disabled",
                                "\n[yellow]Note: Ollama scanning is currently disabled[/yellow]",
                            )
                        )
                        enable_choices = [
                            ("yes", tr("model_dirs.option_enable_now", "Yes, enable now")),
                            (
                                "no",
                                tr("model_dirs.option_keep_disabled", "No, keep disabled"),
                            ),
                        ]
                        enable_choice = prompt_choice(
                            "enable_ollama",
                            tr(
                                "model_dirs.enable_ollama_confirm",
                                "Enable Ollama scanning?",
                            ),
                            enable_choices,
                            allow_back=False,
                        )

                        if enable_choice == "yes":
                            self.api.toggle_ollama_scanning()
                            console.print(
                                tr(
                                    "model_dirs.ollama_scanning_turned_on",
                                    "[green]✓ Ollama scanning enabled[/green]",
                                )
                            )
                else:
                    console.print(
                        tr(
                            "model_dirs.failed_add_ollama_directory",
                            "\n[red]Failed to add Ollama directory (may already exist)[/red]",
                        )
                    )
            except Exception as e:
                console.print(
                    tr(
                        "model_dirs.error_adding_ollama_directory",
                        "\n[red]Error adding Ollama directory: {error}[/red]",
                        error=e,
                    )
                )

            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            return

        # Add regular directory
        try:
            success = self.api.add_directory(path, dir_type)
            if success:
                console.print(
                    tr(
                        "model_dirs.directory_added",
                        "\n[green]✓ Successfully added directory:[/green] {path}",
                        path=path,
                    )
                )

                # Inform about manifest file
                manifest_path = Path(path) / "models_manifest.json"
                console.print(
                    Panel.fit(
                        tr(
                            "model_dirs.manifest_note",
                            "[bold yellow]Note about Model Manifest:[/bold yellow]\n\n"
                            "A [cyan]models_manifest.json[/cyan] file will be "
                            "auto-generated at:\n"
                            "[dim]{path}[/dim]\n\n"
                            "You can manually edit this file to customize:\n"
                            "  • Custom display names\n"
                            "  • Model descriptions\n"
                            "  • Publisher information\n"
                            "  • Model categories\n\n"
                            "[dim]The manifest helps organize and customize how "
                            "models appear.[/dim]",
                            path=manifest_path,
                        ),
                        border_style="yellow",
                    )
                )

                # Offer to scan immediately using unified prompt
                scan_choices = [
                    ("yes", tr("model_dirs.option_scan_now", "Yes, scan now")),
                    ("no", tr("model_dirs.option_scan_later", "No, scan later")),
                ]
                scan_choice = prompt_choice(
                    "scan_now",
                    tr(
                        "model_dirs.scan_directory_confirm",
                        "Scan this directory for models now?",
                    ),
                    scan_choices,
                    allow_back=False,
                )

                if scan_choice == "yes":
                    self._scan_single_directory(path)

                    # After scanning, remind about manifest if models were found
                    console.print(
                        tr(
                            "model_dirs.manifest_generated_tip",
                            "\n[dim]Tip: A models_manifest.json file has been "
                            "auto-generated.[/dim]",
                        )
                    )
                    console.print(
                        tr(
                            "model_dirs.manifest_edit_tip",
                            "[dim]You can edit it to customize how models appear in "
                            "the serving menu.[/dim]",
                        )
                    )
            else:
                console.print(
                    tr(
                        "model_dirs.failed_add_directory",
                        "\n[red]Failed to add directory.[/red]",
                    )
                )
        except Exception as e:
            console.print(
                tr(
                    "model_dirs.error_adding_directory",
                    "\n[red]Error adding directory: {error}[/red]",
                    error=e,
                )
            )

        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")

    def _remove_directory(self):
        """Remove a directory from scanning."""
        directories = self.api.list_directories()

        if not directories:
            console.print(
                tr(
                    "model_dirs.no_directories_to_remove",
                    "\n[yellow]No directories to remove.[/yellow]",
                )
            )
            input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")
            return

        console.print(
            tr(
                "model_dirs.remove_directory_title",
                "\n[bold cyan]Remove Directory[/bold cyan]",
            )
        )

        # Build choices list with directory paths
        dir_choices = []
        for dir_info in directories:
            path = dir_info.get("path", "")
            dir_type = dir_info.get("type", "custom")
            dir_choices.append(f"{path} [{dir_type}]")

        # Use unified prompt for selection
        selected = unified_prompt(
            "remove_dir",
            tr("model_dirs.select_directory_to_remove", "Select directory to remove"),
            dir_choices,
            allow_back=True,
        )

        if not selected or selected == "BACK":
            return

        # Extract the path from the selection
        selected_path = selected.split(" [")[0]  # Remove the [type] suffix

        # Find the matching directory
        for dir_info in directories:
            if dir_info.get("path", "") == selected_path:
                # Confirm removal using unified prompt
                confirm_choices = [
                    (
                        "yes",
                        tr(
                            "model_dirs.option_remove_directory",
                            "Yes, remove this directory",
                        ),
                    ),
                    ("no", tr("model_dirs.option_keep_it", "No, keep it")),
                ]
                confirm = prompt_choice(
                    "confirm_remove",
                    tr(
                        "model_dirs.remove_directory_confirm",
                        "Remove {path}?",
                        path=selected_path,
                    ),
                    confirm_choices,
                    allow_back=False,
                )

                if confirm == "yes":
                    try:
                        # Check if it's an Ollama directory
                        dir_source = dir_info.get("source", "")
                        if dir_info.get("type") == "ollama" and "ollama" in dir_source:
                            # Remove as Ollama directory
                            success = self.api.remove_ollama_directory(selected_path)
                        else:
                            # Remove as regular directory
                            success = self.api.remove_directory(selected_path)

                        if success:
                            console.print(
                                tr(
                                    "model_dirs.directory_removed",
                                    "\n[green]✓ Removed directory: {path}[/green]",
                                    path=selected_path,
                                )
                            )
                        else:
                            console.print(
                                tr(
                                    "model_dirs.failed_remove_directory",
                                    "\n[red]Failed to remove directory.[/red]",
                                )
                            )
                    except Exception as e:
                        console.print(
                            tr(
                                "model_dirs.error_removing_directory",
                                "\n[red]Error removing directory: {error}[/red]",
                                error=e,
                            )
                        )
                else:
                    console.print(
                        tr(
                            "model_dirs.directory_not_removed",
                            "\n[yellow]Directory not removed.[/yellow]",
                        )
                    )
                break

        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")

    def _scan_directories(self):
        """Scan all configured directories for models."""
        console.print(
            tr(
                "model_dirs.scanning_directories",
                "\n[bold cyan]Scanning Directories[/bold cyan]",
            )
        )
        console.print(
            tr(
                "model_dirs.scanning_note",
                "[dim]This may take a moment for large directories...[/dim]\n",
            )
        )

        try:
            # Force refresh to get latest data
            assets = self.api.list_assets(force_refresh=True)

            # Group by type
            models = [a for a in assets if a.get("type") == "model"]
            custom_models = [a for a in assets if a.get("type") == "custom_model"]
            lora_adapters = [a for a in assets if a.get("type") == "lora_adapter"]
            datasets = [a for a in assets if a.get("type") == "dataset"]

            # Display summary
            console.print(tr("model_dirs.scan_results", "[bold]Scan Results:[/bold]"))
            console.print(
                tr(
                    "model_dirs.scan_models",
                    "  Models: [cyan]{count}[/cyan]",
                    count=len(models),
                )
            )
            console.print(
                tr(
                    "model_dirs.scan_custom_models",
                    "  Custom Models: [cyan]{count}[/cyan]",
                    count=len(custom_models),
                )
            )
            console.print(
                tr(
                    "model_dirs.scan_lora_adapters",
                    "  LoRA Adapters: [cyan]{count}[/cyan]",
                    count=len(lora_adapters),
                )
            )
            console.print(
                tr(
                    "model_dirs.scan_datasets",
                    "  Datasets: [cyan]{count}[/cyan]",
                    count=len(datasets),
                )
            )
            console.print(
                tr(
                    "model_dirs.scan_total_assets",
                    "  [bold]Total Assets: [green]{count}[/green][/bold]",
                    count=len(assets),
                )
            )

            # Show top models by size
            if models or custom_models:
                all_models = models + custom_models
                all_models.sort(key=lambda x: x.get("size", 0), reverse=True)

                console.print(
                    tr(
                        "model_dirs.top_models_by_size",
                        "\n[bold]Top Models by Size:[/bold]",
                    )
                )
                for model in all_models[:5]:
                    name = model.get("display_name", model.get("name", "Unknown"))
                    size = model.get("size", 0)
                    size_gb = size / (1024**3)
                    console.print(f"  • {name}: [yellow]{size_gb:.2f} GB[/yellow]")

        except Exception as e:
            console.print(
                tr(
                    "model_dirs.error_scanning_directories",
                    "[red]Error scanning directories: {error}[/red]",
                    error=e,
                )
            )

        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")

    def _scan_single_directory(self, path: str):
        """Scan a single directory for models."""
        console.print(
            tr("model_dirs.scanning_path", "\n[cyan]Scanning: {path}[/cyan]", path=path)
        )

        try:
            assets = self.api.scan_directories([path])

            if assets:
                console.print(
                    tr(
                        "model_dirs.found_assets",
                        "[green]Found {count} asset(s):[/green]",
                        count=len(assets),
                    )
                )
                for asset in assets[:5]:  # Show first 5
                    name = asset.get("display_name", asset.get("name", "Unknown"))
                    asset_type = asset.get("type", "unknown")
                    console.print(f"  • {name} ([yellow]{asset_type}[/yellow])")
                if len(assets) > 5:
                    console.print(
                        tr(
                            "model_dirs.more_assets",
                            "  [dim]... and {count} more[/dim]",
                            count=len(assets) - 5,
                        )
                    )
            else:
                console.print(
                    tr(
                        "model_dirs.no_models_found",
                        "[yellow]No models found in this directory.[/yellow]",
                    )
                )

        except Exception as e:
            console.print(
                tr(
                    "model_dirs.error_scanning_directory",
                    "[red]Error scanning directory: {error}[/red]",
                    error=e,
                )
            )

    def _toggle_ollama_scanning(self):
        """Toggle Ollama model scanning on/off."""
        console.print(
            tr(
                "model_dirs.ollama_model_scanning_title",
                "\n[bold cyan]Ollama Model Scanning[/bold cyan]",
            )
        )

        try:
            # Get current status
            ollama_status = self.api.get_ollama_status()
            current_state = ollama_status.get("scan_enabled", False)

            # Display current state
            if current_state:
                console.print(
                    tr(
                        "model_dirs.ollama_currently_enabled",
                        "\nOllama scanning is currently: [green]✓ Enabled[/green]",
                    )
                )
                console.print(
                    tr(
                        "model_dirs.default_ollama_directories",
                        "\nDefault Ollama directories being scanned:",
                    )
                )
                for dir_path in ollama_status.get("default_directories", []):
                    if Path(dir_path).exists():
                        console.print(f"  • {dir_path}")
            else:
                console.print(
                    tr(
                        "model_dirs.ollama_currently_disabled",
                        "\nOllama scanning is currently: [red]✗ Disabled[/red]",
                    )
                )
                console.print(
                    tr(
                        "model_dirs.enable_to_scan_hint",
                        "\n[dim]Enable to scan Ollama model directories[/dim]",
                    )
                )

            # Ask to toggle
            toggle_choices = [
                (
                    "yes",
                    tr("model_dirs.option_toggle_it", "Yes, toggle it")
                    if current_state
                    else tr("model_dirs.option_enable_it", "Yes, enable it"),
                ),
                (
                    "no",
                    tr(
                        "model_dirs.option_keep_current_setting",
                        "No, keep current setting",
                    ),
                ),
            ]

            toggle_choice = prompt_choice(
                "toggle_ollama",
                tr("model_dirs.disable_ollama_confirm", "Disable Ollama scanning?")
                if current_state
                else tr("model_dirs.enable_ollama_confirm", "Enable Ollama scanning?"),
                toggle_choices,
                allow_back=False,
            )

            if toggle_choice == "yes":
                new_state = self.api.toggle_ollama_scanning()

                if new_state:
                    console.print(
                        tr(
                            "model_dirs.ollama_scanning_on",
                            "\n[green]✓ Ollama scanning enabled[/green]",
                        )
                    )
                    console.print(
                        tr(
                            "model_dirs.ollama_models_included",
                            "[dim]Ollama models will now be included in scans[/dim]",
                        )
                    )
                else:
                    console.print(
                        tr(
                            "model_dirs.ollama_scanning_off",
                            "\n[yellow]Ollama scanning disabled[/yellow]",
                        )
                    )
                    console.print(
                        tr(
                            "model_dirs.ollama_models_excluded",
                            "[dim]Ollama models will be excluded from scans[/dim]",
                        )
                    )

                # Offer to refresh cache
                refresh_choices = [
                    ("yes", tr("model_dirs.option_refresh_now", "Yes, refresh now")),
                    ("no", tr("model_dirs.option_refresh_later", "No, refresh later")),
                ]
                refresh_choice = prompt_choice(
                    "refresh_cache",
                    tr(
                        "model_dirs.refresh_cache_confirm",
                        "Refresh model cache to apply changes?",
                    ),
                    refresh_choices,
                    allow_back=False,
                )

                if refresh_choice == "yes":
                    console.print(
                        tr(
                            "model_dirs.refreshing_cache",
                            "\n[cyan]Refreshing model cache...[/cyan]",
                        )
                    )

                    # Import and use model manager to refresh
                    try:
                        from ..models import get_model_manager

                        model_manager = get_model_manager()
                        model_manager.refresh_cache()
                        console.print(
                            tr(
                                "model_dirs.model_cache_refreshed",
                                "[green]✓ Model cache refreshed[/green]",
                            )
                        )
                    except Exception as e:
                        console.print(
                            tr(
                                "model_dirs.failed_refresh_cache",
                                "[red]Failed to refresh cache: {error}[/red]",
                                error=e,
                            )
                        )
            else:
                console.print(
                    tr(
                        "model_dirs.settings_unchanged",
                        "\n[yellow]Settings unchanged[/yellow]",
                    )
                )

        except Exception as e:
            console.print(
                tr(
                    "model_dirs.error_toggling_ollama",
                    "\n[red]Error toggling Ollama scanning: {error}[/red]",
                    error=e,
                )
            )

        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")

    def _show_statistics(self):
        """Show statistics about managed assets."""
        console.print(
            tr(
                "model_dirs.asset_statistics",
                "\n[bold cyan]Asset Statistics[/bold cyan]\n",
            )
        )

        try:
            stats = self.api.get_statistics()

            # Create statistics panels
            panels = []

            # Models panel
            model_text = Text()
            model_text.append(
                tr("model_dirs.stats_models_title", "Models\n"), style="bold yellow"
            )
            model_text.append(
                tr(
                    "model_dirs.stats_models_total",
                    "Total: {count}\n",
                    count=stats.get("total_models", 0),
                )
            )
            model_text.append(
                tr(
                    "model_dirs.stats_models_custom",
                    "Custom: {count}\n",
                    count=stats.get("custom_models", 0),
                )
            )
            model_text.append(
                tr(
                    "model_dirs.stats_models_lora",
                    "LoRA: {count}",
                    count=stats.get("lora_adapters", 0),
                )
            )
            panels.append(Panel(model_text, border_style="yellow"))

            # Storage panel
            storage_text = Text()
            storage_text.append(
                tr("model_dirs.stats_storage_title", "Storage\n"), style="bold cyan"
            )
            total_size = stats.get("total_size", 0)
            size_gb = total_size / (1024**3)
            storage_text.append(
                tr(
                    "model_dirs.stats_storage_total",
                    "Total: {size} GB\n",
                    size=f"{size_gb:.2f}",
                )
            )
            storage_text.append(
                tr(
                    "model_dirs.stats_storage_datasets",
                    "Datasets: {count}",
                    count=stats.get("dataset_count", 0),
                )
            )
            panels.append(Panel(storage_text, border_style="cyan"))

            # Directories panel
            dir_text = Text()
            dir_text.append(
                tr("model_dirs.stats_directories_title", "Directories\n"),
                style="bold green",
            )
            dir_text.append(
                tr(
                    "model_dirs.stats_directories_monitored",
                    "Monitored: {count}\n",
                    count=stats.get("directories", 0),
                )
            )
            dir_text.append(
                tr("model_dirs.stats_last_scan", "Last scan: Recent")
            )
            panels.append(Panel(dir_text, border_style="green"))

            console.print(Columns(panels))

            # Show breakdown by directory if available
            if "by_directory" in stats:
                console.print(
                    tr("model_dirs.by_directory", "\n[bold]By Directory:[/bold]")
                )
                for dir_path, dir_stats in stats["by_directory"].items():
                    console.print(f"\n  [cyan]{dir_path}[/cyan]")
                    console.print(
                        tr(
                            "model_dirs.directory_models",
                            "    Models: {count}",
                            count=dir_stats.get("models", 0),
                        )
                    )
                    console.print(
                        tr(
                            "model_dirs.directory_size",
                            "    Size: {size} GB",
                            size=f"{dir_stats.get('size', 0) / (1024**3):.2f}",
                        )
                    )

        except Exception as e:
            console.print(
                tr(
                    "model_dirs.error_getting_statistics",
                    "[red]Error getting statistics: {error}[/red]",
                    error=e,
                )
            )

        input(f"\n{tr('common.press_enter', 'Press Enter to continue...')}")


def manage_model_directories() -> str:
    """
    Main entry point for model directory management.

    Returns:
        Action to take after management
    """
    ui = ModelDirectoriesUI()
    return ui.show()
