#!/usr/bin/env python3
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from knowlt.logger import logger
from knowlt.settings import ProjectSettings
from knowlt.stores.duckdb import DuckDBDataRepository
from knowlt.project import ProjectManager
from knowlt.summary import SummaryMode, build_file_summary


def _setup_logging(debug: bool) -> None:
    # Ensure stdlib logger emits records so structlog output is visible
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        # structlog renders the final message; keep stdlib formatter simple.
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setLevel(level)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "source",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
)
@click.argument(
    "file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
)
@click.option(
    "--project-name",
    type=str,
    default=None,
    help="Project name (default: source directory name).",
)
@click.option(
    "--repo-name",
    type=str,
    default=None,
    help="Repository name (default: source directory name).",
)
@click.option(
    "--summary-mode",
    type=click.Choice(["definition", "documentation", "source"], case_sensitive=False),
    default="documentation",
    show_default=True,
    help="Summary granularity: definitions, documentation, or full source.",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging to print scanner performance stats.",
)
def main(
    source: Path,
    file: Path,
    project_name: Optional[str],
    repo_name: Optional[str],
    summary_mode: str,
    debug: bool,
) -> None:
    """
    Parse an individual file and print its summary.
    """
    _setup_logging(debug)

    src = source.resolve()
    file_path = file.resolve()

    try:
        rel_path = file_path.relative_to(src)
    except ValueError:
        click.echo("Error: FILE must be inside SOURCE directory.", err=True)
        raise SystemExit(1)

    default_name = src.name
    project_name = project_name or default_name
    repo_name = repo_name or default_name

    # Normalize summary mode to SummaryMode enum
    try:
        mode_enum = SummaryMode(summary_mode.lower())
    except ValueError:
        click.echo(f"Invalid summary mode: {summary_mode}", err=True)
        raise SystemExit(1)

    async def _run() -> int:
        # Configure project settings: in-memory duckdb
        settings = ProjectSettings(
            project_name=project_name,
            repo_name=repo_name,
            repo_path=str(src),
            repository_backend="duckdb",
            repository_connection=None,  # in-memory
        )

        data = DuckDBDataRepository(settings=settings, db_path=None)

        # Progress callback: rely on debug logs for detailed scanner stats
        def progress_callback(model) -> None:
            try:
                payload = (
                    model.model_dump()
                    if hasattr(model, "model_dump")
                    else dict(model)  # type: ignore[arg-type]
                )
            except Exception:
                payload = {"progress": str(model)}
            logger.debug("scan_progress", **payload)

        pm: Optional[ProjectManager] = None
        try:
            pm = await ProjectManager.create(settings=settings, data=data)

            # Ensure default repo exists and matches the source path
            repo = pm.default_repo

            # Refresh just the target file if supported by the scanner
            await pm.refresh(repo=repo, paths=[str(rel_path)])

            summary = await build_file_summary(
                pm, repo, str(rel_path), summary_mode=mode_enum
            )

            if not summary:
                click.echo(
                    "No summary available for the requested file "
                    "(file might be ignored or unknown to the repository).",
                    err=True,
                )
                return 1

            click.echo(summary.content)
            return 0
        finally:
            # Cleanup
            try:
                if pm is not None:
                    await pm.destroy()
            except Exception:
                pass
            try:
                data.close()
            except Exception:
                pass

    exit_code = asyncio.run(_run())
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()