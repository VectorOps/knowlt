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
from knowlt.data import FileFilter, PackageFilter, NodeFilter, ImportEdgeFilter


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
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging to print scanner performance stats.",
)
def main(
    source: Path, project_name: Optional[str], repo_name: Optional[str], debug: bool
) -> None:
    """
    Parse a project directory into an in-memory DuckDB and print usage stats.
    """
    _setup_logging(debug)

    # Resolve names and ensure path correctness
    src = source.resolve()
    default_name = src.name
    project_name = project_name or default_name
    repo_name = repo_name or default_name

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
                payload = model.model_dump() if hasattr(model, "model_dump") else dict(model)  # type: ignore[arg-type]
            except Exception:
                payload = {"progress": str(model)}
            logger.debug("scan_progress", **payload)

        pm = None
        try:
            pm = await ProjectManager.create(settings=settings, data=data)

            # Ensure repo is present; standard behavior will handle .gitignore, etc.
            await pm.add_repo_path(name=repo_name, path=str(src))

            # Trigger scan/parse
            scan_result = await pm.refresh(progress_callback=progress_callback)

            click.echo(f"Parsed repository: {repo_name}")
            # If ScanResult available, print basic file operation stats
            try:
                if scan_result is not None:
                    added = len(getattr(scan_result, "files_added", []) or [])
                    updated = len(getattr(scan_result, "files_updated", []) or [])
                    deleted = len(getattr(scan_result, "files_deleted", []) or [])
                    click.echo(f"- Files added:   {added}")
                    click.echo(f"- Files updated: {updated}")
                    click.echo(f"- Files deleted: {deleted}")
            except Exception:
                pass

            # Best-effort usage counts via repositories
            try:
                files = await data.file.get_list(FileFilter())
                packages = await data.package.get_list(PackageFilter())
                nodes = await data.node.get_list(NodeFilter())
                imports = await data.importedge.get_list(ImportEdgeFilter())
                click.echo("Usage counts (current DB):")
                click.echo(f"- Projects:      1")
                click.echo(f"- Repos:         1")
                click.echo(f"- Packages:      {len(packages)}")
                click.echo(f"- Files:         {len(files)}")
                click.echo(f"- Nodes:         {len(nodes)}")
                click.echo(f"- Import edges:  {len(imports)}")
            except Exception as e:
                logger.debug("usage_counts_error", error=str(e))

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
