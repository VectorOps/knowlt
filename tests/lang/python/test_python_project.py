from pathlib import Path

import pytest

from knowlt.settings import ProjectSettings
from knowlt import init_project
from knowlt.parsers import CodeParserRegistry
from knowlt.lang.python import PythonCodeParser
from devtools import pprint
from knowlt.data import FileFilter, NodeFilter, PackageFilter


SAMPLES_DIR = Path(__file__).parent / "samples"


@pytest.mark.asyncio
async def test_project_scan_populates_repositories():
    """Project.init + directory scan should create metadata for every sample file."""
    project = await init_project(
        ProjectSettings(
            project_name="test", repo_name="test", repo_path=str(SAMPLES_DIR)
        )
    )
    repo_store = project.data
    repo_meta = project.default_repo

    # ── files ────────────────────────────────────────────────────────────
    files = await repo_store.file.get_list(FileFilter(repo_ids=[repo_meta.id]))
    assert len(files) == 3

    # ── packages ─────────────────────────────────────────────────────────
    pkg_ids = {f.package_id for f in files if f.package_id}
    # current Python parser creates one package per file
    assert len(pkg_ids) == 2

    # ── symbols (spot-check simple.py) ───────────────────────────────────
    simple_meta = next(f for f in files if f.path == "simple.py")
    symbols = await repo_store.node.get_list(NodeFilter(file_ids=[simple_meta.id]))
    symbol_names = {s.name for s in symbols}

    assert {"fn", "Test"}.issubset(symbol_names)
    # every recorded symbol must reference its file
    assert all(s.file_id == simple_meta.id for s in symbols)
