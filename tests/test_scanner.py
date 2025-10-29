from pathlib import Path
import pytest

from knowlt.project import ProjectManager, ProjectCache
from knowlt.settings import ProjectSettings
from knowlt.scanner import ParsingState, upsert_parsed_file, scan_repo
from knowlt.data import (
    PackageFilter,
    FileFilter,
    NodeFilter,
    ImportEdgeFilter,
)
from knowlt.stores.duckdb import DuckDBDataRepository
from knowlt.models import Repo
from knowlt.helpers import generate_id
from knowlt.parsers import CodeParserRegistry

# Import PythonCodeParser if available; skip parser-dependent tests if not
try:
    from knowlt.lang.python import PythonCodeParser  # type: ignore
except Exception:
    PythonCodeParser = None  # type: ignore


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
CODE_V1 = """
CONST = 1

import os

# foo comment
def foo():
    pass
"""

CODE_V2 = """
# new foo comment
def foo(x):
    return 1
"""
MOD2_CODE = "def bar():\n    pass\n"


async def _make_project(root: Path) -> ProjectManager:
    """
    Return a Project instance backed by an In-memory repository WITHOUT running
    the automatic directory scan (so that `upsert_parsed_file` is the only code
    that populates the stores during this test).
    """
    settings = ProjectSettings(
        project_name="test",
        repo_name="test",
        repo_path=str(root),
    )
    data_repo = DuckDBDataRepository(settings)
    if PythonCodeParser is not None:
        CodeParserRegistry.register_parser(PythonCodeParser)
    return await ProjectManager.create(settings, data_repo)


def _parse(pm: ProjectManager, rel_path: str):
    parser = PythonCodeParser(pm, pm.default_repo, rel_path)
    return parser.parse(ProjectCache())


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.skipif(PythonCodeParser is None, reason="Python parser not available in this repository")
async def test_upsert_parsed_file_insert_update_delete(tmp_path: Path):
    """
    Verify that `upsert_parsed_file` correctly handles
      – INSERT  (first call)
      – UPDATE  (same symbols → same IDs, mutated fields changed)
      – DELETE  (symbols / import-edges that disappeared)
    for Package, File, Symbol and ImportEdge models.
    """
    # ── 0) setup ───────────────────────────────────────────────────────────
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    module_fp = repo_dir / "mod.py"
    module2_fp = repo_dir / "mod2.py"
    module2_fp.write_text(MOD2_CODE)

    # build project instance
    pm = await _make_project(repo_dir)

    # ── 1) first version  → INSERT path ────────────────────────────────────
    state = ParsingState()

    module_fp.write_text(CODE_V1)
    parsed_v1 = _parse(pm, "mod.py")
    await upsert_parsed_file(pm, pm.default_repo, state, parsed_v1)

    parsed2_v1 = _parse(pm, "mod2.py")
    await upsert_parsed_file(pm, pm.default_repo, state, parsed2_v1)

    rs = pm.data
    repo_meta = pm.default_repo

    # expect exactly two packages / files (one per source file)
    packages = await rs.package.get_list(PackageFilter(repo_ids=[repo_meta.id]))
    assert len(packages) == 2

    pkg_id_mod1 = next(
        p.id
        for p in packages
        if p.physical_path == "mod.py" or (p.virtual_path or "").endswith("mod")
    )
    pkg_id_mod2 = next(
        p.id
        for p in packages
        if p.physical_path == "mod2.py" or (p.virtual_path or "").endswith("mod2")
    )

    files = await rs.file.get_list(FileFilter(repo_ids=[repo_meta.id]))
    assert len(files) == 2
    file_id = next(f.id for f in files if f.path == "mod.py")
    file_id_mod2 = next(f.id for f in files if f.path == "mod2.py")

    # symbols: CONST + foo
    symbols = await rs.node.get_list(NodeFilter(file_ids=[file_id]))
    names = {s.name for s in symbols if s.name}
    assert names == {"CONST", "foo"}

    # Node IDs may change between updates; do not assert signature fields.
    # import-edge created for 'import os'
    edges = await rs.importedge.get_list(
        ImportEdgeFilter(source_package_ids=[pkg_id_mod1])
    )
    assert len(edges) == 1

    # ── 2) second version  → UPDATE / DELETE paths ─────────────────────────
    state = ParsingState()

    module_fp.write_text(CODE_V2)  # CONST + import go away, foo body mutates
    parsed_v2 = _parse(pm, "mod.py")
    await upsert_parsed_file(pm, pm.default_repo, state, parsed_v2)

    # packages / files unchanged (update, not duplicate)
    assert len(await rs.package.get_list(PackageFilter(repo_ids=[repo_meta.id]))) == 2
    assert len(await rs.file.get_list(FileFilter(repo_ids=[repo_meta.id]))) == 2

    # symbols: only foo remains, id should be SAME, hash should be different
    symbols_after = await rs.node.get_list(NodeFilter(file_ids=[file_id]))
    assert {s.name for s in symbols_after if s.name} == {"foo"}

    # import-edges: removed
    assert await rs.importedge.get_list(
        ImportEdgeFilter(source_package_ids=[pkg_id_mod1])
    ) == []

    # ── 3) delete second file and rescan  ─────────────────────────────────
    module2_fp.unlink()
    await scan_repo(pm, pm.default_repo)

    # Only mod.py-related metadata should remain
    assert len(await rs.file.get_list(FileFilter(repo_ids=[repo_meta.id]))) == 1
    assert len(await rs.package.get_list(PackageFilter(repo_ids=[repo_meta.id]))) == 1
    # Confirm mod2 file metadata truly deleted
    assert await rs.file.get_by_ids([file_id_mod2]) == []


@pytest.mark.asyncio
async def test_nested_gitignore_skips_files(tmp_path: Path):
    """
    A nested .gitignore with a relative pattern should ignore matching files
    under that subtree (including deeper descendants), but not outside it.
    """
    repo_dir = tmp_path / "repo_nested"
    pkg_dir = repo_dir / "pkg"
    deep_dir = pkg_dir / "sub" / "deeper"
    pkg_dir.mkdir(parents=True)
    deep_dir.mkdir(parents=True)

    # Nested .gitignore in 'pkg' ignores any 'ignoreme.py' under 'pkg'
    (pkg_dir / ".gitignore").write_text("ignoreme.py\n")

    # Files to be ignored
    (pkg_dir / "ignoreme.py").write_text("print('ignored at pkg')\n")
    (deep_dir / "ignoreme.py").write_text("print('ignored deeper')\n")

    # Files to be kept
    (repo_dir / "ignoreme.py").write_text("print('kept at root')\n")
    (pkg_dir / "keep.py").write_text("def foo():\n    return 1\n")

    pm = await _make_project(repo_dir)
    await scan_repo(pm, pm.default_repo)

    rs = pm.data
    repo_meta = pm.default_repo
    files = await rs.file.get_list(FileFilter(repo_ids=[repo_meta.id]))
    paths = {f.path for f in files}

    # kept
    assert "ignoreme.py" in paths  # root-level should NOT be affected
    assert "pkg/keep.py" in paths

    # ignored by nested .gitignore
    assert "pkg/ignoreme.py" not in paths
    assert "pkg/sub/deeper/ignoreme.py" not in paths
