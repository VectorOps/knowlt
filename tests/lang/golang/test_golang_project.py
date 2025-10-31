from pathlib import Path

import pytest

from knowlt.settings import ProjectSettings
from knowlt import init_project
from knowlt.data import FileFilter, NodeFilter, PackageFilter


SAMPLES_DIR = Path(__file__).parent / "samples"


@pytest.mark.asyncio
async def test_golang_project_scan_populates_repositories():
    """Project.init + directory scan should create metadata for every Go sample file."""
    project = await init_project(
        ProjectSettings(
            project_name="test",
            repo_name="test",
            repo_path=str(SAMPLES_DIR),
        )
    )
    repo_store = project.data
    repo_meta = project.default_repo

    # ── files (.go only) ────────────────────────────────────────────────
    expected_go_files = [str(p.relative_to(SAMPLES_DIR)) for p in SAMPLES_DIR.rglob("*.go")]
    files = await repo_store.file.get_list(FileFilter(repo_ids=[repo_meta.id]))
    stored_go_files = [f.path for f in files if f.path.endswith(".go")]
    assert len(stored_go_files) == len(expected_go_files)
    assert set(stored_go_files) == set(expected_go_files)

    # ── packages ────────────────────────────────────────────────────────
    # Expect 2 packages: "." (main) and "example.com/m" (from go.mod + subdir)
    pkgs = await repo_store.package.get_list(PackageFilter(repo_ids=[repo_meta.id]))
    vpaths = {p.virtual_path for p in pkgs if p.virtual_path}
    assert vpaths == {".", "example.com/m"}

    # ── symbols (spot-check main.go) ────────────────────────────────────
    main_meta = next(f for f in files if f.path == "main.go")
    symbols = await repo_store.node.get_list(NodeFilter(file_ids=[main_meta.id]))
    symbol_names = {s.name for s in symbols if s.name}

    expected_symbols = {
        "Foobar", "S", "I", "m", "dummy", "main", "E", "Number", "SumIntsOrFloats", "G"
    }
    # every expected symbol should be present
    assert expected_symbols.issubset(symbol_names)
    # every recorded symbol must reference its file
    assert all(s.file_id == main_meta.id for s in symbols)
