from pathlib import Path

import pytest

from knowlt.settings import ProjectSettings
from knowlt import init_project
from knowlt.data import FileFilter, NodeFilter, PackageFilter


SAMPLES_DIR = Path(__file__).parent / "samples"


@pytest.mark.asyncio
async def test_c_project_scan_populates_repositories():
    project = await init_project(
        ProjectSettings(
            project_name="test",
            repo_name="test",
            repo_path=str(SAMPLES_DIR),
        )
    )
    repo_store = project.data
    repo_meta = project.default_repo

    expected_c_files = [
        str(p.relative_to(SAMPLES_DIR))
        for p in SAMPLES_DIR.rglob("*")
        if p.suffix in {".c", ".h"}
    ]
    files = await repo_store.file.get_list(FileFilter(repo_ids=[repo_meta.id]))
    stored_c_files = [f.path for f in files if f.path.endswith((".c", ".h"))]
    assert len(stored_c_files) == len(expected_c_files)
    assert set(stored_c_files) == set(expected_c_files)

    pkgs = await repo_store.package.get_list(PackageFilter(repo_ids=[repo_meta.id]))
    vpaths = {p.virtual_path for p in pkgs if p.virtual_path}
    assert vpaths == {"main.c", "types.h", "sub/feature.h"}

    main_meta = next(f for f in files if f.path == "main.c")
    symbols = await repo_store.node.get_list(NodeFilter(file_ids=[main_meta.id]))
    symbol_names = {s.name for s in symbols if s.name}

    expected_symbols = {"VERSION", "SQUARE", "Size", "Person", "Color", "Point", "add"}
    assert expected_symbols.issubset(symbol_names)
    assert all(s.file_id == main_meta.id for s in symbols)