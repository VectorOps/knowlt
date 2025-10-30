from pathlib import Path

import pytest

from knowlt.settings import ProjectSettings
from knowlt import init_project
from knowlt.summary import build_file_summary, SummaryMode
from knowlt.lang.python import PythonCodeParser


SAMPLES_DIR = Path(__file__).parent / "samples"
SUMMARY_DIR = Path(__file__).parent / "summary"


@pytest.mark.asyncio
async def test_build_file_summary_matches_expected_simple_py():
    # Initialize project and scan samples directory
    project = await init_project(
        ProjectSettings(
            project_name="test",
            repo_name="test",
            repo_path=str(SAMPLES_DIR),
        )
    )

    # Ensure simple.py has been discovered and stored
    files = await project.data.file.get_by_paths(project.default_repo.id, ["simple.py"])
    assert files, "simple.py not found in repository metadata"

    # Build documentation-level summary for simple.py
    file_summary = await build_file_summary(
        project, project.default_repo, "simple.py", SummaryMode.Documentation
    )
    assert file_summary is not None, "Failed to build file summary for simple.py"
    assert file_summary.path == "simple.py"

    # Load expected summary and compare
    expected_text = (SUMMARY_DIR / "simple.py").read_text(encoding="utf-8")
    assert file_summary.content.strip() == expected_text.strip()
