from pathlib import Path

import pytest

from knowlt.settings import ProjectSettings
from knowlt import init_project
from knowlt.summary import build_file_summary, SummaryMode
from knowlt.lang.golang import GolangCodeParser  # ensure registration


SAMPLES_DIR = Path(__file__).parent / "samples"


@pytest.mark.asyncio
async def test_build_file_summary_matches_expected_main_go():
    # Initialize project and scan samples directory
    project = await init_project(
        ProjectSettings(
            project_name="test",
            repo_name="test",
            repo_path=str(SAMPLES_DIR),
        )
    )

    # Ensure main.go has been discovered and stored
    files = await project.data.file.get_by_paths(project.default_repo.id, ["main.go"])
    assert files, "main.go not found in repository metadata"

    # Build documentation-level summary for main.go
    file_summary = await build_file_summary(
        project, project.default_repo, "main.go", SummaryMode.Documentation
    )
    assert file_summary is not None, "Failed to build file summary for main.go"
    assert file_summary.path == "main.go"

    # Load expected summary and compare
    expected_text = (SAMPLES_DIR / "main.go.summary").read_text(encoding="utf-8")
    assert file_summary.content.strip() == expected_text.strip()
