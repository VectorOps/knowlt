# tests/tools/test_filesummary_tool.py
import json
from pathlib import Path
import pytest

from knowlt.stores.duckdb import DuckDBDataRepository
from knowlt.settings import ProjectSettings, ToolOutput
from knowlt.project import ProjectManager
from knowlt.tools.filesummary import SummarizeFilesTool
from knowlt.summary import SummaryMode


SAMPLES_DIR = Path(__file__).parents[1] / "lang" / "python" / "samples"


async def _make_pm():
    settings = ProjectSettings(
        project_name="filesummary-test",
        repo_name="samples",
        repo_path=str(SAMPLES_DIR),
    )
    data_repo = DuckDBDataRepository(settings)
    pm = await ProjectManager.create(settings, data_repo)

    # Ensure a repo exists for the samples path
    repo = await data_repo.repo.get_by_path(str(SAMPLES_DIR))
    if repo is None:
        await pm.add_repo_path("samples", str(SAMPLES_DIR))
        repo = await data_repo.repo.get_by_path(str(SAMPLES_DIR))
    assert repo is not None
    return pm, repo


@pytest.mark.asyncio
async def test_schema_excludes_unsupported_modes():
    tool = SummarizeFilesTool()
    schema = tool.get_openai_schema()
    enum_vals = schema["parameters"]["properties"]["summary_mode"]["enum"]
    # 'source' and 'skip' are not supported by the tool schema
    assert "source" not in enum_vals
    assert "skip" not in enum_vals
    # default should be 'definition'
    assert (
        schema["parameters"]["properties"]["summary_mode"]["default"]
        == SummaryMode.Definition.value
    )


@pytest.mark.asyncio
async def test_execute_summarize_python_sample_file():
    pm, repo = await _make_pm()
    try:
        # Force tool output to JSON for stable assertions
        pm.settings.tools.outputs["summarize_files"] = ToolOutput.JSON

        tool = SummarizeFilesTool()
        out = await tool.execute(
            pm,
            {
                "paths": [
                    "simple.py",
                    "does_not_exist.py",
                ],
                "summary_mode": SummaryMode.Definition.value,
            },
        )

        payload = json.loads(out)
        assert isinstance(payload, list)
        # Find the summary for the existing file
        summ_paths = [item.get("path") for item in payload if isinstance(item, dict)]
        assert "simple.py" in summ_paths
        # Ensure the nonexistent path did not produce a summary
        assert "does_not_exist.py" not in summ_paths
    finally:
        await pm.destroy()


@pytest.mark.asyncio
async def test_execute_raises_on_source_mode():
    pm, repo = await _make_pm()
    try:
        vpath = pm.construct_virtual_path(repo.id, "simple.py")
        tool = SummarizeFilesTool()
        with pytest.raises(ValueError):
            await tool.execute(
                pm,
                {
                    "paths": [vpath],
                    "summary_mode": "source",  # unsupported
                },
            )
    finally:
        await pm.destroy()
