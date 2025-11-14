import json
from pathlib import Path
import pytest

from knowlt.stores.duckdb import DuckDBDataRepository
from knowlt.settings import ProjectSettings, ToolOutput
from knowlt.project import ProjectManager
from knowlt.tools.nodesearch import NodeSearchTool
from knowlt.summary import SummaryMode


SAMPLES_DIR = Path(__file__).parents[1] / "lang" / "python" / "samples"


async def _make_pm():
    settings = ProjectSettings(
        project_name="nodesearch-test",
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
async def test_schema_includes_supported_modes_and_defaults():
    tool = NodeSearchTool()
    schema = await tool.get_openai_schema()
    enum_vals = schema["parameters"]["properties"]["summary_mode"]["enum"]
    # Node search supports all SummaryMode options (including 'source' and 'skip')
    assert "source" in enum_vals
    assert "skip" in enum_vals
    # default should be 'definition'
    assert (
        schema["parameters"]["properties"]["summary_mode"]["default"]
        == SummaryMode.Definition.value
    )
    # visibility should include 'all'
    vis_vals = schema["parameters"]["properties"]["visibility"]["enum"]
    assert "all" in vis_vals


@pytest.mark.asyncio
async def test_execute_nodesearch_skip_mode_returns_list():
    pm, _repo = await _make_pm()
    try:
        # Force tool output to JSON for stable assertions
        pm.settings.tools.outputs["search_project"] = ToolOutput.JSON

        tool = NodeSearchTool()
        out = await tool.execute(
            pm,
            {
                "global_search": True,
                # avoid embedding computation by not passing a free-text query
                "summary_mode": SummaryMode.Skip.value,  # avoid needing language helpers
                "limit": 5,
            },
        )
        payload = json.loads(out)
        assert isinstance(payload, list)
    finally:
        await pm.destroy()


@pytest.mark.asyncio
async def test_execute_raises_on_invalid_visibility():
    pm, _repo = await _make_pm()
    try:
        tool = NodeSearchTool()
        with pytest.raises(ValueError):
            await tool.execute(
                pm,
                {
                    "visibility": "bogus",
                    "summary_mode": SummaryMode.Skip.value,
                },
            )
    finally:
        await pm.destroy()
