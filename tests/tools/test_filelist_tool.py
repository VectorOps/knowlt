from unittest.mock import MagicMock
import json
from pathlib import Path
import pytest

from knowlt.stores.duckdb import DuckDBDataRepository
from knowlt.settings import ProjectSettings, ToolSettings
from knowlt.project import ProjectManager
from knowlt.tools.filelist import ListFilesTool


SAMPLES_DIR = Path(__file__).parents[1] / "lang" / "python" / "samples"


async def _make_pm():
    settings = ProjectSettings(
        project_name="filelist-test",
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
async def test_schema_has_name_and_pattern_string():
    pm = MagicMock()
    pm.settings.tools.file_list_limit = 50
    tool = ListFilesTool(pm)
    schema = await tool.get_openai_schema()
    limit = ToolSettings().file_list_limit
    assert schema["name"] == "list_files"
    assert f"This tool will return up to {limit} files." in schema["description"]
    assert schema["parameters"]["type"] == "object"
    assert "pattern" in schema["parameters"]["properties"]
    assert schema["parameters"]["properties"]["pattern"]["type"] == "string"
    assert "limit" not in schema["parameters"]["properties"]
    assert "required" in schema["parameters"]
    assert schema["parameters"]["required"] == ["pattern"]


@pytest.mark.asyncio
async def test_execute_list_python_files_returns_list():
    pm, _repo = await _make_pm()
    try:
        tool = ListFilesTool(pm)
        out = await tool.execute({"pattern": "*.py"})
        payload = json.loads(out)
        print(payload)
        assert isinstance(payload, list)
        # should include at least one .py file from samples
        assert any(item["path"].endswith(".py") for item in payload)
    finally:
        await pm.destroy()


@pytest.mark.asyncio
async def test_execute_no_matches_returns_empty_list():
    pm, _repo = await _make_pm()
    try:
        tool = ListFilesTool(pm)
        out = await tool.execute({"pattern": "**/*.doesnotexist"})
        payload = json.loads(out)
        assert payload == []
    finally:
        await pm.destroy()
