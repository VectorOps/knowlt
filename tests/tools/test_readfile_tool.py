import base64
import json
from pathlib import Path

import pytest

from knowlt.stores.duckdb import DuckDBDataRepository
from knowlt.settings import ProjectSettings, ToolOutput
from knowlt.project import ProjectManager
from knowlt.tools.readfile import ReadFilesTool


async def _make_pm(root: Path):
    settings = ProjectSettings(
        project_name="readfile-test",
        repo_name="tmp",
        repo_path=str(root),
    )
    data_repo = DuckDBDataRepository(settings)
    pm = await ProjectManager.create(settings, data_repo)

    # Ensure a repo exists for the temp path
    repo = await data_repo.repo.get_by_path(str(root))
    if repo is None:
        await pm.add_repo_path("tmp", str(root))
        repo = await data_repo.repo.get_by_path(str(root))
    assert repo is not None

    # Index files
    await pm.refresh_all()
    return pm, repo


@pytest.mark.asyncio
async def test_execute_read_text_file_json(tmp_path: Path):
    # Arrange: create a simple text file
    text_path = tmp_path / "hello.txt"
    text_body = "hello world"
    text_path.write_text(text_body, encoding="utf-8")

    pm, repo = await _make_pm(tmp_path)
    try:
        # Force JSON output for stable assertions
        pm.settings.tools.outputs["read_files"] = ToolOutput.JSON

        tool = ReadFilesTool()
        vpath = pm.construct_virtual_path(repo.id, "hello.txt")

        # Act
        out = await tool.execute(pm, {"path": vpath})
        payload = json.loads(out)

        # Assert
        assert payload["status"] == 200
        # Headers may be omitted depending on alias population; assert body semantics
        assert payload["body"] == text_body
        # If present, headers should be sensible
        if "content-type" in payload:
            assert "text/plain" in payload["content-type"]
            assert "charset=utf-8" in payload["content-type"]
    finally:
        await pm.destroy()


@pytest.mark.asyncio
async def test_execute_read_binary_file_json(tmp_path: Path):
    # Arrange: create a small binary file (PNG signature)
    bin_path = tmp_path / "img.png"
    bin_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00"
    bin_path.write_bytes(bin_bytes)

    pm, repo = await _make_pm(tmp_path)
    try:
        pm.settings.tools.outputs["read_files"] = ToolOutput.JSON

        tool = ReadFilesTool()
        vpath = pm.construct_virtual_path(repo.id, "img.png")

        # Act
        out = await tool.execute(pm, {"path": vpath})
        payload = json.loads(out)

        # Assert
        assert payload["status"] == 200
        # Body should be base64 of the original bytes
        expected_b64 = base64.b64encode(bin_bytes).decode("ascii")
        assert payload["body"] == expected_b64
        # If present, verify headers
        if "content-encoding" in payload:
            assert payload["content-encoding"] == "base64"
        if "content-type" in payload:
            assert payload["content-type"].startswith("image/png")
    finally:
        await pm.destroy()


@pytest.mark.asyncio
async def test_execute_invalid_and_empty_path_errors(tmp_path: Path):
    pm, _repo = await _make_pm(tmp_path)
    try:
        pm.settings.tools.outputs["read_files"] = ToolOutput.JSON
        tool = ReadFilesTool()

        # Invalid (non-virtual) path => 404
        out_404 = await tool.execute(pm, {"path": "not-a-virtual-path.txt"})
        payload_404 = json.loads(out_404)
        assert payload_404["status"] == 404
        assert "error" in payload_404

        # Empty path => 400
        out_400 = await tool.execute(pm, {"path": ""})
        payload_400 = json.loads(out_400)
        assert payload_400["status"] == 400
        assert "error" in payload_400
    finally:
        await pm.destroy()


@pytest.mark.asyncio
async def test_structured_text_output_headers(tmp_path: Path):
    # Arrange: create a simple text file
    text_path = tmp_path / "greet.txt"
    text_body = "hi there"
    text_path.write_text(text_body, encoding="utf-8")

    pm, repo = await _make_pm(tmp_path)
    try:
        # Default for ReadFilesTool is structured text; do not override
        tool = ReadFilesTool()
        vpath = pm.construct_virtual_path(repo.id, "greet.txt")

        # Act
        out = await tool.execute(pm, {"path": vpath})

        # Assert: HTTP-like headers and body
        assert out.startswith("Status: 200 OK")
        # Blank line between headers and body, then body content
        assert "\n\n" in out
        assert out.split("\n\n", 1)[1].endswith(text_body)
    finally:
        await pm.destroy()
