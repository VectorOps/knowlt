import os
import time
from pathlib import Path

import pytest

from knowlt.data import FileFilter, NodeFilter, PackageFilter
from knowlt.lang.python import PythonCodeParser
from knowlt.parsers import CodeParserRegistry
from knowlt.project import ProjectCache, ProjectManager
from knowlt.scanner import ParsingState, upsert_parsed_file
from knowlt.settings import ProjectSettings
from knowlt.stores.duckdb import DuckDBDataRepository


CODE = """
CONST = 1

import os


def foo(x: int) -> int:
    return x + 1


class Bar:
    def method(self) -> str:
        return "ok"
"""


async def _make_project(root: Path) -> ProjectManager:
    settings = ProjectSettings(
        project_name="benchmark",
        repo_name="benchmark",
        repo_path=str(root),
    )
    data_repo = DuckDBDataRepository(settings)
    CodeParserRegistry.register_parser(PythonCodeParser)
    return await ProjectManager.create(settings, data_repo)


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("KNOWLT_RUN_BENCHMARKS") != "1",
    reason="Set KNOWLT_RUN_BENCHMARKS=1 to run parser benchmarks",
)
async def test_python_parse_and_upsert_benchmark(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    module_path = repo_dir / "mod.py"
    module_path.write_text(CODE)

    iterations = 50
    pm = await _make_project(repo_dir)

    try:
        warmup_parser = PythonCodeParser(pm, pm.default_repo, "mod.py")
        warmup_parsed = warmup_parser.parse(ProjectCache())
        await upsert_parsed_file(pm, pm.default_repo, ParsingState(), warmup_parsed)

        parse_durations: list[float] = []
        upsert_durations: list[float] = []

        for idx in range(iterations):
            module_path.write_text(CODE + f"\nMARKER_{idx} = {idx}\n")

            parser = PythonCodeParser(pm, pm.default_repo, "mod.py")

            parse_start = time.perf_counter()
            parsed_file = parser.parse(ProjectCache())
            parse_durations.append(time.perf_counter() - parse_start)

            upsert_start = time.perf_counter()
            await upsert_parsed_file(pm, pm.default_repo, ParsingState(), parsed_file)
            upsert_durations.append(time.perf_counter() - upsert_start)

        files = await pm.data.file.get_list(FileFilter(repo_ids=[pm.default_repo.id]))
        packages = await pm.data.package.get_list(PackageFilter(repo_ids=[pm.default_repo.id]))
        nodes = await pm.data.node.get_list(NodeFilter(file_ids=[files[0].id]))

        assert len(files) == 1
        assert len(packages) == 1
        assert nodes

        parse_total = sum(parse_durations)
        upsert_total = sum(upsert_durations)
        parse_avg_ms = (parse_total / iterations) * 1000
        upsert_avg_ms = (upsert_total / iterations) * 1000

        print(
            (
                "python_parse_upsert_benchmark "
                f"iterations={iterations} "
                f"parse_total_s={parse_total:.6f} "
                f"parse_avg_ms={parse_avg_ms:.3f} "
                f"upsert_total_s={upsert_total:.6f} "
                f"upsert_avg_ms={upsert_avg_ms:.3f}"
            )
        )
    finally:
        await pm.destroy()