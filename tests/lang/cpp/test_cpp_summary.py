from pathlib import Path

import pytest

from knowlt.settings import ProjectSettings
from knowlt import init_project
from knowlt.summary import build_file_summary, SummaryMode
from knowlt.lang.cpp import CppCodeParser
from knowlt.parsers import CodeParserRegistry
from knowlt.models import ProgrammingLanguage, NodeKind
from knowlt.data import NodeFilter
from knowlt.data_helpers import resolve_node_hierarchy


SAMPLES_DIR = Path(__file__).parent / "samples"


@pytest.mark.asyncio
async def test_build_file_summary_matches_expected_main_cpp():
    project = await init_project(
        ProjectSettings(
            project_name="test",
            repo_name="test",
            repo_path=str(SAMPLES_DIR),
        )
    )

    files = await project.data.file.get_by_paths(project.default_repo.id, ["main.cpp"])
    assert files, "main.cpp not found in repository metadata"

    file_summary = await build_file_summary(
        project, project.default_repo, "main.cpp", SummaryMode.Documentation
    )
    assert file_summary is not None, "Failed to build file summary for main.cpp"
    assert file_summary.path == "main.cpp"

    expected_text = (SAMPLES_DIR / "main.cpp.summary").read_text(encoding="utf-8")
    assert file_summary.content.strip() == expected_text.strip()


@pytest.mark.asyncio
async def test_cpp_function_prototype_summary_does_not_add_body_placeholder():
    project = await init_project(
        ProjectSettings(
            project_name="test",
            repo_name="test",
            repo_path=str(SAMPLES_DIR),
        )
    )

    nodes = await project.data.node.get_list(
        NodeFilter(repo_ids=[project.default_repo.id])
    )
    resolve_node_hierarchy(nodes)

    prototype = next(
        (n for n in nodes if n.kind == NodeKind.FUNCTION and n.name == "sum"),
        None,
    )
    assert prototype is not None

    helper = CodeParserRegistry.get_helper(ProgrammingLanguage.CPP)
    assert helper is not None
    summary = helper.get_node_summary(prototype)

    assert summary.strip() == "int sum(int a, int b);"


@pytest.mark.asyncio
async def test_cpp_namespace_summary_with_nested_children(tmp_path: Path):
    src = """namespace outer {
namespace inner {
int helper();
}
}
"""
    (tmp_path / "nested.cpp").write_text(src, encoding="utf-8")

    project = await init_project(
        ProjectSettings(
            project_name="test",
            repo_name="test",
            repo_path=str(tmp_path),
        )
    )

    summary = await build_file_summary(
        project, project.default_repo, "nested.cpp", SummaryMode.Documentation
    )
    assert summary is not None
    assert summary.content.strip() == """namespace outer {
	namespace inner {
		int helper();
	}
}""".strip()