from pathlib import Path

import pytest

from knowlt.settings import ProjectSettings
from knowlt import init_project
from knowlt.summary import build_file_summary, SummaryMode
from knowlt.lang.c import CCodeParser
from knowlt.parsers import CodeParserRegistry
from knowlt.models import ProgrammingLanguage, NodeKind
from knowlt.data import NodeFilter
from knowlt.data_helpers import resolve_node_hierarchy


SAMPLES_DIR = Path(__file__).parent / "samples"


@pytest.mark.asyncio
async def test_build_file_summary_matches_expected_main_c():
    project = await init_project(
        ProjectSettings(
            project_name="test",
            repo_name="test",
            repo_path=str(SAMPLES_DIR),
        )
    )

    files = await project.data.file.get_by_paths(project.default_repo.id, ["main.c"])
    assert files, "main.c not found in repository metadata"

    file_summary = await build_file_summary(
        project, project.default_repo, "main.c", SummaryMode.Documentation
    )
    assert file_summary is not None, "Failed to build file summary for main.c"
    assert file_summary.path == "main.c"

    expected_text = (SAMPLES_DIR / "main.c.summary").read_text(encoding="utf-8")
    assert file_summary.content.strip() == expected_text.strip()


@pytest.mark.asyncio
async def test_c_function_prototype_summary_does_not_add_body_placeholder():
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

    helper = CodeParserRegistry.get_helper(ProgrammingLanguage.C)
    assert helper is not None
    summary = helper.get_node_summary(prototype)

    assert summary.strip() == "int sum(int a, int b)"


@pytest.mark.asyncio
async def test_c_summary_keeps_chained_elif_branches(tmp_path: Path):
    src = """#if defined(FLAG)
int a(void);
#elif OTHER
int b(void);
#elif THIRD
int c(void);
#else
int d(void);
#endif
"""
    (tmp_path / "chain.h").write_text(src, encoding="utf-8")

    project = await init_project(
        ProjectSettings(
            project_name="test",
            repo_name="test",
            repo_path=str(tmp_path),
        )
    )

    summary = await build_file_summary(
        project, project.default_repo, "chain.h", SummaryMode.Documentation
    )
    assert summary is not None
    assert summary.content.strip() == """#if defined(FLAG)
	...
#elif OTHER
	...
#elif THIRD
	...
#else
	...
#endif""".strip()