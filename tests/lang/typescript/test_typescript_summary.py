from pathlib import Path

import pytest

from knowlt.settings import ProjectSettings
from knowlt import init_project
from knowlt.summary import build_file_summary, SummaryMode
from knowlt.parsers import CodeParserRegistry
from knowlt.models import ProgrammingLanguage, NodeKind
from knowlt.data import NodeFilter
from knowlt.data_helpers import resolve_node_hierarchy


SAMPLES_DIR = Path(__file__).parent / "samples"


@pytest.mark.asyncio
async def test_build_file_summary_simple_tsx():
    # Initialize project and scan TypeScript samples directory
    project = await init_project(
        ProjectSettings(
            project_name="test",
            repo_name="test",
            repo_path=str(SAMPLES_DIR),
        )
    )

    # Ensure simple.tsx has been discovered and stored
    files = await project.data.file.get_by_paths(
        project.default_repo.id, ["simple.tsx"]
    )
    assert files, "simple.tsx not found in repository metadata"

    # Build documentation-level summary for simple.tsx
    file_summary = await build_file_summary(
        project, project.default_repo, "simple.tsx", SummaryMode.Documentation
    )
    assert file_summary is not None, "Failed to build file summary for simple.tsx"
    assert file_summary.path == "simple.tsx"

    # Load expected summary and compare
    expected_text = (SAMPLES_DIR / "simple.tsx.summary").read_text(encoding="utf-8")
    assert file_summary.content.strip() == expected_text.strip()


@pytest.mark.asyncio
async def test_get_node_summary_include_parents_for_method():
    # Initialize project and scan TypeScript samples directory
    project = await init_project(
        ProjectSettings(
            project_name="test",
            repo_name="test",
            repo_path=str(SAMPLES_DIR),
        )
    )

    # Ensure simple.tsx is present
    files = await project.data.file.get_by_paths(
        project.default_repo.id, ["simple.tsx"]
    )
    assert files, "simple.tsx not found in repository metadata"

    # Fetch all nodes for the repo and resolve hierarchy
    nodes = await project.data.node.get_list(
        NodeFilter(repo_ids=[project.default_repo.id])
    )
    assert nodes, "No nodes found in repository"
    resolve_node_hierarchy(nodes)

    # Locate class `Foo` and its `bar` method
    cls = next((n for n in nodes if n.kind == NodeKind.CLASS and n.name == "Foo"), None)
    assert cls is not None, "Class 'Foo' not found"
    method = next(
        (c for c in cls.children if c.kind == NodeKind.METHOD and c.name == "bar"),
        None,
    )
    assert method is not None, "Method 'bar' not found under 'Foo'"

    # Build summary including parents
    helper = CodeParserRegistry.get_helper(ProgrammingLanguage.TYPESCRIPT)
    assert helper is not None, "TypeScript language helper not registered"
    summary = helper.get_node_summary(method, include_parents=True)

    print(summary)
    assert "Foo" in summary
    assert "bar" in summary
