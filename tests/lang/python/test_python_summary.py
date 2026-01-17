from pathlib import Path

import pytest

from knowlt.settings import ProjectSettings
from knowlt import init_project
from knowlt.summary import build_file_summary, SummaryMode
from knowlt.lang.python import PythonCodeParser
from knowlt.parsers import CodeParserRegistry
from knowlt.models import ProgrammingLanguage, NodeKind
from knowlt.data import NodeFilter
from knowlt.data_helpers import resolve_node_hierarchy


SAMPLES_DIR = Path(__file__).parent / "samples"


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
    expected_text = (SAMPLES_DIR / "simple.py.summary").read_text(encoding="utf-8")
    assert file_summary.content.strip() == expected_text.strip()


@pytest.mark.asyncio
async def test_get_node_summary_include_parents_for_async_method():
    # Initialize project and scan samples directory
    project = await init_project(
        ProjectSettings(
            project_name="test",
            repo_name="test",
            repo_path=str(SAMPLES_DIR),
        )
    )

    # Ensure simple.py is present
    files = await project.data.file.get_by_paths(project.default_repo.id, ["simple.py"])
    assert files, "simple.py not found in repository metadata"

    # Fetch all nodes for the repo and resolve hierarchy
    nodes = await project.data.node.get_list(
        NodeFilter(repo_ids=[project.default_repo.id])
    )
    assert nodes, "No nodes found in repository"
    resolve_node_hierarchy(nodes)

    # Locate class `Test` and its `async_method`
    cls = next(
        (n for n in nodes if n.kind == NodeKind.CLASS and n.name == "Test"), None
    )
    assert cls is not None, "Class 'Test' not found"
    async_method = next(
        (
            c
            for c in cls.children
            if c.kind == NodeKind.METHOD and c.name == "async_method"
        ),
        None,
    )
    assert async_method is not None, "Method 'async_method' not found under 'Test'"

    # Build summary including parents
    helper = CodeParserRegistry.get_helper(ProgrammingLanguage.PYTHON)
    assert helper is not None, "Python language helper not registered"
    summary = helper.get_node_summary(async_method, include_parents=True)

    expected = "\n".join(
        [
            "class Test:",
            "    ...",
            "    async def async_method(self):",
            "        ...",
        ]
    )
    assert summary == expected


@pytest.mark.asyncio
async def test_class_docstring_not_duplicated_in_summary():
    # Initialize project and scan samples directory
    project = await init_project(
        ProjectSettings(
            project_name="test",
            repo_name="test",
            repo_path=str(SAMPLES_DIR),
        )
    )

    # Fetch all nodes and resolve hierarchy
    nodes = await project.data.node.get_list(
        NodeFilter(repo_ids=[project.default_repo.id])
    )
    assert nodes, "No nodes found in repository"
    resolve_node_hierarchy(nodes)

    # Locate class with a docstring
    outcome_cls = next(
        (
            n
            for n in nodes
            if n.kind == NodeKind.CLASS and n.name == "OutcomeStrategy"
        ),
        None,
    )
    assert outcome_cls is not None, "Class 'OutcomeStrategy' not found"

    helper = CodeParserRegistry.get_helper(ProgrammingLanguage.PYTHON)
    assert helper is not None, "Python language helper not registered"

    summary = helper.get_node_summary(outcome_cls, include_docs=True)

    doc_line = (
        "Node outcome strategy. Either a marker in the output (TAG) or an "
        "expected function call (FUNCTION)"
    )

    # Docstring text should appear exactly once in the summary
    assert doc_line in summary
    assert summary.count(doc_line) == 1

    # Class body constants should still be present
    assert 'TAG = "tag"' in summary
    assert 'FUNCTION = "function"' in summary
