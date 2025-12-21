from pathlib import Path

import pytest

from knowlt.settings import ProjectSettings
from knowlt import init_project
from knowlt.summary import build_file_summary, SummaryMode
from knowlt.lang.terraform import TerraformCodeParser  # ensure registration
from knowlt.parsers import CodeParserRegistry
from knowlt.models import ProgrammingLanguage, NodeKind
from knowlt.data import NodeFilter
from knowlt.data_helpers import resolve_node_hierarchy


SAMPLES_DIR = Path(__file__).parent / "samples"


@pytest.mark.asyncio
async def test_terraform_file_summary_includes_full_node_bodies():
    # Initialize project and scan Terraform samples directory
    project = await init_project(
        ProjectSettings(
            project_name="tf-test",
            repo_name="tf-test",
            repo_path=str(SAMPLES_DIR),
        )
    )

    # Ensure kitchen_sink.tf is present
    files = await project.data.file.get_by_paths(
        project.default_repo.id, ["kitchen_sink.tf"]
    )
    assert files, "kitchen_sink.tf not found in repository metadata"

    # Build documentation-level summary for kitchen_sink.tf
    file_summary = await build_file_summary(
        project,
        project.default_repo,
        "kitchen_sink.tf",
        SummaryMode.Documentation,
    )
    assert file_summary is not None, "Failed to build file summary for kitchen_sink.tf"
    content = file_summary.content

    # The summary should include full node text, not just block headers.
    # These lines only appear inside node bodies, not in the headers.
    assert 'required_version = ">= 1.0"' in content
    assert 'bucket = "my-terraform-state-bucket"' in content
    assert 'Project = "Example"' in content
    assert 'dynamic "ingress"' in content
    assert 'dynamic "ipv6_cidr_blocks"' in content


@pytest.mark.asyncio
async def test_terraform_node_summary_and_include_parents_for_nested_map_property():
    # Initialize project and scan Terraform samples directory
    project = await init_project(
        ProjectSettings(
            project_name="tf-test",
            repo_name="tf-test",
            repo_path=str(SAMPLES_DIR),
        )
    )

    # Ensure kitchen_sink.tf is present
    files = await project.data.file.get_by_paths(
        project.default_repo.id, ["kitchen_sink.tf"]
    )
    assert files, "kitchen_sink.tf not found in repository metadata"

    # Fetch all nodes and resolve parent/child relationships
    nodes = await project.data.node.get_list(
        NodeFilter(repo_ids=[project.default_repo.id])
    )
    assert nodes, "No nodes found in repository"
    resolve_node_hierarchy(nodes)

    helper = CodeParserRegistry.get_helper(ProgrammingLanguage.TERRAFORM)
    assert helper is not None, "Terraform language helper not registered"

    # Locate the variable "default_tags" block
    default_tags_var = next(
        n
        for n in nodes
        if n.kind == NodeKind.BLOCK
        and (n.subtype or "") == "variable"
        and n.name == "default_tags"
    )

    # Direct node summary should include the full body, including nested map entries.
    var_summary = helper.get_node_summary(default_tags_var)
    assert 'type = map(string)' in var_summary
    assert 'default = {' in var_summary
    assert 'Project = "Example"' in var_summary
    assert 'Owner   = "InfraTeam"' in var_summary
    assert 'description = "Default tags applied to resources"' in var_summary

    # Now drill into the nested Project property and test include_parents=True.
    default_attr = next(
        ch
        for ch in default_tags_var.children
        if ch.kind == NodeKind.PROPERTY and ch.name == "default"
    )
    obj_block = next(
        ch
        for ch in default_attr.children
        if ch.kind == NodeKind.BLOCK and ch.subtype == "object"
    )
    project_prop = next(
        ch
        for ch in obj_block.children
        if ch.kind == NodeKind.PROPERTY and ch.name == "Project"
    )

    summary_with_parents = helper.get_node_summary(
        project_prop,
        include_parents=True,
    )

    expected = "\n".join(
        [
            'variable "default_tags" {',
            "    ...",
            "    default = {",
            "        ...",
            '        Project = "Example"',
        ]
    )

    assert summary_with_parents == expected