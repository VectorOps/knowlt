import pytest
from pathlib import Path

from knowlt.settings import ProjectSettings
from knowlt.project import ProjectCache
from knowlt.lang.terraform import TerraformCodeParser
from knowlt.models import ProgrammingLanguage, NodeKind, Repo


class _DummyPM:
    def __init__(self, settings: ProjectSettings):
        self.settings = settings


def _find_block(nodes, subtype: str, name_contains: str | None = None):
    for n in nodes:
        if n.kind == NodeKind.BLOCK and (n.subtype or "") == subtype:
            if name_contains is None:
                return n
            if (n.name and name_contains in n.name) or (
                n.header and name_contains in n.header
            ):
                return n
        child = _find_block(n.children, subtype, name_contains)
        if child is not None:
            return child
    return None


def _find_property(node, name: str):
    return next(
        (
            ch
            for ch in node.children
            if ch.kind == NodeKind.PROPERTY and ch.name == name
        ),
        None,
    )


def _assert_property_children_are_blocks_only(nodes):
    """
    Ensure PROPERTY nodes only have BLOCK children (for nested objects/blocks),
    and never literal/variable children.
    """
    for n in nodes:
        if n.kind == NodeKind.PROPERTY:
            assert all(ch.kind == NodeKind.BLOCK for ch in n.children)
        _assert_property_children_are_blocks_only(n.children)


def test_terraform_parser_on_kitchen_sink():
    samples_dir = Path(__file__).parent / "samples"

    settings = ProjectSettings(
        project_name="tf-test",
        repo_name="tf-test",
        repo_path=str(samples_dir),
    )
    pm = _DummyPM(settings)
    repo = Repo(id="test", name="test", root_path=str(samples_dir))
    cache = ProjectCache()

    parser = TerraformCodeParser(pm, repo, "kitchen_sink.tf")
    parsed_file = parser.parse(cache)
    parser = TerraformCodeParser(pm, repo, "kitchen_sink.tf")
    parsed_file = parser.parse(cache)

    # Basic assertions
    assert parsed_file.path == "kitchen_sink.tf"
    assert parsed_file.package is not None
    assert parsed_file.package.language == ProgrammingLanguage.TERRAFORM

    # Top-level terraform block
    terraform_blk = _find_block(parsed_file.nodes, subtype="terraform")
    assert terraform_blk is not None

    # required_version attribute with a literal child
    req_ver = _find_property(terraform_blk, "required_version")
    assert req_ver is not None
    assert req_ver.children == []
    assert req_ver.header == "required_version ="
    assert ">= 1.0" in req_ver.body

    # backend "s3" block and bucket attribute
    backend_blk = _find_block(terraform_blk.children, subtype="backend")
    assert backend_blk is not None
    # Block name should be the label only, not "backend.s3"
    assert backend_blk.name == "s3"
    bucket_attr = _find_property(backend_blk, "bucket")
    assert bucket_attr is not None
    assert bucket_attr.children == []
    assert bucket_attr.header == "bucket ="
    assert "my-terraform-state-bucket" in bucket_attr.body

    # variable "default_tags" with default object and nested properties
    default_tags_var = _find_block(
        parsed_file.nodes, subtype="variable", name_contains="default_tags"
    )
    assert default_tags_var is not None
    # Variable block name should be the label only
    assert default_tags_var.name == "default_tags"

    default_attr = _find_property(default_tags_var, "default")
    assert default_attr is not None

    obj_child = next(
        (
            ch
            for ch in default_attr.children
            if ch.kind == NodeKind.BLOCK and ch.subtype == "object"
        ),
        None,
    )
    assert obj_child is not None

    project_prop = _find_property(obj_child, "Project")
    owner_prop = _find_property(obj_child, "Owner")
    assert project_prop is not None
    assert owner_prop is not None

    # Deeply nested dynamic blocks in aws_security_group.sg
    sg_block = _find_block(
        parsed_file.nodes, subtype="resource", name_contains="aws_security_group"
    )
    assert sg_block is not None
    # Resource block name includes labels, not the block type
    assert "aws_security_group" in (sg_block.name or "")

    dynamic_ingress = _find_block(sg_block.children, subtype="dynamic")
    assert dynamic_ingress is not None
    assert (
        dynamic_ingress.header is not None
        and 'dynamic "ingress"' in dynamic_ingress.header
    )

    content_block = _find_block(dynamic_ingress.children, subtype="content")
    assert content_block is not None

    ipv6_dynamic = _find_block(content_block.children, subtype="dynamic")
    assert ipv6_dynamic is not None
    assert (
        ipv6_dynamic.header is not None
        and 'dynamic "ipv6_cidr_blocks"' in ipv6_dynamic.header
    )

    # Ensure PROPERTY nodes only have nested BLOCK-like children (no literals)
    _assert_property_children_are_blocks_only(parsed_file.nodes)

    # --- Comment merging assertions -----------------------------------

    # 1) Top-of-file kitchen-sink header comments should be merged into 1 node.
    top_comments = [n for n in parsed_file.nodes if n.kind == NodeKind.COMMENT]
    assert len(top_comments) == 1
    header_comment = top_comments[0]
    body = header_comment.body
    # Match the actual header text in the sample file.
    assert "Terraform HCL kitchen sink" in body
