import pytest

from knowlt.models import NodeKind, ProgrammingLanguage
from knowlt.parsers import (
    ParsedFile,
    ParsedImportEdge,
    ParsedNode,
    ParsedPackage,
)


def test_parsed_import_edge_to_dict_shape_matches_scanner_contract():
    edge = ParsedImportEdge(
        virtual_path="pkg.module",
        external=False,
        raw="import pkg.module",
        alias="mod",
        dot=False,
        physical_path="pkg/module.py",
    )

    assert edge.to_dict() == {
        "to_package_physical_path": "pkg/module.py",
        "to_package_virtual_path": "pkg.module",
        "alias": "mod",
        "dot": False,
        "external": False,
        "raw": "import pkg.module",
    }


def test_parsed_package_to_dict_shape_matches_scanner_contract():
    pkg = ParsedPackage(
        language=ProgrammingLanguage.PYTHON,
        physical_path="pkg/module.py",
        virtual_path="pkg.module",
    )

    assert pkg.to_dict() == {
        "name": "pkg.module",
        "language": ProgrammingLanguage.PYTHON,
        "virtual_path": "pkg.module",
        "physical_path": "pkg/module.py",
    }


def test_parsed_node_to_dict_excludes_children_and_preserves_scalar_fields():
    child = ParsedNode(
        name="child",
        body="pass",
        kind=NodeKind.METHOD,
        start_line=2,
        end_line=2,
        start_byte=10,
        end_byte=14,
    )
    node = ParsedNode(
        name="parent",
        body="def parent():\n    pass",
        kind=NodeKind.FUNCTION,
        subtype="async",
        start_line=1,
        end_line=2,
        start_byte=0,
        end_byte=22,
        header="def parent():",
        docstring='"docs"',
        comment="# comment",
        children=[child],
    )

    assert node.to_dict() == {
        "name": "parent",
        "body": "def parent():\n    pass",
        "kind": NodeKind.FUNCTION,
        "subtype": "async",
        "start_line": 1,
        "end_line": 2,
        "start_byte": 0,
        "end_byte": 22,
        "header": "def parent():",
        "visibility": None,
        "docstring": '"docs"',
        "comment": "# comment",
    }
    assert node.children == [child]


def test_parsed_file_to_dict_shape_matches_scanner_contract():
    pkg = ParsedPackage(
        language=ProgrammingLanguage.PYTHON,
        physical_path="pkg/module.py",
        virtual_path="pkg.module",
    )
    parsed_file = ParsedFile(
        package=pkg,
        path="pkg/module.py",
        docstring='"module docs"',
        file_hash="abc123",
        last_updated=123456789,
    )

    assert parsed_file.to_dict() == {
        "path": "pkg/module.py",
        "file_hash": "abc123",
        "last_updated": 123456789,
    }


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        (
            {"virtual_path": "", "external": False, "raw": "import foo"},
            "ParsedImportEdge.virtual_path must be non-empty",
        ),
        (
            {"virtual_path": "foo", "external": False, "raw": ""},
            "ParsedImportEdge.raw must be non-empty",
        ),
    ],
)
def test_parsed_import_edge_validation(kwargs, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        ParsedImportEdge(**kwargs)


def test_parsed_package_validation_requires_paths():
    with pytest.raises(ValueError, match="ParsedPackage.physical_path must be non-empty"):
        ParsedPackage(
            language=ProgrammingLanguage.PYTHON,
            physical_path="",
            virtual_path="pkg.module",
        )

    with pytest.raises(ValueError, match="ParsedPackage.virtual_path must be non-empty"):
        ParsedPackage(
            language=ProgrammingLanguage.PYTHON,
            physical_path="pkg/module.py",
            virtual_path="",
        )


def test_parsed_node_validation_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="ParsedNode.start_line cannot exceed end_line"):
        ParsedNode(
            body="x",
            kind=NodeKind.LITERAL,
            start_line=3,
            end_line=2,
            start_byte=0,
            end_byte=1,
        )

    with pytest.raises(ValueError, match="ParsedNode.start_byte cannot exceed end_byte"):
        ParsedNode(
            body="x",
            kind=NodeKind.LITERAL,
            start_line=1,
            end_line=1,
            start_byte=5,
            end_byte=4,
        )


def test_parsed_file_validation_requires_path():
    with pytest.raises(ValueError, match="ParsedFile.path must be non-empty"):
        ParsedFile(path="")