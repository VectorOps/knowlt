import pytest
from pathlib import Path
from types import SimpleNamespace

from knowlt.project import ProjectCache
from knowlt.lang.golang import GolangCodeParser
from knowlt.models import ProgrammingLanguage, NodeKind, Repo


# Helpers
def _make_dummy_repo(root_dir: Path) -> Repo:
    return Repo(id="repo1", name="test", root_path=str(root_dir))


# Tests
def test_golang_parser_on_sample_file():
    """
    Parse the sample `main.go` file and ensure that all imports,
    structs, interfaces, methods, functions, and preceding comments
    are extracted correctly.
    """
    samples_dir = Path(__file__).parent / "samples"
    repo = _make_dummy_repo(samples_dir)
    pm = SimpleNamespace()
    cache = ProjectCache()

    parser = GolangCodeParser(pm, repo, "main.go")
    parsed_file = parser.parse(cache)

    # Basic assertions
    assert parsed_file.path == "main.go"
    assert parsed_file.package is not None
    assert parsed_file.package.language == ProgrammingLanguage.GO

    # main.go contains exactly two imports:
    #   k "example.com/m"   → inside the project
    #   "fmt"               → standard-library (external)
    assert len(parsed_file.imports) == 2

    imports = {imp.virtual_path: imp for imp in parsed_file.imports}

    # aliased import
    m_imp = imports["example.com/m"]
    assert m_imp.alias == "k"
    assert m_imp.dot is False
    assert m_imp.external is False
    assert m_imp.physical_path == "m"

    # std-lib import
    fmt_imp = imports["fmt"]
    assert fmt_imp.alias is None
    assert fmt_imp.dot is False
    assert fmt_imp.external is True
    assert fmt_imp.physical_path is None

    # Top-level nodes
    nodes = {sym.name: sym for sym in parsed_file.nodes if sym.name}

    # Types (all mapped to CLASS)
    assert "Foobar" in nodes
    assert nodes["Foobar"].kind == NodeKind.LITERAL

    # Structs and interfaces
    assert "S" in nodes
    assert nodes["S"].kind == NodeKind.CLASS
    assert nodes["S"].subtype == "struct"

    assert "E" in nodes
    assert nodes["E"].kind == NodeKind.CLASS
    assert nodes["E"].subtype == "struct"

    assert "I" in nodes
    assert nodes["I"].kind == NodeKind.CLASS
    assert nodes["I"].subtype == "interface"

    assert "Number" in nodes
    assert nodes["Number"].kind == NodeKind.CLASS
    assert nodes["Number"].subtype == "interface"

    # Type bodies should include the 'type' keyword (full declaration)
    assert nodes["S"].body.strip().startswith("type ")
    assert "type S" in nodes["S"].body
    assert nodes["Foobar"].body.strip().startswith("type ")

    assert "G" in nodes
    assert nodes["G"].kind == NodeKind.CLASS
    assert nodes["G"].subtype == "struct"

    # Struct fields and interface methods should be parsed as child nodes.
    s_node = nodes["S"]
    assert any(
        ch.kind == NodeKind.PROPERTY for ch in s_node.children
    ), "struct S should have property children"
    # Expect at least embedded E and fields a, b, c
    assert len([ch for ch in s_node.children if ch.kind == NodeKind.PROPERTY]) >= 4

    i_node = nodes["I"]
    assert any(
        ch.kind == NodeKind.PROPERTY for ch in i_node.children
    ), "interface I should have children"
    i_child_names = {ch.name for ch in i_node.children if ch.name}
    assert {"m", "b"}.issubset(i_child_names)

    # Method `m` attached to S (registered as top-level method node)
    assert "m" in nodes
    assert nodes["m"].kind == NodeKind.METHOD

    # Function `main` with preceding comment in .comment
    assert "main" in nodes
    assert nodes["main"].kind == NodeKind.FUNCTION
    assert nodes["main"].docstring is not None
    assert "Test comment" in nodes["main"].docstring

    # Extra top-level function
    assert "dummy" in nodes
    assert nodes["dummy"].kind == NodeKind.FUNCTION
    assert nodes["dummy"].docstring is not None
    assert "Just a comment" in nodes["dummy"].docstring

    # Ensure we saw exactly the expected set of top-level symbols
    expected = {
        "Foobar",  # type alias
        "S",  # struct
        "I",  # interface
        "m",  # method attached to S
        "dummy",  # function
        "main",  # function
        "E",
        "Number",
        "SumIntsOrFloats",
        "G",
    }
    assert set(nodes.keys()) == expected
