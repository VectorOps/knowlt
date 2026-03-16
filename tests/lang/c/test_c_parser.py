from pathlib import Path
from types import SimpleNamespace

from knowlt.project import ProjectCache
from knowlt.lang.c import CCodeParser
from knowlt.models import ProgrammingLanguage, NodeKind, Repo


def _make_dummy_repo(root_dir: Path) -> Repo:
    return Repo(id="repo1", name="test", root_path=str(root_dir))


def test_c_parser_on_sample_file():
    samples_dir = Path(__file__).parent / "samples"
    repo = _make_dummy_repo(samples_dir)
    pm = SimpleNamespace()
    cache = ProjectCache()

    parser = CCodeParser(pm, repo, "main.c")
    parsed_file = parser.parse(cache)

    assert parsed_file.path == "main.c"
    assert parsed_file.package is not None
    assert parsed_file.package.language == ProgrammingLanguage.C

    assert len(parsed_file.imports) == 3

    imports = {imp.virtual_path: imp for imp in parsed_file.imports}

    stdio_imp = imports["stdio.h"]
    assert stdio_imp.external is True
    assert stdio_imp.physical_path is None

    types_imp = imports["types.h"]
    assert types_imp.external is False
    assert types_imp.physical_path == "types.h"

    feature_imp = imports["sub/feature.h"]
    assert feature_imp.external is False
    assert feature_imp.physical_path == "sub/feature.h"

    nodes = {sym.name: sym for sym in parsed_file.nodes if sym.name}

    assert "VERSION" in nodes
    assert nodes["VERSION"].kind == NodeKind.LITERAL

    assert "SQUARE" in nodes
    assert nodes["SQUARE"].kind == NodeKind.LITERAL
    assert "((x) * (x))" in nodes["SQUARE"].body

    assert "sum" in nodes
    assert nodes["sum"].kind == NodeKind.FUNCTION
    assert nodes["sum"].subtype == "signature"
    assert nodes["sum"].header == "int sum(int a, int b)"

    assert "global_name" in nodes
    assert nodes["global_name"].kind == NodeKind.VARIABLE
    assert nodes["global_name"].body == 'static const char *global_name = "knowlt"'

    assert "Size" in nodes
    assert nodes["Size"].kind == NodeKind.LITERAL
    assert nodes["Size"].body.strip().startswith("typedef ")

    assert "Person" in nodes
    assert nodes["Person"].kind == NodeKind.CLASS
    assert nodes["Person"].subtype == "struct"
    person_child_names = {ch.name for ch in nodes["Person"].children if ch.name}
    assert {"id", "name"}.issubset(person_child_names)

    assert "Color" in nodes
    assert nodes["Color"].kind == NodeKind.CLASS
    assert nodes["Color"].subtype == "enum"
    color_child_names = {ch.name for ch in nodes["Color"].children if ch.name}
    assert {"RED", "GREEN", "BLUE"}.issubset(color_child_names)

    assert "Value" in nodes
    assert nodes["Value"].kind == NodeKind.CLASS
    assert nodes["Value"].subtype == "union"
    union_child_names = {ch.name for ch in nodes["Value"].children if ch.name}
    assert {"i", "f"}.issubset(union_child_names)

    assert "Point" in nodes
    assert nodes["Point"].kind == NodeKind.CLASS
    assert nodes["Point"].subtype == "struct"

    assert "add" in nodes
    assert nodes["add"].kind == NodeKind.FUNCTION
    assert nodes["add"].docstring is not None
    assert "Add two integers" in nodes["add"].docstring

    preproc_nodes = [
        sym
        for sym in parsed_file.nodes
        if sym.kind == NodeKind.CUSTOM and sym.subtype == "preprocessor"
    ]
    assert len(preproc_nodes) == 1
    preproc = preproc_nodes[0]
    assert preproc.header == "#ifdef ENABLE_FEATURE"
    assert preproc.comment == "#endif"
    child_names = {ch.name for ch in preproc.children if ch.name}
    assert "enabled" in child_names
    else_branch = next(
        (
            ch
            for ch in preproc.children
            if ch.kind == NodeKind.CUSTOM and ch.subtype == "preprocessor"
        ),
        None,
    )
    assert else_branch is not None
    assert else_branch.header == "#else"
    else_names = {ch.name for ch in else_branch.children if ch.name}
    assert "disabled" in else_names

    expected = {
        "VERSION",
        "SQUARE",
        "sum",
        "global_name",
        "Size",
        "Person",
        "Color",
        "Value",
        "Point",
        "add",
    }
    assert set(nodes.keys()) == expected