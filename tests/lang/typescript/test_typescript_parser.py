import pytest
from pathlib import Path
from knowlt.settings import ProjectSettings
from knowlt import init_project
from knowlt.project import ProjectCache
from knowlt.lang.typescript import TypeScriptCodeParser
from knowlt.models import ProgrammingLanguage, NodeKind
from devtools import pprint

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _make_dummy_project(root_dir: Path):
    settings = ProjectSettings(
        project_name="test",
        repo_name="test",
        repo_path=str(root_dir),
    )
    return await init_project(settings, refresh=False)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_typescript_parser_on_simple_file():
    """
    Parse the sample `simple.tsx` file and verify that the most important
    artefacts (imports, symbols) are extracted correctly.
    """
    samples_dir = Path(__file__).parent / "samples"
    project     = await _make_dummy_project(samples_dir)
    cache       = ProjectCache()

    parser      = TypeScriptCodeParser(project, project.default_repo, "simple.tsx")
    parsed_file = parser.parse(cache)

    #pprint(parsed_file)
    #raise

    # basic assertions
    assert parsed_file.path == "simple.tsx"
    assert parsed_file.package is not None
    assert parsed_file.package.language == ProgrammingLanguage.TYPESCRIPT

    # imports
    assert len(parsed_file.imports) == 4
    assert parsed_file.imports[0].raw.startswith("import React")
    assert parsed_file.imports[3].physical_path.endswith("circle.js")

    # verify import resolution
    imp = parsed_file.imports[0]
    assert imp.external is True
    assert imp.virtual_path == "react"

    # top-level symbols
    def _to_map(symbols):
        # ignore symbols that have no name
        return {s.name: s for s in symbols if s.name}

    top_level = _to_map(parsed_file.nodes)

    # Core declarations we expect to be present (parser may emit extra top-level names)
    expected_names = {
        "LabeledValue",       # interface
        "Base",               # abstract class
        "Point",              # type-alias
        "Direction",          # enum
        "GenericIdentityFn",
        "GenericNumber",
        "identity",
    }
    assert expected_names.issubset(set(top_level.keys()))

    # ------------------------------------------------------------------
    # check symbols that live inside compound declarations / assignments
    # ------------------------------------------------------------------
    def _flatten(sym_list):
        for s in sym_list:
            if s.name:
                yield s
            yield from _flatten(s.children)

    flat_map = {s.name: s for s in _flatten(parsed_file.nodes)}

    assert flat_map["fn"].kind   == NodeKind.FUNCTION
    assert flat_map["Test"].kind == NodeKind.CLASS

    # variable & function names introduced by the sample that were
    # previously untested
    nested_expected = {
        "z",
        "j1", "f1",              # exported const + arrow-fn
        "a1", "b1", "c1",        # let-declaration
        "e2", "f",               # var-declaration
        "a",                     # async arrow-fn
        "foo", "method", "value",# class members
        "fn", "Test",            # moved inside `export`
    }
    assert nested_expected.issubset(flat_map.keys())

    # additional sanity checks for the two moved symbols
    assert flat_map["CONST"].kind in (NodeKind.CONST, NodeKind.VARIABLE)
    assert flat_map["z"].kind == NodeKind.VARIABLE

    # kind sanity-checks for a representative subset
    assert flat_map["j1"].kind in (NodeKind.CONST, NodeKind.VARIABLE)
    assert flat_map["f1"].kind == NodeKind.FUNCTION
    assert flat_map["Point"].kind == NodeKind.LITERAL
    assert (flat_map["Point"].subtype or "") == "type_alias"
    assert flat_map["Direction"].kind == NodeKind.ENUM
    assert flat_map["LabeledValue"].kind == NodeKind.INTERFACE
    assert flat_map["Base"].kind == NodeKind.CLASS

    # class children (method + possible variable)
    test_cls_children = _to_map(flat_map["Test"].children)
    assert "method" in test_cls_children
    assert test_cls_children["method"].kind == NodeKind.METHOD


@pytest.mark.asyncio
async def test_exported_enum_is_supported(tmp_path: Path):
    """
    Ensure `export enum ...` parses into an EXPORT node with an ENUM child marked exported.
    """
    src = 'export enum Foobar { a = 1, b = 2 };'
    (tmp_path / "export_enum.ts").write_text(src, encoding="utf-8")
    project = await _make_dummy_project(tmp_path)
    cache = ProjectCache()

    parser = TypeScriptCodeParser(project, project.default_repo, "export_enum.ts")
    parsed = parser.parse(cache)

    # find export node
    export_nodes = [s for s in parsed.nodes if s.kind == NodeKind.CUSTOM and (s.subtype or "") == "export"]
    assert len(export_nodes) == 1
    exp = export_nodes[0]

    # it should contain one enum child named Foobar, marked exported
    enums = [c for c in exp.children if c.kind == NodeKind.ENUM]
    assert len(enums) == 1
    enum_sym = enums[0]
    assert enum_sym.name == "Foobar"

    # the enum should include its members
    member_names = {c.name for c in enum_sym.children if c.name}
    assert {"a", "b"}.issubset(member_names)
