import pytest
from pathlib import Path
from knowlt.settings import ProjectSettings
from knowlt import init_project
from knowlt.project import ProjectCache
from knowlt.lang.javascript import JavaScriptCodeParser
from knowlt.models import ProgrammingLanguage, NodeKind

# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
async def _make_dummy_project(root_dir: Path):
    settings = ProjectSettings(
        project_name="test",
        repo_name="test",
        repo_path=str(root_dir),
    )
    return await init_project(settings, refresh=False)

# ------------------------------------------------------------------ #
# tests
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_javascript_parser_on_simple_file():
    samples_dir = Path(__file__).parent / "samples"
    project     = await _make_dummy_project(samples_dir)
    cache       = ProjectCache()

    parser      = JavaScriptCodeParser(project, project.default_repo, "simple.js")
    parsed_file = parser.parse(cache)

    # basic assertions
    assert parsed_file.path == "simple.js"
    assert parsed_file.package is not None
    assert parsed_file.package.language == ProgrammingLanguage.JAVASCRIPT

    # imports
    assert len(parsed_file.imports) == 4
    assert parsed_file.imports[0].raw.startswith("import React")
    # ordering may differ from TypeScript; ensure the circle.js require is present
    assert any(
        (imp.physical_path or "").endswith("circle.js") for imp in parsed_file.imports
    )

    imp = parsed_file.imports[0]
    assert imp.external is True
    assert imp.virtual_path == "react"

    # ---- symbol utils --------------------------------------------
    def _to_map(symbols):
        return {s.name: s for s in symbols if s.name}

    # top-level symbols (named)
    top_level = _to_map(parsed_file.nodes)
    # parser may emit extra top-level names; ensure these are present
    assert {"Base", "identity"}.issubset(set(top_level.keys()))

    # flatten whole tree
    def _flatten(syms):
        for s in syms:
            if s.name:
                yield s
            yield from _flatten(s.children)

    flat_map = {s.name: s for s in _flatten(parsed_file.nodes)}

    # representative kinds & presence
    assert flat_map["fn"].kind     == NodeKind.FUNCTION
    assert flat_map["Test"].kind   == NodeKind.CLASS
    assert flat_map["CONST"].kind  in (NodeKind.CONST, NodeKind.VARIABLE)
    assert flat_map["z"].kind      == NodeKind.VARIABLE
    assert flat_map["j1"].kind     in (NodeKind.CONST, NodeKind.VARIABLE)
    assert flat_map["f1"].kind     == NodeKind.FUNCTION

    # class-expression assigned to const: Foo should be a const with a CLASS child
    foo_node = flat_map["Foo"]
    assert foo_node.kind in (NodeKind.CONST, NodeKind.VARIABLE)
    class_children = [c for c in foo_node.children if c.kind == NodeKind.CLASS]
    assert class_children, "Expected a CLASS child under const Foo"
    cls_node = class_children[0]
    foo_children = _to_map(cls_node.children)
    assert "bar" in foo_children
    assert foo_children["bar"].kind in (NodeKind.METHOD, NodeKind.PROPERTY)

    nested_expected = {
        "CONST", "z", "j1", "f1", "a1", "b1", "c1", "e2", "f",
        "a", "fn", "Test", "Foo", "method"
    }
    assert nested_expected.issubset(flat_map.keys())

    # class member sanity check on exported class
    test_cls_children = _to_map(flat_map["Test"].children)
    assert "method" in test_cls_children
    assert test_cls_children["method"].kind == NodeKind.METHOD
