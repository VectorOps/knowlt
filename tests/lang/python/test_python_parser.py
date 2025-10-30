import pytest
from pathlib import Path
from devtools import pprint

from knowlt.settings import ProjectSettings
from knowlt.project import ProjectCache
from knowlt.lang.python import PythonCodeParser
from knowlt.models import ProgrammingLanguage, NodeKind, Repo

# Helpers
class _DummyPM:
    def __init__(self, settings: ProjectSettings):
        self.settings = settings


# Tests
def test_python_parser_on_simple_file():
    """
    Parse the sample `simple.py` file and assert that the most important
    artefacts (imports, symbols, docstrings…) are extracted correctly.
    """
    samples_dir = Path(__file__).parent / "samples"
    settings     = ProjectSettings(
        project_name="test",
        repo_name="test",
        repo_path=str(samples_dir),
    )
    pm           = _DummyPM(settings)
    repo         = Repo(id="test", name="test", root_path=str(samples_dir))
    cache        = ProjectCache()

    parser       = PythonCodeParser(pm, repo, "simple.py")
    parsed_file  = parser.parse(cache)

    # Basic assertions
    assert parsed_file.path == "simple.py"
    assert parsed_file.package is not None
    assert parsed_file.package.language == ProgrammingLanguage.PYTHON

    # Imports
    # simple.py has three import statements
    assert len(parsed_file.imports) == 4

    # Local-package (relative) import                                    #
    # Ensure that the relative import  `from .foobuz import abc`
    # is recognised as *local* (external == False) and that its path is
    # preserved.
    rel_import = next(
        (imp for imp in parsed_file.imports if imp.virtual_path.startswith(".foobuz")),
        None,
    )
    assert rel_import is not None, "Relative import '.foobuz' not found"
    assert rel_import.external is False          # must be classified local
    assert rel_import.dot is True                # leading dot present
    assert rel_import.physical_path == "foobuz.py"
    assert rel_import.virtual_path == ".foobuz"

    # Top-level symbols
    def symbols_to_map(symbols):
        # ignore literals and any symbol that has no name (e.g. comments)
        return {
            sym.name: sym
            for sym in symbols
            if sym.kind != NodeKind.LITERAL and sym.name
        }

    top_level = symbols_to_map(parsed_file.nodes)

    expected_top_level_names = {
        "fn", "_foo", "decorated", "double_decorated",
        "ellipsis_fn", "async_fn", "Test", "Foobar",
    }
    assert set(top_level.keys()) == expected_top_level_names

    # Kinds
    for fn_name in ("fn", "_foo", "decorated", "double_decorated", "ellipsis_fn", "async_fn"):
        assert top_level[fn_name].kind == NodeKind.FUNCTION
    for cls_name in ("Test", "Foobar"):
        assert top_level[cls_name].kind == NodeKind.CLASS

    # Decorators on top-level symbols
    def header_has_decorators(sym, expected_decorators: list[str]) -> bool:
        hdr = (sym.header or "")
        lines = [ln.strip() for ln in hdr.splitlines() if ln.strip()]
        decos = [ln for ln in lines if ln.startswith("@")]
        return decos == [f"@{d}" for d in expected_decorators]

    assert header_has_decorators(top_level["decorated"], ["abc"])
    assert header_has_decorators(top_level["double_decorated"], ["abc", "fed"])
    assert header_has_decorators(top_level["Foobar"], ["dummy"])

    # Class Test children
    test_cls = top_level["Test"]
    children_map = symbols_to_map(test_cls.children)

    expected_test_children = {
        "__init__", "method", "get",
        "async_method", "multi_decorated", "ellipsis_method",
    }
    assert set(children_map.keys()) == expected_test_children

    # Kinds of children
    method_kinds = {
        "__init__", "method", "async_method", "multi_decorated", "ellipsis_method"
    }
    for m in method_kinds:
        assert children_map[m].kind == NodeKind.METHOD

    assert children_map["get"].kind == NodeKind.METHOD

    # Decorators on multi_decorated method
    multi_decorated_sym = children_map["multi_decorated"]
    assert header_has_decorators(multi_decorated_sym, ["abc", "fed"])

    # Docstrings
    # Current parser extracts function and method docstrings
    assert top_level["fn"].docstring == "\"docstring!\""
    assert top_level["_foo"].docstring is not None
    assert "Multiline" in (top_level["_foo"].docstring or "")
    assert test_cls.docstring is None
    # Docstring on ellipsis_method
    ellipsis_method_sym = children_map["ellipsis_method"]
    assert "Test me" in (ellipsis_method_sym.docstring or "")

    # Inheritance edges for Foobar (optional)
    foobar_cls = top_level["Foobar"]
    if foobar_cls.header:
        assert "(Foo, Bar, Buzz)" in foobar_cls.header

    # Optional: ensure file-level docstring captured
    assert parsed_file.docstring is not None

    # If/Elif/Else chain headers should include conditions
    if_chain = next(
        (n for n in parsed_file.nodes if n.kind.value == NodeKind.CUSTOM.value and n.subtype == "if"),
        None,
    )
    assert if_chain is not None, "Top-level if/elif/else chain not parsed"
    assert len(if_chain.children) == 3, "Expected if, elif and else blocks"
    headers = [blk.header for blk in if_chain.children]
    assert headers == [
        'if __name__ == "__main__":',
        'elif __name__ == "__buzz__":',
        'else:',
    ]
