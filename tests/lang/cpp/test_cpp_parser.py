from pathlib import Path
from types import SimpleNamespace

from knowlt.project import ProjectCache
from knowlt.lang.cpp import CppCodeParser
from knowlt.models import ProgrammingLanguage, NodeKind, Repo


def _make_dummy_repo(root_dir: Path) -> Repo:
    return Repo(id="repo1", name="test", root_path=str(root_dir))


def _flatten(symbols):
    for sym in symbols:
        yield sym
        yield from _flatten(sym.children)


def test_cpp_parser_on_sample_file():
    samples_dir = Path(__file__).parent / "samples"
    repo = _make_dummy_repo(samples_dir)
    pm = SimpleNamespace()
    cache = ProjectCache()

    parser = CppCodeParser(pm, repo, "main.cpp")
    parsed_file = parser.parse(cache)

    assert parsed_file.path == "main.cpp"
    assert parsed_file.package is not None
    assert parsed_file.package.language == ProgrammingLanguage.CPP

    assert len(parsed_file.imports) == 3
    imports = {imp.virtual_path: imp for imp in parsed_file.imports}

    vector_imp = imports["vector"]
    assert vector_imp.external is True
    assert vector_imp.physical_path is None

    types_imp = imports["types.hpp"]
    assert types_imp.external is False
    assert types_imp.physical_path == "types.hpp"

    feature_imp = imports["sub/feature.hpp"]
    assert feature_imp.external is False
    assert feature_imp.physical_path == "sub/feature.hpp"

    top_level = {sym.name: sym for sym in parsed_file.nodes if sym.name}
    assert set(top_level.keys()) == {"VERSION", "SQUARE", "Size", "outer"}

    assert top_level["VERSION"].kind == NodeKind.LITERAL
    assert top_level["SQUARE"].kind == NodeKind.LITERAL
    assert "((x) * (x))" in top_level["SQUARE"].body
    assert top_level["Size"].kind == NodeKind.LITERAL
    assert top_level["Size"].body.strip().startswith("typedef ")

    outer = top_level["outer"]
    assert outer.kind == NodeKind.NAMESPACE

    inner = next(
        ch for ch in outer.children if ch.kind == NodeKind.NAMESPACE and ch.name == "inner"
    )
    detail = next(
        ch for ch in outer.children if ch.kind == NodeKind.NAMESPACE and ch.name == "detail"
    )
    impl = next(
        ch for ch in detail.children if ch.kind == NodeKind.NAMESPACE and ch.name == "impl"
    )
    assert any(ch.name == "helper" for ch in impl.children)

    inner_named = {ch.name: ch for ch in inner.children if ch.name}
    assert {"Box", "IntBox", "Color", "Value", "Point", "sum", "add"}.issubset(
        inner_named.keys()
    )

    box = inner_named["Box"]
    assert box.kind == NodeKind.CLASS
    assert box.subtype == "class"
    assert box.header == "template <typename T>\nclass Box {"
    assert box.docstring is not None
    assert "Template box" in box.docstring

    access_blocks = [ch for ch in box.children if ch.kind == NodeKind.BLOCK]
    assert [ch.subtype for ch in access_blocks] == [
        "private",
        "public",
        "private",
        "protected",
        "public",
    ]

    first_private_names = {
        ch.name for ch in _flatten(access_blocks[0].children) if ch.name
    }
    public_names = {ch.name for ch in _flatten(access_blocks[1].children) if ch.name}
    second_private_names = {
        ch.name for ch in _flatten(access_blocks[2].children) if ch.name
    }
    protected_names = {
        ch.name for ch in _flatten(access_blocks[3].children) if ch.name
    }
    last_public_names = {
        ch.name for ch in _flatten(access_blocks[4].children) if ch.name
    }
    assert {"HiddenType", "hidden_value"}.issubset(first_private_names)
    assert {"ValueType", "Box", "get"}.issubset(public_names)
    assert {"value"}.issubset(second_private_names)
    assert {"touch"}.issubset(protected_names)
    assert {"set"}.issubset(last_public_names)

    get_method = next(
        ch for ch in _flatten(access_blocks[1].children) if ch.name == "get"
    )
    assert get_method.kind == NodeKind.METHOD
    assert get_method.subtype == "signature"
    assert get_method.header == "T get() const"

    touch_method = next(
        ch for ch in _flatten(access_blocks[3].children) if ch.name == "touch"
    )
    assert touch_method.kind == NodeKind.METHOD
    assert touch_method.subtype == "signature"
    assert touch_method.header == "void touch() const"

    set_method = next(
        ch for ch in _flatten(access_blocks[4].children) if ch.name == "set"
    )
    assert set_method.kind == NodeKind.METHOD
    assert set_method.subtype is None
    assert set_method.header == "void set(T value)"

    assert inner_named["IntBox"].kind == NodeKind.LITERAL
    assert inner_named["IntBox"].body.strip() == "using IntBox = Box<int>;"

    color = inner_named["Color"]
    assert color.kind == NodeKind.CLASS
    assert color.subtype == "enum"
    color_child_names = {ch.name for ch in color.children if ch.name}
    assert {"RED", "GREEN", "BLUE"}.issubset(color_child_names)

    value = inner_named["Value"]
    assert value.kind == NodeKind.CLASS
    assert value.subtype == "union"
    value_child_names = {ch.name for ch in value.children if ch.name}
    assert {"i", "f"}.issubset(value_child_names)

    point = inner_named["Point"]
    assert point.kind == NodeKind.CLASS
    assert point.subtype == "struct"
    assert point.docstring is not None
    assert "Points in 2D space" in point.docstring

    sum_fn = inner_named["sum"]
    assert sum_fn.kind == NodeKind.FUNCTION
    assert sum_fn.subtype == "signature"
    assert sum_fn.header == "int sum(int a, int b)"

    add_fn = inner_named["add"]
    assert add_fn.kind == NodeKind.FUNCTION
    assert add_fn.header == "template <typename T>\nT add(T a, T b)"
    assert add_fn.docstring is not None
    assert "Add two values" in add_fn.docstring

    preproc = next(
        ch
        for ch in inner.children
        if ch.kind == NodeKind.CUSTOM and ch.subtype == "preprocessor"
    )
    assert preproc.header == "#ifdef ENABLE_FEATURE"
    assert preproc.comment == "#endif"
    assert any(ch.name == "enabled" for ch in preproc.children)
    else_branch = next(
        ch
        for ch in preproc.children
        if ch.kind == NodeKind.CUSTOM and ch.subtype == "preprocessor"
    )
    assert else_branch.header == "#else"
    assert any(ch.name == "disabled" for ch in else_branch.children)