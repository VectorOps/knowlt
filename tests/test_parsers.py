import pytest

from knowlt.lang.markdown import MarkdownCodeParser
from knowlt.lang.python import PythonCodeParser
from knowlt.lang.text import TextParser
from knowlt.parsers import CodeParserRegistry
from knowlt.settings import ProjectSettings


def test_code_parser_registry_has_default_extension_map():
    parser_map = CodeParserRegistry.get_extension_map()

    assert parser_map[".py"] is PythonCodeParser
    assert parser_map[".md"] is MarkdownCodeParser
    assert parser_map[".txt"] is TextParser


def test_code_parser_registry_uses_explicit_parser_keys():
    parser_keys = {parser.parser_key for parser in CodeParserRegistry.get_parsers()}

    assert "python" in parser_keys
    assert "text" in parser_keys


def test_code_parser_registry_merges_language_and_parser_extension_settings():
    settings = ProjectSettings(
        parser_extensions={"md": "text", ".custompy": "python"},
        languages={"python": {"extra_extensions": ["pyi"]}},
    )

    parser_map = CodeParserRegistry.get_extension_map(settings)

    assert parser_map[".py"] is PythonCodeParser
    assert parser_map[".pyi"] is PythonCodeParser
    assert parser_map[".custompy"] is PythonCodeParser
    assert parser_map[".md"] is TextParser
    assert parser_map[".txt"] is TextParser


def test_code_parser_registry_rejects_unknown_parser_override():
    settings = ProjectSettings(parser_extensions={".foo": "missing"})

    with pytest.raises(ValueError, match="Unknown parser"):
        CodeParserRegistry.get_extension_map(settings)