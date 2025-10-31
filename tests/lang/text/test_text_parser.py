import pytest
from pathlib import Path

from knowlt.settings import ProjectSettings
from knowlt.project import ProjectCache
from knowlt.lang.text import TextParser
from knowlt.models import Repo, NodeKind


class _DummyPM:
    def __init__(self, settings: ProjectSettings):
        self.settings = settings
        # TextParser uses create_project_chunker(pm), which accesses pm.embeddings
        self.embeddings = None


def test_text_parser_on_simple_file():
    samples_dir = Path(__file__).parent / "samples"
    settings = ProjectSettings(
        project_name="test",
        repo_name="test",
        repo_path=str(samples_dir),
    )
    pm = _DummyPM(settings)
    repo = Repo(id="test", name="test", root_path=str(samples_dir))
    cache = ProjectCache()

    rel_path = "simple.txt"
    parser = TextParser(pm, repo, rel_path)
    parsed_file = parser.parse(cache)

    # Basic assertions
    assert parsed_file.path.endswith(rel_path)
    assert parsed_file.package is None
    assert parsed_file.imports == []

    # Expect a single chunk/node for small files with default chunk settings
    assert len(parsed_file.nodes) == 1
    node = parsed_file.nodes[0]

    # Read source to validate fields
    text = (samples_dir / rel_path).read_text(encoding="utf-8")
    text_bytes = text.encode("utf-8")

    assert node.kind == NodeKind.LITERAL
    assert node.body == text
    assert node.start_line == 1
    assert node.end_line == text.count("\n") + 1
    assert node.start_byte == 0
    assert node.end_byte == len(text_bytes)
    assert node.header is None
    assert node.docstring is None
    assert node.comment is None