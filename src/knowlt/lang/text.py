import os
from typing import Optional, List, Any

from knowlt.chunking import (
    Chunk,
    AbstractChunker,
    create_project_chunker,
)
from knowlt.models import (
    ProgrammingLanguage,
    NodeKind,
    Node,
    Repo,
)
from knowlt.parsers import (
    AbstractCodeParser,
    AbstractLanguageHelper,
    ParsedFile,
    ParsedPackage,
    ParsedNode,
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from knowlt.project import ProjectManager, ProjectCache


class TextParser(AbstractCodeParser):
    language = ProgrammingLanguage.TEXT
    extensions = (".txt",)

    def __init__(self, pm: "ProjectManager", repo: Repo, rel_path: str):
        super().__init__(pm, repo, rel_path)
        self.parser = None
        self.source_bytes: bytes = b""
        self.package: Optional[ParsedPackage] = None
        self.parsed_file: Optional[ParsedFile] = None

        self.chunker: AbstractChunker = create_project_chunker(pm)

    def _rel_to_virtual_path(self, rel_path: str) -> str:
        return os.path.splitext(rel_path)[0].replace(os.sep, ".")

    def parse(self, cache: "ProjectCache") -> ParsedFile:
        if not self.repo.root_path:
            raise ValueError("repo.root_path must be set to parse files")

        file_path = os.path.join(self.repo.root_path, self.rel_path)
        mtime_ns: int = os.stat(file_path).st_mtime_ns
        with open(file_path, "rb") as file:
            self.source_bytes = file.read()

        self.parsed_file = self._create_file(file_path, mtime_ns)

        text = self.source_bytes.decode("utf-8", errors="replace")

        top_chunks = self.chunker.chunk(text)

        self.parsed_file.nodes.extend(self._chunk_to_node(ch, text) for ch in top_chunks)

        return self.parsed_file

    def _chunk_to_node(self, chunk: Chunk, full_text: str) -> ParsedNode:
        start_line = full_text[: chunk.start].count("\n") + 1
        end_line = full_text[: chunk.end].count("\n") + 1

        start_byte = len(full_text[: chunk.start].encode("utf-8"))
        end_byte = len(full_text[: chunk.end].encode("utf-8"))

        return ParsedNode(
            body=chunk.text,
            kind=NodeKind.LITERAL,
            start_line=start_line,
            end_line=end_line,
            start_byte=start_byte,
            end_byte=end_byte,
            header=None,
            docstring=None,
            comment=None,
        )

    def _process_node(
        self, node: Any, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        return []  # Not used by this parser


class TextHelper(AbstractLanguageHelper):
    language = ProgrammingLanguage.TEXT

    def get_node_summary(
        self,
        sym: Node,
        indent: int = 0,
        include_comments: bool = False,
        include_docs: bool = False,
        include_parents: bool = False,
    ) -> str:
        # Text symbols are flat, so we ignore parent/child context
        IND = " " * indent
        return "\n".join(f"{IND}{line}" for line in (sym.body or "").splitlines())

    def get_common_syntax_words(self) -> set[str]:
        return set()
