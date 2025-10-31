import os
from typing import Optional, List, Any

import tree_sitter_markdown as tsmd
from tree_sitter import Parser, Language

from knowlt.chunking import Chunk, AbstractChunker, create_project_chunker
from knowlt.models import (
    ProgrammingLanguage,
    NodeKind,
    Node,
    Repo,
)
from knowlt.parsers import (
    AbstractCodeParser,
    AbstractLanguageHelper,
    ParsedNode,
)
from knowlt.project import ProjectManager, ProjectCache

MARKDOWN_LANGUAGE = Language(tsmd.language())

_parser: Optional[Parser] = None


def _get_parser():
    global _parser
    if not _parser:
        _parser = Parser(MARKDOWN_LANGUAGE)
    return _parser


class MarkdownCodeParser(AbstractCodeParser):
    language = ProgrammingLanguage.MARKDOWN
    extensions = (".md", ".markdown")

    def __init__(self, pm: ProjectManager, repo: Repo, rel_path: str):
        super().__init__(pm, repo, rel_path)
        self.parser = _get_parser()
        self.chunker: AbstractChunker = create_project_chunker(pm)

    def _rel_to_virtual_path(self, rel_path: str) -> str:
        return os.path.splitext(rel_path)[0].replace(os.sep, ".")

    def _chunk_to_node(
        self,
        chunk: Chunk,
        section_body: str,
        section_start_byte: int,
        section_start_line: int,
    ) -> ParsedNode:
        rel_start_line = section_body[: chunk.start].count("\n")
        rel_end_line = section_body[: chunk.end].count("\n")

        start_byte_offset = len(section_body[: chunk.start].encode("utf-8"))
        end_byte_offset = len(section_body[: chunk.end].encode("utf-8"))

        return ParsedNode(
            body=chunk.text,
            kind=NodeKind.LITERAL,
            start_line=section_start_line + rel_start_line,
            end_line=section_start_line + rel_end_line,
            start_byte=section_start_byte + start_byte_offset,
            end_byte=section_start_byte + end_byte_offset,
            children=[],
        )

    def _process_node(
        self, node: Any, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        if node.type == "document":
            return self._handle_document(node, parent)
        elif node.type == "section":
            return self._handle_section(node, parent)
        else:
            return self._handle_generic_block(node, parent)

    def _create_package(self, root_node):
        return None

    def _handle_file(self, root_node: Any) -> None:
        """
        Optional hook for language-specific post-processing at file-level.
        With dispatcher pattern, most logic moved to _handle_document.
        """
        pass

    def _handle_document(
        self, node: Any, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        all_nodes: List[ParsedNode] = []
        # Handle potentially nested document structure from tree-sitter-markdown
        nodes_to_process = node.children
        if len(node.children) == 1 and node.children[0].type == "document":
            nodes_to_process = node.children[0].children

        for child_node in nodes_to_process:
            # Recursively process children and extend the list of symbols
            all_nodes.extend(self._process_node(child_node, parent))
        return all_nodes

    def _handle_section(
        self, node: Any, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        body = self.source_bytes[node.start_byte : node.end_byte].decode("utf-8")
        if not body.strip():
            return []
        parsed_node = self._make_node(
            node,
            kind=NodeKind.LITERAL,
        )

        # A section is "terminal" if it does not contain any sub-sections.
        is_terminal = not any(child.type == "section" for child in node.children)

        if not is_terminal:
            # For non-terminal sections, we parse children to build the hierarchy.
            child_nodes_to_process = node.children
            for child_node in child_nodes_to_process:
                parsed_node.children.extend(
                    self._process_node(child_node, parent=parsed_node)
                )
        else:
            # For terminal sections, we chunk the body if needed and create child nodes.
            top_chunks = self.chunker.chunk(body)

            # Only add children if chunking resulted in splits.
            has_splits = len(top_chunks) > 1
            if has_splits:
                section_start_byte = node.start_byte
                section_start_line = node.start_point[0]
                parsed_node.children.extend(
                    [
                        self._chunk_to_node(
                            chunk, body, section_start_byte, section_start_line
                        )
                        for chunk in top_chunks
                    ]
                )

        return [parsed_node]

    def _handle_generic_block(
        self, node: Any, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        body = self.source_bytes[node.start_byte : node.end_byte].decode("utf-8")
        if not body.strip():
            return []

        parsed_node = self._make_node(
            node,
            kind=NodeKind.LITERAL,
            # body captured via get_node_text by the base parser
        )

        # For generic blocks, we chunk the body if needed and create child nodes.
        top_chunks = self.chunker.chunk(body)

        # Only add children if chunking resulted in splits.
        has_splits = len(top_chunks) > 1
        if has_splits:
            section_start_byte = node.start_byte
            section_start_line = node.start_point[0]
            parsed_node.children.extend(
                [
                    self._chunk_to_node(
                        chunk, body, section_start_byte, section_start_line
                    )
                    for chunk in top_chunks
                ]
            )

        return [parsed_node]


class MarkdownLanguageHelper(AbstractLanguageHelper):
    language = ProgrammingLanguage.MARKDOWN

    def get_node_summary(
        self,
        sym: Node,
        indent: int = 0,
        include_comments: bool = False,
        include_docs: bool = False,
        include_parents: bool = False,
        child_stack: Optional[List[List[Node]]] = None,
    ) -> str:
        # Markdown symbols are flat, so we ignore parent/child context
        IND = " " * indent
        return "\n".join(f"{IND}{line}" for line in (sym.body or "").splitlines())

    def get_common_syntax_words(self) -> set[str]:
        return set()
