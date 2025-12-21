import os
from typing import Optional, List, Dict, Any, Type, Tuple
from abc import ABC, abstractmethod
import inspect
import textwrap
import tree_sitter as ts
from pydantic import BaseModel, Field
from knowlt.models import (
    ProgrammingLanguage,
    NodeKind,
    Visibility,
    Modifier,
    Repo,
    Package,
    File,
    Node,
    ImportEdge,
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from knowlt.project import ProjectManager, ProjectCache
from knowlt.helpers import compute_file_hash
from knowlt.logger import logger


# Parser-specific data structures
class ParsedImportEdge(BaseModel):
    physical_path: Optional[str] = (
        None  # relative physical path to package from project root.
    )
    virtual_path: str  # syntax specific virtual path to package
    alias: Optional[str] = None  # import alias if any
    dot: bool = False  # true for dot-imports (import . "pkg")
    external: bool
    raw: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "to_package_physical_path": self.physical_path,
            "to_package_virtual_path": self.virtual_path,
            "alias": self.alias,
            "dot": self.dot,
            "external": self.external,
            "raw": self.raw,
        }


class ParsedPackage(BaseModel):
    language: ProgrammingLanguage
    physical_path: str  # relative path to package
    virtual_path: str  # syntax specific virtual path to package
    imports: List[ParsedImportEdge] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": (self.virtual_path or "").split("/")[-1],
            "language": self.language,
            "virtual_path": self.virtual_path,
            "physical_path": self.physical_path,
        }


class ParsedNode(BaseModel):
    name: Optional[str] = None
    body: str
    kind: NodeKind
    subtype: Optional[str] = None

    start_line: int
    end_line: int
    start_byte: int
    end_byte: int

    header: Optional[str] = None
    visibility: Optional[Visibility] = None
    docstring: Optional[str] = None
    comment: Optional[str] = None

    children: List["ParsedNode"] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "body": self.body,
            "kind": self.kind,
            "subtype": self.subtype,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "header": self.header,
            "visibility": self.visibility,
            "docstring": self.docstring,
            "comment": self.comment,
        }


class ParsedFile(BaseModel):
    package: Optional[ParsedPackage] = None
    path: str  # relative path
    docstring: Optional[str] = None

    file_hash: Optional[str] = None
    last_updated: Optional[int] = None

    nodes: List[ParsedNode] = Field(default_factory=list)
    imports: List[ParsedImportEdge] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_hash": self.file_hash,
            "last_updated": self.last_updated,
        }


# Abstract base parser class
class AbstractCodeParser(ABC):
    """
    Abstract base class for code parsers.
    """

    language: ProgrammingLanguage
    extensions: List[str]
    pm: "ProjectManager"
    repo: Repo
    rel_path: str
    source_bytes: bytes
    package: ParsedPackage | None
    parsed_file: ParsedFile | None
    parser: Any

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if not inspect.isabstract(cls):
            if not hasattr(cls, "extensions") or not cls.extensions:
                raise ValueError(f"{cls.__name__} missing `extensions`")
            CodeParserRegistry.register_parser(cls)

    def __init__(self, pm: "ProjectManager", repo: Repo, rel_path: str) -> None:
        self.pm = pm
        self.repo = repo
        self.rel_path = rel_path
        self.package = None
        self.parsed_file = None

    @abstractmethod
    def _rel_to_virtual_path(self, rel_path: str) -> str: ...

    @abstractmethod
    def _process_node(
        self, node: Any, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]: ...

    def _handle_file(self, root_node: Any) -> None:
        """
        Optional hook for language-specific post-processing at file-level.
        """
        pass

    # Helpers
    def parse(self, cache: "ProjectCache") -> ParsedFile:
        if not self.repo.root_path:
            raise ValueError("repo.root_path must be set to parse files")

        file_path = os.path.join(self.repo.root_path, self.rel_path)
        mtime_ns: int = os.stat(file_path).st_mtime_ns
        with open(file_path, "rb") as file:
            self.source_bytes = file.read()

        tree = self.parser.parse(self.source_bytes)
        root_node = tree.root_node

        self.package = self._create_package(root_node)
        self.parsed_file = self._create_file(file_path, mtime_ns)

        # Traverse the syntax tree and populate Parsed structures
        for child in root_node.children:
            nodes = self._process_node(child)

            if nodes:
                self.parsed_file.nodes.extend(nodes)
            else:
                # Some handlers intentionally merge multiple syntax nodes into a
                # single symbol (for example, consecutive `comment` nodes in
                # Terraform). In those cases it's expected that later nodes in
                # the merged run produce no symbols, so we skip the warning.
                if child.type == "comment":
                    continue

                logger.warning(
                    "Parser handled node but produced no symbols",
                    path=self.parsed_file.path,
                    node_type=child.type,
                    line=child.start_point[0] + 1,
                    raw=child.text.decode("utf8", errors="replace"),
                )

        self._handle_file(root_node)

        # Sync package-level imports with file-level imports
        if self.package:
            self.package.imports = list(self.parsed_file.imports)

        return self.parsed_file

    def _create_package(self, root_node):
        return ParsedPackage(
            language=self.language,
            physical_path=self.rel_path,
            virtual_path=self._rel_to_virtual_path(self.rel_path),
            imports=[],
        )

    def _create_file(self, file_path, mtime_ns):
        return ParsedFile(
            package=self.package,
            path=self.rel_path,
            file_hash=compute_file_hash(file_path),
            last_updated=mtime_ns,
            nodes=[],
            imports=[],
        )

    def _make_node(
        self,
        node: Any,
        *,
        kind: NodeKind,
        name: Optional[str] = None,
        header: Optional[str] = None,
        subtype: Optional[str] = None,
        comment: Optional[str] = None,
        docstring: Optional[str] = None,
        body: Optional[str] = None,
    ) -> ParsedNode:
        """
        Construct a ParsedNode from a tree-sitter node with common fields.
        """
        text = body if body is not None else (get_node_text(node) or "")
        return ParsedNode(
            name=name,
            body=text,
            kind=kind,
            subtype=subtype,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            header=header,
            docstring=docstring,
            comment=comment,
        )


class AbstractLanguageHelper:
    """
    Abstract base language helper class
    """

    language: ProgrammingLanguage

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if not inspect.isabstract(cls):
            if not hasattr(cls, "language"):
                raise ValueError(f"{cls.__name__} missing `language`")
            CodeParserRegistry.register_helper(cls.language, cls())

    # TODO: Create generic language formatter
    @abstractmethod
    def get_node_summary(
        self,
        sym: Node,
        indent: int = 0,
        include_comments: bool = False,
        include_docs: bool = False,
        include_parents: bool = False,
    ) -> str:
        """
        Generate symbol summary (comment, definition and a docstring if available) as a string
        with correct identation. For functions and methods, function body is replaced
        with a filler.
        """
        ...

    @abstractmethod
    def get_common_syntax_words(self) -> set[str]:
        """
        Return common language constructs that make sense removing from search
        indexes: def, if, else and so on.
        """
        ...


class CodeParserRegistry:
    """
    Singleton registry mapping file extensions to CodeParser implementations.
    """

    _instance = None
    _parsers: List[Type[AbstractCodeParser]] = []
    _lang_helpers: Dict[ProgrammingLanguage, AbstractLanguageHelper] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CodeParserRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register_helper(
        cls, lang: ProgrammingLanguage, helper: AbstractLanguageHelper
    ) -> None:
        cls._lang_helpers[lang] = helper

    @classmethod
    def get_helper(cls, lang: ProgrammingLanguage) -> Optional[AbstractLanguageHelper]:
        return cls._lang_helpers.get(lang)

    @classmethod
    def register_parser(cls, parser: Type[AbstractCodeParser]) -> None:
        if parser not in cls._parsers:
            cls._parsers.append(parser)

    @classmethod
    def get_parsers(cls) -> List[Type[AbstractCodeParser]]:
        return cls._parsers


# Helpers
def dedent_comment(text: str) -> str:
    """
    Dedents a comment string by calculating the minimum indentation
    from all non-empty lines and removing it.
    """
    lines = text.splitlines()
    if not lines:
        return ""

    min_indent: Optional[int] = None

    # Find minimum indentation of non-empty lines
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            if min_indent is None:
                min_indent = indent
            else:
                min_indent = min(min_indent, indent)

    if min_indent is None:
        return "\n".join(lines)  # all lines are empty or whitespace

    # Remove indentation and reconstruct
    dedented_lines = []
    for line in lines:
        dedented_lines.append(line[min_indent:])

    return "\n".join(dedented_lines)


def get_node_text(node) -> str:
    """
    Get text of the tree sitter node
    """
    if not node or not node.text:
        return ""

    return node.text.decode("utf-8")
