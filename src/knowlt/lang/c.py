import os
from typing import Optional, List, Callable

import tree_sitter as ts
import tree_sitter_c as tsc

from knowlt.parsers import (
    AbstractCodeParser,
    AbstractLanguageHelper,
    ParsedNode,
    ParsedImportEdge,
    get_node_text,
)
from knowlt.models import ProgrammingLanguage, NodeKind, Node, Repo
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlt.project import ProjectManager, ProjectCache
from knowlt.logger import logger


C_LANGUAGE = ts.Language(tsc.language())

_parser: Optional[ts.Parser] = None


def _get_parser() -> ts.Parser:
    global _parser
    if _parser is None:
        _parser = ts.Parser(C_LANGUAGE)
    return _parser


class CCodeParser(AbstractCodeParser):
    language = ProgrammingLanguage.C
    extensions = (".c", ".h")

    def __init__(self, pm: "ProjectManager", repo: Repo, rel_path: str):
        super().__init__(pm, repo, rel_path)
        self.parser = _get_parser()
        self.source_bytes: bytes = b""
        self._handlers: dict[
            str, Callable[[ts.Node, Optional[ParsedNode]], List[ParsedNode]]
        ] = {
            "preproc_include": self._handle_include,
            "preproc_def": self._handle_define,
            "preproc_function_def": self._handle_define,
            "preproc_if": self._handle_preprocessor_conditional,
            "preproc_ifdef": self._handle_preprocessor_conditional,
            "preproc_elif": self._handle_preprocessor_conditional,
            "preproc_else": self._handle_preprocessor_conditional,
            "function_definition": self._handle_function,
            "declaration": self._handle_declaration,
            "type_definition": self._handle_type_definition,
            "struct_specifier": self._handle_struct,
            "enum_specifier": self._handle_enum,
            "union_specifier": self._handle_union,
            "field_declaration": self._handle_field_declaration,
            "enumerator": self._handle_enumerator,
            "field_declaration_list": self._handle_field_declaration_list,
            "enumerator_list": self._handle_enumerator_list,
            "comment": self._handle_comment,
        }

    def parse(self, cache: "ProjectCache"):
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

        for child in root_node.named_children:
            nodes = self._process_node(child)

            if nodes:
                self.parsed_file.nodes.extend(nodes)
            elif child.type != "comment":
                logger.warning(
                    "Parser handled node but produced no symbols",
                    path=self.parsed_file.path,
                    node_type=child.type,
                    line=child.start_point[0] + 1,
                    raw=child.text.decode("utf8", errors="replace"),
                )

        self._handle_file(root_node)

        if self.package:
            self.package.imports = list(self.parsed_file.imports)

        return self.parsed_file

    def _handle_file(self, root_node: ts.Node) -> None:
        return None

    def _rel_to_virtual_path(self, rel_path: str) -> str:
        return rel_path.replace(os.sep, "/")

    def _process_node(
        self, node: ts.Node, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        handler = self._handlers.get(node.type)
        if handler is not None:
            return handler(node, parent)
        self._debug_unknown_node(node)
        return [self._literal_node(node)]

    def _get_preceding_comment(self, node: ts.Node) -> Optional[str]:
        prev = node.prev_sibling
        if prev is None:
            return None
        comment_nodes = []
        expected_line = node.start_point[0]
        while prev is not None and prev.type == "comment":
            if prev.end_point[0] + 1 != expected_line:
                break
            comment_nodes.append(prev)
            expected_line = prev.start_point[0]
            prev = prev.prev_sibling
        if not comment_nodes:
            return None
        comment_nodes.reverse()
        parts: list[str] = []
        for c in comment_nodes:
            parts.append((get_node_text(c) or "").strip())
        return "\n".join(parts).strip() or None

    def _literal_node(self, node: ts.Node) -> ParsedNode:
        return self._make_node(node, kind=NodeKind.LITERAL, name=None, header=None)

    def _find_first_descendant(
        self, node: Optional[ts.Node], kinds: tuple[str, ...]
    ) -> Optional[ts.Node]:
        if node is None:
            return None
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur.type in kinds:
                return cur
            stack.extend(reversed(cur.children))
        return None

    def _resolve_include_path(self, include_path: str, node: ts.Node) -> ParsedImportEdge:
        include_node = next(
            (
                child
                for child in node.children
                if child.type in ("string_literal", "system_lib_string")
            ),
            None,
        )
        external = include_node is None or include_node.type == "system_lib_string"
        physical_path: Optional[str] = None

        if not external:
            base_dir = os.path.dirname(os.path.join(self.repo.root_path, self.rel_path))
            abs_target = os.path.normpath(os.path.join(base_dir, include_path))
            if abs_target.startswith(self.repo.root_path) and os.path.isfile(abs_target):
                physical_path = os.path.relpath(abs_target, self.repo.root_path).replace(
                    os.sep, "/"
                )
                external = False
            else:
                repo_target = os.path.join(self.repo.root_path, include_path)
                if os.path.isfile(repo_target):
                    physical_path = include_path.replace(os.sep, "/")
                    external = False
                else:
                    external = True

        return ParsedImportEdge(
            physical_path=physical_path,
            virtual_path=include_path,
            alias=None,
            dot=False,
            external=external,
            raw=(get_node_text(node) or "").strip(),
        )

    def _handle_include(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        include_node = next(
            (
                child
                for child in node.children
                if child.type in ("string_literal", "system_lib_string")
            ),
            None,
        )
        if include_node is not None:
            include_path = (get_node_text(include_node) or "").strip().strip('"<>')
            if include_path:
                assert self.parsed_file is not None
                self.parsed_file.imports.append(
                    self._resolve_include_path(include_path, node)
                )
        return [
            self._make_node(
                node,
                kind=NodeKind.LITERAL,
                name=None,
                header=None,
                subtype="import",
                docstring=self._get_preceding_comment(node),
            )
        ]

    def _handle_define(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        ident = next((child for child in node.children if child.type == "identifier"), None)
        name = get_node_text(ident) if ident is not None else None
        return [
            self._make_node(
                node,
                kind=NodeKind.LITERAL,
                name=name,
                header=None,
                docstring=self._get_preceding_comment(node),
            )
        ]

    def _build_declaration_header(self, node: ts.Node) -> str:
        return (get_node_text(node) or "").strip().rstrip(";")

    def _build_function_header(self, node: ts.Node) -> str:
        body_node = next(
            (child for child in node.children if child.type == "compound_statement"),
            None,
        )
        if body_node is not None:
            return (
                self.source_bytes[node.start_byte : body_node.start_byte]
                .decode("utf8")
                .rstrip()
            )
        return (get_node_text(node) or "").split("{", 1)[0].strip()

    def _handle_function(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        decl = next(
            (child for child in node.children if child.type == "function_declarator"),
            None,
        )
        ident = self._find_first_descendant(decl, ("identifier",))
        if ident is None:
            return [self._literal_node(node)]
        return [
            self._make_node(
                node,
                kind=NodeKind.FUNCTION,
                name=get_node_text(ident),
                header=self._build_function_header(node),
                docstring=self._get_preceding_comment(node),
            )
        ]

    def _extract_declarator_name(self, node: Optional[ts.Node]) -> Optional[str]:
        ident = self._find_first_descendant(node, ("identifier", "field_identifier"))
        return get_node_text(ident) if ident is not None else None

    def _is_declaration_aggregate(self, node: ts.Node) -> bool:
        return any(
            child.type in ("struct_specifier", "enum_specifier", "union_specifier")
            for child in node.children
        )

    def _handle_declaration(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        if self._is_declaration_aggregate(node):
            return [self._literal_node(node)]

        function_decl = next(
            (child for child in node.children if child.type == "function_declarator"),
            None,
        )
        if function_decl is not None:
            name = self._extract_declarator_name(function_decl)
            if name is None:
                return [self._literal_node(node)]
            return [
                self._make_node(
                    node,
                    kind=NodeKind.FUNCTION,
                    name=name,
                    header=self._build_declaration_header(node),
                    subtype="signature",
                    docstring=self._get_preceding_comment(node),
                )
            ]

        declarators = [
            child
            for child in node.children
            if child.type
            in (
                "identifier",
                "init_declarator",
                "pointer_declarator",
                "array_declarator",
                "parenthesized_declarator",
            )
        ]
        if not declarators:
            return [self._literal_node(node)]

        decl_header = self._build_declaration_header(node)
        doc = self._get_preceding_comment(node)
        symbols: List[ParsedNode] = []
        for declarator in declarators:
            name = self._extract_declarator_name(declarator)
            symbols.append(
                self._make_node(
                    node,
                    kind=NodeKind.VARIABLE,
                    name=name,
                    header=None,
                    docstring=doc,
                    body=decl_header,
                )
            )
        return symbols

    def _extract_aggregate_name(
        self, specifier: ts.Node, container: Optional[ts.Node] = None
    ) -> Optional[str]:
        spec_name = self._find_first_descendant(specifier, ("type_identifier",))
        if container is None:
            return get_node_text(spec_name) if spec_name is not None else None

        trailing_type_identifiers = [
            child for child in container.children if child.type == "type_identifier"
        ]
        if trailing_type_identifiers:
            return get_node_text(trailing_type_identifiers[-1])
        return get_node_text(spec_name) if spec_name is not None else None

    def _build_aggregate_header(self, node: ts.Node, body_node: ts.Node) -> str:
        prefix = (
            self.source_bytes[node.start_byte : body_node.start_byte].decode("utf8").rstrip()
        )
        if not prefix.endswith("{"):
            prefix = f"{prefix} {{"
        return prefix.strip()

    def _build_aggregate_node(
        self,
        node: ts.Node,
        body_node: ts.Node,
        *,
        name: Optional[str],
        subtype: str,
        comment: Optional[str],
    ) -> ParsedNode:
        sym = self._make_node(
            node,
            kind=NodeKind.CLASS,
            name=name,
            header=self._build_aggregate_header(node, body_node),
            subtype=subtype,
            docstring=comment,
        )
        for child in body_node.named_children:
            sym.children.extend(self._process_node(child, sym))
        return sym

    def _handle_type_definition(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        comment = self._get_preceding_comment(node)
        specifier = next(
            (
                child
                for child in node.children
                if child.type in ("struct_specifier", "enum_specifier", "union_specifier")
            ),
            None,
        )
        if specifier is None:
            name_node = next(
                (child for child in reversed(node.children) if child.type == "type_identifier"),
                None,
            )
            name = get_node_text(name_node) if name_node is not None else None
            return [
                self._make_node(
                    node,
                    kind=NodeKind.LITERAL,
                    name=name,
                    header=None,
                    docstring=comment,
                )
            ]

        subtype = {
            "struct_specifier": "struct",
            "enum_specifier": "enum",
            "union_specifier": "union",
        }[specifier.type]
        body_type = (
            "enumerator_list" if subtype == "enum" else "field_declaration_list"
        )
        body_node = next(
            (child for child in specifier.children if child.type == body_type),
            None,
        )
        name = self._extract_aggregate_name(specifier, node)
        if body_node is None:
            return [
                self._make_node(
                    node,
                    kind=NodeKind.LITERAL,
                    name=name,
                    header=None,
                    docstring=comment,
                )
            ]
        return [
            self._build_aggregate_node(
                node,
                body_node,
                name=name,
                subtype=subtype,
                comment=comment,
            )
        ]

    def _handle_struct(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        body_node = next(
            (child for child in node.children if child.type == "field_declaration_list"),
            None,
        )
        if body_node is None:
            return [self._literal_node(node)]
        return [
            self._build_aggregate_node(
                node,
                body_node,
                name=self._extract_aggregate_name(node),
                subtype="struct",
                comment=self._get_preceding_comment(node),
            )
        ]

    def _handle_union(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        body_node = next(
            (child for child in node.children if child.type == "field_declaration_list"),
            None,
        )
        if body_node is None:
            return [self._literal_node(node)]
        return [
            self._build_aggregate_node(
                node,
                body_node,
                name=self._extract_aggregate_name(node),
                subtype="union",
                comment=self._get_preceding_comment(node),
            )
        ]

    def _preprocessor_header_only(self, node: ts.Node) -> str:
        raw = (get_node_text(node) or "").strip()
        lines = raw.splitlines()
        if not lines:
            return raw
        return lines[0].strip()

    def _preprocessor_footer(self, node: ts.Node) -> Optional[str]:
        raw = (get_node_text(node) or "").strip()
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            return None
        last = lines[-1]
        return last if last.startswith("#endif") else None

    def _handle_preprocessor_conditional(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        sym = self._make_node(
            node,
            kind=NodeKind.CUSTOM,
            name=None,
            header=self._preprocessor_header_only(node),
            subtype="preprocessor",
            docstring=self._get_preceding_comment(node),
        )
        footer = self._preprocessor_footer(node)
        if footer:
            sym.comment = footer
        for child in node.named_children:
            if child.type in ("identifier", "preproc_defined"):
                continue
            sym.children.extend(self._process_node(child, sym))
        return [sym]

    def _handle_enum(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        body_node = next(
            (child for child in node.children if child.type == "enumerator_list"),
            None,
        )
        if body_node is None:
            return [self._literal_node(node)]
        return [
            self._build_aggregate_node(
                node,
                body_node,
                name=self._extract_aggregate_name(node),
                subtype="enum",
                comment=self._get_preceding_comment(node),
            )
        ]

    def _handle_field_declaration(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        name_node = self._find_first_descendant(
            node,
            ("field_identifier", "identifier"),
        )
        return [
            self._make_node(
                node,
                kind=NodeKind.PROPERTY,
                name=get_node_text(name_node) if name_node is not None else None,
                header=None,
                docstring=self._get_preceding_comment(node),
            )
        ]

    def _handle_enumerator(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        name_node = self._find_first_descendant(node, ("identifier",))
        return [
            self._make_node(
                node,
                kind=NodeKind.PROPERTY,
                name=get_node_text(name_node) if name_node is not None else None,
                header=None,
                docstring=self._get_preceding_comment(node),
            )
        ]

    def _handle_field_declaration_list(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        out: List[ParsedNode] = []
        for child in node.named_children:
            out.extend(self._process_node(child, parent))
        return out

    def _handle_enumerator_list(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        out: List[ParsedNode] = []
        for child in node.named_children:
            out.extend(self._process_node(child, parent))
        return out

    def _handle_comment(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        return [self._make_node(node, kind=NodeKind.COMMENT, name=None, header=None)]

    def _debug_unknown_node(
        self,
        node: ts.Node,
        *,
        context: Optional[str] = None,
        parent_type: Optional[str] = None,
    ) -> None:
        path = self.parsed_file.path if self.parsed_file else self.rel_path
        fields = dict(
            path=path,
            node_type=node.type,
            line=node.start_point[0] + 1,
            raw=(get_node_text(node) or "")[:200],
        )
        if context is not None:
            fields["context"] = context
        if parent_type is not None:
            fields["parent_type"] = parent_type
        logger.debug("Unknown C node type", **fields)


class CLanguageHelper(AbstractLanguageHelper):
    language = ProgrammingLanguage.C

    def get_node_summary(
        self,
        sym: Node,
        indent: int = 0,
        include_comments: bool = False,
        include_docs: bool = False,
        include_parents: bool = False,
        child_stack: Optional[List[List[Node]]] = None,
    ) -> str:
        if include_docs and sym.kind == NodeKind.COMMENT:
            return ""

        if include_parents:
            if sym.parent_ref:
                return self.get_node_summary(
                    sym.parent_ref,
                    indent,
                    include_comments,
                    include_docs,
                    include_parents,
                    (child_stack or []) + [[sym]],
                )
            include_parents = False

        only_children = child_stack.pop() if child_stack else None
        IND = "\t" * indent
        lines: List[str] = []

        def emit_children(children: List[Node], child_indent: int) -> None:
            if only_children:
                lines.append(f"{IND}\t...")
            body_symbols_added = False
            for ch in children:
                if only_children and ch not in only_children:
                    continue
                if not include_comments and ch.kind == NodeKind.COMMENT:
                    continue
                ch_sum = self.get_node_summary(
                    ch,
                    indent=child_indent,
                    include_comments=include_comments,
                    include_docs=include_docs,
                )
                if ch_sum:
                    lines.append(ch_sum)
                    body_symbols_added = True
            if not body_symbols_added:
                lines.append(f"{IND}\t...")

        if include_comments and sym.comment:
            for ln in (sym.comment or "").splitlines():
                lines.append(f"{IND}{ln.strip()}")

        if sym.kind in (NodeKind.CLASS, NodeKind.FUNCTION, NodeKind.METHOD):
            header = (sym.header or "").strip()
            subtype = (sym.subtype or "").lower()
            if (
                sym.kind in (NodeKind.FUNCTION, NodeKind.METHOD)
                and subtype not in ("signature",)
                and header
                and not header.endswith("{")
            ):
                header = f"{header} {{"
            if include_docs and sym.docstring:
                for ln in sym.docstring.splitlines():
                    lines.append(f"{IND}{ln.strip()}")
            if subtype in ("signature",) and header and not header.endswith(";"):
                header = f"{header};"
            if header:
                for ln in header.splitlines():
                    lines.append(f"{IND}{ln.strip()}")
            if subtype in ("signature",):
                return "\n".join(lines)
            if sym.children:
                emit_children(sym.children, indent + 1)
            else:
                lines.append(f"{IND}\t...")
            lines.append(f"{IND}}}")
            return "\n".join(lines)

        if sym.kind == NodeKind.CUSTOM and (sym.subtype or "").lower() == "preprocessor":
            if include_docs and sym.docstring:
                for ln in sym.docstring.splitlines():
                    lines.append(f"{IND}{ln.strip()}")
            header = (sym.header or sym.body or "").strip()
            if header:
                lines.append(f"{IND}{header}")
            visible_children = [
                ch for ch in sym.children if (not only_children or ch in only_children)
            ]
            branch_children = [
                ch
                for ch in visible_children
                if ch.kind == NodeKind.CUSTOM
                and (ch.subtype or "").lower() == "preprocessor"
            ]
            content_children = [ch for ch in visible_children if ch not in branch_children]
            if content_children:
                lines.append(f"{IND}\t...")
            elif not branch_children:
                lines.append(f"{IND}\t...")
            for ch in branch_children:
                ch_sum = self.get_node_summary(
                    ch,
                    indent=indent,
                    include_comments=include_comments,
                    include_docs=include_docs,
                )
                if ch_sum:
                    lines.append(ch_sum)
            footer = (sym.comment or "").strip()
            if footer:
                lines.append(f"{IND}{footer}")
            return "\n".join(lines)

        if sym.kind == NodeKind.BLOCK and sym.children:
            label = (sym.subtype or "block").lower()
            lines.append(f"{IND}{label}:")
            emit_children(sym.children, indent + 1)
            return "\n".join(lines)

        if sym.children:
            emit_children(sym.children, indent)
            return "\n".join(lines)

        if include_docs and sym.docstring:
            for ln in sym.docstring.splitlines():
                lines.append(f"{IND}{ln.strip()}")
        body = (sym.body or "").strip()
        if body:
            lines.append(f"{IND}{body}")
        return "\n".join(lines)

    def get_common_syntax_words(self) -> set[str]:
        return {
            "auto",
            "break",
            "case",
            "const",
            "continue",
            "default",
            "define",
            "do",
            "else",
            "enum",
            "extern",
            "for",
            "goto",
            "if",
            "include",
            "inline",
            "register",
            "return",
            "sizeof",
            "static",
            "struct",
            "switch",
            "typedef",
            "void",
            "volatile",
            "while",
        }