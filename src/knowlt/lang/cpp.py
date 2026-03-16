import os
from typing import Optional, List, Callable

import tree_sitter as ts
import tree_sitter_cpp as tscpp

from knowlt.lang.c import CCodeParser, CLanguageHelper
from knowlt.parsers import ParsedNode, get_node_text
from knowlt.models import ProgrammingLanguage, NodeKind, Node, Repo
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlt.project import ProjectManager, ProjectCache
from knowlt.logger import logger


CPP_LANGUAGE = ts.Language(tscpp.language())

_parser: Optional[ts.Parser] = None


def _get_parser() -> ts.Parser:
    global _parser
    if _parser is None:
        _parser = ts.Parser(CPP_LANGUAGE)
    return _parser


class CppCodeParser(CCodeParser):
    language = ProgrammingLanguage.CPP
    extensions = (".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".ipp", ".tpp")

    def __init__(self, pm: "ProjectManager", repo: Repo, rel_path: str):
        super().__init__(pm, repo, rel_path)
        self.parser = _get_parser()
        self._handlers.update(
            {
                "template_declaration": self._handle_template,
                "namespace_definition": self._handle_namespace,
                "class_specifier": self._handle_class,
                "alias_declaration": self._handle_alias_declaration,
                "using_declaration": self._handle_using_declaration,
                "friend_declaration": self._handle_friend_declaration,
            }
        )

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
        logger.debug("Unknown C++ node type", **fields)

    def _find_last_descendant(
        self, node: Optional[ts.Node], kinds: tuple[str, ...]
    ) -> Optional[ts.Node]:
        if node is None:
            return None

        found: Optional[ts.Node] = None
        if node.type in kinds:
            found = node
        for child in node.children:
            child_found = self._find_last_descendant(child, kinds)
            if child_found is not None:
                found = child_found
        return found

    def _extract_declarator_name(self, node: Optional[ts.Node]) -> Optional[str]:
        if node is None:
            return None

        if node.type == "function_declarator":
            declarator = next(
                (
                    child
                    for child in node.children
                    if child.type
                    not in (
                        "parameter_list",
                        "type_qualifier",
                        "reference_qualifier",
                        "noexcept",
                        "requires_clause",
                        "trailing_return_type",
                        "attribute_specifier",
                        "virtual_specifier",
                    )
                ),
                None,
            )
            return self._extract_declarator_name(declarator)

        if node.type in ("qualified_identifier", "template_function"):
            name_node = self._find_last_descendant(
                node,
                ("identifier", "field_identifier", "type_identifier", "destructor_name", "operator_name"),
            )
            return get_node_text(name_node) if name_node is not None else None

        name_node = self._find_last_descendant(
            node,
            (
                "identifier",
                "field_identifier",
                "type_identifier",
                "destructor_name",
                "operator_name",
            ),
        )
        if name_node is not None:
            return get_node_text(name_node)

        if node.type == "operator_cast":
            raw = (get_node_text(node) or "").strip()
            return raw.split("(", 1)[0].strip() or None
        return None

    def _is_declaration_aggregate(self, node: ts.Node) -> bool:
        return any(
            child.type
            in (
                "struct_specifier",
                "enum_specifier",
                "union_specifier",
                "class_specifier",
            )
            for child in node.children
        )

    def _is_method_declarator(
        self, node: Optional[ts.Node], parent: Optional[ParsedNode]
    ) -> bool:
        if parent is not None and parent.kind == NodeKind.CLASS:
            return True

        qualified = self._find_first_descendant(node, ("qualified_identifier",))
        return qualified is not None and "::" in (get_node_text(qualified) or "")

    def _build_namespace_header(self, name: Optional[str]) -> str:
        if name:
            return f"namespace {name} {{"
        return "namespace {"

    def _build_namespace_chain(
        self,
        node: ts.Node,
        names: List[Optional[str]],
        comment: Optional[str],
    ) -> ParsedNode:
        root = self._make_node(
            node,
            kind=NodeKind.NAMESPACE,
            name=names[0],
            header=self._build_namespace_header(names[0]),
            docstring=comment,
        )
        cur = root
        for name in names[1:]:
            child = self._make_node(
                node,
                kind=NodeKind.NAMESPACE,
                name=name,
                header=self._build_namespace_header(name),
            )
            cur.children.append(child)
            cur = child
        return root

    def _prefix_template(self, sym: ParsedNode, prefix: str, comment: Optional[str]) -> None:
        prefix = prefix.strip()
        if not prefix:
            return
        if sym.header:
            sym.header = f"{prefix}\n{sym.header}"
        sym.body = f"{prefix}\n{sym.body}".strip()
        if sym.docstring is None and comment:
            sym.docstring = comment

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
        current_block: Optional[ParsedNode] = None
        for child in body_node.named_children:
            if body_node.type == "field_declaration_list" and child.type == "access_specifier":
                current_block = self._make_node(
                    child,
                    kind=NodeKind.BLOCK,
                    name=None,
                    header=None,
                    subtype=(get_node_text(child) or "").strip(),
                )
                sym.children.append(current_block)
                continue

            if (
                body_node.type == "field_declaration_list"
                and subtype == "class"
                and current_block is None
            ):
                current_block = self._make_node(
                    node,
                    kind=NodeKind.BLOCK,
                    name=None,
                    header=None,
                    subtype="private",
                )
                sym.children.append(current_block)

            processed = self._process_node(child, sym)
            if current_block is not None and body_node.type == "field_declaration_list":
                current_block.children.extend(processed)
            else:
                sym.children.extend(processed)
        return sym

    def _handle_namespace(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        body = next((child for child in node.children if child.type == "declaration_list"), None)
        if body is None:
            return [self._literal_node(node)]

        nested = next(
            (child for child in node.children if child.type == "nested_namespace_specifier"),
            None,
        )
        if nested is not None:
            names = [
                get_node_text(child) or None
                for child in nested.children
                if child.type == "namespace_identifier"
            ]
        else:
            name_node = next(
                (child for child in node.children if child.type == "namespace_identifier"),
                None,
            )
            names = [get_node_text(name_node) or None] if name_node is not None else [None]

        root = self._build_namespace_chain(node, names, self._get_preceding_comment(node))
        leaf = root
        while leaf.children and leaf.children[-1].kind == NodeKind.NAMESPACE:
            leaf = leaf.children[-1]

        for child in body.named_children:
            leaf.children.extend(self._process_node(child, leaf))
        return [root]

    def _handle_template(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        inner = next(
            (
                child
                for child in reversed(node.named_children)
                if child.type != "template_parameter_list"
            ),
            None,
        )
        if inner is None:
            return [self._literal_node(node)]

        prefix = self.source_bytes[node.start_byte : inner.start_byte].decode("utf8").strip()
        comment = self._get_preceding_comment(node)
        syms = self._process_node(inner, parent)
        for sym in syms:
            self._prefix_template(sym, prefix, comment)
        return syms

    def _handle_class(
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
                subtype="class",
                comment=self._get_preceding_comment(node),
            )
        ]

    def _handle_alias_declaration(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        name_node = next(
            (child for child in node.children if child.type == "type_identifier"),
            None,
        )
        return [
            self._make_node(
                node,
                kind=NodeKind.LITERAL,
                name=get_node_text(name_node) if name_node is not None else None,
                header=None,
                docstring=self._get_preceding_comment(node),
            )
        ]

    def _handle_using_declaration(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        name_node = self._find_last_descendant(
            node,
            (
                "identifier",
                "field_identifier",
                "namespace_identifier",
                "type_identifier",
            ),
        )
        return [
            self._make_node(
                node,
                kind=NodeKind.LITERAL,
                name=get_node_text(name_node) if name_node is not None else None,
                header=None,
                docstring=self._get_preceding_comment(node),
            )
        ]

    def _handle_friend_declaration(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        return [
            self._make_node(
                node,
                kind=NodeKind.LITERAL,
                name=None,
                header=None,
                docstring=self._get_preceding_comment(node),
            )
        ]

    def _handle_function(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        decl = next(
            (
                child
                for child in node.children
                if child.type in ("function_declarator", "operator_cast")
            ),
            None,
        )
        if decl is None:
            return [self._literal_node(node)]

        name = self._extract_declarator_name(decl)
        kind = (
            NodeKind.METHOD if self._is_method_declarator(decl, parent) else NodeKind.FUNCTION
        )
        body_node = next(
            (
                child
                for child in node.children
                if child.type in ("compound_statement", "default_method_clause", "delete_method_clause")
            ),
            None,
        )
        subtype = None if body_node is not None and body_node.type == "compound_statement" else "signature"
        return [
            self._make_node(
                node,
                kind=kind,
                name=name,
                header=self._build_function_header(node),
                subtype=subtype,
                docstring=self._get_preceding_comment(node),
            )
        ]

    def _handle_declaration(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        if self._is_declaration_aggregate(node):
            return [self._literal_node(node)]

        function_decl = next(
            (
                child
                for child in node.children
                if child.type in ("function_declarator", "operator_cast")
            ),
            None,
        )
        if function_decl is not None:
            name = self._extract_declarator_name(function_decl)
            if name is None:
                return [self._literal_node(node)]
            kind = (
                NodeKind.METHOD
                if self._is_method_declarator(function_decl, parent)
                else NodeKind.FUNCTION
            )
            return [
                self._make_node(
                    node,
                    kind=kind,
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
                "reference_declarator",
            )
        ]
        if not declarators:
            return [self._literal_node(node)]

        decl_header = self._build_declaration_header(node)
        doc = self._get_preceding_comment(node)
        kind = NodeKind.PROPERTY if parent is not None and parent.kind == NodeKind.CLASS else NodeKind.VARIABLE
        symbols: List[ParsedNode] = []
        for declarator in declarators:
            symbols.append(
                self._make_node(
                    node,
                    kind=kind,
                    name=self._extract_declarator_name(declarator),
                    header=None,
                    docstring=doc,
                    body=decl_header,
                )
            )
        return symbols

    def _handle_field_declaration(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        function_decl = self._find_first_descendant(node, ("function_declarator", "operator_cast"))
        if function_decl is not None:
            return [
                self._make_node(
                    node,
                    kind=NodeKind.METHOD,
                    name=self._extract_declarator_name(function_decl),
                    header=self._build_declaration_header(node),
                    subtype="signature",
                    docstring=self._get_preceding_comment(node),
                )
            ]

        return [
            self._make_node(
                node,
                kind=NodeKind.PROPERTY,
                name=self._extract_declarator_name(node),
                header=None,
                docstring=self._get_preceding_comment(node),
            )
        ]


class CppLanguageHelper(CLanguageHelper):
    language = ProgrammingLanguage.CPP

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

        if sym.kind in (NodeKind.CLASS, NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.NAMESPACE):
            header = (sym.header or "").strip()
            subtype = (sym.subtype or "").lower()
            if (
                sym.kind in (NodeKind.FUNCTION, NodeKind.METHOD)
                and subtype not in ("signature",)
                and header
                and not header.endswith("{")
            ):
                header = f"{header} {{"
            elif sym.kind == NodeKind.NAMESPACE and header and not header.endswith("{"):
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
        return super().get_common_syntax_words().union(
            {
                "class",
                "concept",
                "constexpr",
                "consteval",
                "constinit",
                "delete",
                "explicit",
                "export",
                "final",
                "friend",
                "mutable",
                "namespace",
                "new",
                "noexcept",
                "nullptr",
                "operator",
                "override",
                "private",
                "protected",
                "public",
                "template",
                "this",
                "typename",
                "using",
                "virtual",
            }
        )