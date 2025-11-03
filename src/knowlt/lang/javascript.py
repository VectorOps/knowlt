import os
from pathlib import Path
from typing import Optional, List, Callable

import tree_sitter as ts
import tree_sitter_javascript as tsjs

from knowlt.parsers import (
    AbstractCodeParser,
    AbstractLanguageHelper,
    ParsedNode,
    ParsedImportEdge,
    get_node_text,
)
from knowlt.models import ProgrammingLanguage, NodeKind, Node
from knowlt.project import ProjectManager, Repo
from knowlt.logger import logger

JS_LANGUAGE = ts.Language(tsjs.language())
_parser: Optional[ts.Parser] = None


def _get_parser() -> ts.Parser:
    global _parser
    if _parser is None:
        _parser = ts.Parser(JS_LANGUAGE)
    return _parser


class JavaScriptCodeParser(AbstractCodeParser):
    language = ProgrammingLanguage.JAVASCRIPT
    extensions = [".js", ".jsx", ".mjs"]
    _RESOLVE_SUFFIXES = (".js", ".jsx", ".mjs")

    def __init__(self, pm: ProjectManager, repo: Repo, rel_path: str) -> None:
        super().__init__(pm, repo, rel_path)
        self.parser = _get_parser()
        self._handlers: dict[
            str, Callable[[ts.Node, Optional[ParsedNode]], List[ParsedNode]]
        ] = {
            "import_statement": self._handle_import,
            "export_statement": self._handle_export,
            "function_declaration": self._handle_function,
            "function_expression": self._handle_function_expression,
            "arrow_function": self._handle_arrow_function_top,
            "class_declaration": self._handle_class,
            "class": self._handle_class_expr_top,
            "method_definition": self._handle_method,
            # Class fields (e.g., value = 0; foo = () => { ... })
            "variable_declaration": self._handle_lexical,
            "lexical_declaration": self._handle_lexical,
            "comment": self._handle_comment,
            "expression_statement": self._handle_expression,
            "call_expression": self._handle_call_expression,
            # Literals and commonly noisy nodes
            "empty_statement": self._literal_handler(NodeKind.LITERAL),
            "hash_bang_line": self._literal_handler(NodeKind.LITERAL),
            "number": self._literal_handler(NodeKind.LITERAL),
            "string": self._literal_handler(NodeKind.LITERAL),
            "null": self._literal_handler(NodeKind.LITERAL),
            "new_expression": self._literal_handler(NodeKind.LITERAL),
            "binary_expression": self._literal_handler(NodeKind.LITERAL),
            "ternary_expression": self._literal_handler(NodeKind.LITERAL),
            "identifier": self._literal_handler(NodeKind.LITERAL),
            "template_string": self._literal_handler(NodeKind.LITERAL),
            "statement_block": self._literal_handler(NodeKind.LITERAL),
            "array": self._literal_handler(NodeKind.LITERAL),
            "object": self._literal_handler(NodeKind.LITERAL),
            "member_expression": self._literal_handler(NodeKind.LITERAL),
            # Generic statements treated as literals
            "ambient_declaration": self._literal_handler(NodeKind.LITERAL),
            "declare_statement": self._literal_handler(NodeKind.LITERAL),
            "decorator": self._literal_handler(NodeKind.LITERAL),
            "for_statement": self._literal_handler(NodeKind.LITERAL),
            "for_in_statement": self._literal_handler(NodeKind.LITERAL),
            "for_of_statement": self._literal_handler(NodeKind.LITERAL),
            "if_statement": self._literal_handler(NodeKind.LITERAL),
            "while_statement": self._literal_handler(NodeKind.LITERAL),
            "do_statement": self._literal_handler(NodeKind.LITERAL),
            "switch_statement": self._literal_handler(NodeKind.LITERAL),
            "break_statement": self._literal_handler(NodeKind.LITERAL),
            "continue_statement": self._literal_handler(NodeKind.LITERAL),
            "return_statement": self._literal_handler(NodeKind.LITERAL),
            "throw_statement": self._literal_handler(NodeKind.LITERAL),
            "try_statement": self._literal_handler(NodeKind.LITERAL),
            "debugger_statement": self._literal_handler(NodeKind.LITERAL),
            "labeled_statement": self._literal_handler(NodeKind.LITERAL),
            "with_statement": self._literal_handler(NodeKind.LITERAL),
        }

    def _handle_file(self, root_node: ts.Node) -> None:
        return

    def _rel_to_virtual_path(self, rel_path: str) -> str:
        p = Path(rel_path)
        return ".".join(p.with_suffix("").parts)

    def _process_node(
        self, node: ts.Node, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        # Ignore punctuation and unnamed tokens to reduce noise and warnings
        if (not getattr(node, "is_named", True)) or node.type in (
            "{",
            "}",
            "(",
            ")",
            "[",
            "]",
            ";",
            ",",
        ):
            return []
        handler = self._handlers.get(node.type)
        if handler is not None:
            try:
                return handler(node, parent)
            except Exception as ex:
                logger.warning(
                    "JS handler error; falling back to literal",
                    path=self.rel_path,
                    node_type=node.type,
                    line=node.start_point[0] + 1,
                    error=str(ex),
                )
                return [self._literal_node(node)]
        self._debug_unknown_node(node)
        return [self._literal_node(node)]

    # --- helpers ----------------------------------------------------
    def _literal_node(self, node: ts.Node) -> ParsedNode:
        return self._make_node(node, kind=NodeKind.LITERAL, name=None, header=None)

    def _literal_handler(self, kind: NodeKind):
        def _h(node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
            return [self._make_node(node, kind=kind, name=None, header=None)]

        return _h

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
        logger.debug("Unknown JavaScript node type", **fields)

    def _get_preceding_comment(self, node: ts.Node) -> Optional[str]:
        comments: List[str] = []
        sib = node.prev_sibling
        while sib is not None:
            if node.start_point[0] - sib.end_point[0] > 2:
                break
            if sib.type == "comment":
                raw = get_node_text(sib) or ""
                comments.append(raw.strip())
                sib = sib.prev_sibling
                continue
            if sib.type in (";",):
                sib = sib.prev_sibling
                continue
            break
        if comments:
            comments.reverse()
            merged = "\n".join(comments).strip()
            return merged or None
        return None

    def _build_fn_like_header(
        self, node: ts.Node, *, prefix: str, name: Optional[str]
    ) -> str:
        params = get_node_text(node.child_by_field_name("parameters")) or "()"
        async_prefix = ""
        raw = (get_node_text(node) or "").lstrip()
        if raw.startswith("async"):
            async_prefix = "async "
        nm = name or ""
        # JavaScript: no type parameters/return types in grammar
        return f"{async_prefix}{prefix}{' ' if prefix else ''}{nm}{params}"

    def _build_class_like_header(self, node: ts.Node, *, keyword: str) -> str:
        code = get_node_text(node) or ""
        head = code.split("{", 1)[0].strip().rstrip(";")
        return head or keyword

    def _build_arrow_header_only(self, arrow_node: ts.Node) -> str:
        body_node = arrow_node.child_by_field_name("body")
        if body_node is not None:
            return (
                self.source_bytes[arrow_node.start_byte : body_node.start_byte]
                .decode("utf8")
                .strip()
            )
        return (get_node_text(arrow_node) or "").split("{", 1)[0].strip()

    def _build_function_expression_header_only(self, fn_node: ts.Node) -> str:
        body_node = fn_node.child_by_field_name("body") or fn_node.child_by_field_name(
            "statement"
        )
        if body_node is not None:
            return (
                self.source_bytes[fn_node.start_byte : body_node.start_byte]
                .decode("utf8")
                .strip()
            )
        return (get_node_text(fn_node) or "").split("{", 1)[0].strip()

    def _resolve_module(self, module: str) -> tuple[Optional[str], str, bool]:
        if module.startswith("."):
            base_dir = os.path.dirname(self.rel_path)
            rel_candidate = os.path.normpath(os.path.join(base_dir, module))
            if not rel_candidate.endswith(self._RESOLVE_SUFFIXES):
                for suf in self._RESOLVE_SUFFIXES:
                    cand = f"{rel_candidate}{suf}"
                    if self.repo.root_path and os.path.exists(
                        os.path.join(self.repo.root_path, cand)
                    ):
                        rel_candidate = cand
                        break
            return rel_candidate, module, False
        return None, module, True

    def _resolve_arrow_function_name(self, holder_node: ts.Node) -> Optional[str]:
        name_node = holder_node.child_by_field_name("name")
        if name_node:
            name = get_node_text(name_node) or ""
            if name:
                return name.split(".")[-1]
        lhs_node = holder_node.child_by_field_name(
            "left"
        ) or holder_node.child_by_field_name("name")
        if lhs_node:
            lhs = get_node_text(lhs_node) or ""
            if lhs:
                return lhs.split(".")[-1]
        stack = [holder_node]
        while stack:
            cur = stack.pop()
            if cur.type in ("identifier", "property_identifier"):
                val = get_node_text(cur) or ""
                if val:
                    return val.split(".")[-1]
            stack.extend(list(cur.children))
        return None

    def _resolve_class_expression_name(self, holder_node: ts.Node) -> Optional[str]:
        name_node = holder_node.child_by_field_name("name")
        if name_node:
            name = get_node_text(name_node) or ""
            if name:
                return name.split(".")[-1]
        lhs_node = holder_node.child_by_field_name(
            "left"
        ) or holder_node.child_by_field_name("name")
        if lhs_node:
            lhs = get_node_text(lhs_node) or ""
            if lhs:
                return lhs.split(".")[-1]
        stack = [holder_node]
        while stack:
            cur = stack.pop()
            if cur.type in ("identifier", "property_identifier"):
                val = get_node_text(cur) or ""
                if val:
                    return val.split(".")[-1]
            stack.extend(list(cur.children))
        return None

    # --- handlers ---------------------------------------------------
    def _handle_import(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        raw_stmt = get_node_text(node) or ""
        src = node.child_by_field_name("source") or next(
            (c for c in node.children if c.type == "string"), None
        )
        module = (get_node_text(src) or "").strip("\"'")
        if not module:
            self._debug_unknown_node(node, context="import.missing_source")
            return [self._literal_node(node)]
        physical, virtual, external = self._resolve_module(module)
        alias = None
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            alias = get_node_text(name_node) or None
        else:
            ns = next((c for c in node.children if c.type == "namespace_import"), None)
            if ns is not None:
                alias_node = ns.child_by_field_name("name")
                alias = get_node_text(alias_node) or None
        assert self.parsed_file is not None
        self.parsed_file.imports.append(
            ParsedImportEdge(
                physical_path=physical,
                virtual_path=virtual,
                alias=alias,
                dot=False,
                external=external,
                raw=raw_stmt,
            )
        )
        imp_node = self._make_node(
            node, kind=NodeKind.LITERAL, name=None, header=None, subtype="import"
        )
        imp_node.comment = self._get_preceding_comment(node)
        return [imp_node]

    def _handle_export(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        raw_stmt = get_node_text(node) or ""
        # Re-export: export ... from "module"
        source_node = node.child_by_field_name("source") or next(
            (c for c in node.children if c.type in ("from_clause", "string")), None
        )
        if source_node and source_node.type == "from_clause":
            source_node = source_node.child_by_field_name("source")
        # Case 1: re-export (export ... from "module")
        if source_node and source_node.type == "string":
            module = (get_node_text(source_node) or "").strip("\"'")
            physical, virtual, external = self._resolve_module(module)
            assert self.parsed_file is not None
            self.parsed_file.imports.append(
                ParsedImportEdge(
                    physical_path=physical,
                    virtual_path=virtual,
                    alias=None,
                    dot=False,
                    external=external,
                    raw=raw_stmt,
                )
            )
            exp = self._make_node(
                node,
                kind=NodeKind.CUSTOM,
                name=None,
                header=raw_stmt.strip(),
                subtype="export",
            )
            return [exp]
        # Case 2: named export without source: `export { a, b as c }`
        if any(c.type == "export_clause" for c in node.named_children):
            exp = self._make_node(
                node,
                kind=NodeKind.CUSTOM,
                name=None,
                header=raw_stmt.strip(),
                subtype="export",
            )
            return [exp]
        # Case 3: local export — delegate inner nodes
        is_default = "export default" in raw_stmt
        exp = self._make_node(
            node,
            kind=NodeKind.CUSTOM,
            name=None,
            header="export default" if is_default else "export",
            subtype="export",
        )
        for ch in node.named_children:
            if ch.type in ("export_clause", "from_clause", "string"):
                continue
            if ch.type in ("identifier", "property_identifier"):
                txt = (get_node_text(ch) or "").strip() or None
                exp.children.append(
                    self._make_node(ch, kind=NodeKind.LITERAL, name=txt, header=txt)
                )
                continue
            exp.children.extend(self._process_node(ch, parent=parent))
        return [exp]

    def _handle_function(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []
        fn_name = get_node_text(name_node) or None
        header = self._build_fn_like_header(node, prefix="function", name=fn_name)
        kind = (
            NodeKind.METHOD
            if parent and parent.kind == NodeKind.CLASS
            else NodeKind.FUNCTION
        )
        sym = self._make_node(node, kind=kind, name=fn_name, header=header)
        return [sym]

    def _handle_function_expression(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        name = get_node_text(name_node) or None
        header = self._build_fn_like_header(node, prefix="function", name=name)
        kind = (
            NodeKind.METHOD
            if parent and parent.kind == NodeKind.CLASS
            else NodeKind.FUNCTION
        )
        return [self._make_node(node, kind=kind, name=name, header=header)]

    def _handle_arrow_function_top(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        header = self._build_arrow_header_only(node)
        kind = (
            NodeKind.METHOD
            if parent and parent.kind == NodeKind.CLASS
            else NodeKind.FUNCTION
        )
        return [self._make_node(node, kind=kind, name=None, header=header)]

    def _handle_class_expr_top(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        header = self._build_class_like_header(node, keyword="class")
        cls = self._make_node(node, kind=NodeKind.CLASS, name=None, header=header)
        body = next((c for c in node.children if c.type == "class_body"), None)
        for ch in (body.named_children if body is not None else node.named_children):
            cls.children.extend(self._process_node(ch, parent=cls))
        return [cls]

    def _handle_function_expression_in_holder(
        self, holder_node: ts.Node, fn_node: ts.Node, parent: Optional[ParsedNode]
    ) -> ParsedNode:
        name = self._resolve_arrow_function_name(holder_node)
        header = self._build_fn_like_header(fn_node, prefix="function", name=name)
        kind = (
            NodeKind.METHOD
            if parent and parent.kind == NodeKind.CLASS
            else NodeKind.FUNCTION
        )
        return self._make_node(fn_node, kind=kind, name=name, header=header)

    def _handle_arrow_function_in_holder(
        self, holder_node: ts.Node, arrow_node: ts.Node, parent: Optional[ParsedNode]
    ) -> ParsedNode:
        """
        Handle arrow functions assigned within a holder (e.g., variable assignment or export).
        Resolves the name from the holder and uses an arrow-only header for summaries.
        """
        name = self._resolve_arrow_function_name(holder_node)
        header = self._build_arrow_header_only(arrow_node)
        kind = (
            NodeKind.METHOD
            if parent and parent.kind == NodeKind.CLASS
            else NodeKind.FUNCTION
        )
        return self._make_node(arrow_node, kind=kind, name=name, header=header)

    def _handle_class_expression(
        self, holder_node: ts.Node, class_node: ts.Node, parent: Optional[ParsedNode]
    ) -> ParsedNode:
        name = self._resolve_class_expression_name(holder_node)
        header = self._build_class_like_header(class_node, keyword="class")
        cls = self._make_node(class_node, kind=NodeKind.CLASS, name=name, header=header)
        body = next((c for c in class_node.children if c.type == "class_body"), None)
        body_children = (
            body.named_children if body is not None else class_node.named_children
        )
        for ch in body_children:
            cls.children.extend(self._process_node(ch, parent=cls))
        return cls

    def _handle_class(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []
        cls_name = get_node_text(name_node) or None
        header = self._build_class_like_header(node, keyword="class")
        cls = self._make_node(node, kind=NodeKind.CLASS, name=cls_name, header=header)
        body = next((c for c in node.children if c.type == "class_body"), None)
        body_children = body.children if body is not None else node.children
        for ch in body_children:
            cls.children.extend(self._process_node(ch, parent=cls))
        return [cls]

    def _handle_method(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        name = get_node_text(name_node) or None
        body_node = node.child_by_field_name("body") or node.child_by_field_name(
            "statement"
        )
        if body_node is not None:
            header = (
                self.source_bytes[node.start_byte : body_node.start_byte]
                .decode("utf8")
                .strip()
            )
        else:
            header = self._build_fn_like_header(node, prefix="", name=name)
        return [
            self._make_node(
                node, kind=NodeKind.METHOD, name=name, header=header, subtype=None
            )
        ]

    def _handle_comment(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        return [self._make_node(node, kind=NodeKind.COMMENT, name=None, header=None)]

    def _handle_call_expression(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        # Collect require() usage and otherwise treat as literal expression
        self._collect_require_calls(node, alias=None)
        expr = self._make_node(node, kind=NodeKind.LITERAL, name=None, header=None)
        expr.comment = self._get_preceding_comment(node)
        return [expr]

    def _handle_expression(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        for ch in node.named_children:
            if ch.type == "assignment_expression":
                lhs = ch.child_by_field_name("left")
                rhs = ch.child_by_field_name("right")
                # CommonJS exports
                if lhs and self._is_commonjs_export(lhs):
                    exp = self._make_node(
                        ch,
                        kind=NodeKind.CUSTOM,
                        name=None,
                        header="export",
                        subtype="export",
                    )
                    if rhs and rhs.type == "arrow_function":
                        fn = self._handle_arrow_function_in_holder(ch, rhs, parent)
                        exp.children.append(fn)
                        return [exp]
                    if rhs and rhs.type in (
                        "function_declaration",
                        "function_expression",
                    ):
                        fn = self._handle_function_expression_in_holder(ch, rhs, parent)
                        exp.children.append(fn)
                        return [exp]
                    if rhs and rhs.type in ("class", "class_declaration"):
                        cls = self._handle_class_expression(ch, rhs, parent)
                        exp.children.append(cls)
                        return [exp]
                    return [exp]
                # require() aliasing
                if rhs and rhs.type == "call_expression":
                    self._collect_require_calls(
                        rhs, alias=(get_node_text(lhs) or "").split(".")[-1] or None
                    )
                # Holder-aware assignment handling outside export
                if rhs and rhs.type == "arrow_function":
                    return [self._handle_arrow_function_in_holder(ch, rhs, parent)]
                if rhs and rhs.type in ("function_declaration", "function_expression"):
                    return [self._handle_function_expression_in_holder(ch, rhs, parent)]
                if rhs and rhs.type in ("class", "class_declaration"):
                    return [self._handle_class_expression(ch, rhs, parent)]
                # Plain assignment: set variable name from LHS
                vname = self._resolve_arrow_function_name(ch)
                var = self._make_node(
                    ch, kind=NodeKind.VARIABLE, name=vname, header=None
                )
                var.comment = self._get_preceding_comment(ch)
                return [var]
            elif ch.type == "call_expression":
                return self._handle_call_expression(ch, parent)
        expr = self._make_node(node, kind=NodeKind.LITERAL, name=None, header=None)
        expr.comment = self._get_preceding_comment(node)
        return [expr]

    def _handle_lexical(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        # Group multi-declarator statements under a single lexical node with the keyword header.
        raw = (get_node_text(node) or "").lstrip()
        if raw.startswith("const"):
            keyword = "const"
        elif raw.startswith("let"):
            keyword = "let"
        elif raw.startswith("var"):
            keyword = "var"
        else:
            keyword = (raw.split(None, 1)[0] or "").strip() or "const"
        group = self._make_node(
            node, kind=NodeKind.CUSTOM, name=None, header=keyword, subtype="lexical"
        )
        for ch in node.named_children:
            if ch.type != "variable_declarator":
                continue
            # LHS
            name_node = ch.child_by_field_name("name") or next(
                (
                    c
                    for c in ch.named_children
                    if c.type in ("identifier", "property_identifier")
                ),
                None,
            )
            vname = get_node_text(name_node) or None
            value_node = ch.child_by_field_name("value")
            lhs_header = (get_node_text(name_node) or "").strip()
            # JS: no type annotation nodes; still check for presence
            type_node = ch.child_by_field_name("type") or next(
                (c for c in ch.named_children if c.type == "type_annotation"),
                None,
            )
            if type_node is not None:
                type_txt = (get_node_text(type_node) or "").strip()
                if type_txt:
                    lhs_header = (
                        f"{lhs_header}{type_txt}"
                        if type_txt.startswith(":")
                        else f"{lhs_header}: {type_txt}"
                    )
            if value_node is not None:
                lhs_header = f"{lhs_header} ="
            # Declarator kind
            decl_kind = NodeKind.CONST if raw.startswith("const") else NodeKind.VARIABLE
            if value_node is not None and value_node.type in (
                "arrow_function",
                "function_expression",
                "function_declaration",
            ):
                decl_kind = NodeKind.FUNCTION
            decl = self._make_node(ch, kind=decl_kind, name=vname, header=lhs_header)
            # RHS
            if value_node is not None:
                rhs_nodes = self._process_node(value_node, parent=decl)
                for rn in rhs_nodes:
                    decl.children.append(rn)
            group.children.append(decl)
        return [group]

    def _is_commonjs_export(self, lhs: ts.Node) -> bool:
        node = lhs
        while node and node.type == "member_expression":
            prop = node.child_by_field_name("property")
            obj = node.child_by_field_name("object")
            if obj and obj.type == "identifier":
                oname = get_node_text(obj)
                if oname == "exports":
                    return True
                if oname == "module" and prop and (get_node_text(prop) == "exports"):
                    return True
            node = obj
        return False

    def _collect_require_calls(self, node: ts.Node, alias: Optional[str]) -> None:
        if node.type != "call_expression":
            return
        fn = node.child_by_field_name("function")
        if fn and fn.type == "identifier" and (get_node_text(fn) or "") == "require":
            args = node.child_by_field_name("arguments")
            if not args:
                return
            str_node = next((c for c in args.children if c.type == "string"), None)
            if not str_node:
                return
            module = (get_node_text(str_node) or "").strip("\"'")
            phys, virt, ext = self._resolve_module(module)
            assert self.parsed_file is not None
            self.parsed_file.imports.append(
                ParsedImportEdge(
                    physical_path=phys,
                    virtual_path=virt,
                    alias=alias,
                    dot=False,
                    external=ext,
                    raw=get_node_text(node) or "",
                )
            )


class JavaScriptLanguageHelper(AbstractLanguageHelper):
    language = ProgrammingLanguage.JAVASCRIPT

    def get_node_summary(
        self,
        sym: Node,
        indent: int = 0,
        include_comments: bool = False,
        include_docs: bool = False,
        include_parents: bool = False,
        child_stack: Optional[List[List[Node]]] = None,
    ) -> str:
        # Follow TypeScript summary logic/formatting
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
            else:
                include_parents = False

        only_children = child_stack.pop() if child_stack else None

        IND = " " * indent
        lines: List[str] = []

        def emit_children(children: List[Node], child_indent: int) -> None:
            CH_IND = " " * child_indent
            if only_children:
                lines.append(f"{CH_IND}...")
            included = [
                c for c in children if (not only_children or c in only_children)
            ]
            if included:
                for ch in included:
                    if (not include_comments) and ch.kind == NodeKind.COMMENT:
                        continue
                    ch_sum = self.get_node_summary(
                        ch,
                        indent=child_indent,
                        include_comments=include_comments,
                        include_docs=include_docs,
                    )
                    if ch_sum:
                        lines.append(ch_sum)
            else:
                lines.append(f"{CH_IND}...")

        if include_comments and sym.comment:
            for ln in (sym.comment or "").splitlines():
                lines.append(f"{IND}{ln.strip()}")

        header = (sym.header or sym.name or "").strip()

        if sym.kind in (NodeKind.CLASS,):
            if header and not header.endswith("{"):
                header = f"{header} {{"
            lines.append(f"{IND}{header}" if header else f"{IND}class {{")
            if sym.children:
                emit_children(sym.children, indent + 2)
            else:
                lines.append(f"{IND}  ...")
            lines.append(f"{IND}}}")
            return "\n".join(lines)

        if sym.kind in (NodeKind.INTERFACE, NodeKind.NAMESPACE):
            if header and not header.endswith("{"):
                header = f"{header} {{"
            lines.append(f"{IND}{header}" if header else f"{IND}<decl> {{")
            if sym.children:
                emit_children(sym.children, indent + 2)
            else:
                lines.append(f"{IND}  ...")
            lines.append(f"{IND}}}")
            return "\n".join(lines)

        if sym.kind == NodeKind.ENUM:
            if header and not header.endswith("{"):
                header = f"{header} {{"
            lines.append(f"{IND}{header}" if header else f"{IND}enum {{")
            if sym.children:
                included_children = [
                    ch
                    for ch in sym.children
                    if (not only_children or ch in only_children)
                ]
                for idx, ch in enumerate(included_children):
                    ch_sum = self.get_node_summary(
                        ch,
                        indent=indent + 2,
                        include_comments=include_comments,
                        include_docs=include_docs,
                    )
                    if not ch_sum:
                        continue
                    first, *rest = ch_sum.splitlines()
                    first = first.rstrip()
                    if idx < len(included_children) - 1 and not first.endswith(","):
                        first = f"{first},"
                    lines.append(first)
                    for ln in rest:
                        lines.append(ln)
            else:
                lines.append(f"{IND}  ...")
            lines.append(f"{IND}}}")
            return "\n".join(lines)

        if sym.kind == NodeKind.METHOD:
            head = header or ""
            st = (sym.subtype or "").lower()
            if st in ("abstract", "signature"):
                lines.append(f"{IND}{head}" if head else f"{IND}<method>")
            else:
                if head and not head.endswith("{"):
                    head = f"{head} {{ ... }}"
                lines.append(f"{IND}{head}" if head else f"{IND}<method>")
            return "\n".join(lines)

        if sym.kind == NodeKind.FUNCTION:
            head = header or ""
            if head and not head.endswith("{"):
                head = f"{head} {{ ... }}"
            lines.append(f"{IND}{head}" if head else f"{IND}<function>")
            return "\n".join(lines)

        if sym.kind == NodeKind.CUSTOM and (sym.subtype or "").lower() == "lexical":
            keyword = (sym.header or "").strip()
            if not sym.children:
                return f"{IND}{keyword}"
            out_lines: List[str] = []
            decls = [
                d for d in sym.children if (not only_children or d in only_children)
            ]
            if only_children and not decls:
                return f"{IND}{keyword} ..."
            for idx, decl in enumerate(decls):
                lhs = (decl.header or decl.name or "").strip()
                if not decl.children:
                    lhs = lhs.rstrip("= ").strip()
                    if idx == 0:
                        if lhs:
                            out_lines.append(f"{IND}{keyword} {lhs}".rstrip())
                    else:
                        if out_lines:
                            out_lines[-1] = f"{out_lines[-1]}, {lhs}".rstrip()
                    continue
                rhs = decl.children[0]
                rhs_sum = self.get_node_summary(
                    rhs,
                    indent=0,
                    include_comments=include_comments,
                    include_docs=include_docs,
                )
                rhs_lines = rhs_sum.splitlines() if rhs_sum else [""]
                first_rhs = (rhs_lines[0] if rhs_lines else "").lstrip()
                if idx == 0:
                    out_lines.append(
                        f"{IND}{keyword} {lhs.rstrip()} {first_rhs}".rstrip()
                    )
                else:
                    if out_lines:
                        out_lines[-1] = (
                            f"{out_lines[-1]}, {lhs.rstrip()} {first_rhs}".rstrip()
                        )
                for ln in rhs_lines[1:]:
                    out_lines.append(f"{IND}{ln}")
            return "\n".join(out_lines)

        if sym.kind == NodeKind.CUSTOM and (sym.subtype or "").lower() == "export":
            if sym.children:
                out_lines: List[str] = []
                prefix = (sym.header or "export").strip()
                included_children = [
                    ch
                    for ch in sym.children
                    if (not only_children or ch in only_children)
                ]
                for ch in included_children:
                    ch_sum = self.get_node_summary(
                        ch,
                        indent=indent,
                        include_comments=include_comments,
                        include_docs=include_docs,
                    )
                    if not ch_sum:
                        continue
                    first, *rest = ch_sum.splitlines()
                    first_trim = first.lstrip()
                    if first_trim.startswith("exports.") or first_trim.startswith(
                        "module.exports"
                    ):
                        out_lines.append(f"{IND}{first_trim}")
                    else:
                        out_lines.append(f"{IND}{prefix} {first_trim}")
                    for ln in rest:
                        out_lines.append(f"{IND}{ln}")
                return "\n".join(out_lines)
            bare = (sym.header or sym.body or "export").strip()
            return f"{IND}{bare}"

        if sym.children:
            lines.append(
                f"{IND}{header}" if header else f"{IND}{(sym.body or '').strip()}"
            )

            def emit_children_only(children):
                CH_IND = " " * (indent + 2)
                if only_children:
                    lines.append(f"{CH_IND}...")
                included = [
                    c for c in children if (not only_children or c in only_children)
                ]
                if included:
                    for ch in included:
                        if (not include_comments) and ch.kind == NodeKind.COMMENT:
                            continue
                        ch_sum = self.get_node_summary(
                            ch,
                            indent=indent + 2,
                            include_comments=include_comments,
                            include_docs=include_docs,
                        )
                        if ch_sum:
                            lines.append(ch_sum)
                else:
                    lines.append(f"{CH_IND}...")

            emit_children_only(sym.children)
            return "\n".join(lines)

        body = (sym.body or "").strip()
        if body:
            lines.append(f"{IND}{body}")
        return "\n".join(lines)

    def get_common_syntax_words(self) -> set[str]:
        return {
            "break",
            "case",
            "catch",
            "class",
            "const",
            "continue",
            "do",
            "else",
            "export",
            "extends",
            "false",
            "finally",
            "for",
            "function",
            "if",
            "import",
            "in",
            "instanceof",
            "let",
            "new",
            "null",
            "return",
            "super",
            "switch",
            "this",
            "throw",
            "true",
            "try",
            "typeof",
            "undefined",
            "var",
            "void",
            "while",
            "with",
            "async",
            "await",
        }
