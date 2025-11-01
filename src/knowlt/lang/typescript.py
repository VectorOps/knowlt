import os
from pathlib import Path
from typing import Optional, List, Callable

import tree_sitter as ts
import tree_sitter_typescript as tsts

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

TS_LANGUAGE = ts.Language(tsts.language_tsx())
_parser: Optional[ts.Parser] = None


def _get_parser() -> ts.Parser:
    global _parser
    if _parser is None:
        _parser = ts.Parser(TS_LANGUAGE)
    return _parser


class TypeScriptCodeParser(AbstractCodeParser):
    language = ProgrammingLanguage.TYPESCRIPT
    extensions = [".ts", ".tsx"]

    _RESOLVE_SUFFIXES = (".ts", ".tsx", ".js", ".jsx")

    def __init__(self, pm: ProjectManager, repo: Repo, rel_path: str) -> None:
        super().__init__(pm, repo, rel_path)
        self.parser = _get_parser()
        self._handlers: dict[str, Callable[[ts.Node, Optional[ParsedNode]], List[ParsedNode]]] = {
            "import_statement": self._handle_import,
            "import_equals_declaration": self._handle_import_equals,
            "export_statement": self._handle_export,
            "function_declaration": self._handle_function,
            "function_expression": self._handle_function_expression,
            "arrow_function": self._handle_arrow_function_top,
            "call_expression": self._handle_call_expression,
            "for_in_statement": self._handle_for_in,
            "if_statement": self._handle_if,
            "satisfies_expression": self._handle_satisfies_expression,
            "ambient_declaration": self._handle_ambient_declaration,
            "class_declaration": self._handle_class,
            "abstract_class_declaration": self._handle_class,
            "interface_declaration": self._handle_interface,
            "enum_declaration": self._handle_enum,
            "namespace_declaration": self._handle_namespace,
            "module_declaration": self._handle_namespace,
            "internal_module": self._handle_namespace,
            "method_definition": self._handle_method,
            "abstract_method_signature": self._handle_method,
            "variable_declaration": self._handle_lexical,
            "lexical_declaration": self._handle_lexical,
            "public_field_declaration": self._handle_class_field,
            "public_field_definition": self._handle_class_field,
            "comment": self._handle_comment,
            "expression_statement": self._handle_expression,
            "empty_statement": self._literal_handler(NodeKind.LITERAL),
            "hash_bang_line": self._literal_handler(NodeKind.LITERAL),
            "type_alias_declaration": self._handle_type_alias,
        }

    def _handle_file(self, root_node: ts.Node) -> None:
        return

    def _rel_to_virtual_path(self, rel_path: str) -> str:
        p = Path(rel_path)
        parts = p.with_suffix("").parts
        return ".".join(parts)

    def _process_node(
        self, node: ts.Node, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        # Ignore punctuation and unnamed tokens to reduce noise and warnings
        if (not getattr(node, "is_named", True)) or node.type in ("{", "}", "(", ")", "[", "]", ";", ","):
            return []
        handler = self._handlers.get(node.type)
        if handler is not None:
            try:
                return handler(node, parent)
            except Exception as ex:
                logger.warning(
                    "TS handler error; falling back to literal",
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
        self, node: ts.Node, *, context: Optional[str] = None, parent_type: Optional[str] = None
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
        logger.debug("Unknown TypeScript node type", **fields)

    def _extract_type_parameters(self, node: ts.Node) -> Optional[str]:
        tp_node = node.child_by_field_name("type_parameters") or next(
            (c for c in node.children if c.type in ("type_parameters", "type_parameter_list")), None
        )
        if tp_node:
            return (get_node_text(tp_node) or "").strip() or None
        hdr = (get_node_text(node) or "").split("{", 1)[0]
        lt = hdr.find("<")
        gt = hdr.find(">", lt + 1)
        if 0 <= lt < gt:
            return hdr[lt:gt + 1].strip()
        return None

    def _build_fn_like_header(self, node: ts.Node, *, prefix: str, name: Optional[str]) -> str:
        params = get_node_text(node.child_by_field_name("parameters")) or "()"
        ret = ""
        rt = node.child_by_field_name("return_type") or node.child_by_field_name("type")
        if rt is not None:
            rtxt = (get_node_text(rt) or "").lstrip(":").strip()
            if rtxt:
                ret = f": {rtxt}"
        tparams = self._extract_type_parameters(node) or ""
        async_prefix = ""
        raw = (get_node_text(node) or "").lstrip()
        if raw.startswith("async"):
            async_prefix = "async "
        nm = name or "<anonymous>"
        return f"{async_prefix}{prefix}{' ' if prefix else ''}{nm}{tparams}{params}{ret}"

    def _resolve_module(self, module: str) -> tuple[Optional[str], str, bool]:
        if module.startswith("."):
            base_dir = os.path.dirname(self.rel_path)
            rel_candidate = os.path.normpath(os.path.join(base_dir, module))
            if not rel_candidate.endswith(self._RESOLVE_SUFFIXES):
                for suf in self._RESOLVE_SUFFIXES:
                    cand = f"{rel_candidate}{suf}"
                    if self.repo.root_path and os.path.exists(os.path.join(self.repo.root_path, cand)):
                        rel_candidate = cand
                        break
            return rel_candidate, module, False
        return None, module, True

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

    # --- handlers ---------------------------------------------------
    def _handle_import(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        raw_stmt = get_node_text(node) or ""
        src = node.child_by_field_name("source") or next((c for c in node.children if c.type == "string"), None)
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
        imp_node = self._make_node(node, kind=NodeKind.LITERAL, name=None, header=None, subtype="import")
        imp_node.comment = self._get_preceding_comment(node)
        return [imp_node]

    def _handle_import_equals(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        alias_node = node.child_by_field_name("name")
        req_node = node.child_by_field_name("module")
        str_node = next((c for c in req_node.children if c.type == "string"), None) if req_node else None
        alias = get_node_text(alias_node) or None
        module = (get_node_text(str_node) or "").strip("\"'")
        if not (alias and module):
            return [self._literal_node(node)]
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
        return [self._make_node(node, kind=NodeKind.LITERAL, name=None, header=None, subtype="import")]

    def _handle_export(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        # Re-export: export ... from "module"
        source_node = node.child_by_field_name("source") or next((c for c in node.children if c.type in ("from_clause", "string")), None)
        if source_node and source_node.type == "from_clause":
            source_node = source_node.child_by_field_name("source")
        results: List[ParsedNode] = []
        if source_node and source_node.type == "string":
            raw_stmt = get_node_text(node) or ""
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
            # Emit a CUSTOM export node to mirror old EXPORT behavior
            exp = self._make_node(node, kind=NodeKind.CUSTOM, name=None, header=raw_stmt.strip(), subtype="export")
            results.append(exp)
            return results
        # Local export: wrap declarations in a CUSTOM export node and mark inner nodes exported.
        exp = self._make_node(node, kind=NodeKind.CUSTOM, name=None, header="export", subtype="export")
        results.append(exp)
        for ch in node.named_children:
            if ch.type in ("function_declaration", "function_expression"):
                decls = self._handle_function(ch, parent=parent)
                for d in decls:
                    exp.children.append(d)
            elif ch.type in ("arrow_function",):
                decls = self._handle_arrow_function_top(ch, parent)
                for d in decls:
                    exp.children.append(d)
            elif ch.type == "satisfies_expression":
                # e.g., export default { ... } satisfies Config
                for d in self._handle_satisfies_expression(ch, parent):
                    exp.children.append(d)
            elif ch.type == "type_alias_declaration":
                decls = self._handle_type_alias(ch, parent=parent)
                for d in decls:
                    exp.children.append(d)
            elif ch.type == "call_expression":
                # Support patterns like: export default defineConfig(...)
                for d in self._handle_call_expression(ch, parent):
                    exp.children.append(d)
            elif ch.type in ("class_declaration", "abstract_class_declaration"):
                decls = self._handle_class(ch, parent=parent)
                for d in decls:
                    exp.children.append(d)
            elif ch.type == "interface_declaration":
                decls = self._handle_interface(ch, parent=parent)
                for d in decls:
                    exp.children.append(d)
            elif ch.type == "enum_declaration":
                decls = self._handle_enum(ch, parent=parent)
                for d in decls:
                    exp.children.append(d)
            elif ch.type in ("variable_declaration", "lexical_declaration"):
                decls = self._handle_lexical(ch, parent=parent)
                for d in decls:
                    exp.children.append(d)
            elif ch.type in ("identifier", "property_identifier"):
                # e.g., export default ChatMessage
                txt = (get_node_text(ch) or "").strip() or None
                exp.children.append(self._make_node(ch, kind=NodeKind.LITERAL, name=txt, header=txt))
            elif ch.type in ("export_clause",):
                # e.g., export { a, b }
                pass
            else:
                self._debug_unknown_node(ch, context="export.inner")
        return results

    def _handle_function(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []
        fn_name = get_node_text(name_node) or None
        header = self._build_fn_like_header(node, prefix="function", name=fn_name)
        kind = NodeKind.METHOD if parent and parent.kind == NodeKind.CLASS else NodeKind.FUNCTION
        sym = self._make_node(node, kind=kind, name=fn_name, header=header)
        return [sym]

    def _handle_function_expression(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        name = get_node_text(name_node) or None
        header = self._build_fn_like_header(node, prefix="function", name=name)
        kind = NodeKind.METHOD if parent and parent.kind == NodeKind.CLASS else NodeKind.FUNCTION
        return [self._make_node(node, kind=kind, name=name, header=header)]

    def _resolve_arrow_function_name(self, holder_node: ts.Node) -> Optional[str]:
        name_node = holder_node.child_by_field_name("name")
        if name_node:
            name = get_node_text(name_node) or ""
            if name:
                return name.split(".")[-1]
        lhs_node = holder_node.child_by_field_name("left") or holder_node.child_by_field_name("name")
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

    def _handle_arrow_function_top(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        header = self._build_fn_like_header(node, prefix="", name=None)
        kind = NodeKind.METHOD if parent and parent.kind == NodeKind.CLASS else NodeKind.FUNCTION
        return [self._make_node(node, kind=kind, name=None, header=header)]

    def _handle_arrow_function_in_holder(self, holder_node: ts.Node, arrow_node: ts.Node, parent: Optional[ParsedNode]) -> ParsedNode:
        name = self._resolve_arrow_function_name(holder_node)
        body_node = arrow_node.child_by_field_name("body")
        if body_node:
            raw_header = self.source_bytes[holder_node.start_byte:body_node.start_byte].decode("utf8").rstrip()
        else:
            raw_header = (get_node_text(holder_node) or "").split("{", 1)[0].strip().rstrip(";")
        kind = NodeKind.METHOD if parent and parent.kind == NodeKind.CLASS else NodeKind.FUNCTION
        return self._make_node(arrow_node, kind=kind, name=name, header=raw_header)

    def _resolve_class_expression_name(self, holder_node: ts.Node) -> Optional[str]:
        # Mirror arrow name resolution: prefer declarator/assignment LHS identifiers/properties
        name_node = holder_node.child_by_field_name("name")
        if name_node:
            name = get_node_text(name_node) or ""
            if name:
                return name.split(".")[-1]
        lhs_node = holder_node.child_by_field_name("left") or holder_node.child_by_field_name("name")
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

    def _handle_class_expression(self, holder_node: ts.Node, class_node: ts.Node, parent: Optional[ParsedNode]) -> ParsedNode:
        # Build a class symbol and name it from the holder (e.g., const Foo = class { ... })
        name = self._resolve_class_expression_name(holder_node)
        # Prefer explicit header "class <Name><TParams>" for anonymous class expressions
        tparams = self._extract_type_parameters(class_node) or ""
        header = f"class {name}{tparams}" if name else "class"
        cls = self._make_node(class_node, kind=NodeKind.CLASS, name=name, header=header)
        body = next((c for c in class_node.children if c.type == "class_body"), None)
        body_children = body.named_children if body is not None else class_node.named_children
        for ch in body_children:
            cls.children.extend(self._process_node(ch, parent=cls))
        return cls

    def _handle_function_expression_in_holder(self, holder_node: ts.Node, fn_node: ts.Node, parent: Optional[ParsedNode]) -> ParsedNode:
        # Name function expressions by the LHS target (e.g., x = function(...) { ... })
        # Reuse arrow name resolver to extract identifier/property from holder.
        name = self._resolve_arrow_function_name(holder_node)
        header = self._build_fn_like_header(fn_node, prefix="function", name=name)
        kind = NodeKind.METHOD if parent and parent.kind == NodeKind.CLASS else NodeKind.FUNCTION
        return self._make_node(fn_node, kind=kind, name=name, header=header)

    def _build_class_like_header(self, node: ts.Node, *, keyword: str, name: str) -> str:
        code = (get_node_text(node) or "")
        head = code.split("{", 1)[0].strip()
        if head:
            return head
        tparams = self._extract_type_parameters(node) or ""
        return f"{keyword} {name}{tparams}"

    def _handle_class(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []
        cls_name = get_node_text(name_node) or None
        header = self._build_class_like_header(node, keyword="class", name=cls_name or "<anonymous>")
        cls = self._make_node(node, kind=NodeKind.CLASS, name=cls_name, header=header)
        body = next((c for c in node.children if c.type == "class_body"), None)
        body_children = body.children if body is not None else node.children
        for ch in body_children:
            cls.children.extend(self._process_node(ch, parent=cls))
        return [cls]

    def _handle_interface(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []
        itf_name = get_node_text(name_node) or None
        header = self._build_class_like_header(node, keyword="interface", name=itf_name or "<anonymous>")
        itf = self._make_node(node, kind=NodeKind.INTERFACE, name=itf_name, header=header)
        body = next((c for c in node.children if c.type == "interface_body"), None)
        if body:
            for ch in body.named_children:
                if ch.type in ("method_signature",):
                    mname_node = ch.child_by_field_name("name") or ch.child_by_field_name("property")
                    mname = get_node_text(mname_node) or None
                    header = self._build_fn_like_header(ch, prefix="", name=mname)
                    itf.children.append(self._make_node(ch, kind=NodeKind.METHOD, name=mname, header=header))
                elif ch.type in ("property_signature",):
                    pname_node = ch.child_by_field_name("name") or ch.child_by_field_name("property")
                    pname = get_node_text(pname_node) or None
                    itf.children.append(self._make_node(ch, kind=NodeKind.PROPERTY, name=pname, header=None))
                else:
                    itf.children.append(self._make_node(ch, kind=NodeKind.LITERAL, name=None, header=None))
        return [itf]

    def _handle_type_alias(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []
        name = get_node_text(name_node) or None
        header = (get_node_text(node) or "").strip().rstrip(";")
        alias = self._make_node(node, kind=NodeKind.LITERAL, name=name, header=header, subtype="type_alias")
        return [alias]

    def _handle_enum(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []
        en_name = get_node_text(name_node) or None
        header = self._build_class_like_header(node, keyword="enum", name=en_name or "<anonymous>")
        en = self._make_node(node, kind=NodeKind.ENUM, name=en_name, header=header)
        body = next((c for c in node.children if c.type == "enum_body"), None)
        if body:
            for member in body.named_children:
                if member.type == "enum_assignment":
                    mname_node = member.child_by_field_name("name")
                    mname = get_node_text(mname_node) or None
                    if mname:
                        en.children.append(self._make_node(member, kind=NodeKind.CONST, name=mname, header=None))
                elif member.type in ("property_identifier", "identifier"):
                    mname = get_node_text(member) or None
                    if mname:
                        en.children.append(self._make_node(member, kind=NodeKind.CONST, name=mname, header=None))
                else:
                    self._debug_unknown_node(member, context="enum.member")
        return [en]

    def _handle_namespace(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name") or next(
            (c for c in node.named_children if c.type in ("identifier", "property_identifier")), None
        )
        name = get_node_text(name_node) or None
        header = (get_node_text(node) or "").split("{", 1)[0].strip()
        ns = self._make_node(node, kind=NodeKind.NAMESPACE, name=name, header=header)
        body = next((c for c in node.children if c.type == "statement_block"), None)
        if body:
            for ch in body.named_children:
                ns.children.extend(self._process_node(ch, parent=ns))
        return [ns]

    def _handle_method(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        name = get_node_text(name_node) or None
        header = self._build_fn_like_header(node, prefix="", name=name)
        return [self._make_node(node, kind=NodeKind.METHOD, name=name, header=header)]

    def _handle_class_field(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name") or next((c for c in node.named_children if c.type in ("identifier","property_identifier")), None)
        pname = get_node_text(name_node) or None
        value_node = node.child_by_field_name("value")
        out: List[ParsedNode] = []
        if value_node and value_node.type == "arrow_function":
            out.append(self._handle_arrow_function_in_holder(node, value_node, parent or None))
        elif value_node and value_node.type in ("class", "class_declaration", "abstract_class_declaration"):
            out.extend(self._handle_class(value_node, parent))
        else:
            out.append(self._make_node(node, kind=NodeKind.PROPERTY, name=pname, header=None))
        return out

    def _handle_comment(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        return [self._make_node(node, kind=NodeKind.COMMENT, name=None, header=None)]

    def _handle_call_expression(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        # Collect require() usage and otherwise treat as literal expression
        self._collect_require_calls(node, alias=None)
        expr = self._make_node(node, kind=NodeKind.LITERAL, name=None, header=None)
        expr.comment = self._get_preceding_comment(node)
        return [expr]

    def _handle_for_in(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        # Handles both `for (... in ...)` and `for (... of ...)` in tree-sitter typescript
        header = (get_node_text(node) or "").split("{", 1)[0].strip()
        loop = self._make_node(node, kind=NodeKind.LITERAL, name=None, header=header)
        body = node.child_by_field_name("body") or node.child_by_field_name("statement") \
            or next((c for c in node.children if c.type in ("statement_block",)), None)
        if body:
            for ch in body.named_children:
                loop.children.extend(self._process_node(ch, parent=loop))
        return [loop]

    def _handle_if(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        # Represent the if as a literal header and traverse branches
        header = (get_node_text(node) or "").split("{", 1)[0].strip()
        ifn = self._make_node(node, kind=NodeKind.LITERAL, name=None, header=header)
        cons = node.child_by_field_name("consequence") or node.child_by_field_name("body")
        alt = node.child_by_field_name("alternative")
        if cons:
            for ch in (cons.named_children or []):
                ifn.children.extend(self._process_node(ch, parent=ifn))
        if alt:
            for ch in (alt.named_children or []):
                ifn.children.extend(self._process_node(ch, parent=ifn))
        return [ifn]

    def _handle_satisfies_expression(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        # Unwrap and process inner value; useful in `export default { ... } satisfies Config`
        left = node.child_by_field_name("left") or node.child_by_field_name("value")
        if left is not None:
            return self._process_node(left, parent)
        return [self._literal_node(node)]

    def _handle_ambient_declaration(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        # Handle `declare ...` blocks (e.g., `declare global { ... }`)
        name_node = next((c for c in node.named_children if c.type in ("identifier", "property_identifier")), None)
        name = get_node_text(name_node) or None
        header = (get_node_text(node) or "").split("{", 1)[0].strip()
        amb = self._make_node(node, kind=NodeKind.CUSTOM, name=name, header=header, subtype="ambient")
        block = next((c for c in node.children if c.type == "statement_block"), None)
        if block:
            for ch in block.named_children:
                amb.children.extend(self._process_node(ch, parent=amb))
        return [amb]

    def _handle_expression(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        for ch in node.named_children:
            if ch.type == "assignment_expression":
                lhs = ch.child_by_field_name("left")
                rhs = ch.child_by_field_name("right")
                # CommonJS exports
                if lhs and self._is_commonjs_export(lhs):
                    # Wrap the exported right-hand symbol(s) in a CUSTOM export node
                    exp = self._make_node(ch, kind=NodeKind.CUSTOM, name=None, header="export", subtype="export")
                    if rhs and rhs.type == "arrow_function":
                        fn = self._handle_arrow_function_in_holder(ch, rhs, parent)
                        exp.children.append(fn)
                        return [exp]
                    if rhs and rhs.type in ("function_declaration", "function_expression"):
                        # Name function from assignment LHS
                        fn = self._handle_function_expression_in_holder(ch, rhs, parent)
                        exp.children.append(fn)
                        return [exp]
                    if rhs and rhs.type in ("class", "class_declaration", "abstract_class_declaration"):
                        # Name class from assignment LHS
                        cls = self._handle_class_expression(ch, rhs, parent)
                        exp.children.append(cls)
                        return [exp]
                    return [exp]
                # require() aliasing
                if rhs and rhs.type == "call_expression":
                    self._collect_require_calls(rhs, alias=(get_node_text(lhs) or "").split(".")[-1] or None)
                # Holder-aware assignment handling outside export:
                if rhs and rhs.type == "arrow_function":
                    return [self._handle_arrow_function_in_holder(ch, rhs, parent)]
                if rhs and rhs.type in ("function_declaration", "function_expression"):
                    return [self._handle_function_expression_in_holder(ch, rhs, parent)]
                if rhs and rhs.type in ("class", "class_declaration", "abstract_class_declaration"):
                    return [self._handle_class_expression(ch, rhs, parent)]
                # Plain assignment: set variable name from LHS
                vname = self._resolve_arrow_function_name(ch)
                var = self._make_node(ch, kind=NodeKind.VARIABLE, name=vname, header=None)
                var.comment = self._get_preceding_comment(ch)
                return [var]
            elif ch.type == "call_expression":
                # Direct call inside an expression statement; handle uniformly
                return self._handle_call_expression(ch, parent)
        expr = self._make_node(node, kind=NodeKind.LITERAL, name=None, header=None)
        expr.comment = self._get_preceding_comment(node)
        return [expr]

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

    def _handle_lexical(self, node: ts.Node, parent: Optional[ParsedNode]) -> List[ParsedNode]:
        out: List[ParsedNode] = []
        for ch in node.named_children:
            if ch.type != "variable_declarator":
                continue
            value_node = ch.child_by_field_name("value")
            if value_node is not None and value_node.type == "arrow_function":
                out.append(self._handle_arrow_function_in_holder(ch, value_node, parent))
                continue
            if value_node is not None and value_node.type in ("class", "class_declaration", "abstract_class_declaration"):
                # Handle anonymous class expressions by naming from declarator LHS
                out.append(self._handle_class_expression(ch, value_node, parent))
                continue
            if value_node is not None and value_node.type == "call_expression":
                alias = None
                name_node = ch.child_by_field_name("name") or next((c for c in ch.named_children if c.type in ("identifier", "property_identifier")), None)
                if name_node is not None:
                    alias = get_node_text(name_node) or None
                self._collect_require_calls(value_node, alias=alias)
            name_node = ch.child_by_field_name("name") or next((c for c in ch.named_children if c.type in ("identifier", "property_identifier")), None)
            vname = get_node_text(name_node) or None
            kind = NodeKind.CONST if (get_node_text(node) or "").lstrip().startswith("const") else NodeKind.VARIABLE
            out.append(self._make_node(ch, kind=kind, name=vname, header=None))
        return out

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


class TypeScriptLanguageHelper(AbstractLanguageHelper):
    language = ProgrammingLanguage.TYPESCRIPT

    def get_node_summary(
        self,
        sym: Node,
        indent: int = 0,
        include_comments: bool = False,
        include_docs: bool = False,
        include_parents: bool = False,
    ) -> str:
        IND = " " * indent
        lines: List[str] = []

        if include_comments and sym.comment:
            for ln in (sym.comment or "").splitlines():
                lines.append(f"{IND}{ln.strip()}")

        header = (sym.header or sym.name or "").strip()

        if sym.kind in (NodeKind.CLASS,):
            if header and not header.endswith("{"):
                header = f"{header} {{"
            lines.append(f"{IND}{header}" if header else f"{IND}class {{")
            if sym.children:
                for ch in sym.children:
                    ch_sum = self.get_node_summary(ch, indent=indent + 2, include_comments=include_comments, include_docs=include_docs)
                    if ch_sum:
                        lines.append(ch_sum)
            else:
                lines.append(f"{IND}  ...")
            lines.append(f"{IND}}}")
            return "\n".join(lines)

        if sym.kind in (NodeKind.INTERFACE, NodeKind.ENUM, NodeKind.NAMESPACE):
            if header and not header.endswith("{"):
                header = f"{header} {{"
            lines.append(f"{IND}{header}" if header else f"{IND}<decl> {{")
            if sym.children:
                for ch in sym.children:
                    ch_sum = self.get_node_summary(ch, indent=indent + 2, include_comments=include_comments, include_docs=include_docs)
                    if ch_sum:
                        lines.append(ch_sum)
            else:
                lines.append(f"{IND}  ...")
            lines.append(f"{IND}}}")
            return "\n".join(lines)

        if sym.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
            head = header or ""
            if head and not head.endswith("{"):
                head = f"{head} {{ ... }}"
            lines.append(f"{IND}{head}" if head else f"{IND}<function>")
            return "\n".join(lines)

        # Handle custom export node: prefix "export " to child summaries
        if sym.kind == NodeKind.CUSTOM and (sym.subtype or "").lower() == "export":
            if sym.children:
                out_lines: List[str] = []
                for ch in sym.children:
                    ch_sum = self.get_node_summary(
                        ch,
                        indent=indent,
                        include_comments=include_comments,
                        include_docs=include_docs,
                    )
                    if not ch_sum:
                        continue
                    first, *rest = ch_sum.splitlines()
                    out_lines.append(f"{IND}export {first.lstrip()}")
                    for ln in rest:
                        out_lines.append(f"{IND}{ln}")
                return "\n".join(out_lines)
            # bare export (e.g. `export * from "..."`)
            bare = (sym.header or sym.body or "export").strip()
            return f"{IND}{bare}"

        if sym.children:
            lines.append(f"{IND}{header}" if header else f"{IND}{(sym.body or '').strip()}")
            for ch in sym.children:
                ch_sum = self.get_node_summary(ch, indent=indent + 2, include_comments=include_comments, include_docs=include_docs)
                if ch_sum:
                    lines.append(ch_sum)
            return "\n".join(lines)

        body = (sym.body or "").strip()
        if body:
            lines.append(f"{IND}{body}")
        return "\n".join(lines)

    def get_common_syntax_words(self) -> set[str]:
        return {
            "abstract","as","asserts","break","case","catch","class","const","continue",
            "declare","default","do","else","export","extends","false","finally","for",
            "function","if","import","in","instanceof","interface","let","module",
            "namespace","new","null","override","private","protected","public","readonly",
            "return","static","super","switch","this","throw","true","try","typeof",
            "undefined","var","void","while","with","async",
        }
