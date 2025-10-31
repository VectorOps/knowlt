import os
from pathlib import Path
from typing import Optional, List, Callable

import tree_sitter as ts
import tree_sitter_go as tsgo

from knowlt.parsers import (
    AbstractCodeParser,
    AbstractLanguageHelper,
    ParsedNode,
    ParsedImportEdge,
    get_node_text,
)
from knowlt.models import ProgrammingLanguage, NodeKind, Node, ImportEdge, Repo
from knowlt.project import ProjectManager, ProjectCache
from knowlt.logger import logger


GO_LANGUAGE = ts.Language(tsgo.language())

_parser: Optional[ts.Parser] = None


def _get_parser() -> ts.Parser:
    global _parser
    if _parser is None:
        _parser = ts.Parser(GO_LANGUAGE)
    return _parser


class GolangCodeParser(AbstractCodeParser):
    language = ProgrammingLanguage.GO
    extensions = (".go",)

    def __init__(self, pm: ProjectManager, repo: Repo, rel_path: str):
        super().__init__(pm, repo, rel_path)
        self.parser = _get_parser()
        self.source_bytes: bytes = b""
        self.module_path: Optional[str] = None
        self.module_root_abs_path: Optional[str] = None
        # Node-type -> handler mapping
        self._handlers: dict[
            str, Callable[[ts.Node, Optional[ParsedNode]], List[ParsedNode]]
        ] = {
            # top-level
            "import_declaration": self._handle_import,
            "function_declaration": self._handle_function,
            "method_declaration": self._handle_method,
            "type_declaration": self._handle_type,
            "comment": self._handle_comment,
            # struct/interface members
            "field_declaration": self._handle_field_declaration,
            "embedded_field": self._handle_embedded_field,
            "method_elem": self._handle_method_elem,
            "method_spec": self._handle_method_spec,
            "type_elem": self._handle_type_elem,
            "type_element": self._handle_type_element,
            # containers: recurse
            "field_declaration_list": self._handle_field_declaration_list,
            "type_element_list": self._handle_type_element_list,
            "method_spec_list": self._handle_method_spec_list,
        }

    def parse(self, cache: ProjectCache):
        self._load_module_path(cache)
        return super().parse(cache)

    # minimal file handler (no-op)
    def _handle_file(self, root_node: ts.Node) -> None:
        return None

    def _rel_to_virtual_path(self, rel_path: str) -> str:
        rel_dir = os.path.dirname(rel_path).replace(os.sep, "/").strip("/")
        return rel_dir or "."

    def _create_package(self, root_node):
        virtual_path = self._build_virtual_package_path(root_node)
        rel_dir = os.path.dirname(self.rel_path).replace(os.sep, "/").strip("/")
        physical_path = rel_dir or "."
        # rely on AbstractCodeParser's package structure; construct minimal object
        pkg_cls = super()._create_package(root_node).__class__
        return pkg_cls(
            language=self.language,
            physical_path=physical_path,
            virtual_path=virtual_path,
            imports=[],
        )

    def _process_node(
        self, node: ts.Node, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        handler = self._handlers.get(node.type)
        if handler is not None:
            return handler(node, parent)
        self._debug_unknown_node(node)
        return [self._literal_node(node)]

    def _load_module_path(self, cache: ProjectCache) -> None:
        project_path = self.repo.root_path
        cache_key = f"go.project.gomods::{self.repo.id}"
        gomods_map = cache.get(cache_key) if cache is not None else None

        if gomods_map is None:
            gomods_map = {}
            for root, _, files in os.walk(project_path):
                if "go.mod" in files:
                    gomod_path = os.path.join(root, "go.mod")
                    try:
                        with open(gomod_path, "r", encoding="utf8") as fh:
                            for ln in fh:
                                ln = ln.strip()
                                if ln.startswith("module"):
                                    parts = ln.split()
                                    if len(parts) >= 2:
                                        found_module_path = parts[1]
                                        gomods_map[os.path.normpath(root)] = (
                                            found_module_path
                                        )
                                    break
                    except OSError:
                        pass
            if cache is not None:
                cache.set(cache_key, gomods_map)

        file_abs_dir = os.path.dirname(os.path.join(project_path, self.rel_path))
        module_path = None
        module_root_abs_path = None
        best_match_len = -1
        for mod_dir_abs, mod_path in gomods_map.items():
            norm_file_dir = os.path.normpath(file_abs_dir)
            norm_mod_dir = os.path.normpath(mod_dir_abs)
            if norm_file_dir.startswith(norm_mod_dir):
                if len(norm_mod_dir) > best_match_len:
                    best_match_len = len(norm_mod_dir)
                    module_root_abs_path = mod_dir_abs
                    module_path = mod_path

        self.module_path = module_path
        self.module_root_abs_path = module_root_abs_path
        return None

    def _extract_package_name(self, root_node: ts.Node) -> Optional[str]:
        for node in root_node.children:
            if node.type == "package_clause":
                ident = next(
                    (
                        c
                        for c in node.children
                        if c.type in ("identifier", "package_identifier")
                    ),
                    None,
                )
                if ident is not None:
                    return get_node_text(ident)
        return None

    def _build_virtual_package_path(self, root_node: ts.Node) -> str:
        pkg_ident = self._extract_package_name(root_node)
        if pkg_ident == "main":
            rel_dir = os.path.dirname(self.rel_path).replace(os.sep, "/").strip("/")
            return rel_dir or "."

        if self.module_path and self.module_root_abs_path:
            project_path = self.repo.root_path
            file_abs_dir = os.path.dirname(os.path.join(project_path, self.rel_path))
            rel_dir_from_module_root = os.path.relpath(
                file_abs_dir, self.module_root_abs_path
            ).replace(os.sep, "/")
            if rel_dir_from_module_root == ".":
                rel_dir_from_module_root = ""
            full_path = self.module_path + (
                "/" + rel_dir_from_module_root if rel_dir_from_module_root else ""
            )
            if full_path:
                return full_path
            if pkg_ident:
                return pkg_ident
            return self._rel_to_virtual_path(self.rel_path)
        else:
            rel_dir = os.path.dirname(self.rel_path).replace(os.sep, "/").strip("/")
            return rel_dir or "."

    # comment collection
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
            raw = get_node_text(c).strip()
            parts.append(raw)
        return "\n".join(parts).strip() or None

    # --- handlers -------------------------------------------------------
    def _handle_import(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        assert self.parsed_file is not None

        def _walk(n: ts.Node) -> None:
            for child in n.children:
                if child.type == "import_spec":
                    self._process_import_spec(child)
                elif child.type == "import_spec_list":
                    _walk(child)

        _walk(node)
        imp_node = self._make_node(
            node,
            kind=NodeKind.LITERAL,
            name=None,
            header=None,
            subtype="import",
            comment=self._get_preceding_comment(node),
        )
        return [imp_node]

    def _process_import_spec(self, spec_node: ts.Node) -> None:
        assert self.parsed_file is not None
        raw_str: str = get_node_text(spec_node).strip()
        alias: Optional[str] = None
        dot = False
        import_path: Optional[str] = None
        for ch in spec_node.children:
            if ch.type in ("package_identifier", "identifier"):
                alias = get_node_text(ch)
            elif ch.type == "dot":
                dot = True
            elif ch.type == "blank_identifier":
                alias = "_"
            elif ch.type == "interpreted_string_literal":
                import_path = get_node_text(ch).strip()
        if import_path is None:
            return
        if import_path and import_path[0] in '"`' and import_path[-1] in '"`':
            import_path = import_path[1:-1]

        physical_path: Optional[str] = None
        external = True
        # relative
        if import_path.startswith((".", "./", "../")):
            abs_target = os.path.normpath(
                os.path.join(
                    os.path.dirname(
                        os.path.join(self.repo.root_path, self.parsed_file.path)
                    ),
                    import_path,
                )
            )
            if abs_target.startswith(self.repo.root_path) and os.path.isdir(abs_target):
                physical_path = os.path.relpath(
                    abs_target, self.repo.root_path
                ).replace(os.sep, "/")
                external = False
        # module-local
        elif (
            self.module_path
            and self.module_root_abs_path
            and (
                import_path == self.module_path
                or import_path.startswith(self.module_path + "/")
            )
        ):
            sub_path = import_path[len(self.module_path) :].lstrip("/")
            abs_target = os.path.join(self.module_root_abs_path, sub_path)
            if os.path.isdir(abs_target):
                physical_path = os.path.relpath(
                    abs_target, self.repo.root_path
                ).replace(os.sep, "/")
                external = False
        else:
            abs_target = os.path.join(self.repo.root_path, import_path)
            if os.path.isdir(abs_target):
                physical_path = import_path
                external = False

        self.parsed_file.imports.append(
            ParsedImportEdge(
                physical_path=physical_path,
                virtual_path=import_path,
                alias=None if dot else alias,
                dot=dot,
                external=external,
                raw=raw_str,
            )
        )

    def _build_header(self, node: ts.Node, keyword: str) -> str:
        code = get_node_text(node) or ""
        idx = code.find(f"{keyword} ")
        # For type specs, the "type" keyword may live in the parent; synthesize it
        if idx == -1 and keyword == "type":
            payload = (get_node_text(node) or "").lstrip()
            code = f"{keyword} {payload}"
            idx = 0
        if idx == -1:
            return keyword
        depth = 0
        for i in range(idx, len(code)):
            ch = code[i]
            if ch in "([":  # track parameter/receiver lists
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif ch == "{" and depth == 0:
                return code[idx : i + 1].strip()
            elif ch == "\n" and depth == 0:
                return code[idx:i].strip()
        return code[idx:].strip()

    def _handle_function(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        ident_node = next((c for c in node.children if c.type == "identifier"), None)
        if ident_node is None:
            return [self._literal_node(node)]
        name = get_node_text(ident_node)
        header = self._build_header(node, "func")
        comment = self._get_preceding_comment(node)
        sym = self._make_node(
            node, kind=NodeKind.FUNCTION, name=name, header=header, comment=comment
        )
        return [sym]

    def _handle_method(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        ident_node = next(
            (c for c in node.children if c.type in ("field_identifier", "identifier")),
            None,
        )
        if ident_node is None:
            return [self._literal_node(node)]
        name = get_node_text(ident_node)
        header = self._build_header(node, "func")
        comment = self._get_preceding_comment(node)
        sym = self._make_node(
            node, kind=NodeKind.METHOD, name=name, header=header, comment=comment
        )
        return [sym]

    def _handle_type(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        # Gather all type specs, including aliases and grouped specs
        def _collect_specs(n: ts.Node, acc: List[ts.Node]) -> None:
            for ch in n.children:
                if ch.type in ("type_spec", "type_alias", "type_alias_spec"):
                    acc.append(ch)
                elif ch.type in ("type_spec_list",):
                    _collect_specs(ch, acc)

        specs: List[ts.Node] = []
        _collect_specs(node, specs)
        if not specs:
            specs = [node]

        symbols: List[ParsedNode] = []
        for spec in specs:
            ident = next(
                (
                    c
                    for c in spec.children
                    if c.type in ("identifier", "type_identifier")
                ),
                None,
            )
            if ident is None:
                continue

            name = get_node_text(ident)
            header = self._build_header(spec, "type")
            comment = self._get_preceding_comment(spec)
            subtype = None
            type_node = next(
                (
                    c
                    for c in spec.children
                    if c is not ident
                    and c.type
                    not in (
                        "comment",
                        "=",
                    )
                ),
                None,
            )

            kind = NodeKind.CLASS

            if type_node is not None:
                if type_node.type == "interface_type":
                    subtype = "interface"
                elif type_node.type == "struct_type":
                    subtype = "struct"
                elif type_node.type == "type_identifier":
                    kind = NodeKind.LITERAL
                    header = None
            # Ensure the body includes the full type declaration text.
            # type_spec nodes don't include the 'type' keyword; synthesize it.
            spec_text = (get_node_text(spec) or "").lstrip()
            if spec_text.startswith("type "):
                full_body = spec_text
            else:
                full_body = f"type {spec_text}"

            sym = self._make_node(
                spec,
                kind=kind,
                name=name,
                header=header,
                comment=comment,
                subtype=subtype,
                body=full_body,
            )
            # Recursively attach struct/interface children via dispatcher with prefilter.
            if type_node is not None:
                allowed_struct = {
                    "field_declaration_list",
                    "field_declaration",
                    "embedded_field",
                    "comment",
                }
                allowed_iface = {
                    "method_elem",
                    "method_spec",
                    "type_elem",
                    "type_element",
                    "field_declaration",
                    "method_spec_list",
                    "type_element_list",
                    "field_declaration_list",
                    "comment",
                }
                ignore_tokens = {"{", "}", "struct", "interface"}
                allowed = (
                    allowed_struct if type_node.type == "struct_type" else allowed_iface
                )
                for ch in type_node.children:
                    print(
                        get_node_text(ch),
                        ch,
                        ch.type,
                        ch.type in allowed,
                        ch.type in ignore_tokens,
                    )
                    if ch.type in allowed:
                        sym.children.extend(self._process_node(ch, sym))
                    elif ch.type in ignore_tokens:
                        # skip punctuation/keywords, do not dispatch
                        continue
                    else:
                        # defensive: warn on unexpected named nodes
                        if getattr(ch, "is_named", False):
                            logger.warning(
                                "Unexpected Go type child node",
                                parent_type=type_node.type,
                                node_type=ch.type,
                                line=ch.start_point[0] + 1,
                            )
            symbols.append(sym)
        return symbols

    def _handle_comment(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        return [self._make_node(node, kind=NodeKind.COMMENT, name=None, header=None)]

    def _literal_node(self, node: ts.Node) -> ParsedNode:
        return self._make_node(node, kind=NodeKind.LITERAL, name=None, header=None)

    def _debug_unknown_node(self, node: ts.Node) -> None:
        path = self.parsed_file.path if self.parsed_file else self.rel_path
        logger.debug(
            "Unknown Go node type; emitting literal",
            path=path,
            node_type=node.type,
            line=node.start_point[0] + 1,
            raw=(get_node_text(node) or "")[:200],
        )

    # ---- member and container handlers via dispatcher ------------------
    def _handle_field_declaration(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        name_node = next(
            (
                c
                for c in node.children
                if c.type
                in (
                    "field_identifier",
                    "identifier",
                    "type_identifier",
                    "qualified_type",
                )
            ),
            None,
        )
        name = get_node_text(name_node) if name_node is not None else None
        return [
            self._make_node(
                node,
                kind=NodeKind.PROPERTY,
                name=name,
                comment=self._get_preceding_comment(node),
            )
        ]

    def _handle_embedded_field(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        return self._handle_field_declaration(node, parent)

    def _handle_method_elem(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        ident = next(
            (c for c in node.children if c.type in ("field_identifier", "identifier")),
            None,
        )
        name = get_node_text(ident) if ident is not None else None
        return [
            self._make_node(
                node,
                kind=NodeKind.PROPERTY,
                name=name,
                comment=self._get_preceding_comment(node),
            )
        ]

    def _handle_method_spec(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        return self._handle_method_elem(node, parent)

    def _handle_type_elem(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        return [
            self._make_node(
                node,
                kind=NodeKind.PROPERTY,
                name=None,
                comment=self._get_preceding_comment(node),
            )
        ]

    def _handle_type_element(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        return self._handle_type_elem(node, parent)

    def _handle_field_declaration_list(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        out: List[ParsedNode] = []
        for ch in node.children:
            if ch.type in ("{", "}"):
                continue

            out.extend(self._process_node(ch, parent))
        return out

    def _handle_type_element_list(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        out: List[ParsedNode] = []
        for ch in node.children:
            if ch.type in ("{", "}"):
                continue
            out.extend(self._process_node(ch, parent))
        return out

    def _handle_method_spec_list(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        out: List[ParsedNode] = []
        for ch in node.children:
            if ch.type in ("{", "}"):
                continue
            out.extend(self._process_node(ch, parent))
        return out


class GolangLanguageHelper(AbstractLanguageHelper):
    language = ProgrammingLanguage.GO

    def get_node_summary(
        self,
        sym: Node,
        indent: int = 0,
        include_comments: bool = False,
        include_docs: bool = False,
        include_parents: bool = False,
        child_stack: Optional[List[List[Node]]] = None,
    ) -> str:
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
            if (
                sym.kind in (NodeKind.FUNCTION, NodeKind.METHOD)
                and header
                and not header.endswith("{")
            ):
                header = f"{header} {{"
            if header:
                for ln in header.splitlines():
                    lines.append(f"{IND}{ln.strip()}")
            if include_docs and getattr(sym, "docstring", None):
                for ln in (sym.docstring or "").splitlines():
                    lines.append(f"{IND}\t{ln.strip()}")
            if sym.children:
                emit_children(sym.children, indent + 1)
            else:
                lines.append(f"{IND}\t...")
            lines.append(f"{IND}}}")
            return "\n".join(lines)

        if sym.kind == NodeKind.BLOCK and sym.children:
            label = (sym.subtype or "block").lower()
            lines.append(f"{IND}{label}:")
            emit_children(sym.children, indent + 1)
            return "\n".join(lines)

        if sym.children:
            emit_children(sym.children, indent)
            return "\n".join(lines)

        body = (sym.body or "").strip()
        if body:
            lines.append(f"{IND}{body}")
        return "\n".join(lines)

    def get_common_syntax_words(self) -> set[str]:
        return {
            "break",
            "case",
            "const",
            "continue",
            "default",
            "else",
            "false",
            "for",
            "func",
            "if",
            "import",
            "interface",
            "map",
            "nil",
            "package",
            "range",
            "return",
            "struct",
            "switch",
            "true",
            "type",
            "var",
        }
