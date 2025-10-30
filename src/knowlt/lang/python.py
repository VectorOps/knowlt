import os
from pathlib import Path
from typing import Optional, List, Callable

import tree_sitter as ts
import tree_sitter_python as tspython

from knowlt.parsers import (
    AbstractCodeParser,
    AbstractLanguageHelper,
    ParsedNode,
    ParsedImportEdge,
    get_node_text,
)
from knowlt.models import ProgrammingLanguage, NodeKind, Node, ImportEdge, Repo
from knowlt.project import ProjectManager, ProjectCache
from knowlt.settings import PythonSettings
from knowlt.logger import logger

PY_LANGUAGE = ts.Language(tspython.language())
_parser: Optional[ts.Parser] = None


def _get_parser() -> ts.Parser:
    global _parser
    if _parser is None:
        _parser = ts.Parser(PY_LANGUAGE)
    return _parser


class PythonCodeParser(AbstractCodeParser):
    language = ProgrammingLanguage.PYTHON
    extensions = [".py"]
    settings: PythonSettings

    def __init__(self, pm: ProjectManager, repo: Repo, rel_path: str) -> None:
        super().__init__(pm, repo, rel_path)
        self.parser = _get_parser()
        lang_settings = self.pm.settings.languages.get(
            self.language.value, PythonSettings()
        )
        if not isinstance(lang_settings, PythonSettings):
            logger.warning(
                "Python language settings are not of the correct type, using defaults.",
                actual_type=type(lang_settings).__name__,
            )
            lang_settings = PythonSettings()
        self.settings = lang_settings

        # Node-type -> handler mapping
        self._handlers: dict[
            str, Callable[[ts.Node, Optional[ParsedNode]], List[ParsedNode]]
        ] = {
            "import_statement": self._handle_import,
            "import_from_statement": self._handle_import,
            "future_import_statement": self._handle_import,
            "function_definition": self._handle_function,
            "async_function_definition": self._handle_function,
            "class_definition": self._handle_class,
            "decorated_definition": self._handle_decorated_definition,
            "if_statement": self._handle_if,
            "try_statement": self._handle_try,
            "comment": self._handle_comment,
            "assignment": self._handle_assignment,
            "expression_statement": self._handle_expression,
        }

    def _handle_file(self, root_node: ts.Node) -> None:
        if self.parsed_file:
            self.parsed_file.docstring = self._extract_docstring(root_node)

    def _rel_to_virtual_path(self, rel_path: str) -> str:
        p = Path(rel_path)
        if p.name == "__init__.py":
            parts = p.parent.parts
        else:
            parts = p.with_suffix("").parts
        return ".".join(parts)

    def _process_node(
        self, node: ts.Node, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        handler = self._handlers.get(node.type)
        if handler is not None:
            return handler(node, parent)
        self._debug_unknown_node(node)
        return [self._literal_node(node)]

    # Handlers
    def _handle_import(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        raw_stmt = get_node_text(node) or ""
        import_path: Optional[str] = None
        alias: Optional[str] = None
        dot = False
        if node.type == "import_statement":
            dotted = next((c for c in node.children if c.type == "dotted_name"), None)
            aliased = next(
                (c for c in node.children if c.type == "aliased_import"), None
            )
            if aliased is not None:
                name_node = aliased.child_by_field_name("name")
                alias_node = aliased.child_by_field_name("alias")
                import_path = get_node_text(name_node) or None
                alias = get_node_text(alias_node) or None
            elif dotted is not None:
                import_path = get_node_text(dotted) or None
        elif node.type == "import_from_statement":
            rel_node = next(
                (c for c in node.children if c.type == "relative_import"), None
            )
            mod_node = next((c for c in node.children if c.type == "dotted_name"), None)
            rel_txt = get_node_text(rel_node) or ""
            mod_txt = get_node_text(mod_node) or ""
            import_path = f"{rel_txt}{mod_txt}" if (rel_txt or mod_txt) else None
            dot = bool(rel_node)
            aliased = next(
                (c for c in node.children if c.type == "aliased_import"), None
            )
            if aliased is not None:
                alias_node = aliased.child_by_field_name("alias")
                alias = get_node_text(alias_node) or None
        stripped = (import_path or "").lstrip(".")
        is_local = bool(stripped) and self._is_local_import(stripped)
        resolved_path: Optional[str] = None
        if is_local:
            resolved_path = self._resolve_local_import_path(stripped)
        assert self.parsed_file is not None
        self.parsed_file.imports.append(
            ParsedImportEdge(
                physical_path=resolved_path,
                virtual_path=import_path or raw_stmt,
                alias=alias,
                dot=dot,
                external=not is_local,
                raw=raw_stmt,
            )
        )
        # Also return a node so summaries can include import statements.
        # Use a LITERAL node with subtype "import" and the raw body.
        imp_node = self._make_node(
            node,
            kind=NodeKind.LITERAL,
            name=None,
            header=None,
            subtype="import",
        )
        imp_node.comment = self._get_preceding_comment(node)
        return [imp_node]

    def _handle_function(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []
        fn_name = get_node_text(name_node)
        # If wrapped in a decorated_definition, use it to collect decorator lines
        wrapper = (
            node.parent
            if (node.parent is not None and node.parent.type == "decorated_definition")
            else node
        )
        base_header = self._build_function_header(node)
        decorator_texts: List[str] = [
            (get_node_text(c) or "").strip()
            for c in wrapper.children
            if c.type == "decorator"
        ] if wrapper is not node else []
        header = (
            f'{"\n".join(decorator_texts)}\n{base_header}'
            if decorator_texts
            else base_header
        )
        comment = self._get_preceding_comment(wrapper)
        doc = self._extract_docstring(node)
        kind = (
            NodeKind.METHOD
            if parent and parent.kind == NodeKind.CLASS
            else NodeKind.FUNCTION
        )
        sym = self._make_node(
            node, kind=kind, name=fn_name, header=header, comment=comment, docstring=doc
        )
        return [sym]

    def _handle_class(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []
        cls_name = get_node_text(name_node)
        # If wrapped in a decorated_definition, use it to collect decorator lines
        wrapper = (
            node.parent
            if (node.parent is not None and node.parent.type == "decorated_definition")
            else node
        )
        base_header = self._build_class_header(node)
        decorator_texts: List[str] = [
            (get_node_text(c) or "").strip()
            for c in wrapper.children
            if c.type == "decorator"
        ] if wrapper is not node else []
        header = (
            f'{"\n".join(decorator_texts)}\n{base_header}'
            if decorator_texts
            else base_header
        )
        comment = self._get_preceding_comment(wrapper)
        doc = self._extract_docstring(node)
        cls = self._make_node(
            node,
            kind=NodeKind.CLASS,
            name=cls_name,
            header=header,
            comment=comment,
            docstring=doc,
        )
        block = next((c for c in node.children if c.type == "block"), None)
        body_children = block.children if block is not None else node.children

        for child in body_children:
            cls.children.extend(self._process_node(child, parent=cls))

        return [cls]

    def _handle_decorated_definition(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        inner: ts.Node | None = next(
            (
                c
                for c in node.children
                if c.type
                in (
                    "function_definition",
                    "async_function_definition",
                    "class_definition",
                )
            ),
            None,
        )
        if inner is None:
            self._debug_unknown_node(node)
            return [self._literal_node(node)]
        if inner.type in ("function_definition", "async_function_definition"):
            return self._handle_function(inner, parent)
        if inner.type == "class_definition":
            return self._handle_class(inner, parent)
        self._debug_unknown_node(node)
        return [self._literal_node(node)]

    def _handle_if(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        # CUSTOM parent for the whole chain
        chain = self._make_node(
            node, kind=NodeKind.CUSTOM, name=None, header=None, subtype="if"
        )

        def add_block(block_node: ts.Node, subtype: str, header_text: Optional[str]) -> None:
            # BLOCK child; header includes condition (if/elif) or 'else:'
            blk = self._make_node(
                block_node,
                kind=NodeKind.BLOCK,
                name=None,
                header=header_text,
                subtype=subtype,
            )
            for ch in block_node.children:
                blk.children.extend(self._process_node(ch, parent=blk))
            chain.children.append(blk)

        # if <condition>: <consequence>
        cons = node.child_by_field_name("consequence")
        cond_node = node.child_by_field_name("condition")
        if cons is not None:
            cond_text = (get_node_text(cond_node) or "").strip()
            add_block(cons, "if", f"if {cond_text}:")

        # elif / else clauses
        for alt in node.children:
            if alt.type == "elif_clause":
                block = alt.child_by_field_name("consequence")
                if block is not None:
                    elif_cond = (get_node_text(alt.child_by_field_name("condition")) or "").strip()
                    add_block(block, "elif", f"elif {elif_cond}:")
            elif alt.type == "else_clause":
                block = alt.child_by_field_name("body") or alt.child_by_field_name(
                    "consequence"
                )
                if block is not None:
                    add_block(block, "else", "else:")
        return [chain]

    def _handle_try(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        # CUSTOM parent; no header for non-def constructs
        tryn = self._make_node(
            node, kind=NodeKind.CUSTOM, name=None, header=None, subtype="try"
        )

        def add_block(block_node: ts.Node, subtype: str) -> None:
            # BLOCK child; no header for non-def constructs; use subtype to differentiate
            blk = self._make_node(
                block_node, kind=NodeKind.BLOCK, name=None, header=None, subtype=subtype
            )
            for ch in block_node.children:
                blk.children.extend(self._process_node(ch, parent=blk))
            tryn.children.append(blk)

        for ch in node.children:
            if ch.type == "block":
                add_block(ch, "try")
            elif ch.type == "except_clause":
                blk = next((c for c in ch.children if c.type == "block"), None)
                if blk is not None:
                    add_block(blk, "except")
            elif ch.type == "else_clause":
                blk = next((c for c in ch.children if c.type == "block"), None)
                if blk is not None:
                    add_block(blk, "else")
            elif ch.type == "finally_clause":
                blk = next((c for c in ch.children if c.type == "block"), None)
                if blk is not None:
                    add_block(blk, "finally")
        return [tryn]

    def _handle_comment(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        return [self._make_node(node, kind=NodeKind.COMMENT, name=None, header=None)]

    def _handle_assignment(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        return [self._literal_node(node)]

    def _handle_expression(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        expr = self._make_node(node, kind=NodeKind.LITERAL, name=None, header=None)
        expr.comment = self._get_preceding_comment(node)
        return [expr]

    # Utilities
    def _literal_node(self, node: ts.Node) -> ParsedNode:
        return self._make_node(node, kind=NodeKind.LITERAL, name=None, header=None)

    def _build_function_header(self, node: ts.Node) -> str:
        is_async = node.type == "async_function_definition" or (
            get_node_text(node) or ""
        ).lstrip().startswith("async def")
        name_node = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        ret_node = node.child_by_field_name("return_type")
        name = get_node_text(name_node) or "<anonymous>"
        params = get_node_text(params_node) or "()"
        ret = (get_node_text(ret_node) or "").strip()
        prefix = "async def" if is_async else "def"
        return f"{prefix} {name}{params}{(' -> ' + ret) if ret else ''}:"

    def _build_class_header(self, node: ts.Node) -> str:
        code = get_node_text(node) or ""
        idx = code.find("class ")
        if idx == -1:
            return "class:"
        depth = 0
        for i in range(idx, len(code)):
            ch = code[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == ":" and depth == 0:
                return code[idx : i + 1].strip()
        return code[idx:].strip()

    def _extract_docstring(self, node: ts.Node) -> Optional[str]:
        if node.type in ("function_definition", "class_definition"):
            block = next((c for c in node.children if c.type == "block"), None)
            if block is not None:
                return self._extract_docstring(block)
            return None
        for ch in node.children:
            if (
                ch.type == "expression_statement"
                and ch.children
                and ch.children[0].type == "string"
            ):
                return (get_node_text(ch.children[0]) or "").strip()
            if ch.type == "string":
                return (get_node_text(ch) or "").strip()
            if ch.type not in ("comment",):
                break
        return None

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
            if sib.type in ("newline",):
                sib = sib.prev_sibling
                continue
            break
        if comments:
            comments.reverse()
            merged = "\n".join(comments).strip()
            return merged or None
        return None

    def _locate_module_path(self, import_path: str) -> Optional[Path]:
        if not import_path:
            return None
        project_root = Path(self.repo.root_path).resolve()
        parts = import_path.split(".")
        found: Optional[Path] = None
        for idx in range(1, len(parts) + 1):
            base = project_root.joinpath(*parts[:idx])
            if any(seg in self.settings.venv_dirs for seg in base.parts):
                continue
            for suffix in self.settings.module_suffixes:
                file_cand = base.with_suffix(suffix)
                if file_cand.exists():
                    found = file_cand
                    break
            else:
                if base.is_dir() and (base / "__init__.py").exists():
                    found = base / "__init__.py"
        return found

    def _resolve_local_import_path(self, import_path: str) -> Optional[str]:
        path_obj = self._locate_module_path(import_path)
        if path_obj is None:
            return None
        project_root = Path(self.repo.root_path).resolve()
        return path_obj.relative_to(project_root).as_posix()

    def _is_local_import(self, import_path: str) -> bool:
        return self._locate_module_path(import_path) is not None

    def _debug_unknown_node(self, node: ts.Node) -> None:
        path = self.parsed_file.path if self.parsed_file else self.rel_path
        logger.debug(
            "Unknown node type; emitting literal",
            path=path,
            node_type=node.type,
            line=node.start_point[0] + 1,
            raw=(get_node_text(node) or "")[:200],
        )


class PythonLanguageHelper(AbstractLanguageHelper):
    language = ProgrammingLanguage.PYTHON

    def get_node_summary(
        self,
        sym: Node,
        indent: int = 0,
        include_comments: bool = False,
        include_docs: bool = False,
    ) -> str:
        IND = " " * indent
        lines: List[str] = []

        def emit_children(children: List[Node], child_indent: int) -> None:
            for ch in children:
                ch_sum = self.get_node_summary(
                    ch,
                    indent=child_indent,
                    include_comments=include_comments,
                    include_docs=include_docs,
                )
                if ch_sum:
                    lines.append(ch_sum)

        # Optional preceding comments
        if include_comments and sym.comment:
            for ln in (sym.comment or "").splitlines():
                lines.append(f"{IND}{ln.strip()}")

        # Definitions: emit header and then either children or ellipsis
        if sym.kind in (NodeKind.CLASS, NodeKind.FUNCTION, NodeKind.METHOD):
            header = sym.header or ""
            if header and not header.endswith(":"):
                header = f"{header}:"
            if header:
                for ln in header.splitlines():
                    lines.append(f"{IND}{ln.strip()}")
            if include_docs and sym.docstring:
                for ln in (sym.docstring or "").splitlines():
                    lines.append(f"{IND}    {ln.strip()}")
            if sym.children:
                emit_children(sym.children, indent + 4)
            else:
                lines.append(f"{IND}    ...")
            return "\n".join(lines)

        # Special handling for if/elif/else chains
        if sym.kind == NodeKind.CUSTOM and (sym.subtype or "") == "if":
            # children are BLOCK nodes with subtype in {"if","elif","else"}
            for blk in sym.children:
                label = (blk.subtype or "").lower()
                if label in {"if", "elif", "else"}:
                    header_line = (blk.header or f"{label}:").strip()
                    lines.append(f"{IND}{header_line}")
                    emit_children(blk.children, indent + 4)
                else:
                    # Fallback if unexpected subtype
                    lines.append(f"{IND}if:")
                    emit_children(blk.children, indent + 4)
            return "\n".join(lines)

        # Special handling for try/except/else/finally
        if sym.kind == NodeKind.CUSTOM and (sym.subtype or "") == "try":
            for blk in sym.children:
                label = (blk.subtype or "").lower()
                if label in {"try", "except", "else", "finally"}:
                    lines.append(f"{IND}{label}:")
                    emit_children(blk.children, indent + 4)
                else:
                    # Fallback if unexpected subtype
                    lines.append(f"{IND}try:")
                    emit_children(blk.children, indent + 4)
            return "\n".join(lines)

        # If summarizing a BLOCK standalone (outside parent handlers), emit its subtype header then children
        if sym.kind == NodeKind.BLOCK and sym.children:
            label = (sym.subtype or "block").lower()
            lines.append(f"{IND}{label}:")
            emit_children(sym.children, indent + 4)
            return "\n".join(lines)

        # Generic: if there are children but no special header, recurse
        if sym.children:
            emit_children(sym.children, indent)
            return "\n".join(lines)

        # Leaf: print available body
        body = (sym.body or "").strip()
        if body:
            lines.append(f"{IND}{body}")
        return "\n".join(lines)

    def get_common_syntax_words(self) -> set[str]:
        return {
            "and",
            "as",
            "break",
            "class",
            "continue",
            "def",
            "del",
            "elif",
            "else",
            "except",
            "false",
            "finally",
            "for",
            "from",
            "if",
            "import",
            "in",
            "is",
            "none",
            "not",
            "or",
            "pass",
            "raise",
            "return",
            "true",
            "try",
            "while",
            "with",
            "self",
        }
