import os
from typing import Optional, List

import tree_sitter as ts
import tree_sitter_hcl as tshcl

from knowlt.parsers import (
    AbstractCodeParser,
    AbstractLanguageHelper,
    ParsedNode,
    get_node_text,
)
from knowlt.models import ProgrammingLanguage, NodeKind, Node, Repo
from typing import TYPE_CHECKING

from knowlt.logger import logger

if TYPE_CHECKING:
    from knowlt.project import ProjectManager, ProjectCache


HCL_LANGUAGE = ts.Language(tshcl.language())
_parser: Optional[ts.Parser] = None


def _get_parser() -> ts.Parser:
    global _parser
    if _parser is None:
        _parser = ts.Parser(HCL_LANGUAGE)
    return _parser


class TerraformCodeParser(AbstractCodeParser):
    """
    Terraform/HCL parser based on tree-sitter-hcl.

    Rules:
    - Walk the entire tree (no top-level-only parsing).
    - Represent block-like constructs and objects as nested nodes.
    - Attributes and object elements are PROPERTIES; they only have children
      when their values contain nested block-like structures (e.g. objects),
      not for simple literal or variable values.
    - Literal/variable-like leaves are emitted as individual nodes without
      recursing into their internal structure (string_lit, bool_lit, etc.).
    """

    language = ProgrammingLanguage.TERRAFORM
    extensions = [".tf", ".tfvars"]

    def __init__(self, pm: "ProjectManager", repo: Repo, rel_path: str) -> None:
        super().__init__(pm, repo, rel_path)
        self.parser = _get_parser()

    def _rel_to_virtual_path(self, rel_path: str) -> str:
        # For Terraform, treat the module directory as the virtual path anchor.
        # Example: "modules/example/main.tf" -> "modules/example/main"
        base, _ = os.path.splitext(rel_path)
        return base.replace(os.sep, "/")

    def _process_node(
        self, node: ts.Node, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        t = node.type

        # Punctuation and template-delimiter tokens that tree-sitter-hcl
        # exposes as standalone nodes. They do not carry semantic meaning
        # for our model and should be ignored rather than logged as unknown
        # node types.
        if t in (
            "(",
            ")",
            "[",
            "]",
            ",",
            ".",
            "quoted_template_start",
            "quoted_template_end",
            "template_interpolation_start",
            "template_interpolation_end",
        ):
            return []

        # Structural containers: just recurse into children; no own node.
        if t in ("config_file", "body"):
            results: List[ParsedNode] = []
            for ch in node.children:
                results.extend(self._process_node(ch, parent=parent))
            return results

        if t == "block":
            blk = self._handle_block(node, parent)
            return [blk] if blk is not None else []

        if t == "attribute":
            prop = self._handle_attribute(node, parent)
            return [prop] if prop is not None else []

        if t == "object":
            obj = self._handle_object(node, parent)
            return [obj] if obj is not None else []

        if t == "object_elem":
            return self._handle_object_elem(node, parent)
        if t == "comment":
            return self._handle_comment(node, parent)

        # Leaf-like values: stop recursion here.
        if t == "literal_value":
            return [self._make_node(node, kind=NodeKind.LITERAL)]

        if t == "variable_expr":
            return [self._make_node(node, kind=NodeKind.VARIABLE)]

        # Expressions and collections: transparent containers, except for
        # template expressions, which we treat as single literal leaves.
        if t == "expression":
            # Template expressions such as "example-sg-${local.env}" are
            # represented by a mix of template_* tokens under an expression
            # node. For these, we do not recurse into the internal template
            # structure and instead emit a single LITERAL node that spans
            # the whole expression.
            if any(
                ch.type
                in (
                    "quoted_template_start",
                    "quoted_template_end",
                    "template_interpolation_start",
                    "template_interpolation_end",
                    "template_literal",
                    "quoted_template",
                    "template_expr",
                )
                for ch in node.children
            ):
                return [self._make_node(node, kind=NodeKind.LITERAL)]

            results: List[ParsedNode] = []
            for ch in node.children:
                results.extend(self._process_node(ch, parent=parent))
            # If no children produced nodes, fall back to a single literal node.
            return results or [self._make_node(node, kind=NodeKind.LITERAL)]

        if t in ("collection_value", "tuple", "heredoc_template"):
            results: List[ParsedNode] = []
            for ch in node.children:
                results.extend(self._process_node(ch, parent=parent))
            return results

        # Default: recurse into children, but generate minimal nodes.
        results: List[ParsedNode] = []
        for ch in node.children:
            results.extend(self._process_node(ch, parent=parent))

        if results:
            return results

        # Leaf of unknown type -> treat as a literal.
        logger.debug(
            "Unknown Terraform node type",
            path=self.parsed_file.path if self.parsed_file else self.rel_path,
            node_type=node.type,
            line=node.start_point[0] + 1,
            raw=(get_node_text(node) or "")[:200],
        )
        return [self._make_node(node, kind=NodeKind.LITERAL)]

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _strip_quotes(text: str) -> str:
        text = text.strip()
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            return text[1:-1]
        return text

    def _build_block_header(self, node: ts.Node) -> str:
        # Take text up to the first '{' as header (inclusive).
        src = get_node_text(node) or ""
        brace_idx = src.find("{")
        if brace_idx != -1:
            return src[: brace_idx + 1].strip()
        # Fallback to first line
        return src.splitlines()[0].strip() if src else ""

    def _extract_block_type_and_name(
        self, node: ts.Node
    ) -> tuple[Optional[str], Optional[str]]:
        block_type: Optional[str] = None
        labels: List[str] = []

        for ch in node.children:
            if ch.type == "identifier" and block_type is None:
                block_type = (get_node_text(ch) or "").strip()
                continue
            if ch.type in ("string_lit", "identifier") and block_type is not None:
                lbl = self._strip_quotes(get_node_text(ch) or "")
                if lbl:
                    labels.append(lbl)
                continue
            if ch.type == "block_start":
                break
        # Name is derived purely from labels, not including the block type.
        # Examples:
        #   terraform { }              -> block_type="terraform", name=None
        #   backend "s3" { }           -> block_type="backend",   name="s3"
        #   variable "default_tags" { } -> block_type="variable",  name="default_tags"
        #   resource "aws_security_group" "sg" { } -> block_type="resource", name="aws_security_group.sg"
        if labels:
            name = ".".join(labels)
        else:
            name = None

        return block_type, name

    def _build_attribute_header(self, node: ts.Node) -> str:
        src = get_node_text(node) or ""
        eq_idx = src.find("=")
        if eq_idx != -1:
            return src[: eq_idx + 1].rstrip()
        return src.splitlines()[0].strip() if src else ""

    # ------------------------------------------------------------------ handlers

    def _handle_block(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> Optional[ParsedNode]:
        block_type, name = self._extract_block_type_and_name(node)
        header = self._build_block_header(node)

        blk = self._make_node(
            node,
            kind=NodeKind.BLOCK,
            name=name,
            header=header,
            subtype=(block_type or None),
        )

        # Recurse into the block body.
        body = next((c for c in node.children if c.type == "body"), None)
        if body is not None:
            for ch in body.children:
                blk.children.extend(self._process_node(ch, parent=blk))
        else:
            # Fallback: traverse all children except structural tokens.
            for ch in node.children:
                if ch.type in ("identifier", "block_start", "block_end"):
                    continue
                blk.children.extend(self._process_node(ch, parent=blk))

        return blk

    def _handle_attribute(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> Optional[ParsedNode]:
        name_node = next((c for c in node.children if c.type == "identifier"), None)
        name = (
            (get_node_text(name_node) or "").strip() if name_node is not None else None
        )
        header = self._build_attribute_header(node)

        prop = self._make_node(
            node,
            kind=NodeKind.PROPERTY,
            name=name,
            header=header,
        )

        # Attribute value is the first expression child.
        # We only keep nested BLOCK-like children (e.g. objects); simple
        # literals/variables do not become separate child nodes.
        expr = next((c for c in node.children if c.type == "expression"), None)
        if expr is not None:
            value_nodes = self._process_node(expr, parent=prop)
            block_children = [ch for ch in value_nodes if ch.kind == NodeKind.BLOCK]
            prop.children.extend(block_children)

        return prop

    def _handle_object(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> Optional[ParsedNode]:
        # Treat an object value as a block-like node with subtype "object".
        obj = self._make_node(
            node,
            kind=NodeKind.BLOCK,
            name=None,
            header="{",
            subtype="object",
        )
        for ch in node.children:
            if ch.type == "object_elem":
                obj.children.extend(self._handle_object_elem(ch, parent=obj))
        return obj

    def _handle_object_elem(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        # Children look like: expression (key), '=', expression (value)
        expr_children = [c for c in node.children if c.type == "expression"]
        key_expr = expr_children[0] if expr_children else None
        val_expr = expr_children[1] if len(expr_children) > 1 else None

        key_text = (
            (get_node_text(key_expr) or "").strip() if key_expr is not None else ""
        )
        key_text = self._strip_quotes(key_text)

        header_src = get_node_text(node) or ""
        eq_idx = header_src.find("=")
        if eq_idx != -1:
            header = header_src[: eq_idx + 1].rstrip()
        else:
            header = key_text

        prop = self._make_node(
            node,
            kind=NodeKind.PROPERTY,
            name=key_text or None,
            header=header,
            subtype="object_elem",
        )

        if val_expr is not None:
            value_nodes = self._process_node(val_expr, parent=prop)
            block_children = [ch for ch in value_nodes if ch.kind == NodeKind.BLOCK]
            prop.children.extend(block_children)

        return [prop]

    def _handle_comment(
        self, node: ts.Node, parent: Optional[ParsedNode]
    ) -> List[ParsedNode]:
        """
        Merge consecutive `comment` siblings into a single ParsedNode.

        This applies at any level (top-level, inside blocks, etc.):
        only the first comment in a run produces a node; following
        comments are folded into its body.
        """
        # If the previous sibling is also a comment, this node has already been
        # merged into the group starting at that sibling.
        prev = node.prev_sibling
        if prev is not None and prev.type == "comment":
            return []

        # Collect this comment and all immediately following comment siblings.
        comment_nodes: List[ts.Node] = []
        cur = node
        while cur is not None and cur.type == "comment":
            comment_nodes.append(cur)
            cur = cur.next_sibling

        # Combine text from all comment nodes, preserving line order.
        parts: List[str] = []
        for c in comment_nodes:
            txt = get_node_text(c) or ""
            parts.append(txt.rstrip())
        combined = "\n".join(parts)

        first = comment_nodes[0]
        last = comment_nodes[-1]

        # Build node anchored at the first comment, but extend its end range
        # to cover the whole merged comment block.
        comment_node = self._make_node(
            first,
            kind=NodeKind.COMMENT,
            body=combined,
        )
        comment_node.end_line = last.end_point[0] + 1
        comment_node.end_byte = last.end_byte

        return [comment_node]


class TerraformLanguageHelper(AbstractLanguageHelper):
    """
    Summary generator for Terraform/HCL nodes.

    Traverses arbitrarily nested BLOCK and PROPERTY nodes and emits a compact
    summary suitable for search and preview.
    """

    language = ProgrammingLanguage.TERRAFORM
    def get_node_summary(
        self,
        sym: Node,
        indent: int = 0,
        include_comments: bool = False,
        include_docs: bool = False,
        include_parents: bool = False,
        child_stack: Optional[List[List[Node]]] = None,
    ) -> str:
        # These knobs are currently ignored for Terraform summaries.
        _ = include_comments, include_docs, child_stack

        def _first_body_line(node: Node) -> str:
            text = (node.body or "").strip()
            if not text:
                return ""
            return text.splitlines()[0].strip()

        # When include_parents=True, build a compact summary from the root
        # ancestor down to the target node, showing only the relevant chain.
        if include_parents:
            # Build ancestor chain: [root, ..., parent, sym]
            chain: List[Node] = []
            current: Optional[Node] = sym
            while current is not None:
                chain.append(current)
                current = current.parent_ref
            chain.reverse()

            # Hide synthetic object blocks so nested maps read naturally:
            # variable -> default (attribute) -> Project (map key)
            visible: List[Node] = []
            for node in chain:
                if node.kind == NodeKind.BLOCK and (node.subtype or "") == "object":
                    continue
                visible.append(node)
            if not visible:
                visible = [sym]

            base_indent = indent
            lines: List[str] = []

            for i, node in enumerate(visible):
                node_indent = base_indent + i * 4
                line = _first_body_line(node)
                if not line:
                    continue
                lines.append(" " * node_indent + line)

                # For all but the last node, add an ellipsis at the next level
                # to indicate omitted siblings/children under this ancestor.
                if i < len(visible) - 1:
                    child_indent = base_indent + (i + 1) * 4
                    lines.append(" " * child_indent + "...")

            return "\n".join(lines)

        # Non-parent mode: node summaries should return the full node text,
        # without performing any custom recursion over children.
        text = (sym.body or "").rstrip()
        if not text:
            return ""
        # We intentionally ignore `indent` here, because Terraform nodes already
        # carry their original indentation in `body`.
        return text

    def get_common_syntax_words(self) -> set[str]:
        # Common Terraform/HCL keywords to de-emphasize in search.
        return {
            "terraform",
            "provider",
            "resource",
            "data",
            "module",
            "variable",
            "output",
            "locals",
            "backend",
            "moved",
            "import",
            "check",
            "dynamic",
            "content",
            "lifecycle",
            "provisioner",
            "connection",
            "for_each",
            "count",
            "depends_on",
            "precondition",
            "postcondition",
            "true",
            "false",
            "null",
        }
