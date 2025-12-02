import os
from typing import Sequence, Optional, List
from enum import Enum
from pydantic import BaseModel, Field

from knowlt.logger import logger
from knowlt.parsers import CodeParserRegistry, AbstractLanguageHelper
from knowlt.project import ProjectManager
from knowlt.models import ImportEdge, Visibility, Node, NodeKind, Repo
from knowlt.data import ImportEdgeFilter, NodeFilter
from knowlt.data_helpers import resolve_node_hierarchy


class SummaryMode(str, Enum):
    Skip = "skip"
    Definition = "definition"
    Documentation = "documentation"
    Source = "source"


class FileSummary(BaseModel):
    """Represents a generated summary for a single file and the mode used to produce it."""

    path: str = Field(..., description="The project-relative path of the file.")
    content: str = Field(..., description="The generated summary content of the file.")
    summary_mode: SummaryMode = Field(
        ...,
        description="Summary granularity used to produce the content. One of: 'definition', 'documentation', or 'source'.",
    )


async def build_file_summary(
    pm: ProjectManager,
    repo: Repo,
    rel_path: str,
    summary_mode: SummaryMode = SummaryMode.Definition,
) -> Optional[FileSummary]:
    """
    Return a FileSummary for *rel_path* or ``None`` when the file is
    unknown to the repository.
    """
    if summary_mode is SummaryMode.Skip:
        return None

    files = await pm.data.file.get_by_paths(repo.id, [rel_path])
    fm = files[0] if files else None
    if not fm:
        logger.debug("File not found in repository – skipped.", path=rel_path)
        return None
    # Retrieve package if available
    pkg = None
    if fm.package_id:
        pkgs = await pm.data.package.get_by_ids([fm.package_id])
        pkg = pkgs[0] if pkgs else None

    # If there is no language or package, we can't generate a summary,
    # so we return the full file content instead.
    if (summary_mode is SummaryMode.Source) or (pkg is None) or (pkg.language is None):
        abs_path = os.path.join(repo.root_path, rel_path)
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                return FileSummary(
                    path=rel_path, content=f.read(), summary_mode=summary_mode
                )
        except OSError as exc:
            logger.error("Unable to read file", path=abs_path, exc=exc)
            return None
    include_docs = summary_mode is SummaryMode.Documentation

    node_repo = pm.data.node
    helper: AbstractLanguageHelper | None = (
        CodeParserRegistry.get_helper(pkg.language) if pkg.language else None
    )

    if not helper:
        return None

    symbols = await node_repo.get_list(NodeFilter(file_ids=[fm.id]))

    resolve_node_hierarchy(symbols)

    # Top-level nodes are those without a recorded parent
    top_level = [s for s in symbols if not s.parent_node_id]
    # Preserve source order
    top_level.sort(
        key=lambda s: (getattr(s, "start_line", 0), getattr(s, "start_byte", 0))
    )

    if not include_docs:
        top_level = [s for s in top_level if s.kind != NodeKind.COMMENT]

    # TODO: Figure out if we return anything or return nothing
    sections = [
        helper.get_node_summary(
            s,
            include_docs=include_docs,
        )
        for s in top_level
    ]

    return FileSummary(
        path=rel_path,
        content="\n".join(s for s in sections if s),
        summary_mode=summary_mode,
    )
