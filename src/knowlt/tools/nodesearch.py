from typing import Sequence, Optional, Any
import json

from pydantic import BaseModel, Field

from knowlt.data import NodeSearchQuery
from knowlt.models import NodeKind, Visibility
from typing import TYPE_CHECKING
from knowlt.consts import VIRTUAL_PATH_PREFIX

if TYPE_CHECKING:
    from knowlt.project import ProjectManager
from .base import BaseTool
from knowlt.summary import SummaryMode
from knowlt.data_helpers import populate_packages_for_files, post_process_search_results
from knowlt.parsers import (
    CodeParserRegistry,
    AbstractCodeParser,
    AbstractLanguageHelper,
)
from knowlt.models import File
from knowlt.settings import ToolOutput


class NodeSearchReq(BaseModel):
    global_search: bool = Field(
        default=True, description="Search through all repos in the project."
    )
    visibility: Optional[Visibility | str] = Field(
        default="all",
        description=(
            "Restrict by visibility modifier (`public`, `protected`, `private`) or use `all` to include "
            "every symbol. Defaults to `all`."
        ),
    )
    query: Optional[str] = Field(
        default=None,
        description=(
            "Natural-language search string evaluated against docstrings, comments, and code with both "
            "full-text and vector search. Use when you don’t know the exact name."
        ),
    )
    limit: Optional[int] = Field(
        default=10, description="Maximum number of results to return."
    )
    offset: Optional[int] = Field(
        default=0, description="Number of results to skip. Used for pagination."
    )
    summary_mode: SummaryMode | str = Field(
        default=SummaryMode.Documentation,
        description="Amount of source code to include with each match",
    )


class NodeSearchResult(BaseModel):
    visibility: Optional[str] = Field(
        default=None,
        description="The visibility of the symbol (e.g., 'public', 'private').",
    )
    file_path: Optional[str] = Field(
        default=None, description="The path to the file containing the symbol."
    )
    body: Optional[str] = Field(
        default=None,
        description="The summary or body of the symbol, depending on the summary_mode.",
    )
    start_line: Optional[int] = Field(
        default=None,
        description="1-based starting line number of the symbol in the file.",
    )
    end_line: Optional[int] = Field(
        default=None,
        description="1-based ending line number of the symbol in the file (inclusive).",
    )


class NodeSearchTool(BaseTool):
    tool_name = "search_project"
    tool_input = NodeSearchReq
    default_output = ToolOutput.STRUCTURED_TEXT

    async def execute(
        self,
        req: Any,
    ) -> str:
        req = self.parse_input(req)
        await self.pm.maybe_refresh()

        # visibility
        vis = None
        if isinstance(req.visibility, Visibility):
            vis = req.visibility
        elif isinstance(req.visibility, str):
            if req.visibility.lower() == "all":
                vis = None
            else:
                try:
                    vis = Visibility(req.visibility)
                except ValueError:
                    valid_vis = [v.value for v in Visibility] + ["all"]
                    raise ValueError(
                        f"Invalid visibility '{req.visibility}'. Valid values are: {valid_vis}"
                    )

        # summary_mode
        summary_mode = req.summary_mode
        if isinstance(summary_mode, str):
            try:
                summary_mode = SummaryMode(summary_mode)
            except ValueError:
                valid_modes = [m.value for m in SummaryMode]
                raise ValueError(
                    f"Invalid summary_mode '{req.summary_mode}'. Valid values are: {valid_modes}"
                )

        # transform free-text query -> embedding vector (if requested)
        embedding_vec = None
        if req.query:
            embedding_vec = await self.pm.compute_embedding(req.query)

        if req.global_search:
            repo_ids = self.pm.repo_ids
        else:
            repo_ids = [self.pm.default_repo.id]

        final_limit = req.limit or 25

        query = NodeSearchQuery(
            repo_ids=repo_ids,
            visibility=vis,
            needle=req.query,
            embedding_query=embedding_vec,
            boost_repo_id=self.pm.default_repo.id,
            repo_boost_factor=self.pm.settings.search.default_repo_boost,
            limit=final_limit,
            offset=req.offset,
        )

        nodes = await self.pm.data.node.search(query)
        # Use the node repository for post-processing so that descendant/parent
        # expansion works correctly.
        nodes = await post_process_search_results(self.pm.data.node, nodes, final_limit)

        file_repo = self.pm.data.file
        package_repo = self.pm.data.package

        # Batch load files for all nodes, then batch load packages referenced by those files.
        file_ids = [s.file_id for s in nodes if getattr(s, "file_id", None)]
        files: list[File] = (
            await file_repo.get_by_ids(list(set(file_ids))) if file_ids else []
        )
        file_by_id = {f.id: f for f in files}
        if files:
            await populate_packages_for_files(package_repo, files)

        results: list[NodeSearchResult] = []
        for s in nodes:
            helper: Optional[AbstractLanguageHelper] = None
            fm: Optional[File] = None
            file_path = None
            if s.file_id and s.file_id in file_by_id:
                fm = file_by_id[s.file_id]
                file_path = self.pm.construct_virtual_path(s.repo_id, fm.path)
                # get language from package if available
                if fm.package and getattr(fm.package, "language", None):
                    helper = CodeParserRegistry.get_helper(fm.package.language)

            sym_body: Optional[str] = None
            if summary_mode != SummaryMode.Skip:
                if summary_mode == SummaryMode.Source:
                    sym_body = s.body
                elif helper is not None:
                    include_docs = summary_mode == SummaryMode.Documentation
                    include_comments = summary_mode == SummaryMode.Documentation
                    sym_body = helper.get_node_summary(
                        s,
                        include_comments=include_comments,
                        include_docs=include_docs,
                        include_parents=True,
                    )  # type: ignore[call-arg]

            results.append(
                NodeSearchResult(
                    # visibility is not tracked on Node model; omit or set None
                    visibility=None,
                    file_path=file_path,
                    body=sym_body,
                    start_line=s.start_line or None,
                    end_line=s.end_line or None,
                )
            )
        return self.encode_output(results)

    async def get_openai_schema(self) -> dict:
        visibility_enum = [v.value for v in Visibility] + ["all"]

        return {
            "name": self.tool_name,
            "description": (
                "Search for code blocks (functions, classes, variables, etc.) in the current repository. "
                f"All supplied filters are combined with logical **AND**. If the file path contains {VIRTUAL_PATH_PREFIX} "
                "then it is not part of the current repository and should be only considered as an external "
                "dependency."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary_mode": {
                        "type": "string",
                        "enum": [m.value for m in SummaryMode],
                        "description": "Amount of source code to include with each match",
                        "default": SummaryMode.Documentation.value,
                    },
                    "visibility": {
                        "type": "string",
                        "enum": visibility_enum,
                        "description": (
                            "Restrict by visibility modifier (`public`, `protected`, `private`) "
                            "or use `all` to include every symbol. Defaults to `all`."
                        ),
                        "default": "all",
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural-language search string evaluated against docstrings, comments, and code "
                            "with both full-text and vector search. Use when you don’t know the exact name. "
                            "Search does not support any special operators such as logical conditions, double "
                            "quotes and similar."
                        ),
                    },
                },
            },
        }
