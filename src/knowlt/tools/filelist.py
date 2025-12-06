import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from knowlt.project import ProjectManager, VIRTUAL_PATH_PREFIX
from .base import BaseTool


class ListFilesReq(BaseModel):
    """Request model for listing files."""

    pattern: str = Field(
        description="An fnmatch-style glob pattern to match against file paths."
    )
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of files to return. Defaults to the system-wide setting.",
    )


class FileListItem(BaseModel):
    """Represents a file in the project for listing."""

    path: str = Field(description="The path of the file relative to the project root.")


class ListFilesTool(BaseTool):
    """Tool to list files in the project matching a glob pattern."""

    tool_name = "list_files"
    tool_input = ListFilesReq

    async def execute(
        self,
        pm: ProjectManager,
        req: Any,
    ) -> str:
        req_obj = self.parse_input(req)
        """
        Return files whose path matches the supplied glob pattern.

        If `pattern` is None or empty, return an empty list. The matching is
        done fnmatch-style.
        """
        await pm.maybe_refresh()

        file_repo = pm.data.file

        if not req_obj.pattern:
            return self.encode_output(pm, [])

        limit = req_obj.limit or pm.settings.tools.file_list_limit
        matches = await file_repo.glob_search(pm.repo_ids, req_obj.pattern, limit)

        items = [
            FileListItem(path=vpath)
            for fm in matches
            for vpath in [pm.construct_virtual_path(fm.repo_id, fm.path)]
        ]
        return self.encode_output(pm, items)

    async def get_openai_schema(self) -> dict:
        """Return the OpenAI schema for this tool."""
        return {
            "name": self.tool_name,
            "description": (
                "Return project files whose path matches the supplied glob pattern. "
                f"File paths for repos other than the default repo will be prefixed with '{VIRTUAL_PATH_PREFIX}/<repo_name>'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "An fnmatch-style glob pattern (e.g. '**/*.py')."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of files to return.",
                    },
                },
                "required": ["pattern"],
            },
        }
