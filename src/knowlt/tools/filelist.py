import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from typing import TYPE_CHECKING
from knowlt.consts import VIRTUAL_PATH_PREFIX
if TYPE_CHECKING:
    from knowlt.project import ProjectManager
from knowlt.settings import ToolSettings
from .base import BaseTool


class ListFilesReq(BaseModel):
    """Request model for listing files."""

    pattern: str = Field(
        description="An fnmatch-style glob pattern to match against file paths."
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
        req: Any,
    ) -> str:
        req_obj = self.parse_input(req)
        """
        Return files whose path matches the supplied glob pattern.

        If `pattern` is None or empty, return an empty list. The matching is
        done fnmatch-style.
        """
        await self.pm.maybe_refresh()

        file_repo = self.pm.data.file

        if not req_obj.pattern:
            return self.encode_output([])

        limit = self.pm.settings.tools.file_list_limit
        matches = await file_repo.glob_search(self.pm.repo_ids, req_obj.pattern, limit)

        items = [
            FileListItem(path=vpath)
            for fm in matches
            for vpath in [self.pm.construct_virtual_path(fm.repo_id, fm.path)]
        ]
        return self.encode_output(items)

    async def get_openai_schema(self) -> dict:
        """Return the OpenAI schema for this tool."""
        limit = self.pm.settings.tools.file_list_limit
        return {
            "name": self.tool_name,
            "description": (
                "Return a list of project files whose path matches the supplied glob pattern. "
                f"This tool will return up to {limit} files. "
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
                },
                "required": ["pattern"],
            },
        }
