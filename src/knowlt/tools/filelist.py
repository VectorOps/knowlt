import fnmatch
import json
from typing import Sequence, Optional, Any

from pydantic import BaseModel, Field

from knowlt.project import ProjectManager, VIRTUAL_PATH_PREFIX
from knowlt.models import ProgrammingLanguage
from knowlt.data import FileFilter
from .base import BaseTool


class ListFilesReq(BaseModel):
    """Request model for listing files."""

    patterns: Sequence[str] = Field(
        description="List of fnmatch-style glob patterns to match against file paths."
    )


class FileListItem(BaseModel):
    """Represents a file in the project for listing."""

    path: str = Field(description="The path of the file relative to the project root.")


class ListFilesTool(BaseTool):
    """Tool to list files in the project matching glob patterns."""

    tool_name = "list_files"
    tool_input = ListFilesReq

    async def execute(
        self,
        pm: ProjectManager,
        req: Any,
    ) -> str:
        req_obj = self.parse_input(req)
        """
        Return files whose path matches any of the supplied glob patterns.

        If `patterns` is None or empty, return an empty list. The matching is
        done fnmatch-style.
        """
        await pm.maybe_refresh()

        file_repo = pm.data.file

        # TODO: Better search
        all_files = await file_repo.get_list(FileFilter(repo_ids=pm.repo_ids))

        pats = list(req_obj.patterns) if req_obj.patterns else []
        if not pats:
            return self.encode_output(pm, [])

        def _matches(path: str) -> bool:
            return any(fnmatch.fnmatch(path, pat) for pat in pats)

        items = [
            FileListItem(path=vpath, language=fm.language)
            for fm in all_files
            if _matches(vpath := pm.construct_virtual_path(fm.repo_id, fm.path))
        ]
        return self.encode_output(pm, items)

    async def get_openai_schema(self) -> dict:
        """Return the OpenAI schema for this tool."""
        return {
            "name": self.tool_name,
            "description": (
                "Return all project files whose path matches at least one "
                "of the supplied glob patterns. File paths for repos other than the "
                f"default repo will be prefixed with '{VIRTUAL_PATH_PREFIX}/<repo_name>'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of fnmatch-style glob patterns "
                            "(e.g. ['**/*.py', 'src/*.ts'])."
                        ),
                    }
                },
                "required": [],
            },
        }
