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

        If the pattern includes a virtual-path or project-path repo prefix,
        it is decoded to a specific repo and the glob is applied only to that
        repo using a repo-relative pattern.
        """
        await self.pm.maybe_refresh()

        if not req_obj.pattern:
            return self.encode_output([])

        pattern = req_obj.pattern
        search_pattern = pattern
        repo_ids = self.pm.repo_ids

        # Always attempt to decode via ProjectManager.deconstruct_virtual_path.
        # We only scope to a specific repo when the pattern explicitly includes
        # a repo prefix (virtual-path or project-path).
        decoded = self.pm.deconstruct_virtual_path(pattern)
        if decoded is not None:
            repo, relative_pattern = decoded
            explicit = False

            # Explicit virtual-path prefix, e.g. ".virtual-path/repo_name/..."
            if pattern.startswith(VIRTUAL_PATH_PREFIX + "/"):
                explicit = True
            else:
                # Explicit project-path prefix, e.g. "repo_name/..."
                repo_prefix = repo.name + "/"
                if pattern.startswith(repo_prefix):
                    explicit = True

            if explicit:
                if repo.id not in self.pm.repo_ids:
                    return self.encode_output([])
                repo_ids = [repo.id]
                search_pattern = relative_pattern or "**/*"

        file_repo = self.pm.data.file

        limit = self.pm.settings.tools.file_list_limit
        # Boost files from the project's default repository when listing.
        boost_repo_id = self.pm.default_repo.id
        boost_factor = self.pm.settings.search.default_repo_boost

        matches = await file_repo.glob_search(
            repo_ids,
            search_pattern,
            limit,
            boost_repo_id=boost_repo_id,
            repo_boost_factor=boost_factor,
        )

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
                            "An fnmatch-style glob pattern (e.g. '**/*.py'). "
                            "You may optionally prefix the pattern with a virtual-path "
                            f"('{VIRTUAL_PATH_PREFIX}/<repo_name>/...') or project-path "
                            "('<repo_name>/...') repo name to limit the search to that repo."
                        ),
                    },
                },
                "required": ["pattern"],
            },
        }
