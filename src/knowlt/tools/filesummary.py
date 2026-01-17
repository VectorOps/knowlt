from typing import Sequence, Any
from .base import BaseTool
from pydantic import BaseModel
import json

from knowlt.summary import FileSummary, SummaryMode, build_file_summary

from knowlt.settings import ToolOutput
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlt.project import ProjectManager
from knowlt.models import Visibility


class SummarizeFilesReq(BaseModel):
    """Request model for the SummarizeFilesTool."""

    paths: Sequence[str]
    summary_mode: SummaryMode | str = SummaryMode.Documentation


class SummarizeFilesTool(BaseTool):
    """Tool to generate summaries for a list of files."""

    tool_name = "summarize_files"
    tool_input = SummarizeFilesReq
    default_output = ToolOutput.STRUCTURED_TEXT  # new

    async def execute(
        self,
        req: Any,
    ) -> str:
        """Generate summaries for the requested files."""
        req_obj = self.parse_input(req)

        await self.pm.maybe_refresh()

        summary_mode = req_obj.summary_mode
        if isinstance(summary_mode, str):
            summary_mode = SummaryMode(summary_mode)

        if summary_mode is SummaryMode.Source:
            raise ValueError(
                "summary_mode 'source' is not supported. To read the whole file, use a file reading tool."
            )

        summaries: list[FileSummary] = []
        for path in req_obj.paths:
            deconstructed = self.pm.deconstruct_virtual_path(path)
            if not deconstructed:
                continue

            repo, rel_path = deconstructed
            fs = await build_file_summary(
                self.pm, repo, rel_path, summary_mode=summary_mode
            )
            if fs:
                fs.path = path
                summaries.append(fs)

        # Ensure default output is STRUCTURED_TEXT if not explicitly set
        outputs = self.pm.settings.tools.outputs
        if self.tool_name not in outputs:
            outputs[self.tool_name] = ToolOutput.STRUCTURED_TEXT

        return self.encode_output(summaries)

    async def get_openai_schema(self) -> dict:
        """Return the OpenAI schema for the tool."""
        summary_enum = [m.value for m in SummaryMode]

        summary_enum = [
            m.value
            for m in SummaryMode
            if m not in (SummaryMode.Source, SummaryMode.Skip)
        ]

        return {
            "name": self.tool_name,
            "description": (
                "Return a partial summary for each supplied file, consisting of its "
                "import statements and top-level symbol definitions. Use this tool "
                "to get an overview of interesting files. Prefer the default "
                "`definition` mode, but request `documentation` if more detail (like docstrings) is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relative paths of the files to be summarized.",
                    },
                    "summary_mode": {
                        "type": "string",
                        "enum": summary_enum,
                        "description": "Amount of source code to include with each match.",
                        "default": SummaryMode.Definition.value,
                    },
                },
                "required": ["paths"],
            },
        }
