import json
import inspect
import re
import datetime, decimal
from typing import Any, Dict, List, Type
from enum import Enum
from pydantic import BaseModel
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from knowlt.project import ProjectManager
from knowlt.settings import ProjectSettings, ToolOutput
from . import helpers


FENCE_START_RE = re.compile(r"(?m)^`+")


class BaseTool(ABC):
    tool_name: str
    tool_input: Type[BaseModel]
    default_output: ToolOutput = ToolOutput.JSON

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if not inspect.isabstract(cls):
            ToolRegistry.register_tool(cls)

    def __init__(self, pm: "ProjectManager"):
        self.pm = pm

    @abstractmethod
    async def execute(self, req: Any) -> str:
        """
        Execute the tool for the provided request.
        Request can be:
          - dict-like (preferred),
          - a Pydantic BaseModel,
          - or a JSON string.
        Implementations should parse req via `self.parse_input(req)`.
        """
        pass

    @abstractmethod
    async def get_openai_schema(self) -> dict:
        """
        Returns OpenAI function calling schema.
        """
        pass

    def parse_input(self, req: Any) -> BaseModel:
        """
        Parse an arbitrary request payload into this tool's `tool_input` model.
        - If req is a string: attempt json.loads().
        - If req is a BaseModel: coerce into the tool_input model.
        - Otherwise: validate with Pydantic model_validate(req).
        """
        model_cls: Type[BaseModel] = getattr(self, "tool_input", None)
        if model_cls is None:
            raise TypeError(f"{type(self).__name__} missing tool_input model.")

        data: Any
        if isinstance(req, str):
            try:
                data = json.loads(req)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON string for {type(self).__name__}: {e}"
                ) from e
        elif isinstance(req, BaseModel):
            # Convert any BaseModel into plain dict first (use aliases and drop None)
            data = req.model_dump(by_alias=True, exclude_none=True)
        else:
            data = req

        return model_cls.model_validate(data)

    # convenience instance wrapper
    def to_python(self, obj: Any) -> Any:
        return helpers.convert_to_python(obj)

    def get_output_format(
        self,
        pm: "ProjectManager",
    ) -> ToolOutput:
        """
        Resolve the effective output format for this tool:
        - settings.tools.outputs[tool_name] if available
        - otherwise tool's default_output
        """
        settings = pm.settings

        encoding = None
        if settings is not None:
            try:
                encoding = settings.tools.outputs.get(self.tool_name)
            except Exception:
                encoding = None

        return encoding or self.default_output

    def encode_output(self, obj: Any) -> str:
        """
        Convert a tool's execute() return value into a string to send as tool output.
        Uses settings.tools.outputs[tool_name] if provided; otherwise falls back to the tool's
        default_output (usually JSON).
        """
        # Resolve output encoding
        encoding = self.get_output_format(self.pm)

        # Encode by selected format
        if encoding == ToolOutput.STRUCTURED_TEXT:
            return self.format_structured_text(obj)

        # Default JSON
        try:
            if isinstance(obj, BaseModel):
                payload = obj.model_dump(by_alias=True, exclude_none=True)
            else:
                payload = helpers.convert_to_python(obj)
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return str(obj)

    def format_structured_text(
        self,
        obj: Any,
        max_scalar_len: int | None = None,
        record_sep: str = "\n---\n",
    ) -> str:
        """
        Structured text: serialize to dict or list[dict] and print key:value pairs.
        Between records, add an object separator line.
        """

        converted = helpers.convert_to_python(obj)

        # Normalize to list[dict]
        records: list[dict[str, Any]] = []
        if isinstance(converted, dict):
            records = [converted]
        elif isinstance(converted, list):
            if all(isinstance(it, dict) for it in converted):
                records = converted  # list of mappings
            else:
                # list of scalars/mixed
                records = [{"idx": i, "value": it} for i, it in enumerate(converted)]
        else:
            records = [{"value": converted}]

        # Sort keys for deterministic output
        def _sorted_items(d: dict[str, Any]):
            try:
                return sorted(d.items(), key=lambda kv: kv[0])
            except Exception:
                return d.items()

        chunks: list[str] = []

        for rec in records:
            lines: list[str] = []
            for k, v in _sorted_items(rec):
                s = helpers.stringify(v)
                if (
                    max_scalar_len is not None
                    and "\n" not in s
                    and len(s) > max_scalar_len
                ):
                    s = s[:max_scalar_len] + "…"
                if "\n" in s:
                    # choose a fence that won’t collide with content
                    max_ticks = 3
                    # Only increase fence size if a code fence is at the start of a line
                    runs = [len(m.group(0)) for m in FENCE_START_RE.finditer(s)]
                    if any(r >= 3 for r in runs):
                        max_ticks = max(runs) + 1
                    fence = "`" * max(max_ticks, 3)
                    lines.append(f"{k}:\n{fence}text\n{s}\n{fence}")
                else:
                    lines.append(f"{k}: {s}")
            chunks.append("\n".join(lines))

        return record_sep.join(chunks)


class ToolRegistry:
    _tools: Dict[str, BaseTool] = {}

    @classmethod
    def register_tool(cls, tool_cls: Type["BaseTool"]) -> None:
        name = getattr(tool_cls, "tool_name", None)
        if not name:
            raise ValueError(f"{tool_cls.__name__} missing `tool_name`")
        if name not in cls._tools:  # keep singletons
            cls._tools[name] = tool_cls

    @classmethod
    def get_tools(cls) -> Dict[str, "BaseTool"]:
        return cls._tools
