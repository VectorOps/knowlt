from pydantic import BaseModel, Field, model_validator
from enum import Enum
from typing import List, Optional, Dict, Iterable

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ProgrammingLanguage(str, Enum):
    PYTHON = "python"
    GO = "go"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    MARKDOWN = "markdown"
    TEXT = "text"


class NodeKind(str, Enum):
    # Common language-agnostic kinds (used by settings.node_kind_boosts)
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    PROPERTY = "property"
    # Existing kinds
    LITERAL = "literal"
    BLOCK = "block"  # Block with child nodes
    COMMENT = "comment"  # Comment
    CUSTOM = "custom"  # Language specific node, defined by subtype


class Visibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    PACKAGE = "package"
    INTERNAL = "internal"


class Modifier(str, Enum):
    STATIC = "static"
    ASYNC = "async"
    ABSTRACT = "abstract"
    FINAL = "final"
    OVERRIDE = "override"
    GENERIC = "generic"


class EdgeType(str, Enum):
    IMPORTS = "imports"
    EXPORTS = "exports"


# Generic types
ModelId = str
Vector = List[float]


# Core data containers
class Project(BaseModel):
    id: ModelId
    name: str


class ProjectRepo(BaseModel):
    id: ModelId
    project_id: ModelId
    repo_id: ModelId


class Repo(BaseModel):
    id: ModelId
    name: str
    root_path: str = ""
    remote_url: Optional[str] = None
    default_branch: str = "main"
    description: Optional[str] = None


class Package(BaseModel):
    id: ModelId
    repo_id: ModelId

    name: Optional[str] = None
    language: Optional[ProgrammingLanguage] = None
    virtual_path: Optional[str] = None  # import path such as "mypkg/subpkg"
    physical_path: Optional[str] = None  # directory or file relative to repo root
    description: Optional[str] = None

    # Runtime links
    imports: List["ImportEdge"] = Field(default_factory=list, exclude=True, repr=False)
    imported_by: List["ImportEdge"] = Field(
        default_factory=list, exclude=True, repr=False
    )


class File(BaseModel):
    id: ModelId
    repo_id: ModelId
    package_id: Optional[ModelId] = None

    path: str  # project relative path
    file_hash: Optional[str] = None
    last_updated: Optional[int] = None  # POSIX mtime in nanoseconds (st_mtime_ns)

    # Runtime links
    package: Optional[Package] = Field(default=None, exclude=True, repr=False)
    symbols: List["Node"] = Field(default_factory=list, exclude=True, repr=False)


class Node(BaseModel):
    id: ModelId
    repo_id: ModelId
    file_id: Optional[ModelId] = None
    package_id: Optional[ModelId] = None
    parent_node_id: Optional[ModelId] = None

    name: Optional[str] = None
    body: str
    header: Optional[str] = None
    kind: Optional[NodeKind] = None
    subtype: Optional[str] = None
    docstring: Optional[str] = None
    comment: Optional[str] = None
    exported: Optional[bool] = None

    start_line: int = 0
    end_line: int = 0
    start_byte: int = 0
    end_byte: int = 0

    # Embedding
    embedding_code_vec: Optional[Vector] = None
    embedding_model: Optional[str] = None

    # Search boost (persisted)
    search_boost: float = 1.0

    # Runtime links
    file_ref: Optional[File] = Field(default=None, exclude=True, repr=False)
    parent_ref: Optional["Node"] = Field(default=None, exclude=True, repr=False)
    children: List["Node"] = Field(default_factory=list, exclude=True, repr=False)


class ImportEdge(BaseModel):
    id: ModelId
    repo_id: ModelId
    from_package_id: ModelId
    from_file_id: ModelId
    to_package_physical_path: Optional[str]  # physical path for local packages
    to_package_virtual_path: (
        str  # textual path like "fmt"; may not map to a package_id if external
    )
    to_package_id: Optional[ModelId] = (
        None  # filled when the imported package exists in the same repo
    )
    alias: Optional[str] = None  # import alias if any
    dot: bool = False  # true for dot-imports (import . "pkg")
    external: bool
    raw: str

    # Runtime links
    from_package_ref: Optional[Package] = Field(default=None, exclude=True, repr=False)
    to_package_ref: Optional[Package] = Field(default=None, exclude=True, repr=False)
