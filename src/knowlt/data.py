from typing import Optional, Dict, Any, List, TypeVar, Generic, Tuple
from abc import ABC, abstractmethod
from knowlt.models import (
    ModelId,
    Project,
    ProjectRepo,
    Repo,
    Package,
    File,
    Node,
    ImportEdge,
    NodeKind,
    Visibility,
    Vector,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING


T = TypeVar("T")


class AbstractCRUDRepository(ABC, Generic[T]):
    @abstractmethod
    async def get_by_ids(self, item_ids: List[ModelId]) -> List[T]:
        pass

    @abstractmethod
    async def create(self, items: List[T]) -> List[T]:
        pass

    @abstractmethod
    async def update(self, updates: List[Tuple[ModelId, Dict[str, Any]]]) -> List[T]:
        pass

    @abstractmethod
    async def delete(self, item_ids: List[ModelId]) -> bool:
        pass


class AbstractProjectRepository(AbstractCRUDRepository[Project]):
    @abstractmethod
    async def get_by_name(self, name) -> Optional[Project]:
        pass


class AbstractProjectRepoRepository(ABC):
    @abstractmethod
    async def get_repo_ids(self, project_id) -> List[ModelId]:
        pass

    @abstractmethod
    async def add_repo_id(self, project_id, repo_id):
        pass
    @abstractmethod
    async def delete_by_repo_id(self, repo_id: ModelId) -> None:
        """Delete all project-repo mappings for the given repo_id."""
        pass


@dataclass
class RepoFilter:
    project_id: Optional[str] = None


class AbstractRepoRepository(AbstractCRUDRepository[Repo]):
    @abstractmethod
    async def get_list(self, flt: RepoFilter) -> List[Repo]:
        pass

    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[Repo]:
        """Get a repo by its name."""
        pass

    @abstractmethod
    async def get_by_path(self, root_path: str) -> Optional[Repo]:
        """Get a repo by its root path."""
        pass


@dataclass
class PackageFilter:
    repo_ids: Optional[List[str]] = None


class AbstractPackageRepository(AbstractCRUDRepository[Package]):
    @abstractmethod
    async def get_list(self, flt: PackageFilter) -> List[Package]:
        pass

    @abstractmethod
    async def get_by_physical_paths(
        self, repo_id: ModelId, root_paths: List[str]
    ) -> List[Package]:
        """Get a repo by its root path."""
        pass

    @abstractmethod
    async def get_by_virtual_paths(
        self, repo_id: ModelId, root_paths: List[str]
    ) -> List[Package]:
        """Get a repo by its root path."""
        pass

    @abstractmethod
    async def delete_orphaned(self) -> None:
        pass

    @abstractmethod
    async def delete_by_repo_id(self, repo_id: ModelId) -> None:
        pass


@dataclass
class FileFilter:
    repo_ids: Optional[List[ModelId]] = None
    package_id: Optional[ModelId] = None


class AbstractFileRepository(AbstractCRUDRepository[File]):
    @abstractmethod
    async def get_by_paths(self, repo_id: ModelId, paths: List[str]) -> List[File]:
        """Get a file by its project-relative path."""
        pass

    @abstractmethod
    async def get_list(self, flt: FileFilter) -> List[File]:
        pass

    @abstractmethod
    async def filename_complete(
        self, needle: str, repo_ids: Optional[List[str]] = None, limit: int = 5
    ) -> List[File]:
        """
        Fuzzy-complete file paths by name using a Sublime Text–like subsequence match.

        Parameters
        ----------
        needle:
            The search string to match (subsequence).
        repo_ids:
            Optional list of repo IDs to restrict the search to. If None, search
            across all repos.
        limit:
            Maximum number of File objects to return.

        Returns up to `limit` File objects ordered by a relevance score.
        """
        pass

    @abstractmethod
    async def glob_search(
        self,
        repo_ids: Optional[List[ModelId]],
        patterns: List[str],
    ) -> List[File]:
        """
        Return files whose project-relative path matches any of the provided
        glob patterns, optionally restricted to the specified repos.
        """

    @abstractmethod
    async def delete_by_repo_id(self, repo_id: ModelId) -> None:
        pass


# Nodes
@dataclass
class NodeFilter:
    parent_ids: Optional[List[ModelId]] = None
    repo_ids: Optional[List[ModelId]] = None
    file_ids: Optional[List[ModelId]] = None
    package_ids: Optional[List[ModelId]] = None
    visibility: Optional[Visibility] = None
    has_embedding: Optional[bool] = None
    top_level_only: Optional[bool] = False
    limit: Optional[int] = None
    offset: Optional[int] = None


@dataclass
class NodeSearchQuery:
    # Repo filter
    repo_ids: Optional[List[ModelId]] = None
    # Filter by symbol kind
    kind: Optional[NodeKind] = None
    # Filter by symbol visiblity
    visibility: Optional[Visibility] = None
    # Full-text search on symbol documentation or comment
    needle: Optional[str] = None
    # Embedding similarity search
    embedding_query: Optional[Vector] = None
    # ID of a repo whose symbols should be boosted in search results
    boost_repo_id: Optional[ModelId] = None
    # Boost factor to apply. Only used if boost_repo_id is also provided.
    # Values > 1.0 will boost, < 1.0 will penalize. Default is 1.0 (no change).
    repo_boost_factor: float = 1.0
    # Number of records to return. If None is passed, no limit will be applied.
    limit: Optional[int] = None
    # Zero-based offset
    offset: Optional[int] = None


class AbstractNodeRepository(AbstractCRUDRepository[Node]):
    @abstractmethod
    async def delete_by_file_ids(self, file_id: List[ModelId]) -> None:
        pass

    @abstractmethod
    async def delete_by_repo_id(self, repo_id: ModelId) -> None:
        pass

    @abstractmethod
    async def get_list(self, flt: NodeFilter) -> List[Node]:
        pass

    @abstractmethod
    async def search(self, query: NodeSearchQuery) -> List[Node]:
        pass


@dataclass
class ImportEdgeFilter:
    repo_ids: Optional[List[ModelId]] = None
    source_package_ids: Optional[List[ModelId]] = None
    source_file_ids: Optional[List[ModelId]] = None


class AbstractImportEdgeRepository(AbstractCRUDRepository[ImportEdge]):
    @abstractmethod
    async def get_list(self, flt: ImportEdgeFilter) -> List[ImportEdge]:
        pass

    @abstractmethod
    async def delete_by_repo_id(self, repo_id: ModelId) -> None:
        pass


class AbstractDataRepository(ABC):
    @abstractmethod
    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def project(self) -> AbstractProjectRepository:
        pass

    @property
    @abstractmethod
    def project_repo(self) -> AbstractProjectRepoRepository:
        pass

    @property
    @abstractmethod
    def repo(self) -> AbstractRepoRepository:
        pass

    @property
    @abstractmethod
    def package(self) -> AbstractPackageRepository:
        pass

    @property
    @abstractmethod
    def file(self) -> AbstractFileRepository:
        pass

    @property
    @abstractmethod
    def node(self) -> AbstractNodeRepository:
        pass

    @property
    @abstractmethod
    def importedge(self) -> AbstractImportEdgeRepository:
        pass

    @abstractmethod
    def refresh_indexes(self) -> None:
        """Refresh search indexes (noop in back-ends that don’t need it)."""
        pass
