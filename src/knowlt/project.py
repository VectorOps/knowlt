from knowlt.consts import VIRTUAL_PATH_PREFIX
from pathlib import Path
import os
import os.path as op
import datetime
from typing import Any, Optional, Type, Dict, Tuple, List
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from knowlt.models import Project, ProjectRepo, Vector, Repo
from knowlt.data import AbstractDataRepository, NodeSearchQuery
from knowlt.logger import logger
from knowlt.helpers import parse_gitignore, compute_file_hash, generate_id
from knowlt.settings import ProjectSettings
from knowlt.embeddings import EmbeddingWorker
from knowlt.tools import ToolRegistry, BaseTool




@dataclass
class ScanResult:
    """Result object returned by `scan_project_directory`."""

    repo: Repo
    files_added: list[str] = field(default_factory=list)
    files_updated: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)


class ProjectComponent(ABC):
    """Lifecycle-hook interface for project-level components."""

    component_name: str | None = None

    def __init__(self, pm: "ProjectManager"):
        self.pm = pm

    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def refresh(self, scan_result: "ScanResult") -> None: ...

    @abstractmethod
    def destroy(self) -> None: ...


class ProjectCache:
    """
    Mutable project-wide cache for expensive/invariant information
    that code parsers may want to re-use (ex: go.mod content).
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()


class ProjectManager:
    _component_registry: Dict[str, Type[ProjectComponent]] = {}

    @classmethod
    def register_component(
        cls, comp_cls: Type[ProjectComponent], name: str | None = None
    ) -> None:
        if comp_cls.component_name is None:
            raise ValueError(
                f"Cannot register component {comp_cls.__name__} without a `component_name`."
            )
        cls._component_registry[comp_cls.component_name] = comp_cls

    def __init__(
        self,
        settings: ProjectSettings,
        data: AbstractDataRepository,
        embeddings: Optional[EmbeddingWorker] = None,
    ):
        self.settings = settings
        self.data = data
        self.embeddings = embeddings
        self._last_refresh_time: Optional[datetime.datetime] = None
        self._components: dict[str, ProjectComponent] = {}
        # In-memory caches for repos to allow sync helpers to function without async data calls
        self._repos_by_id: dict[str, Repo] = {}
        self._repos_by_name: dict[str, Repo] = {}
        # Tools
        self._tools: List[BaseTool] = []

        if not self.settings.project_name:
            raise ValueError(f"settings.project_name is required.")
        if not self.settings.repo_name:
            raise ValueError(f"settings.repo_name is required.")
    def _cache_repo(self, repo: Repo) -> Repo:
        """
        Store *repo* in all local repo caches and return it.
        """
        self._repos_by_id[repo.id] = repo
        self._repos_by_name[repo.name] = repo
        return repo

    def _uncache_repo(self, repo: Repo) -> None:
        """
        Remove *repo* from all local repo caches.
        """
        self._repos_by_id.pop(repo.id, None)
        existing = self._repos_by_name.get(repo.name)
        if existing is repo:
            self._repos_by_name.pop(repo.name, None)

    async def _get_repo_cached_by_name(self, name: str) -> Optional[Repo]:
        """
        Return repo by *name* from cache when available, otherwise load
        it from the data repository and cache the result.
        """
        repo = self._repos_by_name.get(name)
        if repo is not None:
            return repo

        repo = await self.data.repo.get_by_name(name)
        if repo is not None:
            self._cache_repo(repo)
        return repo
    @classmethod
    async def create(
        self,
        settings: ProjectSettings,
        data: AbstractDataRepository,
        embeddings: Optional[EmbeddingWorker] = None,
    ):
        p = ProjectManager(settings=settings, data=data, embeddings=embeddings)
        await p._init_project()
        return p

    async def _init_project(self):
        # Initialize project
        project = await self.data.project.get_by_name(self.settings.project_name)
        if project is None:
            project = Project(
                id=generate_id(),
                name=self.settings.project_name,
            )
            created = await self.data.project.create([project])
            project = created[0]

        self.project = project
        self.repo_ids = await self.data.project_repo.get_repo_ids(project.id)
        # Preload repo cache so sync helpers can resolve all known repos.
        if self.repo_ids:
            repos = await self.data.repo.get_by_ids(self.repo_ids)
            for repo in repos:
                self._cache_repo(repo)

        # Initialize default repo
        self.default_repo = await self.add_repo_path(
            self.settings.repo_name, self.settings.repo_path
        )

        # Initialize components
        for name, comp_cls in self._component_registry.items():
            try:
                inst = comp_cls(self)
                self._components[name] = inst
                inst.initialize()
            except Exception as exc:
                logger.error("Component failed to initialize", name=name, exc=exc)

        # Initialize tools
        disabled_tools = self.settings.tools.disabled
        self._tools = {
            name: cls(self)
            for name, cls in ToolRegistry.get_tools().items()
            if name not in disabled_tools
        }

    # Simple repo management
    async def add_repo_path(self, name, path: Optional[str] = None):
        repo = await self._get_repo_cached_by_name(name)
        if repo is None:
            if path is None:
                raise ValueError(f"Path is required for a new repo {name}")

            # TODO: Validate if path is valid

            repo = Repo(
                id=generate_id(),
                name=name,
                root_path=path,
            )
            created = await self.data.repo.create([repo])
            repo = created[0]
        elif path and repo.root_path != path:
            logger.warning(
                "Repo path does not match, updating...",
                repo_name=name,
                old_path=repo.root_path,
                new_path=path,
            )
            updated = await self.data.repo.update(
                [
                    (
                        repo.id,
                        {
                            "root_path": path,
                        },
                    )
                ]
            )
            repo = updated[0]

        assert repo is not None
        if repo.id not in self.repo_ids:
            self.repo_ids.append(repo.id)
            await self.data.project_repo.add_repo_id(self.project.id, repo.id)

        # Ensure caches stay in sync
        self._cache_repo(repo)

        return repo

    async def remove_repo(self, repo_id):
        if repo_id in self.repo_ids:
            self.repo_ids.remove(repo_id)
        # Remove from caches
        repo = self._repos_by_id.get(repo_id)
        if repo is not None:
            self._uncache_repo(repo)

        # Data-layer removal may not be supported by the abstract interface
        remover = getattr(
            getattr(self.data, "project_repo", None), "remove_repo_id", None
        )
        if callable(remover):
            try:
                await remover(self.project.id, repo_id)
            except Exception as exc:
                logger.warning(
                    "Failed to remove repo association in data layer",
                    repo_id=repo_id,
                    exc=exc,
                )
        else:
            logger.debug(
                "Data repository does not support removing repo associations",
                repo_id=repo_id,
            )

    # Virtual path helpers
    def construct_virtual_path(self, repo_id: str, path: str) -> str:
        if self.settings.paths.enable_project_paths:
            repo = self._repos_by_id.get(repo_id)
            if repo is None:
                raise ValueError("Repository was not found.")
            return op.join(repo.name, path)

        if repo_id == self.default_repo.id:
            return path

        repo = self._repos_by_id.get(repo_id)
        if repo is None:
            raise ValueError("Repository was not found.")

        return op.join(VIRTUAL_PATH_PREFIX, repo.name, path)

    def deconstruct_virtual_path(self, path) -> Optional[Tuple[Repo, str]]:
        if self.settings.paths.enable_project_paths:
            parts = path.split(os.sep, 1)
            repo_name = parts[0]
            repo = self._repos_by_name.get(repo_name)

            if repo and repo.id in self.repo_ids:
                relative_path = parts[1] if len(parts) > 1 else ""
                return (repo, relative_path)

            return None

        if not path.startswith(VIRTUAL_PATH_PREFIX):
            return (self.default_repo, path)

        path = path[len(VIRTUAL_PATH_PREFIX) + 1 :]
        parts = path.split(os.sep, 1)
        if not parts or not parts[0]:
            return None
        repo = self._repos_by_name.get(parts[0])
        if repo is None:
            return None

        relative_path = parts[1] if len(parts) > 1 else ""
        return (repo, relative_path)

    def get_physical_path(self, virtual_path: str) -> Optional[str]:
        deconstructed = self.deconstruct_virtual_path(virtual_path)
        if deconstructed is None:
            return None

        repo, relative_path = deconstructed
        if repo.root_path is None:
            return None

        return op.join(repo.root_path, relative_path)

    # Single repo refresh helper
    async def refresh(
        self,
        repo=None,
        paths: Optional[list[str]] = None,
        progress_callback: Optional[callable] = None,
    ):
        if repo is None:
            repo = self.default_repo

        from knowlt import scanner

        scan_result = await scanner.scan_repo(
            self, repo, paths=paths, progress_callback=progress_callback
        )
        await self.refresh_components(scan_result)
        self._last_refresh_time = datetime.datetime.now(datetime.timezone.utc)

    async def refresh_all(self) -> None:
        """Refresh all repositories in the project."""
        repos_to_refresh = await self.data.repo.get_by_ids(self.repo_ids)
        for repo in repos_to_refresh:
            await self.refresh(repo)

    async def maybe_refresh(self) -> None:
        if not self.settings.refresh.enabled:
            return

        cooldown_minutes = self.settings.refresh.cooldown_minutes

        if cooldown_minutes > 0:
            if self._last_refresh_time:
                now = datetime.datetime.now(datetime.timezone.utc)
                delta = now - self._last_refresh_time
                if delta < datetime.timedelta(minutes=cooldown_minutes):
                    logger.debug(
                        "Skipping auto-refresh due to cooldown.",
                        since_last_refresh=delta,
                        cooldown_minutes=cooldown_minutes,
                    )
                    return

        if self.settings.refresh.refresh_all_repos:
            logger.debug("Auto-refreshing all associated repositories...")
            await self.refresh_all()
        else:
            logger.debug("Auto-refreshing primary repository...")
            await self.refresh()  # Just refreshes default repo

    async def refresh_components(self, scan_result: ScanResult):
        for name, comp in self._components.items():
            try:
                # TODO: pass repo to refresh
                comp.refresh(scan_result)
            except Exception as exc:
                logger.error("Component failed to refresh", name=name, exc=exc)

    # Tools
    def get_tool(self, name) -> "BaseTool":
        return self._tools.get(name)

    def get_enabled_tools(self) -> List["BaseTool"]:
        return list(self._tools.values())

    # Pluggable lifecycle components
    def add_component(self, name: str, component: ProjectComponent) -> None:
        """Register *component* under *name* and immediately call initialise()."""
        if name in self._components:
            logger.warning("Component already registered – ignored.", name=name)
            return
        self._components[name] = component
        try:
            component.initialize()
        except Exception as exc:
            logger.error("Component failed to initialize", name=name, exc=exc)

    def get_component(self, name: str) -> Optional[ProjectComponent]:
        """Return registered component (or None when unknown)."""
        return self._components.get(name)

    # Embedding helper
    async def compute_embedding(
        self,
        text: str,
    ) -> Optional[Vector]:
        if self.embeddings is None:
            return None

        return await self.embeddings.get_embedding(text)

    # teardown helper
    async def destroy(self, *, timeout: float | None = None) -> None:
        """
        Release every resource held by this Project instance.
        """
        for name, comp in list(self._components.items()):
            try:
                comp.destroy()
            except Exception as exc:
                logger.error("Component failed to destroy", name=name, exc=exc)
        self._components.clear()

        if self.embeddings is not None:
            try:
                self.embeddings.destroy(timeout=timeout)
            except Exception as exc:
                logger.error("Failed to destroy EmbeddingWorker", exc=exc)
            self.embeddings = None

        try:
            self.data.close()
        except Exception as exc:
            logger.error("Failed to close data repository", exc=exc)
