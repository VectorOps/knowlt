import threading
import os
import duckdb
import pandas as pd
import math
import re
import asyncio
import queue
from concurrent.futures import Future
from typing import Optional, Dict, Any, List, Generic, TypeVar, Callable, Tuple, Set
import importlib.resources as pkg_resources
from pypika import (
    Table,
    Query,
    AliasedQuery,
    QmarkParameter,
    CustomFunction,
    functions,
    analytics,
    Order,
    Case,
)
from pypika.terms import LiteralValue, ValueWrapper
from pypika.terms import LiteralValue, ValueWrapper, Criterion, Term

from pydantic import BaseModel

from knowlt.logger import logger
from knowlt.tokenizers import code_tokenizer
from knowlt.models import (
    ModelId,
    Repo,
    Package,
    File,
    Node,
    ImportEdge,
    Modifier,
    Project,
    ProgrammingLanguage,
)
from knowlt.settings import ProjectSettings
from knowlt import data
from knowlt.helpers import generate_id
from knowlt.tokenizers import search_preprocessor_list
from .queue import BaseQueueWorker
from .sql import BaseSQLRepository, RawValue, apply_migrations

T = TypeVar("T", bound=BaseModel)

CREATE_MIGRATIONS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS __migrations__ (
        name TEXT PRIMARY KEY,
        applied_at TIMESTAMP DEFAULT NOW()
    );
"""
INSERT_MIGRATION_SQL = "INSERT INTO __migrations__(name, applied_at) VALUES (?, ?)"

MatchBM25Fn = CustomFunction("fts_main_nodes.match_bm25", ["id", "query"])
ArrayCosineSimilarityFn = CustomFunction("array_cosine_similarity", ["vec", "param"])
RegexpFullMatch = CustomFunction("regexp_full_match", ["s", "pat"])


class Glob(Criterion):
    def __init__(self, left: Term, right: Term):
        super().__init__()
        self.left = left
        self.right = right

    def get_sql(self, **kwargs) -> str:
        l = self.left.get_sql(**kwargs)
        r = self.right.get_sql(**kwargs)
        return f"{l} GLOB {r}"


# Helpers
def _row_to_dict(rel) -> list[dict[str, Any]]:
    """
    Convert a DuckDB relation to List[Dict] via a pandas DataFrame.
    Using DataFrame avoids the manual column-name handling and is faster.
    """
    df = rel.df()
    records = df.to_dict(orient="records")  # [] when df is empty
    for row in records:
        for k, v in row.items():
            # convert scalar NaN / pandas.NA to None, but skip sequences/arrays
            if isinstance(v, float) and math.isnan(v):
                row[k] = None
                continue
            try:
                is_na = pd.isna(v)  # may return array for sequences
            except Exception:
                is_na = False
            if isinstance(is_na, bool) and is_na:
                row[k] = None
    return records


# TODO: Rewrite and batch all operations
# TODO: Add search on body, but with lower weight
async def calc_bm25_fts_index(
    file_repo: data.AbstractFileRepository,
    s: ProjectSettings,
    nodes: List[Node],
) -> Dict[str, str]:
    """
    Batch-compute FTS needles for a list of nodes.
    - Batches file lookups.
    - Defaults language to TEXT (File.language not present in models).
    """
    if not nodes:
        return {}

    # Batch file lookups for file_path enrichment
    file_ids = {n.file_id for n in nodes if n.file_id}
    files_by_id: Dict[str, File] = {}
    if file_ids:
        files = await file_repo.get_by_ids(list(file_ids))
        files_by_id = {f.id: f for f in files}

    # Resolve boosts, default to body:1 if unset
    field_boosts = s.search.fts_field_boosts
    if not field_boosts:
        logger.warning(
            "FTS field boosts are not configured; defaulting to 'body' field only."
        )
        field_boosts = {"body": 1}

    out: Dict[str, str] = {}
    for node in nodes:
        language = ProgrammingLanguage.TEXT  # File.language not in models; keep TEXT
        file_path = None
        if node.file_id:
            f = files_by_id.get(node.file_id)
            if f:
                file_path = f.path

        node_data = node.model_dump()
        node_data["file_path"] = file_path
        if node.header:
            node_data["name"] = node.header

        fts_tokens: List[str] = []
        for field_name, boost in field_boosts.items():
            if field_name not in node_data:
                logger.warning(
                    f"Field '{field_name}' not found on Node model or derived values for FTS indexing."
                )
                continue
            if not boost:
                continue
            field_value = node_data[field_name]
            if field_value:
                processed_tokens = search_preprocessor_list(
                    s, language, str(field_value)
                )
                # Repeat tokens based on integer boost (approximate if float)
                try:
                    mult = int(round(float(boost)))
                except Exception:
                    mult = 1
                mult = max(1, mult)
                fts_tokens.extend(processed_tokens * mult)

        out[node.id] = " ".join(fts_tokens)

    return out


# DuckDB thread wrapper
class DuckDBThreadWrapper(BaseQueueWorker):
    """
    DuckDB is not great with threading, initialize connection and serialize all accesses to DuckDB
    to a separate worker thread. Without this random lockups in execute() were happening even
    with cursor.
    """

    def __init__(self, db_path: Optional[str] = None):
        super().__init__()
        self._db_path = db_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    def _initialize_worker(self) -> None:
        self._conn = duckdb.connect()

        self._conn.execute("INSTALL vss")
        self._conn.execute("LOAD vss")

        self._conn.execute("INSTALL fts")
        self._conn.execute("LOAD fts")

        # TODO: SQL injection?
        if self._db_path:
            self._conn.execute(f"ATTACH '{self._db_path}' as db")
            self._conn.execute("USE db")

        def execute_fn(sql: str, params: Optional[list[Any]] = None):
            assert self._conn is not None
            self._conn.execute(sql, params if params else [])

        def query_fn(
            sql: str, params: Optional[list[Any]] = None
        ) -> list[dict[str, Any]]:
            assert self._conn is not None
            rel = self._conn.execute(sql, params if params else [])
            return _row_to_dict(rel) if rel is not None else []

        apply_migrations(
            execute_fn,
            query_fn,
            "knowlt.migrations.duckdb",
            CREATE_MIGRATIONS_TABLE_SQL,
            INSERT_MIGRATION_SQL,
        )

    def _handle_item(self, item: Any) -> None:
        sql, params, fut = item
        if fut.set_running_or_notify_cancel():
            try:
                assert self._conn is not None
                fut.set_result(_row_to_dict(self._conn.execute(sql, params)))
            except Exception as exc:
                fut.set_exception(exc)

    def _cleanup(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    async def execute(self, sql, params=None):
        # Submit to worker thread and await without blocking the event loop
        fut: Future = Future()
        self._queue.put((sql, params, fut))
        return await asyncio.wrap_future(fut)


# Generic base repository
class _DuckDBBaseRepo(BaseSQLRepository[T]):
    table: str

    def __init__(self, conn: DuckDBThreadWrapper, data_repo: "DuckDBDataRepository"):
        self.conn = conn
        self._data_repo = data_repo
        self._table = Table(self.table)

    @property
    def data_repo(self) -> "DuckDBDataRepository":
        return self._data_repo

    def _get_query(self, q):
        parameter = QmarkParameter()
        sql = q.get_sql(parameter=parameter)
        return sql, parameter.get_parameters()

    async def _execute(self, q):
        sql, args = self._get_query(q)
        return await self.conn.execute(sql, args)

    # CRUD
    async def get_by_ids(self, item_ids: list[ModelId]) -> list[T]:
        if not item_ids:
            return []
        q = Query.from_(self._table).select("*").where(self._table.id.isin(item_ids))
        rows = await self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]
    async def create(self, items: list[T]) -> list[T]:
        if not items:
            return []

        data_list = [
            self._serialize_data(i.model_dump(exclude_none=True)) for i in items
        ]
        keys: list[str] = []

        for d in data_list:
            for k in d.keys():
                if k not in keys:
                    keys.append(k)

        q = Query.into(self._table).columns([self._table[k] for k in keys])
        for d in data_list:
            values = [d.get(k, RawValue("NULL")) for k in keys]
            q = q.insert(values)

        await self._execute(q)

        return items

    async def update(self, updates: List[Tuple[str, Dict[str, Any]]]) -> list[T]:
        out: List[T] = []
        ids: List[ModelId] = []

        for item_id, data in updates:
            ids.append(item_id)

            # TODO: Optimize to do bulk updates
            q = Query.update(self._table).where(self._table.id == item_id)
            for k, v in data.items():
                q = q.set(k, v)

            await self._execute(q)

        return await self.get_by_ids(ids)

    async def delete(self, item_ids: list[str]) -> bool:
        if not item_ids:
            return True
        q = Query.from_(self._table).where(self._table.id.isin(item_ids)).delete()
        await self._execute(q)
        return True


class DuckDBProjectRepo(_DuckDBBaseRepo[Project], data.AbstractProjectRepository):
    table = "projects"
    model = Project

    async def get_by_name(self, name: str) -> Optional[Project]:
        q = Query.from_(self._table).select("*").where(self._table.name == name)
        rows = await self._execute(q)
        return self.model(**self._deserialize_data(rows[0])) if rows else None


class DuckDBProjectRepoRepo(data.AbstractProjectRepoRepository):
    table = "project_repos"

    def __init__(self, conn: "DuckDBThreadWrapper"):
        self.conn = conn
        self._table = Table(self.table)

    def _get_query(self, q):
        parameter = QmarkParameter()
        sql = q.get_sql(parameter=parameter)
        return sql, parameter.get_parameters()

    async def _execute(self, q):
        sql, args = self._get_query(q)
        return await self.conn.execute(sql, args)

    async def get_repo_ids(self, project_id: ModelId) -> List[ModelId]:
        q = (
            Query.from_(self._table)
            .select(self._table.repo_id)
            .where(self._table.project_id == project_id)
        )
        rows = await self._execute(q)
        return [r["repo_id"] for r in rows]

    async def add_repo_id(self, project_id: ModelId, repo_id: ModelId) -> None:
        q = (
            Query.into(self._table)
            .columns(self._table.id, self._table.project_id, self._table.repo_id)
            .insert(generate_id(), project_id, repo_id)
        )
        try:
            await self._execute(q)
        except duckdb.ConstraintException:
            # This can happen if the repo is already associated with the project,
            # which is fine.
            pass
    async def delete_by_repo_id(self, repo_id: ModelId) -> None:
        """
        Remove all project-repo mapping rows for the given repo_id.
        """
        q = Query.from_(self._table).where(self._table.repo_id == repo_id).delete()
        await self._execute(q)


# ---------------------------------------------------------------------------
# concrete repositories
# ---------------------------------------------------------------------------
class DuckDBRepoRepo(_DuckDBBaseRepo[Repo], data.AbstractRepoRepository):
    table = "repos"
    model = Repo

    async def get_by_name(self, name: str) -> Optional[Repo]:
        """Get a repo by its name."""
        q = Query.from_(self._table).select("*").where(self._table.name == name)
        rows = await self._execute(q)
        return self.model(**self._deserialize_data(rows[0])) if rows else None

    async def get_by_path(self, root_path: str) -> Optional[Repo]:
        q = (
            Query.from_(self._table)
            .select("*")
            .where(self._table.root_path == root_path)
        )
        rows = await self._execute(q)
        return self.model(**self._deserialize_data(rows[0])) if rows else None


    async def delete(self, item_ids: list[ModelId]) -> bool:
        """
        Delete repos and cascade by repo_id into dependent tables.
        """
        if not item_ids:
            return True

        for rid in item_ids:
            # Order does not matter much; use explicit repo_id-based deletes.
            await self.data_repo.project_repo.delete_by_repo_id(rid)
            await self.data_repo.importedge.delete_by_repo_id(rid)
            await self.data_repo.node.delete_by_repo_id(rid)
            await self.data_repo.file.delete_by_repo_id(rid)
            await self.data_repo.package.delete_by_repo_id(rid)

        return await super().delete(item_ids)

    async def get_list(self, flt: data.RepoFilter) -> list[Repo]:
        q = Query.from_(self._table).select(self._table.star)
        if flt.project_id:
            project_repos = Table("project_repos")
            subq = (
                Query.from_(project_repos)
                .select(project_repos.repo_id)
                .where(project_repos.project_id == flt.project_id)
            )
            q = q.where(self._table.id.isin(subq))

        rows = await self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]


class DuckDBPackageRepo(_DuckDBBaseRepo[Package], data.AbstractPackageRepository):
    table = "packages"
    model = Package

    def __init__(self, conn: "DuckDBThreadWrapper", data_repo: "DuckDBDataRepository"):
        super().__init__(conn, data_repo)

    async def get_by_physical_paths(
        self, repo_id: ModelId, root_paths: List[str]
    ) -> List[Package]:
        q = (
            Query.from_(self._table)
            .select("*")
            .where(
                (self._table.repo_id == repo_id)
                & (self._table.physical_path.isin(root_paths))
            )
        )
        rows = await self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]

    async def get_by_virtual_paths(
        self, repo_id: ModelId, root_paths: List[str]
    ) -> List[Package]:
        q = (
            Query.from_(self._table)
            .select("*")
            .where(
                (self._table.repo_id == repo_id)
                & (self._table.virtual_path.isin(root_paths))
            )
        )
        rows = await self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]

    async def get_list(self, flt: data.PackageFilter) -> List[Package]:
        q = Query.from_(self._table).select("*")

        if flt.repo_ids:
            q = q.where(self._table.repo_id.isin(flt.repo_ids))

        rows = await self._execute(q)

        return [self.model(**self._deserialize_data(r)) for r in rows]

    async def delete_orphaned(self) -> None:
        files_tbl = Table("files")
        subq = (
            Query.from_(files_tbl)
            .select(files_tbl.package_id)
            .where(files_tbl.package_id.notnull())
        )

        q = Query.from_(self._table).where(self._table.id.notin(subq)).delete()
        await self._execute(q)

    async def delete_by_repo_id(self, repo_id: ModelId) -> None:
        q = Query.from_(self._table).where(self._table.repo_id == repo_id).delete()
        await self._execute(q)


class DuckDBFileRepo(_DuckDBBaseRepo[File], data.AbstractFileRepository):
    table = "files"
    model = File

    # ---------- internal helpers for search index ----------
    # TODO: Optimize and batch
    async def _reindex_file(self, file_id: ModelId, path: str) -> None:
        path_lc = path.lower()
        basename_lc = os.path.basename(path).lower()

        fs_tbl = Table("files_search")
        ft_tbl = Table("file_trigrams")

        # remove old index rows (idempotent)
        q = Query.from_(fs_tbl).where(fs_tbl.file_id == file_id).delete()
        await self._execute(q)
        q = Query.from_(ft_tbl).where(ft_tbl.file_id == file_id).delete()
        await self._execute(q)

        # insert new search rows
        q = (
            Query.into(fs_tbl)
            .columns(fs_tbl.file_id, fs_tbl.path_lc, fs_tbl.basename_lc)
            .insert(file_id, path_lc, basename_lc)
        )
        await self._execute(q)

        # insert distinct trigrams for the path (presence-only)
        if len(path_lc) >= 3:
            seen: Set[str] = set()
            values: list[tuple[str, str]] = []
            for i in range(len(path_lc) - 2):
                tri = path_lc[i : i + 3]
                if tri in seen:
                    continue
                seen.add(tri)
                values.append((file_id, tri))
            if values:
                q = Query.into(ft_tbl).columns(ft_tbl.file_id, ft_tbl.trigram)
                for fid, tri in values:
                    q = q.insert(fid, tri)
                await self._execute(q)

    async def _delete_index_for_ids(self, file_ids: List[ModelId]) -> None:
        if not file_ids:
            return
        fs_tbl = Table("files_search")
        ft_tbl = Table("file_trigrams")
        q = Query.from_(fs_tbl).where(fs_tbl.file_id.isin(file_ids)).delete()
        await self._execute(q)
        q = Query.from_(ft_tbl).where(ft_tbl.file_id.isin(file_ids)).delete()
        await self._execute(q)

    # ---------- CRUD overrides to keep search index in sync ----------
    async def create(self, items: List[File]) -> List[File]:
        out = await super().create(items)
        # TODO: Optimize
        for it in items:
            await self._reindex_file(it.id, it.path)
        return out

    async def update(self, updates: List[Tuple[ModelId, Dict[str, Any]]]) -> List[File]:
        out = await super().update(updates)
        # TODO: Optimize
        for file_id, data in updates:
            if "path" in data:
                await self._reindex_file(file_id, data["path"])
        return out

    async def delete(self, item_ids: List[ModelId]) -> bool:
        await self._delete_index_for_ids(item_ids)
        return await super().delete(item_ids)

    async def delete_by_repo_id(self, repo_id: ModelId) -> None:
        # collect file IDs for this repo to keep search index in sync
        q = (
            Query.from_(self._table)
            .select(self._table.id)
            .where(self._table.repo_id == repo_id)
        )
        rows = await self._execute(q)
        file_ids = [r["id"] for r in rows]
        if not file_ids:
            return
        await self._delete_index_for_ids(file_ids)
        await super().delete(file_ids)

    async def get_by_paths(self, repo_id: ModelId, paths: List[str]) -> Optional[File]:
        q = (
            Query.from_(self._table)
            .select("*")
            .where((self._table.path.isin(paths)) & (self._table.repo_id == repo_id))
        )
        rows = await self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]

    async def get_list(self, flt: data.FileFilter) -> List[File]:
        q = Query.from_(self._table).select("*")

        if flt.repo_ids:
            q = q.where(self._table.repo_id.isin(flt.repo_ids))
        if flt.package_id:
            q = q.where(self._table.package_id == flt.package_id)

        rows = await self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]

    async def glob_search(
        self,
        repo_ids: Optional[List[ModelId]],
        pattern: str,
        limit: Optional[int] = None,
    ) -> List[File]:
        """
        Search files using DuckDB's GLOB operator against files.path.
        Optionally restrict by repo_ids.
        """
        if not pattern:
            return []
        if repo_ids is not None and len(repo_ids) == 0:
            return []

        # Start query
        q = Query.from_(self._table).select(self._table.star)

        # Optional repo filter
        if repo_ids is not None:
            q = q.where(self._table.repo_id.isin(repo_ids))

        q = q.where(Glob(self._table.path, ValueWrapper(pattern))).orderby(
            self._table.path
        )

        if limit is not None:
            q = q.limit(limit)

        rows = await self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]

    async def filename_complete(
        self, needle: str, repo_ids: Optional[list[str]] = None, limit: int = 5
    ) -> list[File]:
        if not needle:
            return []
        if repo_ids is not None and len(repo_ids) == 0:
            return []

        needle_lc = needle.lower()
        # subsequence regex like Sublime Text: a.*b.*c, case-insensitive
        # Use full-match against ".*...*" to match anywhere
        subseq_pat = "(?i).*" + ".*".join(re.escape(ch) for ch in needle_lc) + ".*"

        # build trigram list for the query (for candidate narrowing / scoring)
        q_trigrams: list[str] = []
        if len(needle_lc) >= 3:
            q_trigrams = [needle_lc[i : i + 3] for i in range(len(needle_lc) - 2)]

        files_tbl = self._table
        fs_tbl = Table("files_search")
        ft_tbl = Table("file_trigrams")

        # Optional CTE with trigram hits when we have any trigrams
        q = Query
        tri_hits_ref = None
        if q_trigrams:
            tri_hits = (
                Query.from_(ft_tbl)
                .select(ft_tbl.file_id, functions.Count("*").as_("tri_hits"))
                .where(ft_tbl.trigram.isin(q_trigrams))
                .groupby(ft_tbl.file_id)
            )
            q = q.with_(tri_hits, "trihits")
            tri_hits_ref = AliasedQuery("trihits")

        # Base FROM and optional join to trigram hits
        q = q.from_(fs_tbl)
        if tri_hits_ref is not None:
            q = q.left_join(tri_hits_ref).on(fs_tbl.file_id == tri_hits_ref.file_id)
        q = q.join(files_tbl).on(files_tbl.id == fs_tbl.file_id)

        # Build score components
        subseq = RegexpFullMatch(fs_tbl.path_lc, ValueWrapper(subseq_pat))
        base_subseq = RegexpFullMatch(fs_tbl.basename_lc, ValueWrapper(subseq_pat))
        tri_hits_col = (
            tri_hits_ref.tri_hits if tri_hits_ref is not None else LiteralValue(0)
        )
        tri_hits_val = functions.Coalesce(tri_hits_col, 0)

        score = (
            Case().when(base_subseq, 20).else_(0)
            + Case().when(subseq, 10).else_(0)
            + tri_hits_val * 2
            - (functions.Length(fs_tbl.path_lc) / LiteralValue(100))
        )

        # Candidate filter: require subsequence match; trigram hits only affect scoring
        cond = subseq
        if repo_ids:
            cond = cond & (files_tbl.repo_id.isin(repo_ids))

        q = (
            q.select(files_tbl.star, score.as_("score"))
            .where(cond)
            .orderby(score, order=Order.desc)
            .orderby(files_tbl.path)
            .limit(limit)
        )

        rows = await self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]


class DuckDBNodeRepo(_DuckDBBaseRepo[Node], data.AbstractNodeRepository):
    table = "nodes"
    model = Node

    def __init__(
        self,
        conn: "DuckDBThreadWrapper",
        data_repo: "DuckDBDataRepository",
    ):
        super().__init__(conn, data_repo)
        # use the data repository to access dependencies
        self.file_repo = data_repo.file
        self._settings = data_repo.settings

    async def create(self, items: List[Node]) -> List[Node]:
        if not items:
            return []
        data_list: list[dict[str, Any]] = []
        # Batch compute FTS needles for all items
        fts_map = await calc_bm25_fts_index(
            file_repo=self.file_repo,
            s=self._settings,
            nodes=items,
        )
        for item in items:
            d = self._serialize_data(item.model_dump(exclude_none=True))
            d["fts_needle"] = fts_map.get(item.id, "")
            data_list.append(d)

        keys: list[str] = []
        for d in data_list:
            for k in d.keys():
                if k not in keys:
                    keys.append(k)

        q = Query.into(self._table).columns([self._table[k] for k in keys])
        for d in data_list:
            values = [d.get(k) for k in keys]
            q = q.insert(values)

        await self._execute(q)

        return items

    async def update(self, updates: list[Tuple[str, Dict[str, Any]]]) -> list[Node]:
        if not updates:
            return []
        ids = [item_id for item_id, _ in updates]
        current_list = await self.get_by_ids(ids)
        current_by_id = {n.id: n for n in current_list}

        # Build updated node objects in memory
        updated_nodes_list: list[Node] = []
        for item_id, changes in updates:
            current = current_by_id.get(item_id)
            if not current:
                continue
            updated_nodes_list.append(current.model_copy(update=changes))

        # Batch compute FTS needles
        fts_map = await calc_bm25_fts_index(
            file_repo=self.file_repo,
            s=self._settings,
            nodes=updated_nodes_list,
        )

        # Apply per-row updates (minimal change), but single fetch at the end
        for item_id, changes in updates:
            if item_id not in current_by_id:
                continue
            serialized = self._serialize_data(changes)
            serialized["fts_needle"] = fts_map.get(item_id, "")
            q = Query.update(self._table).where(self._table.id == item_id)
            for k, v in serialized.items():
                q = q.set(k, v)
            await self._execute(q)

        # Fetch all updated records in one go
        return await self.get_by_ids(ids)

    async def search(self, query: data.NodeSearchQuery) -> List[Node]:
        q = Query.from_(self._table)

        # Candidates
        candidates = Query.from_(self._table).select(
            self._table.id,
            self._table.repo_id,
            self._table.embedding_code_vec,
            self._table.kind,
            self._table.search_boost,
        )

        if query.repo_ids:
            q = q.where(self._table.repo_id.isin(query.repo_ids))

        if query.kind:
            q = q.where(self._table.kind == query.kind)

        if query.visibility:
            q = q.where(self._table.visibility == query.visibility)

        # Determine which search dimensions are provided
        has_fts = bool(query.needle)
        has_embedding = bool(query.embedding_query)

        q = q.with_(candidates, "candidates")
        aliased_candidates = AliasedQuery("candidates")

        # unified CTE / query construction
        if has_embedding:
            rank_code_scores = Query.from_(aliased_candidates).select(
                aliased_candidates.id,
                ArrayCosineSimilarityFn(
                    aliased_candidates.embedding_code_vec,
                    functions.Cast(ValueWrapper(query.embedding_query), "FLOAT[1024]"),
                ).as_("dist"),
            )

            aliased = AliasedQuery("rank_code_scores")

            rank_code = (
                Query.from_(aliased)
                .select(
                    aliased.id,
                    analytics.RowNumber()
                    .orderby(aliased.dist, order=Order.desc)
                    .as_("code_rank"),
                )
                .where(
                    aliased.dist >= self._settings.search.embedding_similarity_threshold
                )
            )

            q = q.with_(rank_code_scores, "rank_code_scores").with_(
                rank_code, "rank_code"
            )

        if has_fts:
            assert query.needle
            needle = code_tokenizer(query.needle)
            rank_fts_scores = Query.from_(aliased_candidates).select(
                aliased_candidates.id,
                MatchBM25Fn(aliased_candidates.id, needle).as_("score"),
            )

            aliased = AliasedQuery("rank_fts_scores")

            rank_fts = Query.from_(aliased).select(
                aliased.id,
                analytics.RowNumber()
                .orderby(aliased.score, order=Order.desc)
                .as_("fts_rank"),
            )
            bm25_threshold = self._settings.search.bm25_score_threshold
            if bm25_threshold is not None:
                rank_fts = rank_fts.where(aliased.score >= bm25_threshold)
            else:
                rank_fts = rank_fts.where(aliased.score.notnull())

            q = q.with_(rank_fts_scores, "rank_fts_scores").with_(rank_fts, "rank_fts")

        has_ranking = has_fts or has_embedding
        if has_ranking:
            union_parts = []

            if has_embedding:
                aliased = AliasedQuery("rank_code")

                union_parts.append(
                    Query.from_(aliased).select(
                        aliased.id,
                        (
                            LiteralValue(self._settings.search.rrf_code_weight)
                            / (
                                LiteralValue(self._settings.search.rrf_k)
                                + aliased.code_rank
                            )
                        ).as_("score"),
                    )
                )

            if has_fts:
                aliased = AliasedQuery("rank_fts")

                union_parts.append(
                    Query.from_(aliased).select(
                        aliased.id,
                        (
                            LiteralValue(self._settings.search.rrf_fts_weight)
                            / (
                                LiteralValue(self._settings.search.rrf_k)
                                + aliased.fts_rank
                            )
                        ).as_("score"),
                    )
                )

            rrf_scores = union_parts[0]
            for p in union_parts[1:]:
                rrf_scores = rrf_scores.union_all(p)  # type: ignore[assignment]

            aliased = AliasedQuery("rrf_scores")

            rrf_final = (
                Query.from_(aliased)
                .select(aliased.id, functions.Sum(aliased.score).as_("score"))
                .groupby(aliased.id)
            )

            aliased_scores = AliasedQuery("rrf_final")
            aliased_fused = AliasedQuery("fused")

            score_col = aliased_scores.score * aliased_candidates.search_boost

            if query.boost_repo_id and query.repo_boost_factor != 1.0:
                repo_boost_case = (
                    Case()
                    .when(
                        aliased_candidates.repo_id == query.boost_repo_id,
                        query.repo_boost_factor,
                    )
                    .else_(1.0)
                )
                score_col = score_col * repo_boost_case

            fused = (
                Query.from_(aliased_candidates)
                .join(aliased_scores)
                .on(aliased_candidates.id == aliased_scores.id)
                .select(aliased_scores.id, score_col.as_("rrf_score"))
            )

            q = (
                q.with_(rrf_scores, "rrf_scores")
                .with_(rrf_final, "rrf_final")
                .with_(fused, "fused")
                .join(aliased_fused)
                .on(self._table.id == aliased_fused.id)
                .select(self._table.star, aliased_fused.rrf_score)
                .orderby(aliased_fused.rrf_score, order=Order.desc)
                .orderby(self._table.name)
            )
        else:
            q = (
                q.select("*")
                .join(aliased_candidates)
                .on(self._table.id == aliased_candidates.id)
                .orderby(self._table.name)
            )

        raw_limit = query.limit if query.limit is not None else 20
        offset = query.offset if query.offset is not None else 0

        fetch_limit = raw_limit * 2
        q = q.limit(fetch_limit).offset(offset)

        rows = await self._execute(q)
        results = [self.model(**self._deserialize_data(r)) for r in rows]
        # Enforce pagination limit on primary results. We fetch extra for ranking headroom
        # but must return at most 'raw_limit' items to satisfy API semantics and tests.
        return results[:raw_limit]

    async def delete_by_file_ids(self, file_ids: List[ModelId]) -> None:
        q = Query.from_(self._table).where(self._table.file_id.isin(file_ids)).delete()
        await self._execute(q)

    async def delete_by_repo_id(self, repo_id: ModelId) -> None:
        q = Query.from_(self._table).where(self._table.repo_id == repo_id).delete()
        await self._execute(q)

    async def get_list(self, flt: data.NodeFilter) -> List[Node]:
        q = Query.from_(self._table).select("*")

        if flt.parent_ids:
            q = q.where(self._table.parent_node_id.isin(flt.parent_ids))
        if flt.repo_ids:
            q = q.where(self._table.repo_id.isin(flt.repo_ids))
        if flt.file_ids:
            q = q.where(self._table.file_id.isin(flt.file_ids))
        if flt.package_ids:
            q = q.where(self._table.package_id.isin(flt.package_ids))
        if flt.visibility:
            q = q.where(self._table.visibility == flt.visibility)
        if flt.has_embedding is True:
            q = q.where(self._table.embedding_code_vec.notnull())
        elif flt.has_embedding is False:
            q = q.where(self._table.embedding_code_vec.isnull())
        if flt.top_level_only:
            q = q.where(self._table.parent_node_id.isnull())

        if flt.offset:
            q = q.offset(flt.offset)
        if flt.limit:
            q = q.limit(flt.limit)

        rows = await self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]


class DuckDBImportEdgeRepo(
    _DuckDBBaseRepo[ImportEdge], data.AbstractImportEdgeRepository
):
    table = "import_edges"
    model = ImportEdge

    async def get_list(self, flt: data.ImportEdgeFilter) -> List[ImportEdge]:
        q = Query.from_(self._table).select("*")

        if flt.source_package_ids:
            q = q.where(self._table.from_package_id.isin(flt.source_package_ids))
        if flt.source_file_ids:
            q = q.where(self._table.from_file_id.isin(flt.source_file_ids))
        if flt.repo_ids:
            q = q.where(self._table.repo_id.isin(flt.repo_ids))

        rows = await self._execute(q)
        return [self.model(**self._deserialize_data(r)) for r in rows]

    async def delete_by_repo_id(self, repo_id: ModelId) -> None:
        q = Query.from_(self._table).where(self._table.repo_id == repo_id).delete()
        await self._execute(q)


# Data-repository
class DuckDBDataRepository(data.AbstractDataRepository):
    """
    Main entry point.  Automatically applies pending SQL migrations on first
    construction.
    """

    def __init__(self, settings: ProjectSettings, db_path: Optional[str] = None):
        """
        Parameters
        ----------
        db_path : str | None
            Filesystem path to the DuckDB database.
            • None  ->  use in-memory database.
        """
        if db_path is None:
            db_path = ":memory:"
        # ensure parent dir exists for file-based DBs
        if db_path != ":memory__":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self._conn = DuckDBThreadWrapper(db_path)
        self._conn.start()
        self._settings = settings

        # build repositories
        self._project_repo = DuckDBProjectRepo(self._conn, self)
        self._prj_repo_repo = DuckDBProjectRepoRepo(self._conn)
        self._file_repo = DuckDBFileRepo(self._conn, self)
        self._package_repo = DuckDBPackageRepo(self._conn, self)
        self._repo_repo = DuckDBRepoRepo(self._conn, self)
        self._node_repo = DuckDBNodeRepo(self._conn, self)
        self._edge_repo = DuckDBImportEdgeRepo(self._conn, self)

    def close(self):
        self._conn.close()

    @property
    def settings(self) -> ProjectSettings:
        return self._settings

    @property
    def project(self) -> data.AbstractProjectRepository:
        return self._project_repo

    @property
    def project_repo(self) -> data.AbstractProjectRepoRepository:
        return self._prj_repo_repo

    @property
    def repo(self) -> data.AbstractRepoRepository:
        return self._repo_repo

    @property
    def package(self) -> data.AbstractPackageRepository:
        return self._package_repo

    @property
    def file(self) -> data.AbstractFileRepository:
        return self._file_repo

    @property
    def node(self) -> data.AbstractNodeRepository:
        return self._node_repo

    @property
    def importedge(self) -> data.AbstractImportEdgeRepository:
        return self._edge_repo

    async def refresh_indexes(self) -> None:
        try:
            await self._conn.execute("PRAGMA drop_fts_index('nodes');")
            await self._conn.execute(
                "PRAGMA create_fts_index('nodes', 'id', 'fts_needle');"
            )
        except Exception as ex:
            logger.debug("Failed to refresh DuckDB FTS index", ex=ex)
