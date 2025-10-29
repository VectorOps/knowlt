from collections import defaultdict
from pathlib import Path
from typing import Optional, Any, Type, Callable
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
from enum import Enum
import pathspec
from dataclasses import dataclass, field
import time
import asyncio

from knowlt.project import ScanResult, ProjectManager, ProjectCache
from knowlt.helpers import compute_file_hash, generate_id, parse_gitignore
from knowlt.logger import logger
from knowlt.models import (
    Repo,
    File,
    Package,
    Node,
    ImportEdge,
    NodeKind,
)
from knowlt.data import NodeFilter, ImportEdgeFilter, PackageFilter, FileFilter
from knowlt.parsers import (
    CodeParserRegistry,
    ParsedFile,
    ParsedNode,
    AbstractCodeParser,
)
from knowlt.embedding_helpers import (
    schedule_missing_embeddings,
    schedule_outdated_embeddings,
    schedule_symbol_embedding,
)


@dataclass
class EmbeddingTask:
    symbol_id: str
    text: str


class ScanProgress(BaseModel):
    repo_id: str
    total_files: int
    processed_files: int
    files_added: int
    files_updated: int
    files_deleted: int
    elapsed_seconds: float


class ParsingState:
    def __init__(self) -> None:
        self.pending_import_edges: list[ImportEdge] = []
        self.pending_embeddings: list[EmbeddingTask] = []


class ProcessFileStatus(Enum):
    SKIPPED = "skipped"
    BARE_FILE = "bare_file"
    PARSED_FILE = "parsed_file"
    ERROR = "error"


@dataclass
class ProcessFileResult:
    status: ProcessFileStatus
    duration: float
    suffix: str
    rel_path: Optional[str] = None
    parsed_file: Optional[ParsedFile] = None
    existing_meta: Optional[File] = None
    file_hash: Optional[str] = None
    mod_time: Optional[float] = None
    exception: Optional[Exception] = None


@dataclass
class ProcessFileParams:
    pm: ProjectManager
    repo: Repo
    path: Path
    root: Path
    gitignore: "pathspec.PathSpec"
    cache: ProjectCache
    parser_map: dict[str, Type[AbstractCodeParser]]
    existing_meta: Optional[File] = None


def _get_parser_map(pm: ProjectManager) -> dict[str, Type[AbstractCodeParser]]:
    parser_map = {}
    for parser_cls in CodeParserRegistry.get_parsers():
        for ext in parser_cls.extensions:
            parser_map[ext] = parser_cls

        lang_name = parser_cls.language.value
        if lang_settings := pm.settings.languages.get(lang_name):
            for ext in lang_settings.extra_extensions:
                if not ext.startswith("."):
                    ext = f".{ext}"

                if ext in parser_map and parser_map[ext] != parser_cls:
                    logger.warning(
                        "Overriding parser for extension from settings",
                        extension=ext,
                        language=lang_name,
                        original_parser=parser_map[ext].__name__,
                        new_parser=parser_cls.__name__,
                    )
                parser_map[ext] = parser_cls
    return parser_map


def _process_file(
    params: ProcessFileParams,
) -> ProcessFileResult:
    file_proc_start = time.perf_counter()
    p = params
    rel_path_str = str(p.path.relative_to(p.root))
    suffix = p.path.suffix or "no_suffix"

    try:
        # Skip paths ignored by .gitignore
        if p.gitignore.match_file(str(rel_path_str)):
            duration = time.perf_counter() - file_proc_start
            return ProcessFileResult(
                status=ProcessFileStatus.SKIPPED, duration=duration, suffix=suffix
            )

        mod_time = p.path.stat().st_mtime
        if p.existing_meta and p.existing_meta.last_updated == mod_time:
            duration = time.perf_counter() - file_proc_start
            return ProcessFileResult(
                status=ProcessFileStatus.SKIPPED, duration=duration, suffix=suffix
            )

        file_hash = compute_file_hash(str(p.path))
        if p.existing_meta and p.existing_meta.file_hash == file_hash:
            duration = time.perf_counter() - file_proc_start
            return ProcessFileResult(
                status=ProcessFileStatus.SKIPPED, duration=duration, suffix=suffix
            )

        # File has changed or is new, needs processing
        parser_cls = p.parser_map.get(p.path.suffix.lower())
        if parser_cls is None:
            duration = time.perf_counter() - file_proc_start
            return ProcessFileResult(
                status=ProcessFileStatus.BARE_FILE,
                duration=duration,
                suffix=suffix,
                rel_path=rel_path_str,
                existing_meta=p.existing_meta,
                file_hash=file_hash,
                mod_time=mod_time,
            )

        parser = parser_cls(p.pm, p.repo, rel_path_str)
        parsed_file = parser.parse(p.cache)
        duration = time.perf_counter() - file_proc_start
        return ProcessFileResult(
            status=ProcessFileStatus.PARSED_FILE,
            duration=duration,
            suffix=suffix,
            parsed_file=parsed_file,
            existing_meta=p.existing_meta,
        )

    except Exception as exc:
        duration = time.perf_counter() - file_proc_start
        return ProcessFileResult(
            status=ProcessFileStatus.ERROR,
            duration=duration,
            suffix=suffix,
            rel_path=rel_path_str,
            exception=exc,
        )


async def scan_repo(
    pm: ProjectManager,
    repo: Repo,
    progress_callback: Optional[Callable[[BaseModel], None]] = None,
    paths: Optional[list[str]] = None,
) -> ScanResult:
    """
    Recursively walk the project directory, parse every supported source file
    and store parsing results via the project-wide data repository.
    """
    start_time = time.perf_counter()
    result = ScanResult(repo=repo)
    root_path: str | None = repo.root_path
    if not root_path:
        logger.warning("scan_repo skipped – repo path is not set.")
        return ScanResult(repo=repo)

    processed_paths: set[str] = set()

    root = Path(root_path).resolve()

    cache = ProjectCache()
    state = ParsingState()
    timing_stats: defaultdict[str, dict[str, int | float]] = defaultdict(
        lambda: {"count": 0, "total_time": 0.0}
    )
    upsert_timing_stats: defaultdict[str, float] = defaultdict(float)

    parser_map = _get_parser_map(pm)

    # Collect ignore patterns from all .gitignore files (combined)
    combined_spec = pathspec.PathSpec.from_lines("gitwildmatch", [])
    gitignore_paths: set[Path] = set()
    root_gitignore = root / ".gitignore"
    if root_gitignore.exists():
        gitignore_paths.add(root_gitignore)
    for gi in root.rglob(".gitignore"):
        gitignore_paths.add(gi)
    for gi_path in sorted(gitignore_paths):
        try:
            combined_spec = combined_spec + parse_gitignore(gi_path, root_dir=root)
        except Exception as exc:
            logger.warning("Failed to parse .gitignore", path=str(gi_path), exc=exc)
    gitignore = combined_spec

    if paths:
        all_files = [root / p for p in paths]
    else:
        all_files = list(root.rglob("*"))

    # Prefilter against .gitignore and ignored directories before scheduling work
    filtered_files: list[Path] = []
    for path in all_files:
        try:
            rel_path = path.relative_to(root)
        except Exception:
            # Skip paths outside root (shouldn't happen, but defensive)
            continue
        # Only files
        if not path.is_file():
            continue
        # Skip .gitignore matches
        if gitignore.match_file(str(rel_path)):
            continue
        # Skip ignored directories by settings
        if any(part in pm.settings.ignored_dirs for part in rel_path.parts):
            continue
        filtered_files.append(path)

    num_workers = pm.settings.scanner_num_workers
    if num_workers is None:
        try:
            cpus = os.cpu_count()
            if cpus:
                num_workers = max(1, cpus - 1)
            else:
                num_workers = 4
        except NotImplementedError:
            num_workers = 4  # A reasonable default

    logger.debug("number of workers", count=num_workers)

    # Pre-fetch existing file metadata for change detection
    rel_paths = [str(p.relative_to(root)) for p in filtered_files]
    existing_list = (
        await pm.data.file.get_by_paths(repo.id, rel_paths) if rel_paths else []
    )
    existing_by_path: dict[str, File] = {f.path: f for f in existing_list}

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks: list[asyncio.Future] = []
        for path in filtered_files:
            rel_path = path.relative_to(root)
            processed_paths.add(str(rel_path))
            params = ProcessFileParams(
                pm=pm,
                repo=repo,
                path=path,
                root=root,
                gitignore=gitignore,
                cache=cache,
                parser_map=parser_map,
                existing_meta=existing_by_path.get(str(rel_path)),
            )
            tasks.append(loop.run_in_executor(executor, _process_file, params))

        total_tasks = len(tasks)
        processed = 0
        idx = 0
        async for future in _iter_as_completed(tasks):
            idx += 1
            if (idx + 1) % 100 == 0:
                logger.debug("processing...", num=idx + 1, total=total_tasks)

            res: ProcessFileResult = await future
            processed += 1
            if progress_callback and (processed % 100 == 0):
                try:
                    progress_callback(
                        ScanProgress(
                            repo_id=repo.id,
                            total_files=total_tasks,
                            processed_files=processed,
                            files_added=len(result.files_added),
                            files_updated=len(result.files_updated),
                            files_deleted=len(result.files_deleted),
                            elapsed_seconds=time.perf_counter() - start_time,
                        )
                    )
                except Exception as cb_exc:
                    logger.error("Progress callback failed", exc=cb_exc)

            if res.status == ProcessFileStatus.SKIPPED:
                continue

            # Update timing stats for all processed files (error, bare, parsed)
            timing_stats[res.suffix]["count"] += 1
            timing_stats[res.suffix]["total_time"] += res.duration

            if res.status == ProcessFileStatus.ERROR:
                logger.error(
                    "Failed to parse file", path=res.rel_path, exc=res.exception
                )
                continue

            if res.status == ProcessFileStatus.BARE_FILE:
                assert res.rel_path is not None
                logger.debug(
                    "No parser registered for path – storing bare File.",
                    path=res.rel_path,
                )
                if res.existing_meta is None:
                    assert res.file_hash is not None
                    assert res.mod_time is not None
                    fm = File(
                        id=generate_id(),
                        repo_id=repo.id,
                        package_id=None,
                        path=res.rel_path,
                        file_hash=res.file_hash,
                        last_updated=res.mod_time,
                    )
                    await pm.data.file.create([fm])
                    result.files_added.append(res.rel_path)
                else:
                    await pm.data.file.update(
                        [
                            (
                                res.existing_meta.id,
                                {
                                    "file_hash": res.file_hash,
                                    "last_updated": res.mod_time,
                                },
                            )
                        ]
                    )
                    result.files_updated.append(res.rel_path)

            elif res.status == ProcessFileStatus.PARSED_FILE:
                assert res.parsed_file is not None
                await upsert_parsed_file(
                    pm, repo, state, res.parsed_file, upsert_timing_stats
                )
                if res.existing_meta is None:
                    result.files_added.append(res.parsed_file.path)
                else:
                    result.files_updated.append(res.parsed_file.path)

    # Remove stale metadata for files that have disappeared from disk.
    # Only perform this check on a full scan (when `paths` is not provided).
    if paths is None:
        repo_id = repo.id

        # All files currently stored for this repo
        # TODO: Optimize
        existing_files = await pm.data.file.get_list(FileFilter(repo_ids=[repo_id]))

        for fm in existing_files:
            if fm.path not in processed_paths:
                # 1) delete all symbols that belonged to the vanished file
                await pm.data.node.delete_by_file_ids([fm.id])
                # 2) delete the file metadata itself
                await pm.data.file.delete([fm.id])
                result.files_deleted.append(fm.path)

    #  Remove Package entries that lost all their files
    await pm.data.package.delete_orphaned()

    # Resolve import edges
    await resolve_pending_import_edges(pm, repo, state)

    if pm.embeddings and state.pending_embeddings:
        logger.debug(
            "Scheduling embeddings for new/updated symbols",
            count=len(state.pending_embeddings),
        )
        await asyncio.gather(
            *[
                schedule_symbol_embedding(
                    pm.data.node, pm.embeddings, sym_id=task.symbol_id, body=task.text
                )
                for task in state.pending_embeddings
            ]
        )

    # Refresh any full text indexes
    pm.data.refresh_indexes()

    await schedule_missing_embeddings(pm, repo)
    await schedule_outdated_embeddings(pm, repo)

    duration = time.perf_counter() - start_time
    logger.debug(
        "scan_repo finished.",
        duration=f"{duration:.3f}s",
        files_added=len(result.files_added),
        files_updated=len(result.files_updated),
        files_deleted=len(result.files_deleted),
    )

    if timing_stats:
        logger.debug("File processing summary:")
        for suffix, stats in sorted(timing_stats.items()):
            avg_time = stats["total_time"] / stats["count"]
            logger.debug(
                f"  - Suffix: {suffix:<10} | "
                f"Files: {stats['count']:>4} | "
                f"Total: {stats['total_time']:>7.3f}s | "
                f"Avg: {avg_time * 1000:>8.2f} ms/file"
            )

    if upsert_timing_stats:
        logger.debug("Upsert performance summary:")
        total_upserts = int(upsert_timing_stats.get("total_upsert_count", 0))
        if total_upserts > 0:
            total_time = upsert_timing_stats["total_upsert_time"]
            avg_total_time_ms = total_time / total_upserts * 1000
            logger.debug(
                f"  - Total upserted files: {total_upserts}, "
                f"Total time: {total_time:.3f}s, "
                f"Avg time: {avg_total_time_ms:.2f} ms/file"
            )

            breakdown = {
                "Package": upsert_timing_stats["upsert_package_time"],
                "File": upsert_timing_stats["upsert_file_time"],
                "Import Edges": upsert_timing_stats["upsert_import_edge_time"],
                "Symbols": upsert_timing_stats["upsert_symbol_time"],
                "Symbol Refs": upsert_timing_stats["upsert_symbol_ref_time"],
            }
            if total_time > 0:
                for name, time_val in breakdown.items():
                    if time_val > 0:
                        avg_time_ms = time_val / total_upserts * 1000
                        percentage = (time_val / total_time) * 100
                        logger.debug(
                            f"    - {name:<15}: "
                            f"Total: {time_val:>7.3f}s | "
                            f"Avg: {avg_time_ms:>8.2f} ms/file | "
                            f"Percentage: {percentage:.2f}%"
                        )

    return result


async def upsert_parsed_file(
    pm: ProjectManager,
    repo: Repo,
    state: ParsingState,
    parsed_file: ParsedFile,
    stats: Optional[defaultdict[str, float]] = None,
) -> None:
    """
    Persist *parsed_file* (package → file → symbols) into the
    project's data-repository. If an entity already exists it is
    updated, otherwise it is created (“upsert”).
    """
    upsert_start_time = time.perf_counter()

    # Package
    t_start = time.perf_counter()
    pkg_meta: Optional[Package] = None
    if parsed_file.package:
        existing_pkgs = await pm.data.package.get_by_virtual_paths(
            repo.id, [parsed_file.package.virtual_path]
        )
        pkg_meta = existing_pkgs[0] if existing_pkgs else None

        pkg_data = parsed_file.package.to_dict()
        pkg_data["repo_id"] = repo.id

        if pkg_meta:
            await pm.data.package.update([(pkg_meta.id, pkg_data)])
        else:
            pkg_meta = Package(id=generate_id(), **pkg_data)
            created = await pm.data.package.create([pkg_meta])
            pkg_meta = created[0]
    t_pkg = time.perf_counter()

    # File
    existing_files = await pm.data.file.get_by_paths(repo.id, [parsed_file.path])
    file_meta = existing_files[0] if existing_files else None

    file_data = parsed_file.to_dict()
    file_data.update({"repo_id": repo.id})

    if pkg_meta:
        file_data.update({"package_id": pkg_meta.id})

    if file_meta:
        await pm.data.file.update([(file_meta.id, file_data)])
    else:
        file_meta = File(id=generate_id(), **file_data)
        created_files = await pm.data.file.create([file_meta])
        file_meta = created_files[0]
    t_file = time.perf_counter()

    # Import edges (package-level)
    import_repo = pm.data.importedge

    # Existing edges from this file and package
    existing_edges = await import_repo.get_list(
        ImportEdgeFilter(source_file_ids=[file_meta.id])
    )
    existing_by_key: dict[tuple[str | None, str | None, bool], ImportEdge] = {
        (e.to_package_virtual_path, e.alias, e.dot): e for e in existing_edges
    }

    # Batch-resolve import targets by virtual_path to minimize DB calls
    import_virtual_paths = sorted(
        {
            imp.virtual_path
            for imp in parsed_file.imports
            if not imp.external and imp.virtual_path
        }
    )
    vpath_to_pkg_id: dict[str, str] = {}
    if import_virtual_paths:
        matched_pkgs = await pm.data.package.get_by_virtual_paths(
            repo.id, import_virtual_paths
        )
        for p in matched_pkgs:
            if p.virtual_path:
                vpath_to_pkg_id[p.virtual_path] = p.id

    updates: list[tuple[str, dict[str, Any]]] = []
    creates: list[ImportEdge] = []
    new_keys: set[tuple[str | None, str | None, bool]] = set()

    for imp in parsed_file.imports:
        key = (imp.virtual_path, imp.alias, imp.dot)
        new_keys.add(key)

        to_pkg_id: str | None = (
            vpath_to_pkg_id.get(imp.virtual_path) if imp.virtual_path else None
        )

        kwargs = imp.to_dict()
        kwargs.update(
            {
                "repo_id": repo.id,
                "from_package_id": pkg_meta.id if pkg_meta else None,
                "from_file_id": file_meta.id,
                "to_package_id": to_pkg_id,
            }
        )

        if key in existing_by_key:
            updates.append((existing_by_key[key].id, kwargs))
        else:
            creates.append(ImportEdge(id=generate_id(), **kwargs))

    updated_edges = await import_repo.update(updates) if updates else []
    created_edges = await import_repo.create(creates) if creates else []

    for edge in [*updated_edges, *created_edges]:
        if not edge.external and edge.to_package_id is None:
            state.pending_import_edges.append(edge)

    obsolete_ids = [
        edge.id
        for edge_key, edge in existing_by_key.items()
        if edge_key not in new_keys
    ]
    if obsolete_ids:
        await import_repo.delete(obsolete_ids)
    t_imp = time.perf_counter()

    # Symbols (re-create)
    node_repo = pm.data.node

    # collect existing symbols
    existing_symbols = await node_repo.get_list(NodeFilter(file_ids=[file_meta.id]))

    def _get_embedding_text(body: str, docstring: Optional[str]) -> str:
        if docstring:
            return f"{docstring}\n{body}"
        return body

    _old_by_content: dict[str, Node] = {
        _get_embedding_text(s.body, s.docstring): s for s in existing_symbols
    }

    emb_calc = pm.embeddings

    nodes_to_create: list[Node] = []

    def _insert_symbol(psym: ParsedNode, parent_id: str | None = None) -> str:
        """
        Insert *psym* as Node (recursively handles its children).
        When an old symbol with identical body exists, its embedding vector
        (and model name) are copied instead of re-computing.
        """
        # base attributes coming from the parser
        sm_data: dict[str, Any] = psym.to_dict()
        sm_data.update(
            {
                "id": generate_id(),
                "repo_id": repo.id,
                "file_id": file_meta.id,
                "package_id": pkg_meta.id if pkg_meta else None,
                "parent_node_id": parent_id,
            }
        )
        kind_val = sm_data.get("kind")
        if kind_val is not None:
            kind_key = kind_val.value if hasattr(kind_val, "value") else str(kind_val)
            boost = pm.settings.search.node_kind_boosts.get(kind_key, 1.0)
        else:
            boost = 1.0
        sm_data["search_boost"] = boost

        # reuse embedding if we had an identical symbol earlier
        embedding_text = _get_embedding_text(psym.body, psym.docstring)
        old = _old_by_content.get(embedding_text)
        if old and old.embedding_code_vec is not None:
            sm_data["embedding_code_vec"] = old.embedding_code_vec
            sm_data["embedding_model"] = old.embedding_model
            schedule_emb = False
        else:
            schedule_emb = emb_calc is not None

        sm = Node(**sm_data)
        nodes_to_create.append(sm)

        if schedule_emb:
            state.pending_embeddings.append(
                EmbeddingTask(symbol_id=sm.id, text=embedding_text)
            )

        # recurse into children
        for child in psym.children:
            _insert_symbol(child, sm.id)

        return sm.id

    await node_repo.delete_by_file_ids([file_meta.id])
    for sym in parsed_file.symbols:
        _insert_symbol(sym)
    if nodes_to_create:
        await node_repo.create(nodes_to_create)
    t_sym = time.perf_counter()

    t_ref = time.perf_counter()
    if stats is not None:
        stats["total_upsert_count"] += 1
        stats["total_upsert_time"] += t_ref - upsert_start_time
        stats["upsert_package_time"] += t_pkg - t_start
        stats["upsert_file_time"] += t_file - t_pkg
        stats["upsert_import_edge_time"] += t_imp - t_file
        stats["upsert_symbol_time"] += t_sym - t_imp
        stats["upsert_symbol_ref_time"] += t_ref - t_sym


async def resolve_pending_import_edges(
    pm: ProjectManager, repo: Repo, state: ParsingState
) -> None:
    pkg_repo = pm.data.package
    imp_repo = pm.data.importedge

    # Filter edges that can be resolved and require resolution
    candidates = [
        e
        for e in state.pending_import_edges
        if (not e.external)
        and (e.to_package_id is None)
        and (e.to_package_physical_path is not None)
    ]
    if not candidates:
        state.pending_import_edges.clear()
        return

    # One batched lookup by physical paths
    phys_paths = sorted(
        {e.to_package_physical_path for e in candidates if e.to_package_physical_path}
    )
    matched_pkgs = await pkg_repo.get_by_physical_paths(repo.id, phys_paths)
    phys_to_pkg_id: dict[str, str] = {
        p.physical_path: p.id for p in matched_pkgs if p.physical_path
    }

    # One batched update
    updates: list[tuple[str, dict[str, Any]]] = []
    for e in candidates:
        pkg_id = phys_to_pkg_id.get(e.to_package_physical_path)  # type: ignore[arg-type]
        if pkg_id:
            updates.append((e.id, {"to_package_id": pkg_id}))
            e.to_package_id = pkg_id
    if updates:
        await imp_repo.update(updates)

    # Maintain existing behavior: clear all pending edges after attempt
    state.pending_import_edges.clear()


async def _iter_as_completed(tasks: list[asyncio.Future]):
    for fut in asyncio.as_completed(tasks):
        yield fut
