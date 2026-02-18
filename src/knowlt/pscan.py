from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Iterable

import os

import pathspec

from knowlt.helpers import parse_gitignore


@dataclass
class FileScanCandidate:
    path: Path
    maybe_changed: bool


@dataclass
class PScanResult:
    candidates: list[FileScanCandidate]
    processed_paths: set[str]


def _iter_paths_from_root(root: Path, paths: list[str]) -> Iterable[Tuple[Path, str]]:
    """Yield (absolute_path, relative_str) for the given *paths* under *root*.

    This helper does not apply any filtering; callers are expected to apply
    ignore rules and type checks.
    """
    for p in paths:
        abs_path = (root / p).resolve()
        try:
            rel_path = abs_path.relative_to(root)
        except Exception:
            # Skip anything outside of the root for safety
            continue
        yield abs_path, str(rel_path)


def collect_files(
    root: Path,
    ignored_dirs: set[str],
    last_scanned: Optional[int],
    paths: Optional[list[str]] = None,
) -> PScanResult:
    """Collect and pre-filter files under *root* using os.scandir.

    This function is intended to run in a background thread via
    ``asyncio.to_thread``. It performs an iterative directory traversal
    (no Python recursion) and applies .gitignore-based filtering on the fly,
    combining patterns from .gitignore files discovered along the way.
    """

    root = root.resolve()

    # Base (empty) spec – we always keep a PathSpec instance handy so we can
    # "add" new specs as we discover nested .gitignore files.
    empty_spec = pathspec.PathSpec.from_lines("gitwildmatch", [])

    candidates: list[FileScanCandidate] = []
    processed_paths: set[str] = set()

    # Stack elements: (abs_dir, rel_dir_str, current_spec)
    stack: list[Tuple[Path, str, pathspec.PathSpec]] = []

    if paths:
        # When explicit paths are provided, we seed the stack with the
        # corresponding directories (or handle individual files directly)
        # while still applying .gitignore rules per directory.
        #
        # We do NOT attempt to guess user intent beyond this; semantics are
        # preserved as "scan exactly these paths".
        for abs_path, rel_str in _iter_paths_from_root(root, paths):
            if abs_path.is_dir():
                stack.append((abs_path, rel_str, empty_spec))
            elif abs_path.is_file():
                # For individual files we still want to respect any
                # .gitignore files that may apply. Build the effective
                # spec by walking from root to the file's parent.
                parent = abs_path.parent
                spec = empty_spec
                # Build the chain of directories from root to parent.
                dir_chain: list[Path] = []
                cur = parent
                while True:
                    try:
                        _ = cur.relative_to(root)
                    except Exception:
                        break
                    dir_chain.append(cur)
                    if cur == root:
                        break
                    cur = cur.parent
                for d in reversed(dir_chain):
                    gi_path = d / ".gitignore"
                    if gi_path.is_file():
                        try:
                            spec = spec + parse_gitignore(gi_path, root_dir=root)
                        except Exception:
                            # Ignore malformed .gitignore files
                            pass

                try:
                    rel_path = abs_path.relative_to(root)
                except Exception:
                    continue

                rel_path_str = str(rel_path)

                if spec.match_file(rel_path_str):
                    continue
                if any(part in ignored_dirs for part in rel_path.parts):
                    continue

                try:
                    mod_time = abs_path.stat().st_mtime_ns
                except OSError:
                    maybe_changed = False
                else:
                    maybe_changed = last_scanned is None or mod_time > last_scanned

                candidates.append(
                    FileScanCandidate(path=abs_path, maybe_changed=maybe_changed)
                )
                processed_paths.add(rel_path_str)
    else:
        stack.append((root, "", empty_spec))

    while stack:
        abs_dir, rel_dir_str, cur_spec = stack.pop()

        # Merge in a .gitignore from this directory if present.
        gi_path = abs_dir / ".gitignore"
        spec = cur_spec
        if gi_path.is_file():
            try:
                spec = spec + parse_gitignore(gi_path, root_dir=root)
            except Exception:
                # Ignore malformed .gitignore files but continue traversal.
                pass

        try:
            with os.scandir(abs_dir) as it:
                for entry in it:
                    try:
                        name = entry.name
                    except FileNotFoundError:
                        # Entry vanished between scandir and access; skip.
                        continue

                    abs_path = Path(entry.path)

                    if rel_dir_str:
                        rel_path_str = f"{rel_dir_str}{os.sep}{name}"
                    else:
                        rel_path_str = name

                    # Normalize to Path for parts/relative handling
                    rel_path = Path(rel_path_str)

                    if entry.is_dir(follow_symlinks=False):
                        # Skip directories by explicit ignored name
                        if name in ignored_dirs:
                            continue

                        # If the directory itself is ignored by spec, skip
                        # recursing into it entirely.
                        if spec.match_file(rel_path_str):
                            continue

                        stack.append((abs_path, rel_path_str, spec))
                        continue

                    if not entry.is_file(follow_symlinks=False):
                        continue

                    if spec.match_file(rel_path_str):
                        continue

                    if any(part in ignored_dirs for part in rel_path.parts):
                        continue

                    try:
                        mod_time = abs_path.stat().st_mtime_ns
                    except OSError:
                        maybe_changed = False
                    else:
                        maybe_changed = last_scanned is None or mod_time > last_scanned

                    candidates.append(
                        FileScanCandidate(path=abs_path, maybe_changed=maybe_changed)
                    )
                    processed_paths.add(rel_path_str)
        except FileNotFoundError:
            # Directory disappeared while scanning; skip.
            continue

    return PScanResult(candidates=candidates, processed_paths=processed_paths)
