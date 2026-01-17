import hashlib
import os
import uuid
from pathlib import Path
from typing import Union
import pathspec


def compute_file_hash(abs_path: str) -> str:
    """Compute SHA256 hash of a file's contents."""
    sha256 = hashlib.sha256()
    with open(abs_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_symbol_hash(symbol: Union[str, bytes]) -> str:
    """
    Return the SHA-256 hex-digest of *symbol*.
    Accepts either ``str`` (automatically UTF-8-encoded) or raw ``bytes``.
    """
    sha256 = hashlib.sha256()
    if isinstance(symbol, str):
        symbol = symbol.encode("utf-8")
    sha256.update(symbol)
    return sha256.hexdigest()


def parse_gitignore(
    gitignore_path: str | Path, *, root_dir: str | Path | None = None
) -> "pathspec.PathSpec":
    """
    Parse a .gitignore file at gitignore_path and return a pathspec.PathSpec
    built with the 'gitwildmatch' syntax (same as Git).
    If root_dir is provided, patterns from nested .gitignore files are rewritten
    to be relative to the repository root, approximating Git scoping:
      - '/pat'      -> '<subdir>/pat'
      - 'pat'       -> '<subdir>/**/pat'
      - 'dir/pat'   -> '<subdir>/dir/pat'
    Negations '!' are preserved and rewritten accordingly.
    """
    gitignore_file = Path(gitignore_path)
    if not gitignore_file.exists() or not gitignore_file.is_file():
        return pathspec.PathSpec.from_lines("gitwildmatch", [])

    raw_lines: list[str] = []
    for raw in gitignore_file.read_text().splitlines():
        raw = raw.rstrip()
        if not raw or raw.lstrip().startswith("#"):
            continue
        raw_lines.append(raw)

    # No rewriting needed if no root_dir provided (ex: top-level .gitignore usage)
    if root_dir is None:
        return pathspec.PathSpec.from_lines("gitwildmatch", raw_lines)

    root = Path(root_dir).resolve()
    base_dir = gitignore_file.parent.resolve()
    try:
        dir_rel_path = base_dir.relative_to(root)
        dir_rel = os.sep.join(dir_rel_path.parts) if dir_rel_path.parts else ""
    except Exception:
        # Fallback to absolute path components joined by os.sep
        dir_rel = os.sep.join(base_dir.parts)
    base_prefix = "" if dir_rel in ("", ".") else f"{dir_rel}{os.sep}"

    def _rewrite(pat: str) -> str:
        # Normalize any separators in the pattern to the current OS separator
        p = pat.replace("\\", os.sep).replace("/", os.sep)
        if p.startswith(os.sep):
            # anchored to the .gitignore's directory
            return base_prefix + p.lstrip(os.sep)
        if os.sep in p:
            # path component present: match relative to this directory
            return base_prefix + p
        # no separator: match anywhere under this directory
        return base_prefix + f"**{os.sep}" + p

    rewritten: list[str] = []
    for raw in raw_lines:
        if raw.startswith("!"):
            pat = raw[1:]
            rewritten.append("!" + _rewrite(pat))
        else:
            rewritten.append(_rewrite(raw))

    return pathspec.PathSpec.from_lines("gitwildmatch", rewritten)


def matches_gitignore(path: str | Path, spec: "pathspec.PathSpec") -> bool:
    """
    Return True if *path* (relative to repo root) is ignored by *spec*.
    """
    return spec.match_file(str(path))


_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _uuid_to_base58(u: uuid.UUID) -> str:
    """
    Encode a 128-bit UUID into a compact, URL-safe Base58 string.

    We left-pad with the first alphabet character to produce a fixed
    22-character representation, which is sufficient for 128 bits.
    """
    num = u.int
    if num == 0:
        return _BASE58_ALPHABET[0]

    chars: list[str] = []
    base = len(_BASE58_ALPHABET)
    while num > 0:
        num, rem = divmod(num, base)
        chars.append(_BASE58_ALPHABET[rem])

    encoded = "".join(reversed(chars))
    # 58^21 < 2^128 <= 58^22, so 22 Base58 chars cover the space.
    if len(encoded) < 22:
        encoded = _BASE58_ALPHABET[0] * (22 - len(encoded)) + encoded
    return encoded


def generate_id() -> str:
    """
    Return a new unique identifier as a compact, URL-safe string.

    Encodes a random UUID4 using a fixed-length Base58 alphabet.
    """
    return _uuid_to_base58(uuid.uuid4())
