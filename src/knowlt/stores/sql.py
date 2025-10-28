import json
import zlib
from typing import Any, Callable, Generic, TypeVar, Type, Optional
import importlib.resources as pkg_resources
from datetime import datetime, timezone

from pydantic import BaseModel
from pypika.terms import ValueWrapper


# RawValue override
class RawValue(ValueWrapper):
    def get_value_sql(self, **kwargs: Any) -> str:
        return self.value


# SQL migrations helpers
def apply_migrations(
    execute_fn: Callable[[str, Optional[list[Any]]], None],
    query_fn: Callable[[str, Optional[list[Any]]], list[dict[str, Any]]],
    migrations_pkg: str,
    create_migrations_table_sql: str,
    insert_migration_sql: str,
) -> None:
    """
    Generic migration helper.

    Scans a package for ``.sql`` files and applies them in alphabetical
    order.  Tracks applied migrations in a ``__migrations__`` table.
    """
    # ensure bookkeeping table exists
    execute_fn(create_migrations_table_sql, None)
    applied_rows = query_fn("SELECT name FROM __migrations__", None)
    already_applied = {r["name"] for r in applied_rows}

    try:
        files_source = pkg_resources.files(migrations_pkg)
    except ModuleNotFoundError:
        # no migrations for this backend
        return

    with pkg_resources.as_file(files_source) as mig_root:
        sql_files = sorted(p for p in mig_root.iterdir() if p.suffix == ".sql")

        for file_path in sql_files:
            if file_path.name in already_applied:
                continue
            sql = file_path.read_text()
            execute_fn(sql, None)

            execute_fn(
                insert_migration_sql, [file_path.name, datetime.now(timezone.utc)]
            )


# Base class for SQL repositories
T = TypeVar("T", bound=BaseModel)


class BaseSQLRepository(Generic[T]):
    model: Type[T]

    _json_fields: set[str] = set()
    _field_parsers: dict[str, Callable[[Any], Any]] = {}
    _compress_fields: set[str] = set()

    def _serialize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        for fld in self._json_fields:
            if fld in data and data[fld] is not None:
                val = data[fld]
                # support Pydantic models
                if hasattr(val, "model_dump"):
                    val = val.model_dump(exclude_none=False)
                data[fld] = json.dumps(val)

        for fld in self._compress_fields:
            if fld in data and data[fld] is not None:
                raw = data[fld]
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                data[fld] = _compress_blob(bytes(raw))

        for k, v in data.items():
            if isinstance(v, list):
                data[k] = RawValue(v)

        return data

    def _deserialize_data(self, row: dict[str, Any]) -> dict[str, Any]:
        for fld in self._json_fields:
            if fld in row and row[fld] is not None:
                parsed = json.loads(row[fld])
                parser = self._field_parsers.get(fld)
                row[fld] = parser(parsed) if parser else parsed

        for fld in self._compress_fields:
            if fld in row and row[fld] is not None:
                blob = bytes(row[fld])
                text = _decompress_blob(blob)
                try:
                    row[fld] = text.decode("utf-8")
                except Exception:
                    row[fld] = text

        return row


# Compression helpers
_UNCOMPRESSED_PREFIX = b"\x00"  # 1-byte marker → raw payload follows
_COMPRESSED_PREFIX = b"\x01"  # 1-byte marker → zlib-compressed payload
_MIN_COMPRESS_LEN = 50  # threshold in *bytes*


def _compress_blob(data: bytes) -> bytes:
    """Return data prefixed & (optionally) zlib-compressed for storage."""
    if len(data) <= _MIN_COMPRESS_LEN:
        return _UNCOMPRESSED_PREFIX + data
    return _COMPRESSED_PREFIX + zlib.compress(data)


def _decompress_blob(blob: bytes) -> bytes:
    """Undo `_compress_blob`."""
    if not blob:
        return blob
    prefix, payload = blob[:1], blob[1:]
    if prefix == _COMPRESSED_PREFIX:
        return zlib.decompress(payload)
    if prefix == _UNCOMPRESSED_PREFIX:
        return payload
    # legacy / unknown prefix → return as-is
    return blob
