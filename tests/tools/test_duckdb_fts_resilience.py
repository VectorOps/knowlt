import pytest

import knowlt.stores.duckdb as duckdb_store
from knowlt.data import NodeSearchQuery
from knowlt.models import Node, NodeKind, Repo
from knowlt.settings import ProjectSettings
from knowlt.stores.duckdb import DuckDBDataRepository


@pytest.mark.asyncio
async def test_reopen_recovers_missing_fts_macro(tmp_path):
    db_path = tmp_path / "fts-recovery.duckdb"
    settings = ProjectSettings(repository_connection=str(db_path))

    repo = DuckDBDataRepository(settings, db_path=str(db_path))
    try:
        await repo.repo.create(
            [Repo(id="repo-1", name="repo-1", root_path=str(tmp_path))]
        )
        await repo.node.create(
            [
                Node(
                    id="node-1",
                    repo_id="repo-1",
                    name="hello_world",
                    body="hello world search text",
                    kind=NodeKind.FUNCTION,
                )
            ]
        )
        await repo._conn.execute("PRAGMA drop_fts_index('nodes')")
    finally:
        repo.close()

    reopened = DuckDBDataRepository(settings, db_path=str(db_path))
    try:
        probe = await reopened._conn.execute(
            "SELECT fts_main_nodes.match_bm25(?, ?)",
            ["node-1", "hello"],
        )
        assert len(probe) == 1
    finally:
        reopened.close()


def test_startup_raises_explicit_error_for_fts_initialization_failure(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "fts-broken-startup.duckdb"
    settings = ProjectSettings(repository_connection=str(db_path))

    repo = DuckDBDataRepository(settings, db_path=str(db_path))
    repo.close()

    def broken_ensure(execute_sql, *, force_rebuild: bool = False):
        raise RuntimeError("simulated FTS init failure")

    monkeypatch.setattr(duckdb_store, "_ensure_fts_index", broken_ensure)

    with pytest.raises(RuntimeError) as excinfo:
        DuckDBDataRepository(settings, db_path=str(db_path))

    msg = str(excinfo.value)
    assert "Delete the database file and restart" in msg
    assert str(db_path) in msg
    assert "simulated FTS init failure" in msg


@pytest.mark.asyncio
async def test_refresh_indexes_does_not_drop_live_fts_macro(tmp_path):
    db_path = tmp_path / "fts-refresh-live.duckdb"
    settings = ProjectSettings(repository_connection=str(db_path))

    repo = DuckDBDataRepository(settings, db_path=str(db_path))
    try:
        await repo.repo.create(
            [Repo(id="repo-1", name="repo-1", root_path=str(tmp_path))]
        )
        await repo.node.create(
            [
                Node(
                    id="node-1",
                    repo_id="repo-1",
                    name="hello_world",
                    body="hello world search text",
                    kind=NodeKind.FUNCTION,
                )
            ]
        )

        before = await repo._conn.execute(
            "SELECT fts_main_nodes.match_bm25(?, ?)",
            ["node-1", "hello"],
        )
        assert len(before) == 1

        await repo.refresh_indexes()

        probe = await repo._conn.execute(
            "SELECT fts_main_nodes.match_bm25(?, ?)",
            ["node-1", "hello"],
        )
        assert len(probe) == 1
    finally:
        repo.close()