import pytest

import knowlt.stores.duckdb as duckdb_store
from knowlt.data import NodeSearchQuery
from knowlt.models import Node, NodeKind, Repo
from knowlt.project import ProjectManager
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


@pytest.mark.asyncio
async def test_refresh_indexes_keeps_fts_macro_after_reopen(tmp_path):
    db_path = tmp_path / "fts-refresh-after-reopen.duckdb"
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
        await repo.refresh_indexes()
    finally:
        repo.close()

    reopened = DuckDBDataRepository(settings, db_path=str(db_path))
    try:
        before = await reopened._conn.execute(
            "SELECT fts_main_nodes.match_bm25(?, ?)",
            ["node-1", "hello"],
        )
        assert len(before) == 1

        await reopened.refresh_indexes()

        after = await reopened._conn.execute(
            "SELECT fts_main_nodes.match_bm25(?, ?)",
            ["node-1", "hello"],
        )
        assert len(after) == 1
    finally:
        reopened.close()

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


@pytest.mark.asyncio
async def test_refresh_indexes_rebuilds_fts_for_new_nodes(tmp_path):
    db_path = tmp_path / "fts-refresh-new-nodes.duckdb"
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

        before = await repo.node.search(
            NodeSearchQuery(repo_ids=["repo-1"], needle="search", limit=5)
        )
        assert before == []

        await repo.refresh_indexes()

        after = await repo.node.search(
            NodeSearchQuery(repo_ids=["repo-1"], needle="search", limit=5)
        )
        assert [n.id for n in after] == ["node-1"]
    finally:
        repo.close()


@pytest.mark.asyncio
async def test_refresh_waits_for_search_indexes_before_returning(tmp_path):
    db_path = tmp_path / "fts-refresh-after-scan.duckdb"
    root = tmp_path / "repo"
    root.mkdir()
    (root / "sample.py").write_text("def hello_world():\n    return 'hello world'\n")

    settings = ProjectSettings(
        project_name="search-test",
        repo_name="repo",
        repo_path=str(root),
        repository_connection=str(db_path),
    )

    repo = DuckDBDataRepository(settings, db_path=str(db_path))
    pm = await ProjectManager.create(settings, repo)
    try:
        await pm.refresh()

        results = await repo.node.search(
            NodeSearchQuery(repo_ids=[pm.default_repo.id], needle="sample", limit=5)
        )
        assert any(n.name == "hello_world" for n in results)
    finally:
        await pm.destroy()