import pytest

from knowlt import data
from knowlt.models import Node, NodeKind, Repo
from knowlt.settings import ProjectSettings
from knowlt.stores.duckdb import DuckDBDataRepository


@pytest.mark.asyncio
async def test_node_search_recovers_when_fts_macro_is_missing(tmp_path):
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

        await repo.node.search(data.NodeSearchQuery(needle="hello", limit=5))
        probe = await repo._conn.execute(
            "SELECT fts_main_nodes.match_bm25(?, ?)",
            ["node-1", "hello"],
        )
        assert len(probe) == 1
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