from pathlib import Path
from typing import Dict, Any
import pytest
from knowlt.stores.duckdb import DuckDBDataRepository
from knowlt.models import (
    Repo,
    Package,
    File,
    Node,
    ImportEdge,
)
from knowlt.settings import ProjectSettings
from knowlt.data import (
    NodeSearchQuery,
    PackageFilter,
    FileFilter,
    NodeFilter,
    ImportEdgeFilter,
)
from knowlt.helpers import generate_id


def make_id() -> str:
    return generate_id()


@pytest.fixture(params=[DuckDBDataRepository])
def data_repo(request, tmp_path: Path):
    """Yield a fresh data-repository instance (in-memory or DuckDB)."""
    cls = request.param
    settings = ProjectSettings(
        project_name="test",
        repo_name="test",
        repo_path=str(tmp_path),
    )
    return cls(settings)


@pytest.mark.asyncio
async def test_repo_metadata_repository(data_repo):
    repo_repo = data_repo.repo

    rid = make_id()
    obj = Repo(id=rid, name="repo1", root_path="/tmp/repo1")

    # create / fetch
    created = await repo_repo.create([obj])
    assert created == [obj]
    assert await repo_repo.get_by_ids([rid]) == [obj]
    # specialised method
    assert await repo_repo.get_by_path("/tmp/repo1") == obj
    # update / delete
    updated = await repo_repo.update([(rid, {"name": "repo2"})])
    assert updated and updated[0].name == "repo2"
    assert await repo_repo.delete([rid]) is True
    assert await repo_repo.get_by_ids([rid]) == []


@pytest.mark.asyncio
async def test_repo_delete_cascades_by_repo_id(data_repo):
    repo_repo = data_repo.repo
    pkg_repo = data_repo.package
    file_repo = data_repo.file
    node_repo = data_repo.node
    edge_repo = data_repo.importedge

    # two repos: rid1 will be deleted, rid2 is control
    rid1, rid2 = make_id(), make_id()
    await repo_repo.create(
        [
            Repo(id=rid1, name="r1", root_path="/tmp/r1"),
            Repo(id=rid2, name="r2", root_path="/tmp/r2"),
        ]
    )

    # objects for rid1
    pkg1_id = make_id()
    file1_id = make_id()
    node1_id = make_id()
    edge1_id = make_id()

    await pkg_repo.create(
        [
            Package(
                id=pkg1_id,
                name="pkg1",
                virtual_path="pkg1",
                physical_path="pkg1.py",
                repo_id=rid1,
            )
        ]
    )
    await file_repo.create(
        [File(id=file1_id, repo_id=rid1, package_id=pkg1_id, path="src/pkg1/file.py")]
    )
    await node_repo.create(
        [
            Node(
                id=node1_id,
                name="sym1",
                repo_id=rid1,
                file_id=file1_id,
                package_id=pkg1_id,
                body="def sym1(): pass",
            )
        ]
    )
    await edge_repo.create(
        [
            ImportEdge(
                id=edge1_id,
                repo_id=rid1,
                from_package_id=pkg1_id,
                from_file_id=file1_id,
                to_package_physical_path="pkg/other",
                to_package_virtual_path="pkg/other",
                raw="import pkg.other",
                external=False,
            )
        ]
    )

    # objects for rid2 (should survive)
    pkg2_id = make_id()
    file2_id = make_id()
    node2_id = make_id()
    edge2_id = make_id()

    await pkg_repo.create(
        [
            Package(
                id=pkg2_id,
                name="pkg2",
                virtual_path="pkg2",
                physical_path="pkg2.py",
                repo_id=rid2,
            )
        ]
    )
    await file_repo.create(
        [File(id=file2_id, repo_id=rid2, package_id=pkg2_id, path="src/pkg2/file.py")]
    )
    await node_repo.create(
        [
            Node(
                id=node2_id,
                name="sym2",
                repo_id=rid2,
                file_id=file2_id,
                package_id=pkg2_id,
                body="def sym2(): pass",
            )
        ]
    )
    await edge_repo.create(
        [
            ImportEdge(
                id=edge2_id,
                repo_id=rid2,
                from_package_id=pkg2_id,
                from_file_id=file2_id,
                to_package_physical_path="pkg/other2",
                to_package_virtual_path="pkg/other2",
                raw="import pkg.other2",
                external=False,
            )
        ]
    )

    # sanity: data exists for both repos
    assert await pkg_repo.get_list(PackageFilter(repo_ids=[rid1]))
    assert await file_repo.get_list(FileFilter(repo_ids=[rid1]))
    assert await node_repo.get_list(NodeFilter(repo_ids=[rid1]))
    assert await edge_repo.get_list(ImportEdgeFilter(repo_ids=[rid1]))

    assert await pkg_repo.get_list(PackageFilter(repo_ids=[rid2]))
    assert await file_repo.get_list(FileFilter(repo_ids=[rid2]))
    assert await node_repo.get_list(NodeFilter(repo_ids=[rid2]))
    assert await edge_repo.get_list(ImportEdgeFilter(repo_ids=[rid2]))

    # delete rid1 and ensure cascade
    assert await repo_repo.delete([rid1]) is True

    # rid1 repo and all its data gone
    assert await repo_repo.get_by_ids([rid1]) == []
    assert await pkg_repo.get_list(PackageFilter(repo_ids=[rid1])) == []
    assert await file_repo.get_list(FileFilter(repo_ids=[rid1])) == []
    assert await node_repo.get_list(NodeFilter(repo_ids=[rid1])) == []
    assert await edge_repo.get_list(ImportEdgeFilter(repo_ids=[rid1])) == []

    # file search index for rid1 should also be empty
    assert await file_repo.filename_complete("file", repo_ids=[rid1]) == []

    # rid2 data still present
    assert await repo_repo.get_by_ids([rid2]) != []
    assert await pkg_repo.get_list(PackageFilter(repo_ids=[rid2])) != []
    assert await file_repo.get_list(FileFilter(repo_ids=[rid2])) != []
    assert await node_repo.get_list(NodeFilter(repo_ids=[rid2])) != []
    assert await edge_repo.get_list(ImportEdgeFilter(repo_ids=[rid2])) != []


@pytest.mark.asyncio
async def test_package_metadata_repository(data_repo):
    pkg_repo, file_repo = data_repo.package, data_repo.file

    orphan_id = make_id()
    used_id = make_id()
    rid = make_id()
    await pkg_repo.create(
        [
            Package(
                id=orphan_id,
                name="orphan",
                virtual_path="pkg/orphan",
                physical_path="pkg/orphan.py",
                repo_id=rid,
            ),
            Package(
                id=used_id,
                name="used",
                virtual_path="pkg/used",
                physical_path="pkg/used.go",
                repo_id=rid,
            ),
        ]
    )

    # add a file that references the “used” package, leaving the first one orphaned
    await file_repo.create([File(id=make_id(), repo_id=rid, path="pkg/used/a.py", package_id=used_id)])

    v_list = await pkg_repo.get_by_virtual_paths(rid, ["pkg/used"])
    assert any(v.id == used_id for v in v_list)
    p_list = await pkg_repo.get_by_physical_paths(rid, ["pkg/used.go"])
    assert any(p.id == used_id for p in p_list)

    # list retrievals unchanged
    lst = await pkg_repo.get_list(PackageFilter(repo_ids=[rid]))
    assert {p.id for p in lst} == {orphan_id, used_id}
    # delete_orphaned should remove only the orphan package
    await pkg_repo.delete_orphaned()
    assert await pkg_repo.get_by_ids([orphan_id]) == []
    assert await pkg_repo.get_by_ids([used_id]) != []
    lst2 = await pkg_repo.get_list(PackageFilter(repo_ids=[rid]))
    assert [p.id for p in lst2] == [used_id]
    # update / delete
    upd = await pkg_repo.update([(used_id, {"name": "renamed"})])
    assert upd and upd[0].name == "renamed"
    assert await pkg_repo.delete([used_id]) is True
    assert await pkg_repo.get_list(PackageFilter(repo_ids=[rid])) == []


@pytest.mark.asyncio
async def test_file_metadata_repository(data_repo):
    file_repo = data_repo.file
    rid, pid, fid = make_id(), make_id(), make_id()
    obj = File(id=fid, repo_id=rid, package_id=pid, path="src/file.py")

    await file_repo.create([obj])
    res = await file_repo.get_by_paths(rid, ["src/file.py"])
    assert res and res[0] == obj

    assert await file_repo.get_list(FileFilter(repo_ids=[rid])) == [obj]
    assert await file_repo.get_list(FileFilter(package_id=pid)) == [obj]
    upd = await file_repo.update([(fid, {"path": "src/other.py"})])
    assert upd and upd[0].path == "src/other.py"
    assert await file_repo.delete([fid]) is True


@pytest.mark.asyncio
async def test_node_metadata_repository(data_repo):
    repo_repo = data_repo.repo
    rid = make_id()
    await repo_repo.create([Repo(id=rid, name="test", root_path=f"/tmp/{rid}")])

    node_repo = data_repo.node
    fid, sid = make_id(), make_id()

    body = "def sym(a: int) -> str\n\treturn 'a'"

    # create with signature
    await node_repo.create(
        [
            Node(
                id=sid,
                name="sym",
                file_id=fid,
                repo_id=rid,
                body=body,
            )
        ]
    )

    # read back (by id and by file_id) and ensure signature persisted
    got = await node_repo.get_by_ids([sid])
    assert got and got[0].body == body
    by_file = await node_repo.get_list(NodeFilter(file_ids=[fid]))
    assert by_file and by_file[0].body == body

    # update body
    new_body = "def sym(a: int) -> str\n\treturn 'b'"
    upd = await node_repo.update([(sid, {"body": new_body})])
    assert upd and upd[0].body == new_body

    # delete
    assert await node_repo.delete([sid]) is True


@pytest.mark.asyncio
async def test_import_edge_repository(data_repo):
    edge_repo = data_repo.importedge
    rid, eid, fid, from_pid = make_id(), make_id(), make_id(), make_id()
    await edge_repo.create(
        [
            ImportEdge(
                id=eid,
                repo_id=rid,
                from_package_id=from_pid,
                from_file_id=fid,
                to_package_physical_path="pkg/other",
                to_package_virtual_path="pkg/other",
                raw="import pkg.other",
                external=False,
            )
        ]
    )

    by_src = await edge_repo.get_list(ImportEdgeFilter(source_package_ids=[from_pid]))
    assert by_src and by_src[0].id == eid
    by_repo = await edge_repo.get_list(ImportEdgeFilter(repo_ids=[rid]))
    assert by_repo and by_repo[0].id == eid
    upd = await edge_repo.update([(eid, {"alias": "aliaspkg"})])
    assert upd and upd[0].alias == "aliaspkg"
    assert await edge_repo.delete([eid]) is True
    assert await edge_repo.get_list(ImportEdgeFilter(repo_ids=[rid])) == []


@pytest.mark.asyncio
async def test_node_search(data_repo):
    repo_repo, file_repo, node_repo = data_repo.repo, data_repo.file, data_repo.node

    # ---------- minimal repo / file scaffolding ----------
    rid = make_id()
    fid = make_id()
    await repo_repo.create([Repo(id=rid, name="test", root_path="/tmp/rid")])
    await file_repo.create([File(id=fid, repo_id=rid, path="src/a.py")])

    # ---------- seed three nodes ----------
    await node_repo.create(
        [
            Node(
                id=make_id(),
                name="Alpha",
                repo_id=rid,
                file_id=fid,
                body="def Alpha(): pass",
                docstring="Compute foo and bar.",
            ),
            Node(
                id=make_id(),
                name="Beta",
                repo_id=rid,
                file_id=fid,
                body="class Beta(): pass",
                docstring="Baz qux docs.",
            ),
            Node(
                id=make_id(),
                name="Gamma",
                repo_id=rid,
                file_id=fid,
                body="Gamma = 10",
                docstring="Alpha-numeric helper.",
            ),
        ]
    )
    await data_repo.refresh_indexes()

    # ---------- no-filter search: default ordering (name ASC) ----------
    res = await node_repo.search(NodeSearchQuery(repo_ids=[rid]))
    assert [s.name for s in res] == ["Alpha", "Beta", "Gamma"]

    # ---------- name substring (case-insensitive) ----------
    res_alpha = await node_repo.search(NodeSearchQuery(repo_ids=[rid], needle="alpha"))
    assert any(s.name == "Alpha" for s in res_alpha)

    # ---------- docstring / comment full-text search ----------
    res_foo = await node_repo.search(NodeSearchQuery(repo_ids=[rid], needle="foo"))
    assert any(s.name == "Alpha" for s in res_foo)

    # ---------- pagination ----------
    assert len(await node_repo.search(NodeSearchQuery(repo_ids=[rid], limit=2))) == 2
    assert [s.name for s in await node_repo.search(NodeSearchQuery(repo_ids=[rid], limit=2, offset=2))] == ["Gamma"]


# ---------------------------------------------------------------------------
# embedding-similarity search
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_symbol_embedding_search(data_repo):
    repo_repo, file_repo, node_repo = data_repo.repo, data_repo.file, data_repo.node

    rid = make_id()
    fid = make_id()

    await repo_repo.create([Repo(id=rid, name="test", root_path="/tmp/emb_repo")])
    await file_repo.create([File(id=fid, repo_id=rid, path="src/vec.py")])

    # seed three symbols with simple, orthogonal 3-d vectors
    await node_repo.create(
        [
            Node(
                id=make_id(),
                name="VecA",
                repo_id=rid,
                file_id=fid,
                body="def VecA(): pass",
                embedding_code_vec=[1.0, 0.0, 0.0] + [0] * 1021,
            ),
            Node(
                id=make_id(),
                name="VecB",
                repo_id=rid,
                file_id=fid,
                body="def VecB(): pass",
                embedding_code_vec=[0.0, 1.0, 0.0] + [0] * 1021,
            ),
            Node(
                id=make_id(),
                name="VecC",
                repo_id=rid,
                file_id=fid,
                body="def VecC(): pass",
                embedding_code_vec=[0.0, 0.0, 1.0] + [0] * 1021,
            ),
        ]
    )

    # query vector identical to VecA  ->  VecA must rank first
    res = await node_repo.search(
        NodeSearchQuery(
            repo_ids=[rid], embedding_query=[1.0, 0.0, 0.0] + [0] * 1021, limit=3
        )
    )

    assert res[0].name == "VecA"


@pytest.mark.asyncio
async def test_file_filename_complete(data_repo):
    repo_repo, file_repo = data_repo.repo, data_repo.file

    rid = make_id()
    await repo_repo.create([Repo(id=rid, name="fuzzy", root_path="/tmp/fuzzy")])

    f1 = File(id=make_id(), repo_id=rid, path="src/alpha/beta/cappa.py")
    f2 = File(id=make_id(), repo_id=rid, path="src/abc_utils.py")
    f3 = File(id=make_id(), repo_id=rid, path="src/random.py")
    f4 = File(id=make_id(), repo_id=rid, path="docs/AnotherBigCase.md")

    await file_repo.create([f1, f2, f3, f4])

    # “abc” should fuzzy-match both contiguous and subsequence-across-folders
    res = await file_repo.filename_complete("abc")
    paths = [f.path for f in res]

    assert any("alpha/beta/cappa.py" in p for p in paths)
    assert any("abc_utils.py" in p for p in paths)
    # default limit should cap results
    assert len(res) <= 5

    # --- verify repo_id filtering ---
    repo_repo = data_repo.repo
    rid2 = make_id()
    await repo_repo.create([Repo(id=rid2, name="fuzzy2", root_path="/tmp/fuzzy2")])
    f5 = File(id=make_id(), repo_id=rid2, path="src/abc_match.ts")
    await file_repo.create([f5])

    # Filter to first repo only: results should all have repo_id == rid
    res_r1 = await file_repo.filename_complete("abc", repo_ids=[rid])
    assert all(ff.repo_id == rid for ff in res_r1)

    # Filter to second repo only: the new file should be present
    res_r2 = await file_repo.filename_complete("abc", repo_ids=[rid2])
    assert any(ff.id == f5.id for ff in res_r2)
    assert all(ff.repo_id == rid2 for ff in res_r2)


@pytest.mark.asyncio
async def test_file_filename_complete_short_needles_return_empty(data_repo):
    repo_repo, file_repo = data_repo.repo, data_repo.file

    rid = make_id()
    await repo_repo.create([Repo(id=rid, name="short", root_path="/tmp/short")])

    f = File(id=make_id(), repo_id=rid, path="src/abc_utils.py")
    await file_repo.create([f])

    # 1- and 2-character needles should not crash and should return no results
    assert await file_repo.filename_complete("a") == []
    assert await file_repo.filename_complete("ab") == []

@pytest.mark.asyncio
async def test_file_index_sync_on_update(data_repo):
    repo_repo, file_repo = data_repo.repo, data_repo.file

    rid = make_id()
    await repo_repo.create([Repo(id=rid, name="upd", root_path="/tmp/upd")])

    fid = make_id()
    f = File(id=fid, repo_id=rid, path="src/abc_utils.py")
    await file_repo.create([f])

    # initial search finds the file
    res = await file_repo.filename_complete("abc")
    assert any(ff.id == fid for ff in res)

    # update path to something that should not match "abc"
    out = await file_repo.update([(fid, {"path": "src/zzz.py"})])
    assert out and out[0].path == "src/zzz.py"

    res = await file_repo.filename_complete("abc")
    assert all(ff.id != fid for ff in res)


@pytest.mark.asyncio
async def test_file_filename_complete_strict_subsequence(data_repo):
    repo_repo, file_repo = data_repo.repo, data_repo.file

    rid = make_id()
    await repo_repo.create([Repo(id=rid, name="strict", root_path="/tmp/strict")])

    # Expected match
    f_match = File(id=make_id(), repo_id=rid, path="tests/test_buf.py")
    # Near misses that should NOT match the subsequence "t e s t b u f . p y"
    f_near1 = File(id=make_id(), repo_id=rid, path="src/vocode/ui/terminal/buf.py")
    f_near2 = File(id=make_id(), repo_id=rid, path="tests/test_graph.py")
    f_near3 = File(id=make_id(), repo_id=rid, path="tests/test_runner.py")
    f_near4 = File(id=make_id(), repo_id=rid, path="src/vocode/testing.py")

    await file_repo.create([f_match, f_near1, f_near2, f_near3, f_near4])

    res = await file_repo.filename_complete("testbuf.py", limit=10)
    paths = [f.path for f in res]

    assert "tests/test_buf.py" in paths
    # Ensure non-subsequence matches are excluded
    assert "src/vocode/ui/terminal/buf.py" not in paths
    assert "tests/test_graph.py" not in paths
    assert "tests/test_runner.py" not in paths
    assert "src/vocode/testing.py" not in paths

    # update path to something that does match subsequence "a.*b.*c"
    out = await file_repo.update([(f_near1.id, {"path": "docs/alpha/beta/cappa.py"})])
    assert out and "alpha/beta/cappa.py" in out[0].path

    res = await file_repo.filename_complete("abc")
    assert any(ff.id == f_near1.id and "alpha/beta/cappa.py" in ff.path for ff in res)


@pytest.mark.asyncio
async def test_file_index_sync_on_delete(data_repo):
    repo_repo, file_repo = data_repo.repo, data_repo.file

    rid = make_id()
    await repo_repo.create([Repo(id=rid, name="del", root_path="/tmp/del")])

    fid = make_id()
    f = File(id=fid, repo_id=rid, path="src/abc_utils.py")
    await file_repo.create([f])

    # present before delete
    res = await file_repo.filename_complete("abc")
    assert any(ff.id == fid for ff in res)

    # delete and ensure it’s gone from index
    assert await file_repo.delete([fid]) is True

    res = await file_repo.filename_complete("abc")
    assert all(ff.id != fid for ff in res)
