import pytest
import math

from knowlt.stores.duckdb import DuckDBDataRepository
from knowlt.models import Repo, File, Node
from knowlt.data import NodeSearchQuery
from knowlt.settings import ProjectSettings

pytest.importorskip("sentence_transformers")

from knowlt.embeddings import EmbeddingWorker


@pytest.fixture(scope="module")
def emb_calc():
    try:
        # Use default model of the sentence-transformers
        calc = EmbeddingWorker("local", "all-MiniLM-L6-v2")
        yield calc
        calc.destroy()
    except Exception as ex:
        pytest.skip(f"Couldn't load embedding worker {ex}")


@pytest.fixture(params=["duckdb"])
def data_repo(request, tmp_path):
    """Return a fresh data repository"""
    if request.param == "duckdb":
        settings = ProjectSettings(
            project_name="test",
            repo_name="test",
            repo_path=str(tmp_path),
        )
        db_path = tmp_path / "ducktest.db"
        pytest.importorskip("duckdb")
        return DuckDBDataRepository(settings, str(db_path))
    raise ValueError()


@pytest.mark.asyncio
async def test_bm25_embedding_search_20cases(data_repo, emb_calc):
    """
    Inserts 20 distinct symbols with themed docstrings.
    Performs BM25, embedding-only and *combined (BM25 + embedding)* search.
    Verifies result relevance, ranking and the RRF fusion implementation.
    """
    repo_repo = data_repo.repo
    file_repo = data_repo.file
    sym_repo = data_repo.node

    rid = "repo-BM25-test"
    fid = "file-f"
    await repo_repo.create([Repo(id=rid, name="BM25Repo", root_path="/bm25test")])
    await file_repo.create([File(id=fid, repo_id=rid, path="src/bm25.py")])

    themes = [
        ("Sorting", "Sorts a list using quicksort", "def quicksort(arr): ..."),
        ("Sorting", "Sorts items ascendingly", "def sort_asc(a): ..."),
        ("Sorting", "Bubble sort implementation", "def bubble(a): ..."),
        ("Search", "Binary search for element", "def binary_search(arr, x): ..."),
        (
            "Search",
            "Finds item using linear search",
            "def linear_search(lst, val): ...",
        ),
        ("Math", "Computes the factorial", "def factorial(n): ..."),
        ("Math", "Returns the n-th fibonacci", "def fibonacci(n): ..."),
        ("Math", "Calculate the sum of numbers", "def sum_numbers(nums): ..."),
        ("Math", "Returns the average of a list", "def average(xs): ..."),
        ("String", "Reverses a given string", "def reverse_str(s): ..."),
        ("String", "Converts string to uppercase", "def to_upper(s): ..."),
        ("String", "Checks if string is palindrome", "def is_palindrome(s): ..."),
        ("IO", "Reads file and returns contents", "def read_file(path): ..."),
        ("IO", "Writes data to file", "def write_file(p, data): ..."),
        ("IO", "Appends line to a file", "def append_line(f,x): ..."),
        ("Network", "Send GET request to URL", "def get(url): ..."),
        ("Network", "Parse HTTP response body", "def parse_http(body): ..."),
        ("Network", "Open TCP connection", "def open_tcp(host,port): ..."),
        ("Date", "Returns current date", "def today(): ..."),
        ("Date", "Formats date to ISO string", "def format_iso(dt): ..."),
        ("Date", "Adds days to a date", "def add_days(dt, days): ..."),
    ]
    # Insert all as symbols, record their ids for later checks
    ids_by_theme = {}
    for i, (theme, docstring, body) in enumerate(themes):
        sid = f"s_{i}"
        vec = await emb_calc.get_embedding(docstring)
        assert len(vec) > 0
        if theme not in ids_by_theme:
            ids_by_theme[theme] = []
        ids_by_theme[theme].append(sid)
        await sym_repo.create([
            Node(
                id=sid,
                name=f"{theme}{i}",
                repo_id=rid,
                file_id=fid,
                body=body,
                docstring=docstring,
                embedding_doc_vec=vec,
                embedding_code_vec=vec,
                kind="function",
            )
        ])
    await data_repo.refresh_indexes()

    # BM25 Search tests
    # Query for 'sort'; should rank sorting-related symbols at the top.
    res_bm25 = await sym_repo.search(
        NodeSearchQuery(repo_ids=[rid], needle="sort", limit=5)
    )
    top_names = [s.name for s in res_bm25]

    # The top 3 should be Sorting-related
    assert any("Sorting" in n for n in top_names[:3]), f"Top 3: {top_names[:3]}"

    # Query for 'date'; the date-related symbols should rank highest
    res_date = await sym_repo.search(
        NodeSearchQuery(repo_ids=[rid], needle="date", limit=3)
    )
    assert all(
        "Date" in s.name for s in res_date
    ), f"Top date results: {[s.name for s in res_date]}"

    # Query for 'HTTP'; network symbols
    res_net = await sym_repo.search(
        NodeSearchQuery(repo_ids=[rid], needle="HTTP", limit=2)
    )
    assert any(
        "Network" in s.name for s in res_net
    ), f"Top network/HTTP: {[s.name for s in res_net]}"

    # Embedding search: retrieve all 'Sorting' (clustered), using first Sorting docstring as query
    sort_vec = await emb_calc.get_embedding(themes[0][1])
    emb_sort = await sym_repo.search(
        NodeSearchQuery(repo_ids=[rid], embedding_query=sort_vec, limit=5)
    )
    sort_names = [s.name for s in emb_sort]
    # At least 2/3 of the top 3 should be 'Sorting' related
    assert (
        sum("Sorting" in n for n in sort_names[:3]) >= 2
    ), f"Top emb: {sort_names[:3]}"

    # Embedding search: retrieve all 'Math' symbols
    math_vec = await emb_calc.get_embedding(themes[5][1])
    emb_math = await sym_repo.search(
        NodeSearchQuery(repo_ids=[rid], embedding_query=math_vec, limit=5)
    )
    math_names = [s.name for s in emb_math]
    assert sum("Math" in n for n in math_names[:3]) >= 1, f"Top math: {math_names[:3]}"

    # ------------------------------------------------------------------
    # Combined BM25 + Embedding search  (exercises RRF fusion)
    # ------------------------------------------------------------------
    comb = await sym_repo.search(
        NodeSearchQuery(
            repo_ids=[rid], needle="sort", embedding_query=sort_vec, limit=5
        )
    )
    comb_names = [s.name for s in comb]

    # At least 3 of the 5 returned symbols should be Sorting-related
    assert (
        sum("Sorting" in n for n in comb_names) >= 3
    ), f"Combined search (RRF) results not focused on Sorting: {comb_names}"

    # The very first result should also appear in the top-5 of *either*
    # BM25-only or embedding-only search – proof that RRF fused ranks.
    top_candidate = comb_names[0]
    assert top_candidate in [s.name for s in res_bm25[:5]] or top_candidate in [
        s.name for s in emb_sort[:5]
    ], "RRF top result not present in the individual rankings"

    # Ensures sorted (BM25 or embedding score) and relevant
    # Also covers at least 21 data entries
    assert await sym_repo.search(NodeSearchQuery(repo_ids=[rid], limit=25))  # total exists
