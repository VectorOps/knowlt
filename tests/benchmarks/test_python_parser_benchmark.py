import os
import time
from pathlib import Path

import pytest

from knowlt.lang.python import PythonCodeParser
from knowlt.models import Repo
from knowlt.project import ProjectCache
from knowlt.settings import ProjectSettings


class _DummyPM:
    def __init__(self, settings: ProjectSettings):
        self.settings = settings


def _build_parser(sample_root: Path, rel_path: str) -> PythonCodeParser:
    settings = ProjectSettings(
        project_name="benchmark",
        repo_name="benchmark",
        repo_path=str(sample_root),
    )
    pm = _DummyPM(settings)
    repo = Repo(id="benchmark", name="benchmark", root_path=str(sample_root))
    return PythonCodeParser(pm, repo, rel_path)


@pytest.mark.skipif(
    os.environ.get("KNOWLT_RUN_BENCHMARKS") != "1",
    reason="Set KNOWLT_RUN_BENCHMARKS=1 to run parser benchmarks",
)
def test_python_parser_benchmark_baseline():
    """
    Opt-in baseline benchmark for the current Python parser.

    This is intentionally lightweight and uses an existing parser fixture so it
    can live in the test suite without requiring extra dependencies.
    """
    samples_dir = Path(__file__).resolve().parents[1] / "lang" / "python" / "samples"
    rel_path = "simple.py"
    iterations = 200

    parser = _build_parser(samples_dir, rel_path)

    warmup_cache = ProjectCache()
    warmup_result = parser.parse(warmup_cache)
    assert warmup_result.path == rel_path
    assert warmup_result.nodes

    durations: list[float] = []
    for _ in range(iterations):
        parser = _build_parser(samples_dir, rel_path)
        cache = ProjectCache()
        start = time.perf_counter()
        parsed_file = parser.parse(cache)
        durations.append(time.perf_counter() - start)
        assert parsed_file.path == rel_path

    total = sum(durations)
    avg_ms = (total / iterations) * 1000
    min_ms = min(durations) * 1000
    max_ms = max(durations) * 1000

    print(
        (
            "python_parser_benchmark "
            f"iterations={iterations} "
            f"total_s={total:.6f} "
            f"avg_ms={avg_ms:.3f} "
            f"min_ms={min_ms:.3f} "
            f"max_ms={max_ms:.3f}"
        )
    )
