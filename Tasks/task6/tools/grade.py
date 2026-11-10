"""Run the homework tests and add up the points (10 in total). A line scores only if all of its tests pass.

    uv run python tools/grade.py              # everything, ~5 min on a laptop
    uv run python tools/grade.py -x           # extra arguments go to pytest
"""

import sys

import pytest

PARALLEL = "tests/test_parallel.py::test_cp_matches_single_process"
GROUPS = [  # (part, what, points, test id prefixes)
    (1, "cp_shard", 0.5, ["tests/test_cp.py::test_cp_shard"]),
    (1, "allgather_attention: forward + backward, N=4, 8", 1.0, ["tests/test_cp.py::test_allgather_attention"]),
    (1, "training with allgather == 1 process", 1.5, [f"{PARALLEL}[allgather-"]),
    (2, "ring_attention: forward + backward, N=4, 8, P2P only", 1.0, ["tests/test_cp.py::test_ring_attention"]),
    (2, "training with ring == 1 process", 1.0, [f"{PARALLEL}[ring-"]),
    (3, "all-to-all: equal splits", 1.0, ["tests/test_cp.py::test_a2a_equal_splits"]),
    (3, "all-to-all: uneven and empty splits, subgroups", 1.0, ["tests/test_cp.py::test_a2a_uneven_splits_and_subgroups"]),
    (3, "all-to-all: autograd", 1.0, ["tests/test_cp.py::test_a2a_autograd"]),
    (4, "ulysses_attention: forward + backward, N=4, 8", 1.0, ["tests/test_cp.py::test_ulysses_attention"]),
    (4, "training with ulysses == 1 process", 1.0, [f"{PARALLEL}[ulysses-"]),
]


class Outcomes:
    def __init__(self):
        self.collected, self.passed, self.failed = [], set(), set()

    def pytest_collection_modifyitems(self, items):
        self.collected = [item.nodeid for item in items]

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.failed.add(report.nodeid)
        elif report.when == "call" and report.passed:
            self.passed.add(report.nodeid)


if __name__ == "__main__":
    outcomes = Outcomes()
    pytest.main(["-q", "tests/test_cp.py", PARALLEL, *sys.argv[1:]], plugins=[outcomes])
    total = 0.0
    print()
    for part, what, points, prefixes in GROUPS:
        tests = [t for t in outcomes.collected if any(t.startswith(p) for p in prefixes)]
        ok = bool(tests) and all(t in outcomes.passed and t not in outcomes.failed for t in tests)
        total += points * ok
        print(f"part {part}  {'+' if ok else '-'}{points:<4} {what}  ({sum(t in outcomes.passed for t in tests)}/{len(tests)} tests)")
    print(f"total: {total:g} / 10")
