"""Run the homework tests and add up the points of the automatic part (8 of 10; part 3, the report, is graded by
hand). A line scores only if all of its tests pass.

    uv run python tools/grade.py              # everything, ≈3 min on a laptop
    uv run python tools/grade.py -x           # extra arguments go to pytest
"""

import sys

import pytest

UNIT = "tests/test_my_fsdp.py::"
TRAIN = "tests/test_parallel.py::"
GROUPS = [  # (part, what, points, test id prefixes; "[" ends a name, so test_forward[ is not test_forward_prefetch)
    ("1.1", "sharding: DTensor shards, N=4, 8", 0.5, [f"{UNIT}test_shard["]),
    ("1.2", "forward: all-gather, free after forward, N=4, 8", 1.5, [f"{UNIT}test_forward["]),
    ("1.3", "backward: all-gather, reduce-scatter, gradient accumulation, N=4, 8", 1.0, [f"{UNIT}test_backward["]),
    ("1.3", "training == 1 process and FSDP2, N=4, 8", 1.5, [f"{TRAIN}test_mine_matches_single_process_and_fsdp2["]),
    ("1.4", "mixed precision bf16 == FSDP2", 0.5, [f"{UNIT}test_mixed_precision[", f"{TRAIN}test_mine_bf16_matches_fsdp2"]),
    ("2.1", "forward prefetch, N=4, 8", 1.0, [f"{UNIT}test_forward_prefetch["]),
    ("2.2", "backward prefetch, N=4, 8", 1.0, [f"{UNIT}test_backward_prefetch["]),
    ("2.3", "async reduce-scatter; training with overlap == 1 process and FSDP2", 1.0,
     [f"{UNIT}test_async_reduce_scatter[", f"{TRAIN}test_mine_with_overlap_matches_single_process_and_fsdp2["]),
]
AUTO = sum(points for _, _, points, _ in GROUPS)


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
    pytest.main(["-q", "tests/test_my_fsdp.py", f"{TRAIN}test_mine_matches_single_process_and_fsdp2",
                 f"{TRAIN}test_mine_with_overlap_matches_single_process_and_fsdp2", f"{TRAIN}test_mine_bf16_matches_fsdp2",
                 *sys.argv[1:]], plugins=[outcomes])
    total = 0.0
    print()
    for part, what, points, prefixes in GROUPS:
        tests = [t for t in outcomes.collected if any(t.startswith(p) for p in prefixes)]
        ok = bool(tests) and all(t in outcomes.passed and t not in outcomes.failed for t in tests)
        total += points * ok
        score = f"{points * ok:g} / {points:g}"
        print(f"part {part}  {score:<9} {what}  ({sum(t in outcomes.passed for t in tests)}/{len(tests)} tests)")
    print("part 3    report: comparison with FSDP2 (2 points), graded by hand")
    print(f"total: {total:g} / {AUTO:g} (automatic part)")
