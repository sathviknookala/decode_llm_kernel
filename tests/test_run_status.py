import json

import pytest

from benchmarks.benchmark_utils import ResumableRun, lock_path, write_csv
from benchmarks.run_status import render, status

TREE = {"git_sha": "a" * 40, "git_tree_digest": "d" * 64}
META = {**TREE, "timestamp_utc": "2026-08-15T12:00:00+00:00",
        "cli_args": {"out": "p.csv", "resume": False, "iters": 100}}


@pytest.fixture
def out(tmp_path):
    return str(tmp_path / "probe.csv")


def partial(out, done=3, planned=6):
    run = ResumableRun(out, META)
    run.declare([f"cfg_{i}" for i in range(planned)])
    for i in range(done):
        run.add(f"cfg_{i}", [{"unit": f"cfg_{i}", "ms": 1.0}])
    return run


def test_a_killed_run_reports_what_is_outstanding(out):
    """The point of the tool: this answer used to cost a relaunch of the run itself."""
    partial(out)
    s = status(out, tree=TREE)
    assert s["complete"] is False
    assert s["units"] == 3
    assert s["units_planned"] == 6
    assert s["units_missing"] == ["cfg_3", "cfg_4", "cfg_5"]
    assert s["resumable"] is True


def test_a_finished_run_reports_coverage(out):
    run = partial(out, done=6, planned=6)
    run.finish()
    s = status(out, tree=TREE)
    assert (s["complete"], s["covered"]) == (True, True)
    assert s["units_missing"] == []


def test_the_verdict_repeats_the_refusal_the_resume_would_give(out):
    partial(out)
    s = status(out, tree={"git_sha": "b" * 40})
    assert s["resumable"] is False
    assert "two trees" in s["resume_detail"]
    assert "REFUSED" in render(s)


def test_rows_predating_unit_keys_are_named_as_such(out):
    """Every committed artifact is this case, and '0 finished units' alone reads as a failure
    rather than as a file written before restart boundaries existed."""
    write_csv(out, [{"unit": "cfg_a", "ms": 1.0}])
    s = status(out, tree=TREE)
    assert (s["units"], s["unkeyed_rows"]) == (0, 1)
    assert "predating unit keys" in render(s)


def test_a_held_lock_is_reported_with_its_owner(out):
    partial(out)
    with open(lock_path(out), "w") as f:
        json.dump({"pid": 999_999, "host": "otherbox", "since": "earlier"}, f)
    assert "pid 999999 on otherbox" in render(status(out, tree=TREE))


def test_a_probe_that_declared_no_plan_says_so_rather_than_implying_full_coverage(out):
    run = ResumableRun(out, META)
    run.add("cfg_a", [{"unit": "cfg_a", "ms": 1.0}])
    run.finish()
    assert "coverage: not declared" in render(status(out, tree=TREE))


def test_the_missing_list_is_truncated_unless_asked(out):
    partial(out, done=1, planned=30)
    s = status(out, tree=TREE)
    assert "and 21 more" in render(s)
    assert "and 21 more" not in render(s, show_all_missing=True)


def test_a_run_killed_before_its_first_unit_still_says_what_it_intended(out):
    """Otherwise there is nothing on disk at all, and a run killed in warmup cannot be told
    from one that was never launched."""
    partial(out, done=0, planned=6)
    s = status(out, tree=TREE)
    assert s["complete"] is False
    assert s["units_planned"] == 6
    assert len(s["units_missing"]) == 6


def test_resume_segments_are_reported_once_there_is_more_than_one(out):
    partial(out, done=1, planned=6)
    ResumableRun(out, META, resume=True).add("cfg_9", [{"unit": "cfg_9", "ms": 1.0}])
    text = render(status(out, tree=TREE))
    assert "written across 2 processes" in text
    assert "segment 1: 1 rows" in text


def test_units_that_would_not_survive_a_resume_are_reported_as_such(out):
    """run_status counted every keyed unit, ignoring the probe's unit_ok, so it answered
    '64 units, would resume' for a file whose resume re-measures three of them."""
    timed = lambda rows: any(r["ms"] for r in rows)
    run = ResumableRun(out, META, unit_ok=timed)
    run.declare(["cfg_a", "cfg_b"])
    run.add("cfg_a", [{"unit": "cfg_a", "ms": 1.0}])
    run.add("cfg_b", [{"unit": "cfg_b", "ms": ""}])
    run.finish()

    s = status(out, tree=TREE)
    assert s["units"] == 2
    assert s["units_inheritable"] == 1
    assert "only 1 of those would be inherited" in render(s)


def test_a_file_whose_units_all_survive_says_nothing_extra(out):
    run = partial(out, done=2, planned=2)
    run.finish()
    s = status(out, tree=TREE)
    assert s["units_inheritable"] == 2
    assert "would be inherited" not in render(s)
