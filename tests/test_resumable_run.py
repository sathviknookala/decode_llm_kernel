import json

import pytest

from benchmarks.benchmark_utils import (
    UNIT_KEY,
    ResumableRun,
    env_path,
    read_rows,
    read_run_completeness,
    resume_is_safe,
    write_csv,
    write_json,
)

META = {"git_sha": "a" * 40, "gpu_name": "test"}


@pytest.fixture
def out(tmp_path):
    return str(tmp_path / "probe.csv")


def ladder(unit, n=3, ms=1.0):
    return [{"unit": unit, "rung": i, "ms": ms} for i in range(n)]


def test_a_unit_reaches_disk_as_soon_as_it_closes(out):
    """The whole point: five scripts used to hold every row in a list until the last config,
    so a kill at 85% lost 100% of the measurements rather than the tail."""
    run = ResumableRun(out, META)
    run.add("cfg_a", ladder("cfg_a"))
    assert len(read_rows(out)) == 3
    run.add("cfg_b", ladder("cfg_b"))
    assert len(read_rows(out)) == 6


def test_only_whole_units_are_ever_readable(out):
    """A bracketed ladder half on disk would be read as a finished ladder by every consumer,
    and its ordering_drift would be computed from one baseline rung instead of two."""
    run = ResumableRun(out, META)
    rows = ladder("cfg_a")
    for r in rows[:2]:
        r["ms"] = 2.0
    run.add("cfg_a", rows)
    on_disk = read_rows(out)
    assert {r[UNIT_KEY] for r in on_disk} == {"cfg_a"}
    assert len(on_disk) == 3


def test_provenance_lands_with_the_first_unit_and_says_the_run_is_unfinished(out):
    run = ResumableRun(out, META)
    run.add("cfg_a", ladder("cfg_a"))
    assert read_run_completeness(env_path(out)) is False
    doc = json.load(open(env_path(out)))
    assert doc["environment"]["gpu_name"] == "test"
    assert doc["rows_written"] == 3
    assert doc["units_written"] == 1


def test_finish_marks_the_run_complete(out):
    run = ResumableRun(out, META)
    run.add("cfg_a", ladder("cfg_a"))
    run.finish()
    assert read_run_completeness(env_path(out)) is True


def test_sidecar_extras_survive_every_checkpoint(out):
    run = ResumableRun(out, META, sidecar={"slot_ladder": [1, 8, 32]})
    run.add("cfg_a", ladder("cfg_a"))
    assert json.load(open(env_path(out)))["slot_ladder"] == [1, 8, 32]
    run.finish({"summary": {"rows_total": 3}})
    doc = json.load(open(env_path(out)))
    assert doc["slot_ladder"] == [1, 8, 32]
    assert doc["summary"]["rows_total"] == 3


def test_resume_inherits_finished_units_and_leaves_the_rest_to_measure(out):
    first = ResumableRun(out, META)
    first.add("cfg_a", ladder("cfg_a"))
    first.add("cfg_b", ladder("cfg_b"))

    second = ResumableRun(out, META, resume=True)
    assert second.done("cfg_a") and second.done("cfg_b")
    assert not second.done("cfg_c")
    assert len(second.rows) == 6


def test_inherited_rows_are_carried_forward_verbatim(out):
    """A resumed run rewrites the whole CSV; the earlier measurements must come back out of it
    unchanged, not reformatted or recomputed."""
    first = ResumableRun(out, META)
    first.add("cfg_a", [{"unit": "cfg_a", "rung": 0, "ms": 0.012590000000001}])
    first.finish()

    second = ResumableRun(out, META, resume=True)
    second.add("cfg_b", [{"unit": "cfg_b", "rung": 0, "ms": 1.0}])
    second.finish()

    rows = read_rows(out)
    assert [r["unit"] for r in rows] == ["cfg_a", "cfg_b"]
    assert rows[0]["ms"] == "0.012590000000001"


def test_a_unit_that_failed_is_retried_rather_than_inherited(out):
    """An OOM or a compile failure is not a result. unit_ok is where each script says so."""
    first = ResumableRun(out, META, unit_ok=lambda rows: any(r["ms"] for r in rows))
    first.add("cfg_a", [{"unit": "cfg_a", "rung": 0, "ms": 1.0}])
    first.add("cfg_b", [{"unit": "cfg_b", "rung": 0, "ms": ""}])

    second = ResumableRun(out, META, resume=True,
                          unit_ok=lambda rows: any(r["ms"] for r in rows))
    assert second.done("cfg_a")
    assert not second.done("cfg_b")
    assert len(second.rows) == 1


def test_a_retried_unit_does_not_leave_its_failed_rows_behind(out):
    first = ResumableRun(out, META, unit_ok=lambda rows: any(r["ms"] for r in rows))
    first.add("cfg_a", [{"unit": "cfg_a", "rung": 0, "ms": ""}])

    second = ResumableRun(out, META, resume=True,
                          unit_ok=lambda rows: any(r["ms"] for r in rows))
    second.add("cfg_a", [{"unit": "cfg_a", "rung": 0, "ms": 2.0}])
    second.finish()

    rows = read_rows(out)
    assert len(rows) == 1
    assert rows[0]["ms"] == "2.0"


def test_resuming_across_two_trees_is_refused(out):
    """Rows from two source trees in one CSV would be compared against each other by every
    consumer of the file, with nothing marking the boundary."""
    ResumableRun(out, META).add("cfg_a", ladder("cfg_a"))
    with pytest.raises(SystemExit, match="two trees"):
        ResumableRun(out, {"git_sha": "b" * 40}, resume=True)


def test_a_flat_env_json_still_yields_its_git_sha(tmp_path):
    """Regression: the committed amdahl_probe.env.json is flat, not {"environment": {...}}, and
    a reader that only looked under "environment" found no sha and allowed the resume it was
    written to refuse."""
    path = str(tmp_path / "probe.csv")
    write_json(env_path(path), {"timestamp_utc": "2026-01-01", "git_sha": "a" * 40})
    ok, why = resume_is_safe(path, {"git_sha": "b" * 40})
    assert ok is False
    assert "two trees" in why


def test_a_fresh_output_path_resumes_trivially(out):
    run = ResumableRun(out, META, resume=True)
    assert run.rows == []
    assert not run.done("cfg_a")


def test_a_csv_predating_unit_keys_is_measured_again_whole(out):
    """Every committed artifact was written before restart boundaries were recorded, so there
    is no way to tell which of its rows belong to a finished unit."""
    write_csv(out, [{"unit": "cfg_a", "rung": 0, "ms": 1.0}])
    run = ResumableRun(out, META, resume=True)
    assert run.rows == []
    assert not run.done("cfg_a")


def test_a_sidecar_lagging_behind_the_csv_still_resumes(out):
    """Completeness is read from the CSV's own unit_key column, so a crash landing between the
    two writes costs nothing."""
    first = ResumableRun(out, META)
    first.add("cfg_a", ladder("cfg_a"))
    write_json(env_path(out), {"environment": META, "complete": False, "rows_written": 0})

    second = ResumableRun(out, META, resume=True)
    assert second.done("cfg_a")
    assert len(second.rows) == 3


class Unwritable:
    def __str__(self):
        raise RuntimeError("boom")


def test_a_failed_write_leaves_the_previous_csv_intact(out):
    """Checkpointing rewrites the whole file once per unit. Without the temp-file rename, a
    crash inside that rewrite would destroy every row already measured."""
    write_csv(out, [{"unit": "cfg_a", "rung": 0, "ms": 1.0}])
    with pytest.raises(RuntimeError, match="boom"):
        write_csv(out, [{"unit": "cfg_b", "rung": 0, "ms": Unwritable()}])
    rows = read_rows(out)
    assert len(rows) == 1
    assert rows[0]["unit"] == "cfg_a"


def test_a_failed_write_leaves_no_temp_file_behind(out):
    with pytest.raises(RuntimeError, match="boom"):
        write_csv(out, [{"unit": "cfg_a", "ms": Unwritable()}])
    import os
    assert not os.path.exists(out + ".tmp")
