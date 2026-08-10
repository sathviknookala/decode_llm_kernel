import csv

from benchmarks.benchmark_utils import write_json
from benchmarks.probe_amdahl import (
    completed_keys,
    env_path,
    prior_rows,
    resume_is_safe,
    row_key,
)

COLUMNS = ["model_id", "batch", "ctx", "mode", "rung", "amortized_step_ms", "error"]


def row(rung, ms, *, mode="hf_static_graph", batch=32, ctx=128, error=""):
    return {"model_id": "m", "batch": str(batch), "ctx": str(ctx), "mode": mode,
            "rung": rung, "amortized_step_ms": ms, "error": error}


def write(tmp_path, rows, name="probe.csv"):
    path = str(tmp_path / name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def test_a_measured_point_is_keyed_by_its_whole_configuration():
    a = row_key(row("full", "61.9"))
    assert a != row_key(row("full", "61.9", ctx=256))
    assert a != row_key(row("full", "61.9", mode="hf_eager"))
    assert a != row_key(row("op_removed", "60.0"))


def test_completed_points_are_skipped_on_resume(tmp_path):
    path = write(tmp_path, [row("full", "61.9"), row("op_removed", "60.0")])
    done = completed_keys(path)
    assert row_key(row("full", "")) in done
    assert row_key(row("op_doubled", "")) not in done


def test_a_row_without_a_timing_is_retried_rather_than_inherited(tmp_path):
    """An OOM or a compile failure left a row behind; resuming must re-measure it, not treat
    the failure as a result."""
    path = write(tmp_path, [row("full", ""), row("op_removed", "60.0", error="OOM")])
    done = completed_keys(path)
    assert row_key(row("full", "")) not in done
    assert row_key(row("op_removed", "")) in done


def test_a_skipped_config_is_not_retried(tmp_path):
    """A configuration that does not fit in the budget will not fit on the next pass either."""
    path = write(tmp_path, [row("skipped", "", mode="")])
    assert row_key(row("skipped", "", mode="")) in completed_keys(path)


def test_prior_rows_are_carried_forward_so_resume_appends(tmp_path):
    path = write(tmp_path, [row("full", "61.9"), row("op_removed", "60.0")])
    assert len(prior_rows(path)) == 2


def test_resuming_across_two_trees_is_refused(tmp_path):
    """Rows from two source trees in one CSV would be compared against each other by every
    consumer of the file, with nothing marking the boundary."""
    path = write(tmp_path, [row("full", "61.9")])
    write_json(env_path(path), {"environment": {"git_sha": "a" * 40}})
    ok, why = resume_is_safe(path, {"git_sha": "b" * 40})
    assert ok is False
    assert "two trees" in why


def test_resuming_onto_the_same_tree_is_allowed(tmp_path):
    path = write(tmp_path, [row("full", "61.9")])
    write_json(env_path(path), {"environment": {"git_sha": "a" * 40}})
    assert resume_is_safe(path, {"git_sha": "a" * 40})[0] is True


def test_a_fresh_output_path_resumes_trivially(tmp_path):
    assert completed_keys(str(tmp_path / "absent.csv")) == set()
    assert resume_is_safe(str(tmp_path / "absent.csv"), {"git_sha": "a"})[0] is True


def test_the_partial_run_this_was_written_for_would_resume(tmp_path):
    """The killed run had four complete configs and part of a fifth; only the unmeasured
    points should be re-run."""
    rows = [row(r, "60.0", ctx=c) for c in (128, 256, 512)
            for r in ("full", "op_removed", "op_doubled")]
    rows += [row("full", "27.9", ctx=1024), row("op_removed", "26.9", ctx=1024)]
    path = write(tmp_path, rows)
    done = completed_keys(path)
    assert len(done) == 11
    assert row_key(row("op_doubled", "", ctx=1024)) not in done
