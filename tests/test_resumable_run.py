import json
import os

import pytest

from benchmarks.benchmark_utils import (
    REPO_ROOT,
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


def test_resuming_onto_the_same_commit_with_a_different_working_tree_is_refused(out):
    """A sha identifies the commit, not the tree. Editing a probe without committing left the
    sha equal on both sides, so the check passed and one CSV held rows from two behaviours."""
    ResumableRun(out, {**META, "git_tree_digest": "d" * 64}).add("cfg_a", ladder("cfg_a"))
    with pytest.raises(SystemExit, match="different working tree"):
        ResumableRun(out, {**META, "git_tree_digest": "e" * 64}, resume=True)


def test_an_unchanged_dirty_tree_still_resumes(out):
    """Refusing every dirty tree would make resume useless mid-development; the digest is what
    distinguishes 'uncommitted' from 'changed since those rows were measured'."""
    meta = {**META, "git_dirty": True, "git_tree_digest": "d" * 64}
    ResumableRun(out, meta).add("cfg_a", ladder("cfg_a"))
    assert ResumableRun(out, meta, resume=True).done("cfg_a")


def test_rows_measured_dirty_without_a_digest_are_refused(out):
    """Nothing recoverable identifies the tree behind them, and default-deny is the safe
    direction: a spurious refusal is loud, a missed one is silent."""
    ResumableRun(out, {**META, "git_dirty": True}).add("cfg_a", ladder("cfg_a"))
    with pytest.raises(SystemExit, match="cannot be identified"):
        ResumableRun(out, META, resume=True)


def test_rows_measured_clean_without_a_digest_are_not_second_guessed(out):
    """Every committed artifact predates the digest; a clean sha does identify its tree."""
    ResumableRun(out, META).add("cfg_a", ladder("cfg_a"))
    assert ResumableRun(out, {**META, "git_tree_digest": "d" * 64}, resume=True).done("cfg_a")


def test_the_tree_digest_moves_with_an_uncommitted_edit():
    """The digest is only worth checking if it actually tracks the working tree."""
    from benchmarks.benchmark_utils import _git_metadata
    before = _git_metadata()
    scratch = os.path.join(REPO_ROOT, ".resume_digest_probe.py")
    with open(scratch, "w") as f:
        f.write("# scratch\n")
    try:
        after = _git_metadata()
    finally:
        os.unlink(scratch)
    assert before["git_sha"] == after["git_sha"]
    assert before["git_tree_digest"] != after["git_tree_digest"]
    assert _git_metadata()["git_tree_digest"] == before["git_tree_digest"]


def test_a_flat_env_json_still_yields_its_git_sha(tmp_path):
    """Regression: the committed amdahl_probe.env.json is flat, not {"environment": {...}}, and
    a reader that only looked under "environment" found no sha and allowed the resume it was
    written to refuse."""
    path = str(tmp_path / "probe.csv")
    write_json(env_path(path), {"timestamp_utc": "2026-01-01", "git_sha": "a" * 40})
    ok, why = resume_is_safe(path, {"git_sha": "b" * 40})
    assert ok is False
    assert "two trees" in why


def args_meta(**cli):
    return {"git_sha": "a" * 40, "cli_args": {"out": "p.csv", "resume": False,
                                              "iters": 100, "warmup": 25, **cli}}


def test_resuming_under_different_measurement_settings_is_refused(out):
    """The tree check catches a changed tree; without this, a run killed at --iters 2000 and
    resumed at --iters 25 produced one CSV holding both, with env.json reporting only 25."""
    ResumableRun(out, args_meta(), unit_ok=lambda r: True).add("cfg_a", ladder("cfg_a"))
    with pytest.raises(SystemExit, match="iters 100 -> 25"):
        ResumableRun(out, args_meta(iters=25), resume=True)


def test_the_refusal_names_every_setting_that_moved(out):
    ResumableRun(out, args_meta()).add("cfg_a", ladder("cfg_a"))
    with pytest.raises(SystemExit) as e:
        ResumableRun(out, args_meta(iters=25, warmup=5), resume=True)
    assert "iters" in str(e.value) and "warmup" in str(e.value)


def test_a_new_setting_is_guarded_the_day_it_is_added(out):
    """Default-deny: an argument nothing has thought about yet still blocks a resume, rather
    than silently changing what the later rows mean."""
    ResumableRun(out, args_meta()).add("cfg_a", ladder("cfg_a"))
    with pytest.raises(SystemExit, match="fresh_args"):
        ResumableRun(out, args_meta(fresh_args=True), resume=True)


def test_the_output_path_and_the_resume_flag_are_not_measurement_settings(out):
    """The first run is launched without --resume and usually names its own --out; refusing on
    those would refuse every resume there is."""
    ResumableRun(out, args_meta()).add("cfg_a", ladder("cfg_a"))
    resumed = ResumableRun(out, {"git_sha": "a" * 40,
                                 "cli_args": {"out": "elsewhere.csv", "resume": True,
                                              "iters": 100, "warmup": 25}}, resume=True)
    assert resumed.done("cfg_a")


def test_a_prior_run_that_recorded_no_arguments_is_not_second_guessed(out):
    """Absence of a record is not evidence the settings differ, and the committed artifacts
    predate this check."""
    ResumableRun(out, {"git_sha": "a" * 40}).add("cfg_a", ladder("cfg_a"))
    assert ResumableRun(out, args_meta(), resume=True).done("cfg_a")


def test_a_resumed_run_can_read_what_the_earlier_one_recorded_about_itself(out):
    """A reference the inherited rows were scored against has to be recoverable, or the resumed
    run will derive a second one and the file will carry both."""
    first = ResumableRun(out, {**args_meta(), "bandwidth_reference": {"reference_gbps": 553.0}})
    first.add("cfg_a", ladder("cfg_a"))
    second = ResumableRun(out, args_meta(), resume=True)
    assert second.prior_meta["bandwidth_reference"]["reference_gbps"] == 553.0


def test_a_fresh_run_has_no_prior_provenance_to_read(out):
    assert ResumableRun(out, args_meta()).prior_meta == {}


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


def test_a_checkpoint_is_durable_before_it_replaces_the_last_one(out, monkeypatch):
    """os.replace only orders the rename against data that is already on the platter. Without
    the fsync a host crash can land the rename over blocks still in the page cache."""
    synced, replaced = [], []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(len(replaced)), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (replaced.append(b), real_replace(a, b))[1])

    write_csv(out, [{"unit": "cfg_a", "ms": 1.0}])

    assert replaced == [out]
    assert synced == [0, 1], "expected the file fsynced before the rename and the dir after"


def test_a_directory_that_refuses_fsync_does_not_fail_the_checkpoint(out, monkeypatch):
    """Some filesystems reject fsync on a directory fd; that must not lose the measurement."""
    real_fsync = os.fsync

    def picky(fd):
        if os.fstat(fd).st_mode & 0o040000:
            raise OSError("directory fsync unsupported")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", picky)
    write_csv(out, [{"unit": "cfg_a", "ms": 1.0}])
    assert read_rows(out)[0]["unit"] == "cfg_a"
