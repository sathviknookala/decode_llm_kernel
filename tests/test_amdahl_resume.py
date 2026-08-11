import pytest

from benchmarks.benchmark_utils import ResumableRun, env_path, write_json
from benchmarks.decode_loop import DecodeConfig
from benchmarks.probe_amdahl import point_measured, unit_key

META = {"git_sha": "a" * 40}


def cfg(batch=32, ctx=128):
    return DecodeConfig("org/m", batch, ctx)


def row(rung, ms, *, error=""):
    return {"rung": rung, "amortized_step_ms": ms, "error": error}


def test_a_measured_point_is_keyed_by_its_whole_configuration():
    a = unit_key(cfg(), "hf_static_graph", "full")
    assert a != unit_key(cfg(ctx=256), "hf_static_graph", "full")
    assert a != unit_key(cfg(), "hf_eager", "full")
    assert a != unit_key(cfg(), "hf_static_graph", "op_removed")


def test_a_point_with_a_timing_is_measured():
    assert point_measured([row("full", 61.9)])


def test_a_point_without_a_timing_is_retried_rather_than_inherited():
    """An OOM or a compile failure left a row behind; resuming must re-measure it, not treat
    the failure as a result."""
    assert not point_measured([row("full", "")])
    assert point_measured([row("op_removed", "", error="OOM")]) is False


def test_a_skipped_config_is_not_retried():
    """A configuration that does not fit in the budget will not fit on the next pass either."""
    assert point_measured([{"rung": "skipped", "amortized_step_ms": ""}])


@pytest.fixture
def partial(tmp_path):
    """The killed run this was written for: four complete configs and part of a fifth."""
    out = str(tmp_path / "amdahl_probe.csv")
    run = ResumableRun(out, META, unit_ok=point_measured)
    for ctx in (128, 256, 512):
        for rung in ("full", "op_removed", "op_doubled"):
            run.add(unit_key(cfg(ctx=ctx), "hf_static_graph", rung), [row(rung, 60.0)])
    for rung in ("full", "op_removed"):
        run.add(unit_key(cfg(ctx=1024), "hf_static_graph", rung), [row(rung, 27.9)])
    return out


def test_only_the_unmeasured_points_are_re_run(partial):
    run = ResumableRun(partial, META, resume=True, unit_ok=point_measured)
    assert len(run.rows) == 11
    assert run.done(unit_key(cfg(ctx=1024), "hf_static_graph", "op_removed"))
    assert not run.done(unit_key(cfg(ctx=1024), "hf_static_graph", "op_doubled"))


def test_the_full_rung_baseline_is_recoverable_from_an_inherited_row(partial):
    """A resumed config whose `full` rung is inherited still needs its timing: every saving in
    the file is against it."""
    run = ResumableRun(partial, META, resume=True, unit_ok=point_measured)
    key = unit_key(cfg(ctx=1024), "hf_static_graph", "full")
    prior = next(r for r in run.rows if r["unit_key"] == key and r["rung"] == "full")
    assert float(prior["amortized_step_ms"]) == 27.9


def test_resuming_across_two_trees_is_refused(partial):
    write_json(env_path(partial), {"environment": {"git_sha": "b" * 40}})
    with pytest.raises(SystemExit, match="two trees"):
        ResumableRun(partial, META, resume=True, unit_ok=point_measured)
