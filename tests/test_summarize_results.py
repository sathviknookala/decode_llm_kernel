import csv

import pytest

from benchmarks.summarize_results import (
    bandwidth_column_sanity,
    build_summary,
    numerical_deltas,
    read_rows,
    graph_launch_share,
    head_dim_response,
    ragged_uniform_ratio,
    render_markdown,
    serving_layout_cost,
    speedup_by_batch,
)

COLUMNS = ["impl", "head_label", "num_q_heads", "num_kv_heads", "head_dim", "num_requests",
           "cache_alloc_len", "dtype_label", "position_mode", "layout", "request_mapping",
           "validation", "device_median_ms", "logical_write_bytes", "pct_of_empirical_bw",
           "pct_of_scattered_write_bw"]


def row(impl, ms, *, head="mha", hkv=32, d=128, b=1, alloc=2048, dtype="bf16",
        mode="ragged", layout="packed", mapping="identity", pct=10.0, spct=20.0,
        validation="pass"):
    return {"impl": impl, "head_label": head, "num_q_heads": "32", "num_kv_heads": str(hkv),
            "head_dim": str(d), "num_requests": str(b), "cache_alloc_len": str(alloc),
            "dtype_label": dtype, "position_mode": mode, "layout": layout,
            "request_mapping": mapping, "validation": validation,
            "device_median_ms": str(ms), "logical_write_bytes": "4096",
            "pct_of_empirical_bw": str(pct), "pct_of_scattered_write_bw": str(spct)}


def write_csv(tmp_path, rows, name="baseline.csv"):
    path = tmp_path / name
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return str(path)


def test_speedup_is_a_median_over_configs_not_a_single_pair():
    rows = [row("eager", 1.0, alloc=128), row("compile", 0.5, alloc=128),
            row("eager", 1.0, alloc=2048), row("compile", 0.1, alloc=2048),
            row("eager", 1.0, dtype="fp32"), row("compile", 0.25, dtype="fp32")]
    got = speedup_by_batch(rows)
    assert len(got) == 1
    assert got[0]["compile"] == pytest.approx(4.0)      # median of 2x, 10x, 4x
    assert got[0]["n"] == 3


def test_speedup_ignores_serving_layout_variants():
    rows = [row("eager", 1.0), row("compile", 0.5),
            row("eager", 1.0, layout="strided"), row("compile", 0.01, layout="strided")]
    got = speedup_by_batch(rows)
    assert got[0]["compile"] == pytest.approx(2.0)
    assert got[0]["n"] == 1


def test_graph_share_pairs_each_graph_impl_with_its_own_base():
    rows = [row("eager", 1.0), row("graph_eager", 0.4),
            row("compile", 0.5), row("graph_compile", 0.1)]
    got = {g["graph_impl"]: g for g in graph_launch_share(rows)}
    assert got["graph_eager"]["median_pct"] == pytest.approx(60.0)
    assert got["graph_compile"]["median_pct"] == pytest.approx(80.0)


def test_graph_share_skips_configs_missing_a_side():
    rows = [row("compile", 0.5), row("graph_compile", 0.1),
            row("compile", 0.5, alloc=128)]
    got = graph_launch_share(rows)
    assert len(got) == 1 and got[0]["n"] == 1


def test_ragged_uniform_pairs_only_identical_configs():
    rows = [row("compile", 1.4, mode="ragged"), row("compile", 1.0, mode="uniform"),
            row("compile", 9.9, mode="ragged", b=32)]
    got = ragged_uniform_ratio(rows)
    assert len(got) == 1
    assert got[0]["median_ratio"] == pytest.approx(1.4)


def test_serving_layout_cost_is_relative_to_packed_identity():
    rows = [row("compile", 1.0), row("compile", 1.1, layout="strided"),
            row("compile", 0.9, mapping="permuted")]
    got = {(r["layout"], r["request_mapping"]): r for r in serving_layout_cost(rows)}
    assert got[("strided", "identity")]["median_ratio"] == pytest.approx(1.1)
    assert got[("packed", "permuted")]["median_ratio"] == pytest.approx(0.9)
    assert ("packed", "identity") not in got


def test_head_dim_response_picks_the_batch_every_layout_reaches():
    rows = []
    for b, heads in ((1, ("mha", "mqa")), (128, ("mha",))):
        for h in heads:
            hkv, d = (32, 128) if h == "mha" else (1, 64)
            rows += [row("compile", 0.03, head=h, hkv=hkv, d=d, b=b)]
    got, ctx = head_dim_response(rows)
    assert ctx["num_requests"] == 1                      # 128 has no mqa row to compare against
    assert [g["head_label"] for g in got] == ["mha", "mqa"]
    assert got[0]["kv_elems_per_token"] == 2 * 32 * 128


def test_bandwidth_sanity_counts_rows_that_exceed_the_reference():
    rows = [row("compile", 0.1, pct=40.0), row("graph_compile", 0.01, pct=180.0, spct=273.0)]
    got = bandwidth_column_sanity(rows)
    assert got["empirical"]["over_100"] == 1
    assert got["empirical"]["max_pct"] == pytest.approx(180.0)
    assert got["empirical"]["row"]["impl"] == "graph_compile"
    assert got["scattered"]["max_pct"] == pytest.approx(273.0)


def test_failed_rows_never_reach_any_statistic(tmp_path):
    rows = [row("eager", 1.0), row("compile", 0.5),
            row("compile", 0.0001, alloc=128, validation="FAIL")]
    path = write_csv(tmp_path, rows)
    summary = build_summary(path, {"git_sha": "abc123", "git_dirty": False}, None)
    assert summary["rows_timed"] == 2
    assert summary["speedup_by_batch"][0]["compile"] == pytest.approx(2.0)


def test_markdown_flags_a_dirty_provenance(tmp_path):
    path = write_csv(tmp_path, [row("eager", 1.0), row("compile", 0.5)])
    clean = render_markdown(build_summary(path, {"git_sha": "a" * 40, "git_dirty": False}, None))
    dirty = render_markdown(build_summary(path, {"git_sha": "a" * 40, "git_dirty": True}, None))
    assert "DIRTY TREE" not in clean
    assert "DIRTY TREE" in dirty


def test_markdown_renders_without_a_profile_summary(tmp_path):
    path = write_csv(tmp_path, [row("eager", 1.0), row("compile", 0.5),
                                row("graph_compile", 0.1)])
    text = render_markdown(build_summary(path, {}, None))
    assert "Speedup vs eager" in text
    assert "Launch structure" not in text


# --- amdahl probe derivation ------------------------------------------------------------

from benchmarks.summarize_results import (  # noqa: E402
    amdahl_gate,
    amdahl_savings,
    read_amdahl_rows,
)

AMDAHL_COLUMNS = ["model_id", "batch", "ctx", "dtype_label", "mode", "rung", "description",
                  "numerically_valid", "amortized_step_ms", "amortized_spread_pct"]


def amdahl_row(mode, rung, ms, *, batch=32, ctx=128, valid=False, spread=""):
    return {"model_id": "m", "batch": str(batch), "ctx": str(ctx), "dtype_label": "bf16",
            "mode": mode, "rung": rung, "description": rung,
            "numerically_valid": str(valid), "amortized_step_ms": str(ms),
            "amortized_spread_pct": str(spread)}


def write_amdahl(tmp_path, rows):
    path = tmp_path / "amdahl.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AMDAHL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return str(path)


def test_savings_are_relative_to_the_full_rung_of_the_same_mode(tmp_path):
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_eager", "full", 100.0), amdahl_row("hf_eager", "op_removed", 90.0),
        amdahl_row("hf_static_graph", "full", 50.0),
        amdahl_row("hf_static_graph", "op_removed", 49.0),
    ])
    got = {(s["mode"], s["rung"]): s["saving_pct"] for s in amdahl_savings(read_amdahl_rows(path))}
    assert got[("hf_eager", "op_removed")] == pytest.approx(10.0)
    assert got[("hf_static_graph", "op_removed")] == pytest.approx(2.0)


def test_rows_without_a_timing_are_dropped_not_counted_as_zero(tmp_path):
    path = write_amdahl(tmp_path, [amdahl_row("hf_eager", "full", 100.0),
                                   amdahl_row("hf_eager", "op_removed", "")])
    assert [s["rung"] for s in amdahl_savings(read_amdahl_rows(path))] == ["full"]


def test_the_gate_reads_the_static_graph_mode_not_eager(tmp_path):
    """Eager flatters the operator. Gating on it would be the careless reading the plan
    pre-registered against."""
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_eager", "full", 100.0), amdahl_row("hf_eager", "op_removed", 80.0),
        amdahl_row("hf_static_graph", "full", 50.0),
        amdahl_row("hf_static_graph", "op_removed", 49.75),
    ])
    gate = amdahl_gate(amdahl_savings(read_amdahl_rows(path)))
    assert gate["gate_mode"] == "hf_static_graph"
    assert gate["saving_pct"] == pytest.approx(0.5)
    assert "dead" in gate["verdict"]


def test_the_gate_picks_the_most_favourable_batch32_config(tmp_path):
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_static_graph", "full", 50.0, ctx=128),
        amdahl_row("hf_static_graph", "op_removed", 49.0, ctx=128),
        amdahl_row("hf_static_graph", "full", 80.0, ctx=1024),
        amdahl_row("hf_static_graph", "op_removed", 79.9, ctx=1024),
    ])
    gate = amdahl_gate(amdahl_savings(read_amdahl_rows(path)))
    assert gate["ctx"] == 128


@pytest.mark.parametrize("saving_ms,expected", [(0.4, "dead"), (2.0, "marginal"), (4.0, "real")])
def test_each_preregistered_band_is_reachable(tmp_path, saving_ms, expected):
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_static_graph", "full", 50.0),
        amdahl_row("hf_static_graph", "op_removed", 50.0 - saving_ms),
    ])
    assert expected in amdahl_gate(amdahl_savings(read_amdahl_rows(path)))["verdict"]


def test_the_gate_reports_the_doubling_slope_when_present(tmp_path):
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_static_graph", "full", 50.0),
        amdahl_row("hf_static_graph", "op_removed", 49.0),
        amdahl_row("hf_static_graph", "op_doubled", 51.0),
    ])
    gate = amdahl_gate(amdahl_savings(read_amdahl_rows(path)))
    assert gate["doubled_cost_pct"] == pytest.approx(2.0)


def test_a_missing_amdahl_csv_leaves_the_summary_renderable(tmp_path):
    assert read_amdahl_rows(str(tmp_path / "nope.csv")) == []
    assert amdahl_gate([]) is None


def test_a_doubling_that_mirrors_the_removal_licenses_the_bound(tmp_path):
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_static_graph", "full", 100.0),
        amdahl_row("hf_static_graph", "op_removed", 93.0),      # saves 7%
        amdahl_row("hf_static_graph", "op_doubled", 106.5),     # costs 6.5% -- mirrors
    ])
    gate = amdahl_gate(amdahl_savings(read_amdahl_rows(path)))
    assert gate["bound_licensed"]
    assert gate["realizable_pct"] == pytest.approx(7.0)
    assert "real" in gate["verdict"]


def test_a_doubling_that_does_not_mirror_demotes_the_bound(tmp_path):
    """The pre-registered rule: removal that is not mirrored by doubling was never on the
    critical path, so the removal saving is an upper bound nobody can realize."""
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_static_graph", "full", 100.0),
        amdahl_row("hf_static_graph", "op_removed", 96.0),      # saves 4% -> would read "real"
        amdahl_row("hf_static_graph", "op_doubled", 100.6),     # costs 0.6% -- does not mirror
    ])
    gate = amdahl_gate(amdahl_savings(read_amdahl_rows(path)))
    assert not gate["bound_licensed"]
    assert gate["saving_pct"] == pytest.approx(4.0)
    assert gate["realizable_pct"] == pytest.approx(0.6)
    assert "dead" in gate["verdict"]


def test_the_gate_reports_the_mirror_ratio_it_judged_on(tmp_path):
    """The cutoff is a number chosen after seeing the data, so the ratio it was compared against
    has to travel with the verdict -- otherwise a reader cannot tell whether the demotion was
    decisive or a coin flip against an arbitrary constant."""
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_static_graph", "full", 100.0),
        amdahl_row("hf_static_graph", "op_removed", 96.0),
        amdahl_row("hf_static_graph", "op_doubled", 101.2),
    ])
    gate = amdahl_gate(amdahl_savings(read_amdahl_rows(path)))
    assert gate["mirror_ratio"] == pytest.approx(0.3)
    assert gate["mirror_cutoff"] == 0.5


@pytest.mark.parametrize("cutoff", [0.35, 0.5, 0.75])
def test_the_verdict_is_stable_across_plausible_cutoffs(tmp_path, monkeypatch, cutoff):
    """A ratio of 0.275 -- what the probe actually measured -- must demote under every cutoff a
    reasonable person would pick, or the verdict is an artifact of the constant."""
    import benchmarks.summarize_results as sr
    monkeypatch.setattr(sr, "AMDAHL_MIRROR_RATIO", cutoff)
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_static_graph", "full", 100.0),
        amdahl_row("hf_static_graph", "op_removed", 97.04),     # 2.96%
        amdahl_row("hf_static_graph", "op_doubled", 100.81),    # 0.81% -> ratio 0.275
    ])
    gate = sr.amdahl_gate(sr.amdahl_savings(sr.read_amdahl_rows(path)))
    assert not gate["bound_licensed"]
    assert "dead" in gate["verdict"]


def test_a_missing_doubling_rung_leaves_the_bound_unlicensed(tmp_path):
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_static_graph", "full", 100.0),
        amdahl_row("hf_static_graph", "op_removed", 96.0),
    ])
    gate = amdahl_gate(amdahl_savings(read_amdahl_rows(path)))
    assert not gate["bound_licensed"]
    assert gate["doubled_cost_pct"] is None


def test_a_saving_larger_than_the_spread_is_resolved(tmp_path):
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_static_graph", "full", 100.0, spread=0.4),
        amdahl_row("hf_static_graph", "op_removed", 97.0, spread=0.3),
    ])
    s = {r["rung"]: r for r in amdahl_savings(read_amdahl_rows(path))}
    assert s["op_removed"]["noise_pct"] == pytest.approx(0.4)
    assert s["op_removed"]["resolved"] is True


def test_a_saving_smaller_than_the_spread_is_not_resolved(tmp_path):
    """0.81% against a 0.4-1.0% spread is the actual situation at the gate, and reporting it as
    a clean number would overclaim what this rig can see."""
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_static_graph", "full", 100.0, spread=1.0),
        amdahl_row("hf_static_graph", "op_doubled", 100.8, spread=0.5),
    ])
    s = {r["rung"]: r for r in amdahl_savings(read_amdahl_rows(path))}
    assert s["op_doubled"]["resolved"] is False


def test_noise_takes_the_larger_of_the_two_spreads(tmp_path):
    """The delta is between two timings, so the noisier of the pair bounds it."""
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_static_graph", "full", 100.0, spread=0.2),
        amdahl_row("hf_static_graph", "op_removed", 97.0, spread=0.9),
    ])
    s = {r["rung"]: r for r in amdahl_savings(read_amdahl_rows(path))}
    assert s["op_removed"]["noise_pct"] == pytest.approx(0.9)


def test_a_csv_without_spread_columns_still_summarises(tmp_path):
    """The committed single-shot CSV predates --repeats; it must not crash the summariser."""
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_static_graph", "full", 100.0),
        amdahl_row("hf_static_graph", "op_removed", 97.0),
    ])
    s = {r["rung"]: r for r in amdahl_savings(read_amdahl_rows(path))}
    assert s["op_removed"]["noise_pct"] is None
    assert s["op_removed"]["resolved"] is None


def test_the_gate_carries_whether_its_two_inputs_were_resolved(tmp_path):
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_static_graph", "full", 100.0, spread=0.5),
        amdahl_row("hf_static_graph", "op_removed", 97.0, spread=0.5),
        amdahl_row("hf_static_graph", "op_doubled", 100.3, spread=0.5),
    ])
    g = amdahl_gate(amdahl_savings(read_amdahl_rows(path)))
    assert g["removal_resolved"] is True
    assert g["doubling_resolved"] is False
    assert "dead" in g["verdict"]


# --- the positive control -----------------------------------------------------------------

from benchmarks.summarize_results import (
    numerical_deltas,
    read_rows,  # noqa: E402
    amdahl_control_check,
    amdahl_mirror_table,
)


def _both_modes(tmp_path, eager_double, compiled_double):
    return write_amdahl(tmp_path, [
        amdahl_row("hf_eager", "full", 100.0),
        amdahl_row("hf_eager", "op_removed", 97.0),
        amdahl_row("hf_eager", "op_doubled", 100.0 + eager_double),
        amdahl_row("hf_eager", "rope_removed", 99.5),
        amdahl_row("hf_eager", "append_removed", 97.5),
        amdahl_row("hf_static_graph", "full", 100.0),
        amdahl_row("hf_static_graph", "op_removed", 97.0),
        amdahl_row("hf_static_graph", "op_doubled", 100.0 + compiled_double),
    ])


def test_the_control_holds_when_eager_mirrors_and_compiled_does_not(tmp_path):
    path = _both_modes(tmp_path, eager_double=3.0, compiled_double=0.3)
    c = amdahl_control_check(amdahl_mirror_table(amdahl_savings(read_amdahl_rows(path))))
    assert c["control_holds"] is True
    assert c["eager_min"] == pytest.approx(1.0, abs=0.05)
    assert c["compiled_max"] == pytest.approx(0.1, abs=0.02)


def test_the_control_fails_when_doubling_registers_nowhere(tmp_path):
    """If op_doubled were simply insensitive it would read ~0 in eager too, and then a near-zero
    compiled ratio would prove nothing. That case has to be detectable."""
    path = _both_modes(tmp_path, eager_double=0.05, compiled_double=0.3)
    c = amdahl_control_check(amdahl_mirror_table(amdahl_savings(read_amdahl_rows(path))))
    assert c["control_holds"] is False


def test_the_mirror_table_splits_rope_from_append(tmp_path):
    path = _both_modes(tmp_path, 3.0, 0.3)
    rows = {r["mode"]: r for r in amdahl_mirror_table(amdahl_savings(read_amdahl_rows(path)))}
    assert rows["hf_eager"]["rope_pct"] == pytest.approx(0.5)
    assert rows["hf_eager"]["append_pct"] == pytest.approx(2.5)
    assert rows["hf_static_graph"]["rope_pct"] is None      # not measured in this fixture


def test_the_control_is_absent_rather_than_wrong_with_one_mode(tmp_path):
    path = write_amdahl(tmp_path, [
        amdahl_row("hf_static_graph", "full", 100.0),
        amdahl_row("hf_static_graph", "op_removed", 97.0),
        amdahl_row("hf_static_graph", "op_doubled", 100.3),
    ])
    assert amdahl_control_check(amdahl_mirror_table(amdahl_savings(read_amdahl_rows(path)))) is None


def test_the_control_paragraph_reaches_the_markdown(tmp_path):
    base = write_csv(tmp_path, [row("eager", 1.0), row("compile", 0.5)])
    text = render_markdown(build_summary(base, {}, None, _both_modes(tmp_path, 3.0, 0.3)))
    assert "Positive control" in text
    assert "holds" in text


def delta_row(impl, dq, dk, *, dtype="bf16", atol=8e-3, rtol=8e-3, v_exact=True,
              intact=True, **kw):
    return {**row(impl, 1.0, dtype=dtype, **kw), "max_abs_diff_q": str(dq),
            "max_abs_diff_k_cache": str(dk), "tolerance_atol": str(atol),
            "tolerance_rtol": str(rtol), "v_byte_exact": str(v_exact),
            "unaddressed_slots_intact": str(intact)}


def write_delta_csv(tmp_path, rows, name="deltas.csv"):
    path = tmp_path / name
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return str(path)


def test_deltas_report_the_worst_row_per_dtype():
    rows = [delta_row("eager", 1e-3, 2e-3), delta_row("compile", 5e-3, 1e-3),
            delta_row("eager", 1e-7, 1e-7, dtype="fp32", atol=1e-6, rtol=1e-6)]
    by_dtype = {d["dtype_label"]: d for d in numerical_deltas(rows)}
    assert by_dtype["bf16"]["max_abs_diff_q"] == pytest.approx(5e-3)
    assert by_dtype["bf16"]["max_abs_diff_k_cache"] == pytest.approx(2e-3)
    assert by_dtype["bf16"]["worst_q_impl"] == "compile"
    assert by_dtype["bf16"]["n"] == 2
    assert by_dtype["fp32"]["atol"] == pytest.approx(1e-6)


def test_headroom_is_the_worst_of_q_and_k_against_atol():
    d = numerical_deltas([delta_row("eager", 2e-3, 4e-3, atol=8e-3)])[0]
    assert d["headroom_pct"] == pytest.approx(50.0)


def test_rows_that_did_not_validate_contribute_no_deltas():
    """A FAIL row carries real deltas, but they describe a wrong implementation; folding them
    into the passing distribution would report another impl's bug as this one's precision."""
    assert numerical_deltas([delta_row("broken", 9.9, 9.9, validation="FAIL")]) == []


def test_a_csv_without_delta_columns_omits_the_section(tmp_path):
    """v3 predates these columns; the summariser must not report absent deltas as zero."""
    base = write_csv(tmp_path, [row("eager", 1.0), row("compile", 0.5)])
    text = render_markdown(build_summary(base, {}, None))
    assert numerical_deltas(read_rows(base)) == []
    assert "Numerical deltas" not in text


def test_the_delta_table_reaches_the_markdown(tmp_path):
    base = write_delta_csv(tmp_path, [delta_row("eager", 1e-3, 2e-3)])
    text = render_markdown(build_summary(base, {}, None))
    assert "Numerical deltas against the oracle" in text
    assert "% of atol" in text


def test_a_passing_row_contradicting_the_gate_is_called_out(tmp_path):
    """v_byte_exact False on a validation=pass row means the gate and the columns disagree."""
    base = write_delta_csv(tmp_path, [delta_row("eager", 1e-3, 1e-3, v_exact=False)])
    text = render_markdown(build_summary(base, {}, None))
    assert "which is a bug in one of them" in text
