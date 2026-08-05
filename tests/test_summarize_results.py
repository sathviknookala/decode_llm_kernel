import csv

import pytest

from benchmarks.summarize_results import (
    bandwidth_column_sanity,
    build_summary,
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
                  "numerically_valid", "amortized_step_ms"]


def amdahl_row(mode, rung, ms, *, batch=32, ctx=128, valid=False):
    return {"model_id": "m", "batch": str(batch), "ctx": str(ctx), "dtype_label": "bf16",
            "mode": mode, "rung": rung, "description": rung,
            "numerically_valid": str(valid), "amortized_step_ms": str(ms)}


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
