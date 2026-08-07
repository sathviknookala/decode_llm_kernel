import argparse
import csv
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmark_utils import REPO_ROOT, read_env_doc

MAIN_LAYOUT = ("packed", "identity")
GRAPH_PAIRS = (("eager", "graph_eager"), ("compile", "graph_compile"))
CONFIG_KEYS = ("head_label", "num_q_heads", "num_kv_heads", "head_dim", "num_requests",
               "cache_alloc_len", "dtype_label", "position_mode", "layout", "request_mapping")


def read_rows(path):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["validation"] == "pass"]


def _ms(row):
    return float(row["device_median_ms"])


def by_config(rows):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in CONFIG_KEYS), {})[r["impl"]] = r
    return out


def _main_layout(key):
    return (key[CONFIG_KEYS.index("layout")],
            key[CONFIG_KEYS.index("request_mapping")]) == MAIN_LAYOUT


def speedup_by_batch(rows, baseline="eager"):
    """Median device-median ratio against the baseline impl, per batch size."""
    grouped = by_config(rows)
    batches = sorted({k[CONFIG_KEYS.index("num_requests")] for k in grouped}, key=int)
    impls = sorted({r["impl"] for r in rows} - {baseline})
    out = []
    for b in batches:
        entry = {"num_requests": int(b), "n": 0}
        for impl in impls:
            ratios = [_ms(m[baseline]) / _ms(m[impl]) for k, m in grouped.items()
                      if k[CONFIG_KEYS.index("num_requests")] == b and _main_layout(k)
                      and baseline in m and impl in m]
            if ratios:
                entry[impl] = st.median(ratios)
                entry["n"] = max(entry["n"], len(ratios))
        out.append(entry)
    return out


def graph_launch_share(rows):
    """Fraction of a direct impl's device median that graph replay removes."""
    grouped = by_config(rows)
    out = []
    for direct, graph in GRAPH_PAIRS:
        pct = [100.0 * (1.0 - _ms(m[graph]) / _ms(m[direct]))
               for m in grouped.values() if direct in m and graph in m]
        if pct:
            out.append({"direct_impl": direct, "graph_impl": graph, "n": len(pct),
                        "median_pct": st.median(pct), "min_pct": min(pct), "max_pct": max(pct)})
    return out


def ragged_uniform_ratio(rows):
    """Ragged over uniform device median at otherwise identical configs."""
    grouped = by_config(rows)
    mode_at = CONFIG_KEYS.index("position_mode")
    batch_at = CONFIG_KEYS.index("num_requests")
    acc = {}
    for key, impls in grouped.items():
        if key[mode_at] != "ragged":
            continue
        twin = grouped.get(key[:mode_at] + ("uniform",) + key[mode_at + 1:], {})
        for impl, row in impls.items():
            if impl in twin:
                acc.setdefault((impl, int(key[batch_at])), []).append(_ms(row) / _ms(twin[impl]))
    return [{"impl": impl, "num_requests": b, "n": len(v), "median_ratio": st.median(v)}
            for (impl, b), v in sorted(acc.items(), key=lambda kv: (kv[0][1], kv[0][0]))]


def serving_layout_cost(rows):
    """Cost of a strided fused-projection read and a permuted request mapping, vs packed/identity."""
    grouped = by_config(rows)
    li, ri = CONFIG_KEYS.index("layout"), CONFIG_KEYS.index("request_mapping")
    acc = {}
    for key, impls in grouped.items():
        variant = (key[li], key[ri])
        if variant == MAIN_LAYOUT:
            continue
        base_key = key[:li] + MAIN_LAYOUT + key[ri + 1:]
        base = grouped.get(base_key, {})
        for impl, row in impls.items():
            if impl in base:
                acc.setdefault((variant, impl), []).append(_ms(row) / _ms(base[impl]))
    return [{"layout": v[0], "request_mapping": v[1], "impl": impl, "n": len(r),
             "median_ratio": st.median(r)}
            for (v, impl), r in sorted(acc.items())]


def head_dim_response(rows, impl="compile"):
    """Latency against KV bytes per token at the largest batch every head layout reaches."""
    grouped = by_config(rows)
    batch_at = CONFIG_KEYS.index("num_requests")
    candidates = [k for k, m in grouped.items()
                  if _main_layout(k) and impl in m
                  and k[CONFIG_KEYS.index("position_mode")] == "ragged"]
    if not candidates:
        return [], {}
    heads = {k[0] for k in candidates}
    shared = [b for b in sorted({k[batch_at] for k in candidates}, key=int)
              if {k[0] for k in candidates if k[batch_at] == b} == heads]
    if not shared:
        return [], {}
    batch = shared[-1]
    dtypes = {k[CONFIG_KEYS.index("dtype_label")] for k in candidates if k[batch_at] == batch}
    dtype = sorted(dtypes)[0]
    allocs = {k[CONFIG_KEYS.index("cache_alloc_len")] for k in candidates
              if k[batch_at] == batch and k[CONFIG_KEYS.index("dtype_label")] == dtype}
    alloc = max(allocs, key=int)
    out = []
    for k in candidates:
        if (k[batch_at], k[CONFIG_KEYS.index("dtype_label")],
                k[CONFIG_KEYS.index("cache_alloc_len")]) != (batch, dtype, alloc):
            continue
        row = grouped[k][impl]
        out.append({
            "head_label": row["head_label"],
            "head_dim": int(row["head_dim"]),
            "num_kv_heads": int(row["num_kv_heads"]),
            "kv_elems_per_token": 2 * int(row["num_kv_heads"]) * int(row["head_dim"]),
            "device_median_ms": _ms(row),
            "pct_of_empirical_bw": float(row["pct_of_empirical_bw"] or 0.0),
        })
    return sorted(out, key=lambda d: -d["kv_elems_per_token"]), {
        "num_requests": int(batch), "dtype_label": dtype, "cache_alloc_len": int(alloc),
        "impl": impl}


def bandwidth_column_sanity(rows):
    """Rows above 100% are proof the logical numerator counts cache-served bytes."""
    def top(col):
        vals = [(float(r[col]), r) for r in rows if r.get(col)]
        if not vals:
            return None
        best = max(vals, key=lambda v: v[0])
        return {"max_pct": best[0], "over_100": sum(1 for v, _ in vals if v > 100.0),
                "n": len(vals), "row": best[1]}
    return {"empirical": top("pct_of_empirical_bw"),
            "scattered": top("pct_of_scattered_write_bw")}


AMDAHL_GATE_MODE = "hf_static_graph"
# Pre-registered before the run. Applied to the op_removed saving at the most operator-favourable
# b=32 configuration, which is the largest saving any fused kernel could realize.
AMDAHL_GATE = ((5.0, "latency case is real: build the fused kernel for latency"),
               (1.0, "marginal: build it as fusion ablation + the paged path"),
               (-1e9, "latency case is dead: pivot Checkpoint C to paged-cache capability"))
# The plan pre-registered op_doubled as the check that licenses quoting op_removed: "if removal
# saves X but doubling costs materially less than X, the operation is partly overlapped and
# op_removed is an optimistic bound". Below this ratio the two do not mirror, so the realizable
# figure is the doubling slope, not the removal saving.
AMDAHL_MIRROR_RATIO = 0.5


def read_amdahl_rows(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("amortized_step_ms")]


def _spread_pct(row):
    v = row.get("amortized_spread_pct")
    return float(v) if v not in (None, "") else None


def amdahl_savings(rows):
    """Saving vs the `full` rung at each (config, mode), from amortized step time.

    `noise_pct` is the observed spread of repeated timings at that (config, mode) -- the larger
    of the rung's own spread and `full`'s. A saving smaller than that is not resolved by this
    instrument and is reported as such rather than as a small number.
    """
    key = lambda r: (r["batch"], r["ctx"], r["mode"])  # noqa: E731
    full = {key(r): r for r in rows if r["rung"] == "full"}
    out = []
    for r in rows:
        base_row = full.get(key(r))
        if not base_row:
            continue
        base = float(base_row["amortized_step_ms"])
        ms = float(r["amortized_step_ms"])
        spreads = [s for s in (_spread_pct(r), _spread_pct(base_row)) if s is not None]
        noise = max(spreads) if spreads else None
        saving = (base - ms) / base * 100.0
        out.append({
            "batch": int(r["batch"]), "ctx": int(r["ctx"]), "mode": r["mode"],
            "rung": r["rung"], "step_ms": ms, "full_ms": base,
            "saving_pct": saving,
            "noise_pct": noise,
            "resolved": None if noise is None else abs(saving) > noise,
            "numerically_valid": r.get("numerically_valid") == "True",
        })
    return out


def amdahl_gate(savings, layers=32):
    """The pre-registered verdict, derived rather than asserted."""
    candidates = [s for s in savings
                  if s["mode"] == AMDAHL_GATE_MODE and s["rung"] == "op_removed"
                  and s["batch"] == 32]
    if not candidates:
        return None
    best = max(candidates, key=lambda s: s["saving_pct"])
    doubled = next((s for s in savings
                    if s["mode"] == best["mode"] and s["batch"] == best["batch"]
                    and s["ctx"] == best["ctx"] and s["rung"] == "op_doubled"), None)
    doubled_pct = None if not doubled else -doubled["saving_pct"]
    licensed = (doubled_pct is not None
                and doubled_pct >= AMDAHL_MIRROR_RATIO * best["saving_pct"])
    realizable = best["saving_pct"] if licensed else (doubled_pct if doubled_pct is not None
                                                      else best["saving_pct"])
    verdict = next(v for threshold, v in AMDAHL_GATE if realizable >= threshold)
    return {
        "gate_mode": AMDAHL_GATE_MODE, "batch": best["batch"], "ctx": best["ctx"],
        "full_ms": best["full_ms"], "saving_pct": best["saving_pct"],
        "saving_ms": best["full_ms"] - best["step_ms"],
        "per_layer_us": (best["full_ms"] - best["step_ms"]) * 1000.0 / layers,
        "doubled_cost_pct": doubled_pct,
        "mirror_ratio": (None if doubled_pct is None or not best["saving_pct"]
                         else doubled_pct / best["saving_pct"]),
        "mirror_cutoff": AMDAHL_MIRROR_RATIO,
        "noise_pct": best.get("noise_pct"),
        "removal_resolved": best.get("resolved"),
        "doubling_resolved": None if not doubled else doubled.get("resolved"),
        "bound_licensed": licensed,
        "realizable_pct": realizable,
        "realizable_per_layer_us": realizable / 100.0 * best["full_ms"] * 1000.0 / layers,
        "verdict": verdict,
    }


def launch_structure(profile_summary):
    if not profile_summary:
        return []
    return [{"impl": s["impl"], "config": s["config"],
             "kernels_per_invocation": s["kernels_per_invocation"],
             "distinct_kernels": s["distinct_kernels"],
             "device_us_per_invocation": s["device_time_per_invocation_us"]}
            for s in profile_summary.get("summaries", [])]


def build_summary(csv_path, env, profile_summary, amdahl_csv=None):
    rows = read_rows(csv_path)
    heads, head_ctx = head_dim_response(rows)
    amdahl = amdahl_savings(read_amdahl_rows(amdahl_csv))
    return {
        "amdahl": {"savings": amdahl, "gate": amdahl_gate(amdahl),
                   "source_csv": (os.path.relpath(amdahl_csv, REPO_ROOT)
                                  if amdahl_csv and os.path.exists(amdahl_csv) else None)},
        "source_csv": os.path.relpath(csv_path, REPO_ROOT),
        "rows_timed": len(rows),
        "environment": {k: env.get(k) for k in
                        ("gpu_name", "git_sha", "git_dirty", "torch_version",
                         "cuda_toolkit_version_nvcc", "timestamp_utc")},
        "speedup_by_batch": speedup_by_batch(rows),
        "graph_launch_share": graph_launch_share(rows),
        "head_dim_response": {"context": head_ctx, "rows": heads},
        "bandwidth_columns": bandwidth_column_sanity(rows),
        "ragged_uniform": ragged_uniform_ratio(rows),
        "serving_layout": serving_layout_cost(rows),
        "launch_structure": launch_structure(profile_summary),
    }


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def render_markdown(s):
    env = s["environment"]
    dirty = " (DIRTY TREE)" if env.get("git_dirty") else ""
    out = [
        "# Results summary",
        "",
        "Generated by `benchmarks/summarize_results.py`. Every number here is derived from the "
        "source files named below -- do not hand-edit.",
        "",
        f"- source: `{s['source_csv']}` ({s['rows_timed']} timed rows)",
        f"- device: {env.get('gpu_name')} | torch {env.get('torch_version')} | "
        f"nvcc {env.get('cuda_toolkit_version_nvcc')}",
        f"- provenance: `{(env.get('git_sha') or '')[:12]}`{dirty} at {env.get('timestamp_utc')}",
        "",
        "## Speedup vs eager (median over configs, packed/identity)",
        "",
    ]
    impls = [k for k in s["speedup_by_batch"][0] if k not in ("num_requests", "n")] \
        if s["speedup_by_batch"] else []
    out += _table(["batch"] + [f"{i} / eager" for i in impls] + ["configs"],
                  [[e["num_requests"]] + [f"{e.get(i, float('nan')):.2f}x" for i in impls] + [e["n"]]
                   for e in s["speedup_by_batch"]])

    out += ["", "## Launch overhead removed by CUDA-graph replay", "",
            "The direct path's device median that disappears when the same work is replayed "
            "from a captured graph. This is measured, not inferred from a kernel/call ratio.", ""]
    out += _table(["direct", "graph", "median removed", "min", "max", "pairs"],
                  [[g["direct_impl"], g["graph_impl"], f"{g['median_pct']:.1f}%",
                    f"{g['min_pct']:.1f}%", f"{g['max_pct']:.1f}%", g["n"]]
                   for g in s["graph_launch_share"]])

    hd = s["head_dim_response"]
    if hd["rows"]:
        c = hd["context"]
        out += ["", "## Does latency follow KV traffic?", "",
                f"`{c['impl']}`, b={c['num_requests']}, {c['dtype_label']}, "
                f"alloc={c['cache_alloc_len']}, ragged, packed/identity.", ""]
        out += _table(["head", "head_dim", "kv heads", "K+V elems/token", "device median ms",
                       "% empirical bw"],
                      [[h["head_label"], h["head_dim"], h["num_kv_heads"],
                        h["kv_elems_per_token"], f"{h['device_median_ms']:.4f}",
                        f"{h['pct_of_empirical_bw']:.1f}%"] for h in hd["rows"]])
        span = hd["rows"][0]["kv_elems_per_token"] / max(1, hd["rows"][-1]["kv_elems_per_token"])
        fastest = min(hd["rows"], key=lambda h: h["device_median_ms"])
        out += ["", f"KV traffic spans **{span:.0f}x** across these layouts. The fastest is "
                    f"`{fastest['head_label']}`, carrying "
                    f"{fastest['kv_elems_per_token']} elements/token. If latency tracked traffic "
                    f"the fastest row would be the lightest one."]

    bw = s["bandwidth_columns"]
    out += ["", "## Bandwidth columns are not efficiency scores", ""]
    for name, d in bw.items():
        if not d:
            continue
        r = d["row"]
        out += [f"- **{name}**: max **{d['max_pct']:.1f}%**, {d['over_100']} of {d['n']} rows "
                f"above 100% (`{r['impl']} {r['head_label']} b{r['num_requests']} "
                f"{r['dtype_label']} {r['position_mode']}`)."]
    out += ["", "A ratio above 100% means the logical byte count is being served from cache. "
                "Treat these columns as orientation only."]

    out += ["", "## Ragged vs uniform positions", ""]
    out += _table(["impl", "batch", "ragged / uniform", "pairs"],
                  [[r["impl"], r["num_requests"], f"{r['median_ratio']:.3f}", r["n"]]
                   for r in s["ragged_uniform"]])

    out += ["", "## Serving layout cost vs packed/identity", ""]
    out += _table(["layout", "request mapping", "impl", "ratio", "pairs"],
                  [[r["layout"], r["request_mapping"], r["impl"],
                    f"{r['median_ratio']:.3f}", r["n"]] for r in s["serving_layout"]])

    am = s.get("amdahl", {})
    if am.get("gate"):
        g = am["gate"]
        out += ["", "## Amdahl fraction: what a fused kernel can win end to end", "",
                f"Source: `{am['source_csv']}`. Substitution ablation on a real decode loop -- "
                "`op_removed` deletes RoPE and the cache write (and their launches), so its "
                "saving is the upper bound on a fused kernel.", ""]
        by_mode = {}
        for r in am["savings"]:
            if r["rung"] == "op_removed":
                by_mode.setdefault(r["mode"], []).append(r)
        out += _table(["mode", "batch", "ctx", "full ms/step", "op_removed saving"],
                      [[r["mode"], r["batch"], r["ctx"], f"{r['full_ms']:.3f}",
                        f"{r['saving_pct']:+.2f}%"]
                       for m in by_mode for r in sorted(by_mode[m], key=lambda x: (x["batch"], x["ctx"]))])
        doubled = ("not measured" if g["doubled_cost_pct"] is None
                   else f"{g['doubled_cost_pct']:+.2f}%")
        out += ["",
                f"**Gate** (pre-registered): `{g['gate_mode']}`, b={g['batch']}, ctx={g['ctx']} -- "
                f"the most operator-favourable configuration measured. Removing the operation "
                f"saves **{g['saving_pct']:.2f}%** of a {g['full_ms']:.2f} ms step "
                f"({g['saving_ms']*1000:.0f} us, {g['per_layer_us']:.1f} us per layer).",
                ""]
        if g["bound_licensed"]:
            out += [f"Validity check passes: doubling the operation costs {doubled}, which "
                    "mirrors the removal, so the operation is serially on the critical path and "
                    "the removal saving is realizable."]
        else:
            out += [f"**Validity check FAILS: doubling the operation costs only {doubled}** "
                    f"against a {g['saving_pct']:.2f}% removal saving. The two do not mirror, so "
                    "most of what removal buys is not the operation's serial cost -- deleting it "
                    "lets the compiler restructure around it, which no faster kernel reproduces. "
                    f"The realizable figure is the doubling slope: **{g['realizable_pct']:.2f}%** "
                    f"({g['realizable_per_layer_us']:.1f} us per layer). Treat "
                    f"{g['saving_pct']:.2f}% as an upper bound that is not achievable.",
                    "",
                    f"Mirror ratio is **{g['mirror_ratio']:.3f}** against a {g['mirror_cutoff']} "
                    f"cutoff, so the demotion holds for any cutoff above "
                    f"{g['mirror_ratio']:.3f}. The cutoff was chosen after seeing the data; this "
                    "ratio is what lets a reader judge how much that choice mattered."]
        if g.get("noise_pct") is not None:
            res = ("above" if g.get("doubling_resolved") else "**inside**")
            out += ["",
                    f"Repeat spread at this configuration is {g['noise_pct']:.2f}% of a step. The "
                    f"removal ({g['saving_pct']:.2f}%) is "
                    f"{'above' if g.get('removal_resolved') else 'inside'} it; the doubling "
                    f"({g['doubled_cost_pct']:+.2f}%) is {res} it."]
            if not g.get("doubling_resolved"):
                out += ["",
                        "So the realizable figure is bounded by the instrument, not measured by "
                        "it: what the probe establishes is that the doubling cost is *at most* "
                        "about the noise floor, which is already well inside the dead band. A "
                        "tighter number would need a lower-variance rig, not a different verdict."]
        out += ["", f"**Verdict: {g['verdict']}**"]

    if s["launch_structure"]:
        out += ["", "## Launch structure (profiler)", ""]
        out += _table(["impl", "config", "kernels/invocation", "distinct", "device us/invocation"],
                      [[l["impl"], l["config"], f"{l['kernels_per_invocation']:.2f}",
                        l["distinct_kernels"], f"{l['device_us_per_invocation']:.1f}"]
                       for l in s["launch_structure"]])
    return "\n".join(out) + "\n"


def _load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description="Derive the headline claims from a baseline CSV, so they are reproducible")
    ap.add_argument("--csv", default=os.path.join(REPO_ROOT, "results/raw/operator_baseline_v3.csv"))
    ap.add_argument("--env", default=None, help="defaults to the CSV's adjacent .env.json")
    ap.add_argument("--profile-summary",
                    default=os.path.join(REPO_ROOT, "results/profiling/profile_summary.json"))
    ap.add_argument("--amdahl-csv",
                    default=os.path.join(REPO_ROOT, "results/raw/amdahl_probe.csv"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results/summary.md"))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    env_path = args.env or (os.path.splitext(args.csv)[0] + ".env.json")
    summary = build_summary(args.csv, read_env_doc(env_path),
                            _load_json(args.profile_summary), args.amdahl_csv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(render_markdown(summary))
    print(f"wrote {args.out}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
