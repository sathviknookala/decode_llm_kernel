# Limitations and blockers, per results file

What each result set in this directory does **not** establish. Read this before quoting any
number from these files. Methodology is in `docs/benchmark_methodology.md`.

Everything here was measured on a single machine: one RTX PRO 4000 Blackwell (`sm_120`,
23.4 GiB, driver 575.64.03), torch 2.11.0+cu128, CUDA runtime 12.8 / nvcc 12.9. Exact
provenance for each run is in the adjacent `*.env.json`.

---

## `raw/operator_baseline_v2.csv` (+ `operator_baseline_v2.env.json`)

**`logical_eff_gbps` and `pct_of_empirical_bw` are not physical DRAM traffic.** The numerator
is a logical floor: it excludes intermediates an implementation materializes (eager's
`rotate_half` concatenation, gathered trig rows, cast temporaries), excludes whole-sector
overhead on scattered writes, and counts bytes that may be served from L2. The denominator is
streaming throughput on a different access pattern. Treat the ratio as orientation, not an
efficiency score. **Blocker:** measured traffic needs Nsight Compute counters, not yet run.

**The timed region is an operator call, not a bare kernel.** It includes PyTorch dispatch and
one iterator step for position rotation. That is identical across implementations, so
comparisons are fair, but these are not pure kernel times.

**No CUDA-graph baseline.** Since ~93% of the compiled path's measured cost is dispatch
(`profiling/profile_summary.json`), a graph-captured variant is the missing measurement that
would cleanly separate launch overhead from kernel cost. Until it exists, "dispatch-bound" is
supported by the profiler split but the size of the recoverable fraction is unquantified.
**This is the highest-value gap for Checkpoint C.**

**Two configurations are unmeasured, not measured-and-fast.** `mha b=128 alloc=2048 fp32`
(both position modes) is skipped: it needs 17.2 GB peak contiguous cache (2 sets × 8.6 GB)
against a 12 GB budget on a 23.4 GiB card. Rows are present with `impl=skipped` and the
reason. This is a preview of why paged caches exist, not a gap in the sweep.

**Coverage bounds of the matrix.** `head_dim` is fixed at 128 (the Llama-2 anchor) and rotary
is always full — no partial-rotary or awkward head dims (Checkpoint D). Contiguous
token-major cache only; no paged, fragmented, or head-major layouts (Checkpoint F). Head
configurations are MHA 32/32 and GQA 32/8 only; MQA is covered in the byte-formula tests but
not benchmarked. `torch.compile` is measured with one backend and default mode, recorded in
the env JSON; other modes are unexplored.

**GQA shapes are real; GQA tensors are not.** The swept GQA layout (32 q heads : 8 kv heads,
`head_dim` 128, `rope_theta` 10000) is the attention shape of **Mistral-7B-v0.1**
(Apache-2.0, ungated; Jiang et al. 2023, arXiv:2310.06825), cross-checked against its
published `config.json` by `tests/test_anchor_models.py`, and our RoPE tables are bit-exact
against `MistralRotaryEmbedding`. That establishes the *configuration* is real, **not** that
the numbers are: every timed tensor is still synthetic `randn`, no model weights are loaded,
and no real activation distribution or decode loop is involved. Latency here is dtype- and
shape-driven, so synthetic inputs are appropriate — but nothing in this file is a measurement
of Mistral-7B-v0.1 the model.

**Timed rows use the identity request mapping and packed Q/K/V.** Non-identity mappings and
fused-projection strides are exercised by the validation gate, not by the timing path, so this
file says nothing about their cost. A strided read is plausibly slower than a packed one; that
delta is unmeasured.

**The attention consumer is not modelled at all.** This operator only writes the cache. A
layout or kernel that speeds writes may slow the attention reads that follow, and that
trade-off is invisible in this file.

**`cache_alloc_len` is an allocation size.** Flat latency across it is the expected result and
is not evidence about launch-bounding — the operator touches one RoPE row and one K/V slot per
request and never scans the context.

---

## `raw/bandwidth_reference.json`

**It is a streaming reference, not a bound on this operator.** Large contiguous `copy_` and
`add_` are the friendliest possible access pattern; the operator does small scattered writes
into a large cache, so it should not be expected to reach this number even when perfectly
implemented.

**Single buffer size, single dtype.** 512 MiB FP32. No sweep over sizes to show the L2→DRAM
transition, so "DRAM-bound" rests on 512 MiB being 10× the 48 MiB L2 rather than on a measured
knee.

**Medians of a short run** (50 iterations after warmup) with no clock-state control — no
locked clocks, no thermal soak — so treat ±few-% differences as noise.

---

## `profiling/` (`profile_summary.json`, `trace_*.json`, `nsys_*_cuda_gpu_kern_sum.csv`)

**Kernel counts are exact; the attribution of the remaining time is inferred.** The traces
establish kernels per invocation and device time per invocation directly. That the rest of the
measured span is host-side dispatch follows from the arithmetic (1.8 µs of GPU work inside a
25.5 µs call) rather than from a direct measurement of dispatch cost. A CUDA-graph comparison
would confirm it.

**Profiling perturbs what it measures.** Reported device times come from profiled runs and are
not the timing path's numbers; use `operator_baseline_v2.csv` for latency.

**Narrow config coverage.** bf16, ragged positions, `alloc=2048`, b∈{1,32}, MHA and GQA only.
Kernel counts for other dtypes and batch sizes are not established, though the inductor fusion
count is expected to vary (already 1 vs 2 kernels between MHA b=32 and GQA b=32).

**nsys traces cover the whole process.** Setup allocation and RNG kernels appear alongside the
operator; the NVTX range marks the measured region. The `.nsys-rep` and `.sqlite` files are
gitignored as large regenerable binaries — regenerate with `benchmarks/nsight_capture.sh`.
Only the kernel-summary CSVs are tracked.

**No Nsight Compute.** No occupancy, register pressure, sector efficiency, or achieved-bandwidth
counters — so no statement here explains *why* a kernel takes the time it does.

---

## `raw/archive/operator_baseline_v1_preB1.csv`

Superseded and **not comparable** to v2. See `raw/archive/README.md` for the five reasons
(byte accounting, latency naming, uniform-only positions, no validation gate, column rename).
Retained only so earlier claims can be traced to the data that produced them.
