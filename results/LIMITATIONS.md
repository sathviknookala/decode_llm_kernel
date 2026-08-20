# Limitations and blockers, per results file

What each result set in this directory does **not** establish. Read this before quoting any
number from these files. Methodology is in `docs/benchmark_methodology.md`.

Everything here was measured on a single machine: one RTX PRO 4000 Blackwell (`sm_120`,
23.4 GiB, driver 575.64.03), torch 2.11.0+cu128, CUDA runtime 12.8 / nvcc 12.9. Exact
provenance for each run is in the adjacent `*.env.json`.

---

## `raw/operator_kernels_v1.csv` (+ `operator_kernels_v1.env.json`)

3812 rows (3808 timed), eight impls, 480/480 declared units, 0 validation failures, 4 budget
skips. Clean `46e47a0`, one `run_segment`, ~27 minutes. This is the file `results/summary.md` is
generated from and the only committed measurement of the custom kernels; the four-impl
`operator_baseline_v3.csv` remains as the independent baseline.

**The fusion ratio is not ordering-controlled.** `benchmark_operator.py` is not a bracketed
ladder -- there is no `ordering_drift` column -- so the median **1.82x** of `cuda_separate` over
`cuda_fused` carries whatever first-rung artifact the impl ordering induces. That artifact was
measured at up to **25%** on this rig for the ladder probes, which is larger than the effects some
sections of this file discuss. Two things bound the risk here and neither removes it: the ratio is
between two impls timed inside one configuration rather than across a ladder, and it reproduces
across 476 configurations, four batch sizes, three dtypes and four head layouts. **Read the
direction as established and the third digit as noise.**

**It is not a decode-latency result, and the file cannot tell you that.** `raw/amdahl_probe.csv`
measured *doubling* this whole operation inside a real Mistral-7B decode step at **0.0-0.9%** under
both compiled modes. A kernel that is 1.8x faster here does not shorten a step measurably. Every
ratio in this file is an operator microbenchmark; quoting one as a serving win would contradict
the artifact two sections down.

**Quote `amortized_call_ms`, never `device_median_ms`, for anything involving a custom rung.**
CUDA-event bracketing costs 4.3 us on a call that launches nothing (`raw/graph_floor_probe.csv`)
and `cuda_fused` is ~3 us of work, so more than half of its event median is instrument. The same
floor is why `device_median_ms` reports `graph_compile` at 8.3x eager where the amortized route
reports 12.4x: the floor sits in the denominator and biases graph and custom ratios *down*.

**The graph twins of the custom rungs are in this file for completeness and are the wrong rungs
to quote.** `graph_cuda_fused` is **1.78x slower** than plain `cuda_fused`, because replay plus the
position `copy_` is ~5 us against a ~3 us kernel -- graphs pay off against eager's 18 launches,
not against one. The fused kernel's best form is ungraphed. `graph_cuda_separate` is likewise
within noise of, or worse than, its direct rung.

**The 19 configurations where `cuda_fused` loses to `graph_compile` are a memory-bound corner, not
a launch-overhead result.** All 19 are b=128 on `mha` and `mha96` (0.68x and 0.77x), the widest KV
layouts, where the kernel already moves 614 GB/s logical against a 553 GB/s streaming reference.
Fusion still pays 1.64-1.76x there; what inverts is the comparison against PyTorch, because
inductor's generated kernel is better on the memory path. By batch the advantage against the
ceiling runs 1.87x / 1.84x / 1.50x / 1.06x at b=1/8/32/128. Through b=32 that is a roughly fixed
absolute gap -- 2.76, 2.68 and 2.08 us -- so the ratio falls because the denominator grows rather
than because the kernel degrades. At b=128 the median gap is only 0.37 us, and that is a genuine
change of regime rather than more of the same: it is +2.04 us on `mqa`, which never leaves the
overhead-bound regime, and negative on `mha` and `mha96`.

**`pct_of_empirical_bw` is worse here than in v3, for the same reason it was wrong there.** 40 of
3808 rows exceed 100%, peaking at **191% empirical / 285% scattered** (`cuda_fused`, mha, b=128,
fp32, uniform). A faster implementation against an unchanged logical byte count inflates the
column further. It is not an efficiency score. **Blocker:** real traffic still needs Nsight
Compute counters, still blocked by `ERR_NVGPUCTRPERM`.

**Every timed tensor is synthetic `randn`, and the cache is contiguous.** The four head layouts
are real models' shapes but no real weights are loaded anywhere in this file. Paged or fragmented
cache addressing is Checkpoint F and is not represented.

**The four skips are a preview of why paged caches exist.** `mha` and `mha96` at
`b=128 alloc=2048 fp32` need 17.2 GB and 12.9 GB of peak contiguous cache against a 12 GB budget,
so they are recorded as `impl=skipped` rows in both position modes rather than silently dropped.
The two head layouts with the most traffic are therefore unmeasured at the largest fp32 point.

---

## `raw/operator_baseline_v3.csv` (+ `operator_baseline_v3.env.json`)

1908 rows (1904 timed), four impls, four head layouts, 0 validation failures. Provenance is a
clean `df96e9c`; each row names the gate cases that cleared it in `validation_cases`.

**Every row in this file is overhead-bound, so the bandwidth columns describe almost nothing.**
At b=128 the four layouts move KV traffic differing by 64x (mha 8192 bytes per token, mqa 128)
and the compiled device medians are 0.0272 ms and 0.0304 ms respectively -- the layout carrying
64x the bytes is the faster one. No configuration here is limited by memory traffic. Do not read
any latency in this file as evidence about a memory system.

**This file's own writes never leave L2, and that turns out not to matter.** The timed loop cycles
8 position sets and reuses one set of input tensors, so it rewrites the same slots on every
invocation -- a write working set of 0.67x L2 even at the largest configuration that fits. The
flatness above could therefore have been an artifact of cache residency rather than a property of
the operator. It is not: `raw/l2_residency_probe.csv` sweeps that working set to 42.67x L2 and
latency moves 1.9%. Read the flatness as real, but read it knowing this file alone could not
establish it.

**`pct_of_empirical_bw` is not an efficiency score, and this file proves it.** 16 rows exceed
100%, peaking at **180% of empirical bandwidth and 273% of the scattered reference**
(`graph_compile`, mha, b=128, fp32, uniform). The numerator is a logical byte count that ignores
where bytes are served from; uniform positions rewrite one slot per request, which stays
L2-resident. v2 reported 88% on this column and it read like saturation. It was not.
**Blocker:** real traffic needs Nsight Compute counters, still not run.

**The launch share is measured; the floor beneath it is decomposed elsewhere.** `graph_eager` and
`graph_compile` remove 62-69% of the corresponding direct path's device median, and
`graph_compile` sits at ~0.0107 ms in essentially every configuration. What that flat number
consists of is not visible in this file: `raw/graph_floor_probe.csv` splits it into a 4.3 us
CUDA-event harness cost, a ~3.0 us position copy, a ~2.0 us replay launch, and 1.4-4.6 us of
operator work depending on size.

**Every `device_median_ms` in this file carries that 4.3 us harness cost.** It is common to all
impls, so the ratios between them are sound, but no absolute latency here should be read as the
operator's cost in a real decode loop, and differences smaller than a few microseconds are below
the instrument. `amortized_call_ms` is the column to use for those.

**The graph path's timed region includes the position copy, deliberately.** Capture binds every
other input pointer, so serving with dynamic positions requires that copy. Presenting replay
alone would measure a workload that cannot be run.

**The ragged-positions finding is real in this file and explained by nothing in it.** Compiled b=1
reproduces v2's penalty at 1.41x ragged-over-uniform, while `eager`, `graph_eager` and
`graph_compile` all read ~1.00x at the same configurations on identical traffic.
`raw/ragged_positions_probe.csv` later swept a 2048x range of position spread and found no effect,
and ruled out tensor identity as well -- so v2's "rewriting one slot kept it L2-resident" is
refuted and **this column is not measuring a response to positions.** What it *is* measuring is
still unknown. Note the sweep times uniform before ragged for every config, so run ordering works
against this effect rather than producing it.

**Strided QKV was measured and costs nothing here.** strided-over-packed device median is 0.993
at b=1 and 1.002 at b=32. That is a real answer to v2's open question, bounded to those two
batch sizes: the serving cross runs only at b in {1, 32}, so nothing is established at b=8 or
b=128.

**Four configurations are unmeasured, not measured-and-fast.** `mha` and `mha96` at b=128,
alloc=2048, fp32 need 17.2 GB and 12.9 GB peak contiguous cache (2 sets each) against a 12 GB
budget. Rows are present with `impl=skipped`. A preview of why paged caches exist.

**The timed region is an operator call, not a bare kernel.** For direct impls it includes
PyTorch dispatch and one iterator step for position rotation. Identical across implementations,
so comparisons are fair, but these are not pure kernel times.

**Clock state is recorded at both boundaries but not controlled.** `nvidia-smi -lgc` is
unavailable to this user (permission denied); the rig logs the denial and continues unlocked.
Treat few-percent differences as noise.

**Shapes are real; tensors are not.** All four layouts are real released models' attention
shapes -- `mha` Llama-2-7B, `gqa` Mistral-7B-v0.1, `mqa` Falcon-7B, `mha96` Phi-3-mini-4k --
cross-checked against published `config.json` by `tests/test_anchor_models.py`, and the RoPE
tables are bit-exact against `MistralRotaryEmbedding`. Every timed tensor is still synthetic
`randn`: no weights, no real activation distribution, no decode loop.

**Coverage bounds.** Rotary is always full -- no partial-rotary (Checkpoint D). Contiguous
token-major cache only; no paged, fragmented, or head-major layouts (Checkpoint F).
`torch.compile` runs one backend at default mode, recorded in the env JSON. `rope_theta` is
10000 for every row, which is why anchors at other thetas were not adopted.

**The attention consumer is not modelled in this file.** This operator only writes the cache, so a
layout or kernel that speeds writes may slow the attention reads that follow. `raw/cache_read_probe.csv`
now prices the read side separately -- token-major-as-view costs the consumer ~nothing while
materializing the conversion costs 2.65x -- but nothing times a write and a read in one step, so
the net effect across a full attention layer is still two numbers added together.

**`cache_alloc_len` is an allocation size.** Flat latency across it is expected and is not
evidence about launch-bounding -- the operator touches one RoPE row and one K/V slot per request
and never scans the context.

---

## `raw/l2_residency_probe.csv` (+ `l2_residency_probe.env.json`)

Sweeps the cache-write working set from 0.00x to 42.67x this device's 50.3 MB L2, by drawing each
cycled position set from its own disjoint slot range. Exists to test whether the baseline sweep's
L2-resident writes were flattering its flat-latency result.

**It answers one question and is not a baseline.** No oracle is allocated and no validation gate
runs -- the impls were already cleared by the v3 sweep, and skipping the second cache set is what
lets `mha b=128 fp32` fit the footprint budget here. Do not quote its latencies as operator
timings; they were measured under a deliberately hostile addressing pattern that no baseline uses.

**Only the write side is swept.** Q/K/V, the trig tables and the graph's captured output buffer
are the same allocations on every invocation, so the read side stays cache-resident throughout.
That is a fair model of decode, where Q/K/V arrive hot from the projection, but it means the probe
bounds cache-write cost specifically, not total memory cost.

**`compile` and `graph_compile` only**, b=128 and alloc=2048 only, ragged packed/identity only,
one seed, 100 iterations per point. Nothing here speaks to small batch, and small batch is where
the operator is most overhead-dominated to begin with.

**The marginal-rate figure is an inference, not a counter reading.** Comparing across head layouts,
a 127x range of cache-write traffic costs 2.0 us, which implies a marginal rate near 2 TB/s and
therefore that the stores are not being paid for at memory speed. That is arithmetic over two
measured medians; confirming *why* still needs Nsight Compute.

**This file predates the ordering control and is therefore uncontrolled.** Its slot ladder was
measured with the baseline rung first and once, so "the first rung is slow" and "the smallest
working set is fast" are the same number here. The ragged-positions probe later measured that
artifact at up to **25%** on this rig -- an order above the +1.9% this file reports. The ladder is
bracketed now (`benchmark_utils.bracketed`, `ordering_drift`) but has not been re-run, so the
+1.9% and +5.9% figures should be read as upper bounds on a real effect, not as the effect.

---

## `raw/graph_floor_probe.csv` (+ `graph_floor_probe.env.json`)

A six-rung ladder isolating what a `graph_compile` sample consists of: a Python call that launches
nothing, the position copy alone, replay of a graph holding one trivial kernel, both launches
together, the operator graph replayed with positions frozen, and `graph_compile` itself. Operator
work is the top rung minus the two-launch rung.

**The two latency definitions disagree, and the file reports both rather than choosing.** By event
median the operator does 0.03 us of work at b=32; by amortized loop it does ~1.4 us. Event
bracketing costs 4.3 us per sample and partially masks GPU execution, so the amortized figures are
the ones to use here. Neither is a counter reading.

**The frozen-position rung cannot serve decode** and is labelled `NOT SERVABLE` in the stage table.
It exists only to separate the copy from the replay. Its cache writes are also L2-resident, worth
about 2% per `raw/l2_residency_probe.csv`, so it is measured under friendlier conditions than the
serving path.

**"Minimal replay" is one trivial kernel, not an empty graph**, so the ~2.0 us attributed to replay
launch includes that kernel's own cost. It is an upper bound on pure replay overhead.

**`compile` lineage only**, five configurations, one seed. Nothing here covers `eager` or
`graph_eager`, and the 4.3 us harness figure is specific to this device and torch build.

**Subtraction across rungs assumes the components do not interact.** They share a stream and
serialise, which is why the arithmetic roughly closes, but the derived numbers carry about
+/- 0.5 us of run-to-run noise -- visible directly in the ladder, where nominally identical rungs
differ by that much between configurations.

---

## `raw/ragged_positions_probe.csv` and `raw/ragged_positions_probe_fresh.csv`

Two arms, 96 rows each, testing whether the v3 sweep's 1.41x ragged/uniform at compiled b=1 is a
response to position values, to the tensor object carrying them, or to neither. The `_fresh` file
is the same ladder with every tensor reallocated per rung, since the sweep timed uniform and
ragged as separate configs and so gave them separate compiles and allocations.

**It is a negative result, and the negative is the point.** Across a 2048x range of position
spread no impl departs from its own baseline by more than about 5% once `ordering_drift` is
subtracted, and spread=1-with-distinct-tensors reads the same as spread=2048. Do not read this as
"the probe found nothing"; it refutes two specific mechanisms (L2 residency of a rewritten slot,
and tensor identity) that were the standing explanations.

**It does not explain the sweep's 1.41x.** That number is still unexplained. What this file
establishes is only that it is not a response to positions. The sweep always times uniform before
ragged, so its own ordering works against its effect rather than producing it -- which rules
ordering out as the explanation there too.

**Read `ordering_drift` before any ratio in this file.** It is what caught this probe's own first
result: `compile` at `mha b=1 fp32` reads 0.760 against its opening baseline and 0.754 against its
closing one, so the ratio *is* the drift. Four configurations, three impls, one seed, no repeats
beyond the bracket -- a single drift figure per group, not a distribution.

**`mha b=1 fp32` is where the instrument is loudest**, and it is also the least representative
configuration in the file: absolute times are ~20-30 us, so a fixed per-run warmup cost is a large
fraction. Nothing here transfers to b=32.

---

## `raw/cache_read_probe.csv` (+ `cache_read_probe.env.json`)

160 rows over 32/32 units, no errors, no skips. Four arms -- SDPA on head-major, SDPA on a
transposed token-major view, the same with the conversion materialized, and a pure read floor --
across four head layouts, b in {1, 32} and ctx in {128, 512, 2048, 8192}, bf16 only. The ladder is
bracketed, so head-major is measured twice per unit and the fifth rung is the control, not a
fifth arm.

**This is the read side measured alone, not a full attention layer.** It prices the consumer of
the cache this operator writes. Nothing here times a write and a read in one step, so a layout's
net effect is still two numbers added together.

**The view figure is now separated from ordering drift, and it survives.** An earlier version of
this file was written before the ladder was bracketed, so `head_major` was both the baseline and
the first arm measured and the 1.007 could not be told apart from run ordering. Re-run bracketed:
`ordering_drift` is a median **1.0018** over 32 ladders (range 0.9907-1.0391), and the transposed
view reads a median **1.006** of head-major (worst 1.123). The effect is inside its own control,
which is the strongest form this claim can take -- it does not show that the view is free, it
shows that any cost is below what this instrument can distinguish from ordering. The **2.65x
median copy penalty (worst 5.62x)** is two orders above the drift and was never at risk. Neither
figure moved in the re-run (1.007 -> 1.006, 2.65 -> 2.65).

**`read_gbps` is not DRAM bandwidth at the narrow end.** Same trap as `pct_of_empirical_bw`: a
cache that fits in L2 is served from L2, and the small configurations exceed the 553 GB/s
streaming reference. Quote `ctx=8192`.

**bf16 only, SDPA only, causal masking absent.** No `attn_mask` or `is_causal` is passed -- a
single decode query attends to all cached keys, which is the decode case, but nothing here covers
prefill or windowed attention. Which SDPA backend served each arm is not recorded, so a
transposed view falling back to a slower kernel would be invisible except in the timing.

**GQA rows go through `enable_gqa`**, never by materializing KV heads, so their `cache_read_bytes`
is the true traffic. An older torch without that argument makes the arm error rather than silently
change the workload.

---

## `raw/amdahl_probe.csv` (+ `amdahl_probe.env.json`)

91 rows (90 timed, 1 skipped, 0 errors) over Mistral-7B-v0.1 in HF transformers 5.12.1: six
ablation rungs x three execution modes x five surviving configurations. Exists to answer the one
question the operator microbenchmarks cannot -- what fraction of a real decode step this operation
is, and therefore what a fused kernel could win end to end.

**There is no serving engine anywhere in this file, and the bias is not one-directional.** The
denominator is HF's own decode loop; vLLM, FlashAttention, SGLang and TensorRT-LLM are not
installed. Attention is PyTorch SDPA, the cache is HF's contiguous `StaticCache`, and there is no
paged KV or continuous batching. Both halves of the ratio differ from a serving stack, in opposite
directions: HF's RoPE + append is unoptimized PyTorch (`op_removed` deletes 672 of 2123 traced
calls) where vLLM has hand-written kernels, which **inflates the numerator**; while at b=32 HF's
step is 62-108 ms against a ~27 ms weight-bandwidth floor, which **inflates the denominator**. At
b=1 the denominator is honest (27.9 ms measured against ~26.2 ms analytic) so only the numerator
caveat applies. **Do not quote any number here as "X% of a serving step"** without naming the
stack; it is a statement about HF's decode path.

**Five of the six rungs are numerically invalid by construction** and are marked
`numerically_valid=False`. `op_removed`, `rope_removed` and `append_removed` corrupt the model's
output; `op_doubled` double-rotates Q, which feeds attention directly. All hold shapes, dtypes and
allocations fixed, and `tests/test_amdahl_probe.py` asserts each one actually changes the logits --
a patch that silently did nothing would report a clean 0% operator share.

**The headline saving is an upper bound that is not achievable, and the file says so.** Removing
the operation saves 2.0-5.3% depending on configuration, but **doubling it costs 0.0-0.9%** under
both compiled modes. Mirror ratio 0.046-0.275. The two do not mirror, so most of what removal buys
is the compiler restructuring around a deleted mutation, which no faster kernel reproduces. The
realizable figure is the doubling slope: **0.81% at the gate configuration**, 15.7 us per layer.

**The same check reads 0.99-1.65 under eager, which is the positive control.** `op_doubled` is not
an insensitive instrument -- in eager, removal and doubling mirror each other, as a serial
operation should. It registers ~0 only under `torch.compile` and cudagraphs, on the same operation
in the same runs. Without those eager rows the demotion would rest on a null result.

**The 0.5 mirror cutoff was chosen after seeing the data.** The measured ratio at the gate is
0.275, so the demotion holds for any cutoff above that; `test_the_verdict_is_stable_across_
plausible_cutoffs` checks 0.35/0.5/0.75. It is not a wide margin and the ratio is reported beside
every verdict so the choice can be judged.

**Sliding-window eviction is disabled, so this models plain-append decode, not Mistral as
shipped.** Mistral-7B-v0.1 declares `sliding_window` 4096, which makes `StaticCache` build
`StaticSlidingWindowLayer` for every layer; that `update()` rolls the whole cache tensor once full,
O(window) per layer per step. Patching it away in a rung would have attributed eviction cost to
RoPE + append and inflated the saving. `decode_loop.load_model` sets `sliding_window = None` and
`layer_types` to all-`full_attention`, every row carries `layer_types_forced=True`, and the
constructed layers are asserted non-sliding. Phi-3-mini-4k declares 2047, so no ungated anchor
escapes this.

**`op_replaced_ref` is this project's reference, and it loses badly** -- 3% slower at b=1, 19% at
b=8, 21-29% at b=32, worsening with batch. The seam is numerically equivalent to HF's own path by
test, so this is a real result about the reference, not an artifact. Two deviations are baked into
it: HF gathers the cos/sin rows once per model step in `MistralRotaryEmbedding` rather than once
per layer, so the per-layer operation does not include the table gather the locked signature
implies; and HF's cache is `[B, Hkv, S, D]`, so the append writes through a transposed view.

**One configuration is unmeasured, not measured-and-fast.** b=32 alloc-equivalent ctx=1024 needs
weights 14.5 + KV 5.8 + ~3.0 GB workspace = 23.2 GB against a 22.5 GB budget on a 23.4 GB card.
It is present with `rung=skipped` and the arithmetic as its reason. The b=32 arm therefore stops
at ctx=512, and the batch arm at b=8/b=1 runs only at ctx=1024.

**The b=32 ctx sweep goes downward on purpose.** The operator's own cost is context-independent
but attention's KV read grows with context, so the operator's *share* peaks at small ctx --
2.96% at 128 falling to 2.02% at 512. Gating at ctx=1024, as the plan first proposed, would have
understated the operator's best case.

**Provenance is a dirty tree, and the delta is known.** `env.json` records `fd89bfed` with
`git_dirty=True`; the uncommitted delta is exactly the crash-safety fix, footprint guard and gate
licensing committed as `9e9f3c4`, which is the state that regenerates this file. Unlike the
archived v2, the producing code is committed and named. `env.json` for this probe is also written
flat rather than nested under an `environment` key as `graph_floor_probe.env.json` is; the code was
left as it ran rather than edited after the fact, so shape and data stay consistent.

**Clocks unlocked but stable**: 2647 MHz SM at both boundaries, throttle reasons `0x0` at both.
`nvidia-smi -lgc` remains denied to this user.

**Timing is amortized step time only.** Step scale is milliseconds so the 4.3 us event floor is
irrelevant here, but no per-layer figure in this file is measured directly -- every one is a step
delta divided by 32. Single seed, single GPU, single model, bf16 only, one prompt length per
configuration, no prefill measurements.

**This file has one timing per rung and therefore no error bars, and the verdict turns on
differences near the noise floor.** The realizable figure is 0.81% of a step; a three-repeat check
run by hand during the session put run-to-run spread around 0.4%. Same order. `--repeats` was
added afterwards and records median/min/max/stdev/spread per rung, and the summariser marks a
saving smaller than the observed spread as unresolved -- but **this CSV predates that flag**, so
`noise_pct` is absent and the summary omits the resolved/unresolved paragraph. Until it is
re-run, read 0.81% as *at most about the noise floor*, which is inside the dead band either way,
rather than as a measured quantity. Regenerate with
`python benchmarks/probe_amdahl.py --repeats 3`.

**A re-run was attempted and did not survive; nothing from it is in this repo.** It was killed at
roughly 85% when its session ended, and because probes wrote provenance only at the end, it had
already overwritten this CSV with 73 partial rows beside a three-day-old `env.json` -- the
committed file was restored from git and the partial rows were not retained. The gate
configuration did complete before the kill and indicated the removal resolving above the spread
while the doubling sat inside it, which is the outcome this section predicts, but **there is no
surviving artifact and those figures must be reproduced before being quoted.** Two things changed
as a result: provenance is now written with the first rows and carries `complete`, and `--resume`
exists. Note `--resume` will *refuse* against this committed CSV, correctly, because its
`env.json` records an older tree -- so the re-run is a fresh ~2 hour job.

Note also that repeats reuse one compiled instance, so even after a re-run the spread is timing
noise only. Rungs are separately compiled -- that is what guarantees the patch is traced -- so a
rung-vs-rung delta carries compile-to-compile variance nothing here captures.

**Every row is Mistral, and the architecture-generic patch path is untested against real
weights.** When this file was produced, `rung_patches` hardcoded Mistral's modeling module, so the
`--model` flag would have silently patched a module a non-Mistral model never consults and
reported a clean 0% share. That is fixed (patch targets resolve from the live model) and covered
by a Phi-3 regression test, and a fused-QKV seam adapter now exists -- but no non-Mistral anchor
has been run end to end. The transfer formula in `benchmark_methodology.md`, which predicts how
the fraction moves with a different weight-read denominator, therefore remains asserted rather
than tested.

---

## `raw/bandwidth_reference.json`

**The continuity reference is streaming, not a bound on this operator.** Large contiguous `copy_`
and `add_` remain the friendliest pattern. `reference_gbps` is deliberately still the 512 MiB copy
median so old and new result files stay comparable.

**The size sweep locates the knee, but only for FP32 copies.** The 1/4/16/64/256/512 MiB sweep
measures cache-resident throughput through 16 MiB and DRAM throughput from 64 MiB upward on this
48 MiB-L2 GPU. It does not generalize the knee or throughput to other dtypes or GPUs.

**The scattered reference matches one important shape, not every row.** It uses random K/V slot
writes at MHA b=128, alloc=2048, FP32 granularity with one 8 GiB K+V cache set. That is the shape
behind the old 88%-of-streaming headline and stays under the 12 GB budget, but batch size, head
layout, dtype, and allocation length differ for other operator rows.

**Medians of a short run** (50 iterations after warmup) with no clock-state provenance or control.
The current rig records boundary state and can request a lock, but this historical file predates
that change; treat its ±few-percent differences as noise.

---

## `profiling/` (`profile_summary.json`, `trace_*.json`, `nsys_*_cuda_gpu_kern_sum.csv`)

`profile_summary.json` and its traces were re-exported together at `d32af14`; the `nsys_*` kernel
summaries are older and were not.

**Kernel counts are exact; the attribution of the remaining time is inferred.** The traces
establish kernels per invocation and device time per invocation directly. That the rest of the
measured span is host-side dispatch follows from the arithmetic (1.8 µs of GPU work inside a
25.5 µs call). For the *measured* recoverable launch share use the baselines' graph-vs-direct
latency instead, which is where 62-69% comes from.

**`distinct_kernels` counted non-kernel device events until `1c05249`.** The name table was keyed
over kernels, memcpys and memsets alike, so the compiled path at b=1 was reported as 2.00 kernels
per invocation and **3 distinct** -- the third being a `Memcpy DtoD`. Both are now keyed off kernel
events and the file was re-exported, but any quote of "distinct kernels" taken from this file
before that commit is high by the number of memcpy and memset names present.

**Profiling perturbs what it measures.** Reported device times come from profiled runs and are
not the timing path's numbers; use `operator_kernels_v1.csv` for latency. Re-exporting moved
device time per invocation by 0.7-4.9% between two runs of identical arguments, which is the scale
of run-to-run variation to expect from these numbers.

**Narrow config coverage.** bf16, ragged positions, `alloc=2048`, b∈{1,32}, MHA and GQA only.
Kernel counts for other dtypes and batch sizes are not established, though the inductor fusion
count is expected to vary (already 1 vs 2 kernels between MHA b=32 and GQA b=32).

**nsys traces cover the whole process.** Setup allocation and RNG kernels appear alongside the
operator; the NVTX range marks the measured region. The `.nsys-rep` and `.sqlite` files are
gitignored as large regenerable binaries — regenerate with `benchmarks/nsight_capture.sh`.
Only the kernel-summary CSVs are tracked.

**No Nsight Compute — and on this machine it is a permission wall, not a missing harness.**
No occupancy, register pressure, sector efficiency, or achieved-bandwidth counters, so no
statement here explains *why* a kernel takes the time it does. `benchmarks/ncu_capture.sh`
requests `dram__bytes_{read,write}.sum` and `lts__t_sectors_op_{read,write}.sum` — the pair that
would settle whether the rows above 100% of empirical bandwidth are L2 service — and
`benchmarks/ncu_report.py` reduces that output against the logical count. Both are written and
the reporter is tested, but `ncu` returns **`ERR_NVGPUCTRPERM`** for this user (verified
2026-08-07, ncu 2025.2.1), the same class of block as `--lock-clocks`. Enabling counters needs a
root-level change (`NVreg_RestrictProfilingToAdminUsers=0` plus a reboot, or running the capture
as root); the capture script probes for the denial up front and exits 3 with those instructions
rather than profiling every configuration first.

**The ncu parser has never seen real ncu output.** Because no capture can run here,
`parse_ncu_csv` is written against the documented `--csv` schema and tested against a fixture
built from it. Treat the first real capture as also testing the parser: a column-name change
between ncu versions would surface as zero metric rows, which the reporter refuses rather than
reporting as zero traffic.

---

## `profiling/kernels/` (`profile_summary.json`, `trace_*.json`)

The custom rungs profiled separately, opt-in exactly as they are in the sweep. Reproduce with
`python benchmarks/profile_operator.py --impls cuda_separate cuda_fused --outdir
results/profiling/kernels`.

**This is the only place the fusion ablation's premise is measured rather than argued.**
`cuda_separate` is exactly 2.00 kernels per invocation and 2 distinct (`rope_kernel`,
`kv_append_kernel`); `cuda_fused` is exactly 1.00 and 1 (`fused_rope_kv_append_kernel`), at every
representative configuration. Before this the launch count was read off the source.

**Its device-time ratio corroborates the sweep and does not replace it.** Summing per-kernel
device time, fusion is 1.63-2.10x here against the sweep's 1.82x from amortized wall clock --
independent instruments (CUPTI kernel spans against a host timing loop) agreeing within the spread
of either. But these are profiled runs at `--iters 20`, so the absolute microsecond figures are
not the timing path's and should not be quoted as latency.

**Same narrow coverage as the sibling directory.** bf16, ragged positions, `alloc=2048`,
b in {1, 32}, MHA and GQA only. The launch counts are structural and will not vary with dtype or
batch, but nothing here covers `mqa`, `mha96`, or the b=128 memory-bound corner where the fused
kernel loses to `graph_compile`.

---

## `raw/archive/operator_baseline_v2_preV3.csv`

Superseded and **not comparable** to v3, and not regenerable from any committed state: its
`env.json` records a dirty tree at `1db04be`, a commit containing neither `validation.py` nor
`workload.py`. See `raw/archive/README.md` for the six reasons (provenance, autograd state,
added impls, added head layouts, added serving-layout columns, gate strength). Several claims
derived from it -- "3.8-4.0x eager at b>=8", "88% of achievable, near-saturated" -- are
contradicted by v3.

---

## `raw/archive/operator_baseline_v1_preB1.csv`

Superseded and **not comparable** to anything later. See `raw/archive/README.md` for the five
reasons (byte accounting, latency naming, uniform-only positions, no validation gate, column
rename).
Retained only so earlier claims can be traced to the data that produced them.
