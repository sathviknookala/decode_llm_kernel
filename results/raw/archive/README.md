# Archived results

## `operator_baseline_v2_preV3.csv` (+ `operator_baseline_v2_preV3.env.json`)

Superseded by `results/raw/operator_baseline_v3.csv`. Do **not** compare the two directly.

The first reason is not a methodology change but a provenance failure, and it is why v2 could
not simply be extended: its `env.json` records `git_sha=1db04be` with `git_dirty=true` over 17
files, and that commit's tree contains no `validation.py` and no `workload.py`. **No committed
state of this repository can regenerate v2.** Every row also reports `sets=2` or `sets=3`, while
the gate that produced those rows was renamed and extended in `bd70564` — so v2's rows were
cleared by a strictly weaker gate than the one the code describes, and nothing in the file says
so. v3 records a clean SHA and names its gate cases per row.

Differences that make v2 numbers non-comparable:

- **Autograd state.** v2 timed and validated with autograd live, so every call paid dispatch
  through the autograd key and version-counter bookkeeping on the in-place cache write. v3 runs
  both under `torch.inference_mode()`. Eager gained ~9%; compile was unchanged. The change is
  per-dispatch, so it did not shift the two paths equally.
- **Implementations.** v2 has `eager` and `compile`. v3 adds `graph_eager` and `graph_compile`,
  which replay a captured CUDA graph. v2's "dispatch share" was arithmetic; v3 measures it.
- **Head layouts.** v2 swept `mha` and `gqa`, both `head_dim` 128. v3 adds `mqa` (Falcon-7B,
  71:1 @ 64) and `mha96` (Phi-3-mini-4k, 32:32 @ 96). Rows are not one-to-one across files.
- **Serving layouts.** v3 adds `layout` and `request_mapping` columns and times strided-QKV and
  permuted-request variants at b ∈ {1, 32}. v2 exercised those only in validation.
- **Bandwidth denominator.** v2 has only `pct_of_empirical_bw` against a streaming copy. v3 adds
  `pct_of_scattered_write_bw` against random K/V slot writes.
- **Gate strength.** v2's rows predate the `permuted-requests` and `strided-qkv` cases. A v3 row
  cleared strictly more than a v2 row with the same config.

Retained only so earlier claims can be traced to the data that produced them. Several of those
claims are wrong; see the v3 section of `results/LIMITATIONS.md`.

## `operator_baseline_v1_preB1.csv`

Superseded by `results/raw/operator_baseline_v2.csv` (Checkpoint B.1). Do **not** compare
the two directly: v1 used different byte accounting and different latency naming.

Differences that make v1 numbers non-comparable:

- **Byte accounting.** v1 hard-coded FP32 for the cos/sin term and derived every other term
  from a single dtype argument. v2 reads `element_size()` from each real tensor.
- **Latency naming.** v1's `op_call_ms` was an amortized loop measurement mislabelled as an
  operator-call latency; v2 reports `amortized_call_ms` and a separate
  `synchronized_call_ms`. v1 also carried a redundant `p50_ms` alongside `median_ms`.
- **Positions.** v1 pinned every request to `max_seq - 1`, so every timed invocation rewrote
  one cache slot (an L2-resident best case). v2 defaults to deterministic ragged positions and
  rotates position sets across invocations.
- **No validation gate.** v1 timed `torch.compile` without checking its output against the
  oracle. v2 refuses to record timings for a configuration that fails validation.
- **Column rename.** v1's `context_len` is v2's `cache_alloc_len` (the operator never scans
  the preceding context; the value only sets cache allocation and the valid position range).
