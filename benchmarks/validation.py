import torch

from decode_kernels.reference import apply_rope, fused_rope_kv_append_ref
from benchmarks import positions as pos
from benchmarks.workload import CACHE_SENTINEL, build_op_args

TOLERANCES = {
    torch.float32: {"atol": 1e-6, "rtol": 1e-6},
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 8e-3, "rtol": 8e-3},
}


class ValidationError(AssertionError):
    pass


def _max_abs_diff(a, b):
    if a.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max())


def _written_mask(cfg_num_requests, cache_alloc_len, request_indices, positions, device):
    mask = torch.zeros(cfg_num_requests, cache_alloc_len, dtype=torch.bool, device=device)
    mask[request_indices, positions] = True
    return mask


def validate_once(candidate, cfg, device, seed, positions):
    """Compare one candidate invocation against the reference oracle on identical inputs.

    Checks rotated Q, rotated K at the addressed slots, byte-exact raw V at the
    addressed slots, and that every unaddressed cache slot still holds the sentinel.
    """
    tol = TOLERANCES[cfg.dtype]
    a_ref = build_op_args(cfg, device, seed, positions, sentinel_cache=True)
    a_cand = build_op_args(cfg, device, seed, positions, sentinel_cache=True)

    q_ref = fused_rope_kv_append_ref(*a_ref)
    q_cand = candidate(*a_cand)

    ri, p = a_cand.request_indices, a_cand.positions
    mask = _written_mask(cfg.num_requests, cfg.cache_alloc_len, ri, p, device)

    k_ref_slots = a_ref.k_cache[ri, p].clone()
    k_cand_slots = a_cand.k_cache[ri, p]
    v_cand_slots = a_cand.v_cache[ri, p]
    del a_ref  # release the oracle's caches; only its extracted slots are still needed

    cos_p = a_cand.cos[p].to(torch.float32)
    sin_p = a_cand.sin[p].to(torch.float32)
    k_expected = apply_rope(a_cand.k.to(torch.float32), cos_p, sin_p).to(a_cand.k_cache.dtype)

    atol, rtol = tol["atol"], tol["rtol"]
    failures = []
    if not torch.allclose(q_cand.float(), q_ref.float(), atol=atol, rtol=rtol):
        failures.append(f"q_rot mismatch vs oracle (max_abs={_max_abs_diff(q_cand, q_ref):.3e})")
    if not torch.allclose(k_cand_slots.float(), k_ref_slots.float(), atol=atol, rtol=rtol):
        failures.append("k_cache written slots mismatch vs oracle "
                        f"(max_abs={_max_abs_diff(k_cand_slots, k_ref_slots):.3e})")
    if not torch.allclose(k_cand_slots.float(), k_expected.float(), atol=atol, rtol=rtol):
        failures.append("k_cache written slots are not rotate(k) "
                        f"(max_abs={_max_abs_diff(k_cand_slots, k_expected):.3e})")
    v_exact = bool(torch.equal(v_cand_slots, a_cand.v))
    if not v_exact:
        failures.append("v_cache written slots are not byte-exact raw v "
                        f"(max_abs={_max_abs_diff(v_cand_slots, a_cand.v):.3e})")

    unwritten_k = a_cand.k_cache[~mask]
    unwritten_v = a_cand.v_cache[~mask]
    intact = bool((unwritten_k == CACHE_SENTINEL).all() and (unwritten_v == CACHE_SENTINEL).all())
    if not intact:
        n_k = int((unwritten_k != CACHE_SENTINEL).sum())
        n_v = int((unwritten_v != CACHE_SENTINEL).sum())
        failures.append(f"unaddressed cache slots modified (k:{n_k} v:{n_v} elements)")

    return {
        "ok": not failures,
        "failures": failures,
        "max_abs_diff_q": _max_abs_diff(q_cand, q_ref),
        "max_abs_diff_k_cache": _max_abs_diff(k_cand_slots, k_ref_slots),
        "v_byte_exact": v_exact,
        "unaddressed_slots_intact": intact,
        "num_addressed_slots": int(mask.sum()),
        "tolerance": tol,
    }


def validation_position_sets(cfg, device, seed):
    """Position sets used for validation: two sets of the timed mode plus a uniform set,
    so every config is validated under both uniform and ragged addressing."""
    sets = pos.build_position_sets(cfg.num_requests, cfg.cache_alloc_len,
                                   cfg.position_mode, seed, 2, device)
    out = list(sets)
    if cfg.position_mode != pos.UNIFORM:
        out += pos.build_position_sets(cfg.num_requests, cfg.cache_alloc_len,
                                       pos.UNIFORM, seed, 1, device)
    return out


def validate_candidate(candidate, cfg, device, seed):
    """Aggregate validation over several position sets. Returns a report; never raises."""
    reports = [validate_once(candidate, cfg, device, seed, p)
               for p in validation_position_sets(cfg, device, seed)]
    failures = [f for r in reports for f in r["failures"]]
    return {
        "ok": not failures,
        "failures": failures,
        "num_position_sets": len(reports),
        "max_abs_diff_q": max(r["max_abs_diff_q"] for r in reports),
        "max_abs_diff_k_cache": max(r["max_abs_diff_k_cache"] for r in reports),
        "v_byte_exact": all(r["v_byte_exact"] for r in reports),
        "unaddressed_slots_intact": all(r["unaddressed_slots_intact"] for r in reports),
        "tolerance": reports[0]["tolerance"],
    }


def validate_or_raise(candidate, cfg, device, seed):
    report = validate_candidate(candidate, cfg, device, seed)
    if not report["ok"]:
        raise ValidationError(f"{cfg.label()}: " + "; ".join(report["failures"]))
    return report
