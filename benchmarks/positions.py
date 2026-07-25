import torch

UNIFORM = "uniform"
RAGGED = "ragged"
MODES = (UNIFORM, RAGGED)


def build_position_sets(num_tokens, cache_alloc_len, mode, seed, num_sets, device):
    """Pre-generate device-resident position tensors; the timed region only cycles this list.

    uniform: every request decodes at the last valid slot (controlled comparison).
    ragged:  seeded nonuniform positions stratified over early/middle/late thirds
             of the allocated cache, so timed invocations touch varied slots.
    """
    if mode == UNIFORM:
        p = torch.full((num_tokens,), cache_alloc_len - 1, dtype=torch.long, device=device)
        return [p] * num_sets
    if mode == RAGGED:
        return _ragged_sets(num_tokens, cache_alloc_len, seed, num_sets, device)
    raise ValueError(f"unknown position mode {mode!r}, expected one of {MODES}")


def _ragged_sets(num_tokens, cache_alloc_len, seed, num_sets, device):
    m = cache_alloc_len
    total = num_sets * num_tokens
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(total, generator=g, dtype=torch.float64)
    if m < 3:
        vals = (u * m).long().clamp_(0, m - 1)
    else:
        bounds = torch.tensor([0, m // 3, 2 * (m // 3), m], dtype=torch.long)
        stratum = torch.arange(total) % 3
        lo = bounds[stratum]
        hi = bounds[stratum + 1]
        vals = (lo + (u * (hi - lo).to(torch.float64)).long()).clamp_(0, m - 1)
    flat = vals.to(device=device, dtype=torch.long)
    return [flat[i * num_tokens:(i + 1) * num_tokens].contiguous() for i in range(num_sets)]


def position_span(position_sets, cache_alloc_len):
    """Coverage report used to assert ragged sets exercise early/middle/late slots."""
    allp = torch.cat([p.reshape(-1).cpu() for p in position_sets])
    m = cache_alloc_len
    third = m // 3
    return {
        "min": int(allp.min()),
        "max": int(allp.max()),
        "distinct": int(torch.unique(allp).numel()),
        "in_bounds": bool((allp >= 0).all() and (allp < m).all()),
        "early": int((allp < third).sum()),
        "middle": int(((allp >= third) & (allp < 2 * third)).sum()),
        "late": int((allp >= 2 * third).sum()),
    }
