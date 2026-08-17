"""What a benchmark CSV holds and whether resuming it would be accepted.

Finding out how far a killed run got meant relaunching it and reading the "already measured"
lines -- a 2 h job to answer a question the sidecar can answer in a second.
"""
import argparse
import glob
import json
import os
import platform
import sys

from benchmarks.benchmark_utils import (
    REPO_ROOT,
    RUN_SEGMENT,
    UNIT_KEY,
    _git_metadata,
    _lock_owner,
    _owner_is_gone,
    env_path,
    lock_path,
    read_env_doc,
    read_rows,
    resume_is_safe,
)

DEFAULT_GLOB = os.path.join(REPO_ROOT, "results/raw/*.csv")


def _lock_state(out):
    path = lock_path(out)
    if not os.path.exists(path):
        return None
    owner = _lock_owner(path)
    state = "dead" if _owner_is_gone(owner, platform.node()) else "live"
    return f"pid {owner.get('pid')} on {owner.get('host')} since {owner.get('since')} ({state})"


def _sidecar(out):
    try:
        with open(env_path(out)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def status(out, *, tree=None):
    rows = read_rows(out)
    doc = _sidecar(out)
    env = read_env_doc(env_path(out))
    units = [r.get(UNIT_KEY, "") for r in rows]
    keyed = sorted({u for u in units if u})
    safe, why = resume_is_safe(out, tree if tree is not None else _git_metadata(), rows=rows)
    return {
        "out": out,
        "rows": len(rows),
        "units": len(keyed),
        # what a resume would actually adopt, which is fewer when a unit recorded only failures
        "units_inheritable": doc.get("units_inheritable"),
        "unkeyed_rows": sum(1 for u in units if not u),
        "complete": doc.get("complete"),
        "covered": doc.get("covered"),
        "units_planned": doc.get("units_planned"),
        "units_missing": doc.get("units_missing") or [],
        "measured_at": env.get("git_sha"),
        "dirty": env.get("git_dirty"),
        "timestamp": env.get("timestamp_utc"),
        "segments": len(env.get("resume_chain") or []),
        "cli_args": env.get("cli_args"),
        "lock": _lock_state(out),
        "resumable": safe,
        "resume_detail": why,
        "segment_counts": _segment_counts(rows),
    }


def _segment_counts(rows):
    counts = {}
    for r in rows:
        counts[r.get(RUN_SEGMENT, "")] = counts.get(r.get(RUN_SEGMENT, ""), 0) + 1
    return counts


def _display_path(out):
    rel = os.path.relpath(out, REPO_ROOT)
    return out if rel.startswith("..") else rel


def render(s, *, show_all_missing=False):
    lines = [_display_path(s["out"])]
    unkeyed = (f", {s['unkeyed_rows']} rows predating unit keys (a resume re-measures those "
               f"whole)" if s["unkeyed_rows"] else "")
    lines.append(f"  {s['rows']} rows in {s['units']} finished units{unkeyed}")
    inheritable = s["units_inheritable"]
    if inheritable is not None and inheritable != s["units"]:
        lines.append(f"  only {inheritable} of those would be inherited: "
                     f"{s['units'] - inheritable} recorded no usable measurement and a resume "
                     f"retries them")

    if s["complete"] is None:
        lines.append("  complete: unrecorded (written before the flag existed)")
    else:
        lines.append(f"  complete: {str(s['complete']).lower()}"
                     + ("  <- the run did not reach its end" if s["complete"] is False else ""))
    if s["covered"] is None:
        lines.append("  coverage: not declared by this probe")
    else:
        lines.append(f"  coverage: {s['units']}/{s['units_planned']} planned units"
                     + ("" if s["covered"] else f", {len(s['units_missing'])} missing"))
        shown = s["units_missing"] if show_all_missing else s["units_missing"][:8]
        for key in shown:
            lines.append(f"    missing: {key}")
        if len(s["units_missing"]) > len(shown):
            lines.append(f"    ... and {len(s['units_missing']) - len(shown)} more "
                         f"(--all-missing to list them)")

    where = f"{(s['measured_at'] or '?')[:12]}{' (dirty)' if s['dirty'] else ''}"
    lines.append(f"  measured at {where} on {s['timestamp'] or '?'}")
    if s["segments"] > 1:
        spread = ", ".join(f"segment {k or '?'}: {v} rows"
                           for k, v in sorted(s["segment_counts"].items()))
        lines.append(f"  written across {s['segments']} processes ({spread})")
    if s["lock"]:
        lines.append(f"  lock held by {s['lock']}")
    verdict = "would resume" if s["resumable"] else "REFUSED"
    lines.append(f"  {verdict}: {s['resume_detail']}")
    # the tree is all this can check: the argument comparison needs the arguments the resuming
    # run would be given, which are not knowable from here
    if s["resumable"] and s["cli_args"]:
        lines.append(f"    against these settings: {_settings(s['cli_args'])}")
    return "\n".join(lines)


def _settings(cli_args):
    return " ".join(f"{k}={v}" for k, v in sorted(cli_args.items())
                    if k not in ("out", "resume"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help=f"CSVs to inspect (default: {DEFAULT_GLOB})")
    ap.add_argument("--all-missing", action="store_true",
                    help="list every outstanding unit rather than the first few")
    args = ap.parse_args()

    paths = args.paths or sorted(glob.glob(DEFAULT_GLOB))
    if not paths:
        raise SystemExit(f"no CSVs found at {DEFAULT_GLOB}")
    # one tree read for the whole listing; it cannot change under us mid-run
    tree = _git_metadata()
    # a run killed in warmup has a sidecar and no CSV yet, which is exactly a state worth asking
    absent = [p for p in paths if not (os.path.exists(p) or os.path.exists(env_path(p)))]
    if absent:
        raise SystemExit(f"no such file: {', '.join(absent)}")

    for i, path in enumerate(paths):
        if i:
            print()
        print(render(status(path, tree=tree), show_all_missing=args.all_missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
