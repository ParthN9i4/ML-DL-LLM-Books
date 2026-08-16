"""Minimal test harness — no pytest dependency.

Usage inside an exercise file:

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from check import check, summary

    check("test name", got, expected)
    check("boolean test", condition)
    check("float test", got, expected, tol=1e-6)
    check("array test", got_array, expected_array, tol=1e-6)

Run every exercise in the ladder:

    python3 check.py            # all tiers
    python3 check.py tier0_math # one tier

Deliberately dependency-free: this file imports nothing but the standard
library, so a failing exercise fails for its own reasons and not because the
harness could not load.
"""

import sys

_pass = _fail = 0


def _as_flat_list(x):
    """Flatten anything array-like to a plain list of floats. Returns None if it isn't.

    Numpy SCALARS (np.float64, and any 0-d array) must be treated as scalars, not
    as one-element arrays: they carry .ravel() and .tolist() just like a real
    array, so a naive duck-type check classifies np.float64(1.5) as array-like
    while the plain float 1.5 is not, and every comparison between the two then
    fails with "one side is array-like and the other is not".
    """
    if getattr(x, "ndim", None) == 0:
        return None
    # numpy array (or anything with .ravel()/.tolist()) without importing numpy
    if hasattr(x, "ravel") and hasattr(x, "tolist"):
        try:
            flat = x.ravel().tolist()
            return [float(v) for v in flat]
        except (TypeError, ValueError):
            return None
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            sub = _as_flat_list(item)
            if sub is None:
                try:
                    out.append(float(item))
                except (TypeError, ValueError):
                    return None
            else:
                out.extend(sub)
        return out
    return None


def _close(got, expected, tol):
    """|got - expected| < tol, elementwise for array-likes.

    tol=0.0 means "bit-exact" and is compared with <=, not <; otherwise a
    request for exact equality could never be satisfied by any input.
    """
    cmp = (lambda d: d <= tol) if tol == 0 else (lambda d: d < tol)
    g_flat = _as_flat_list(got)
    e_flat = _as_flat_list(expected)

    if g_flat is not None and e_flat is not None:
        if len(g_flat) != len(e_flat):
            return False, f"length {len(g_flat)} != {len(e_flat)}"
        worst = 0.0
        for a, b in zip(g_flat, e_flat):
            worst = max(worst, abs(a - b))
        return cmp(worst), f"max|diff| = {worst:.3e}"

    if g_flat is not None or e_flat is not None:
        return False, "one side is array-like and the other is not"

    try:
        diff = abs(float(got) - float(expected))
    except (TypeError, ValueError):
        return False, "not numeric"
    return cmp(diff), f"|diff| = {diff:.3e}"


def check(name, got, expected=True, tol=None):
    """Assert got == expected, or |got - expected| < tol when tol is given.

    With tol set, both scalars and array-likes (numpy arrays, nested lists) are
    compared elementwise and the worst absolute difference is reported — which
    is the number you actually want when a gradient check fails.
    """
    global _pass, _fail

    detail = ""
    if tol is not None:
        ok, detail = _close(got, expected, tol)
    else:
        try:
            ok = bool(got == expected)
        except ValueError:
            # numpy raises on the ambiguous truth value of a multi-element
            # array; fall back to exact elementwise comparison. Note this must
            # test equality, not `< 0`, which no difference can ever satisfy.
            g_flat, e_flat = _as_flat_list(got), _as_flat_list(expected)
            ok = (
                g_flat is not None
                and e_flat is not None
                and len(g_flat) == len(e_flat)
                and all(a == b for a, b in zip(g_flat, e_flat))
            )

    if ok:
        _pass += 1
        print(f"  PASS  {name}" + (f"   [{detail}]" if detail else ""))
    else:
        _fail += 1
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")
        print(f"        got:      {_short(got)}")
        print(f"        expected: {_short(expected)}")


def _short(x, limit=200):
    s = repr(x).replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + " …"


def summary():
    """Print pass/fail counts and exit with an appropriate code."""
    total = _pass + _fail
    print(f"\n{'=' * 46}")
    print(f"  {_pass}/{total} passed", end="")
    if _fail:
        print(f"  ({_fail} FAILED)")
    else:
        print("  — all good!")
    print(f"{'=' * 46}")
    sys.exit(1 if _fail else 0)


# ---------------------------------------------------------------------------
# Runner: execute every exercise as a subprocess and report which ones pass.
# ---------------------------------------------------------------------------

def _run_all(only_tier=None):
    import os
    import subprocess

    root = os.path.dirname(os.path.abspath(__file__))
    tiers = sorted(
        d for d in os.listdir(root)
        if d.startswith("tier") and os.path.isdir(os.path.join(root, d))
    )
    if only_tier:
        tiers = [t for t in tiers if t == only_tier or t.startswith(only_tier)]
        if not tiers:
            print(f"no tier matching {only_tier!r}")
            return 1

    results = []
    for tier in tiers:
        tier_dir = os.path.join(root, tier)
        exercises = sorted(
            f for f in os.listdir(tier_dir)
            if f.startswith("ex") and f.endswith(".py")
        )
        if not exercises:
            continue
        print(f"\n{'─' * 46}\n{tier}\n{'─' * 46}")
        for ex in exercises:
            path = os.path.join(tier_dir, ex)
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True, timeout=600,
            )
            ok = proc.returncode == 0
            results.append((tier, ex, ok))
            print(f"  {'PASS' if ok else 'FAIL'}  {ex}")
            if not ok:
                tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
                for line in tail:
                    print(f"        {line}")

    n_ok = sum(1 for _, _, ok in results if ok)
    print(f"\n{'=' * 46}")
    print(f"  {n_ok}/{len(results)} exercises passed")
    print(f"{'=' * 46}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(_run_all(sys.argv[1] if len(sys.argv) > 1 else None))
