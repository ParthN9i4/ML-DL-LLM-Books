"""Artifact 32.1 -- Ring all-reduce from scratch, plus a ZeRO parallelism planner.

torch.distributed cannot launch multiple processes in this sandbox, so the P
"workers" are simulated as P numpy arrays inside one process, and every byte
that would cross the wire is counted explicitly.  The algorithm and the byte
counts are exactly what NCCL's ring would do; only the transport is fake.
"""
import numpy as np

try:
    import torch
except ImportError:
    torch = None

# ----------------------------- ring all-reduce ------------------------------

def ring_allreduce(bufs):
    """All-reduce (sum) across P simulated workers arranged in a ring.

    bufs : list of P 1-D numpy arrays of equal length n, with P | n.
    Mutates bufs in place; afterwards every worker holds sum_p bufs[p].
    Returns sent[p] = bytes worker p transmitted on its outgoing link.
    """
    P, n = len(bufs), bufs[0].size
    assert n % P == 0, "pad the tensor so P divides its length"
    chunks = [np.split(b, P) for b in bufs]          # views into each buffer
    sent = [0] * P

    # Phase 1: reduce-scatter, P-1 steps.  At step t worker p sends chunk
    # (p - t) mod P to its right neighbour p+1, which adds it in.  After the
    # last step, worker p owns the fully reduced chunk (p + 1) mod P.
    for t in range(P - 1):
        msgs = []
        for p in range(P):
            c = (p - t) % P
            msgs.append(((p + 1) % P, c, chunks[p][c].copy()))
            sent[p] += chunks[p][c].nbytes
        for dst, c, payload in msgs:                 # deliver only after all send
            chunks[dst][c] += payload

    # Phase 2: all-gather, P-1 steps.  At step t worker p forwards the
    # completed chunk (p + 1 - t) mod P to its right neighbour, overwriting.
    for t in range(P - 1):
        msgs = []
        for p in range(P):
            c = (p + 1 - t) % P
            msgs.append(((p + 1) % P, c, chunks[p][c].copy()))
            sent[p] += chunks[p][c].nbytes
        for dst, c, payload in msgs:
            chunks[dst][c][:] = payload
    return sent


def naive_allreduce_root_bytes(P, nbytes):
    """Gather-to-root then broadcast: bytes crossing the ROOT's single link.

    Root receives (P-1) full tensors, then sends (P-1) full copies back out.
    That link is the bottleneck; everyone else moves only 2*nbytes.
    """
    return 2 * (P - 1) * nbytes


# --------------------------- ZeRO memory planner ----------------------------

# Mixed-precision Adam, bytes per parameter (fp16/bf16 compute, fp32 state):
#   2 (bf16 weights) + 2 (bf16 grads) + 4 (fp32 master) + 4 (m) + 4 (v) = 16.
W_LO, G_LO, OPT = 2, 2, 12          # OPT = fp32 master copy + m + v

def per_device_bytes(N, P, stage):
    """Per-device bytes for N params, DP degree P, ZeRO stage 0..3.
    Excludes activations, buffers, and framework overhead."""
    if stage == 0:  return N * (W_LO + G_LO + OPT)           # plain DP
    if stage == 1:  return N * (W_LO + G_LO + OPT / P)       # shard optimizer
    if stage == 2:  return N * (W_LO + (G_LO + OPT) / P)     # + shard grads
    if stage == 3:  return N * (W_LO + G_LO + OPT) / P       # + shard weights
    raise ValueError(stage)

def plan(N, P_list, budget_gb, label):
    print(f"\n--- planner: {label}, N = {N:.1e} params, budget {budget_gb} GB/device ---")
    print(f"{'DP':>5} | " + " | ".join(f"{s:>12}" for s in
          ["DP (ZeRO-0)", "ZeRO-1", "ZeRO-2", "ZeRO-3"]))
    for P in P_list:
        row = []
        for stage in range(4):
            gb = per_device_bytes(N, P, stage) / 1e9
            row.append(f"{gb:8.1f} {'ok ' if gb <= budget_gb else 'XXX'}")
        print(f"{P:>5} | " + " | ".join(f"{r:>12}" for r in row))


# --------------------------------- checks -----------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # (a) exact correctness: integer-valued float64 so summation order is exact
    for P, n in [(2, 8), (4, 16), (5, 40), (8, 64), (16, 4096)]:
        data = [rng.integers(-1000, 1000, n).astype(np.float64) for _ in range(P)]
        truth = np.sum(data, axis=0)
        bufs = [d.copy() for d in data]
        sent = ring_allreduce(bufs)
        for b in bufs:
            assert np.array_equal(b, truth), "ring result != direct sum"

        # (b) bytes on wire per worker == 2*(P-1)/P * S, exactly, every worker
        S = data[0].nbytes
        expect = 2 * (P - 1) * S // P
        assert all(s == expect for s in sent), (sent, expect)

        # (c) gather+broadcast bottleneck moves strictly more than a ring worker
        naive = naive_allreduce_root_bytes(P, S)
        assert naive > expect and naive == P * expect
        print(f"P={P:>3} S={S:>6}B  ring/worker={expect:>6}B  "
              f"= 2(P-1)/P*S exactly; naive root link={naive:>7}B  ({P}x worse)")

    # (d) 16 bytes/param, hand-counted on a concrete 2-layer MLP (8->4->2):
    #     W1 8x4 + b1 4 + W2 4x2 + b2 2  ->  32+4+8+2 = 46 parameters.
    shapes = [(4, 8), (4,), (2, 4), (2,)]
    n_params = sum(int(np.prod(s)) for s in shapes)
    assert n_params == 46
    hand = 0
    for s in shapes:                       # materialize all six tensors and count
        hand += np.zeros(s, np.float16).nbytes      # bf16/fp16 weights
        hand += np.zeros(s, np.float16).nbytes      # low-precision grads
        hand += 3 * np.zeros(s, np.float32).nbytes  # fp32 master + m + v
    assert hand == 16 * n_params == per_device_bytes(n_params, 1, 0) == 736
    print(f"\nAdam mixed-precision hand count: {n_params} params -> {hand} bytes "
          f"= 16 x {n_params}  [exact]")

    # torch cross-check: Adam really does keep two fp32 state tensors per param
    if torch is not None:
        tp = [torch.nn.Parameter(torch.randn(*s)) for s in shapes]
        opt = torch.optim.Adam(tp)
        sum((p ** 2).sum() for p in tp).backward()
        opt.step()
        st = sum(v[k].numel() for v in opt.state.values()
                 for k in ("exp_avg", "exp_avg_sq"))
        assert st == 2 * n_params
        print(f"torch.optim.Adam state elements: {st} = 2 x {n_params}  [matches]")
    else:
        print("[skipped: torch not installed]")

    # (e) the planner on two real model sizes, 80 GB devices (GB = 1e9 bytes)
    plan(7e9,  [8, 64, 512], 80, "7B model")
    plan(70e9, [8, 64, 512], 80, "70B model")
    print("\nAll assertions passed.")
