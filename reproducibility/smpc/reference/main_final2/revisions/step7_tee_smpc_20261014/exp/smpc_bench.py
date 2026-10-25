"""Step 7: secure two-party evaluation of one auction round with CrypTen.

Setting: two non-colluding servers hold additive secret shares of the bids
(semi-honest, CrypTen's default trusted-dealer randomness, i.e. the online
phase is measured). The mechanism weights are public, as auction rules are.
Both parties run as processes on one machine and talk over loopback (gloo).

Measured per auction round, for N clients:
  * mfg : MFG-RegretNet forward pass (input sharing, mean-field statistic, two
          tanh trunks, per-client allocation, budget-scaled sigmoid payments,
          and the budget projection), outputs revealed at the end;
  * sort: oblivious bitonic sort of the N bid keys, the core primitive of the
          sorting-based mechanisms (PAC, VCG, CSRA); a lower bound for them.
Reported: wall time (max over parties), communication rounds and bytes (sum
over parties / 2, CrypTen convention), and the error against plaintext.

Usage: python smpc_bench.py <out_json> <N list> <reps> [ckpt_for_N10]
"""
import json
import math
import sys
import time

import torch
import crypten
import crypten.communicator as comm
import crypten.mpc as mpc

VBAR, EPS_MAX, SIZE_MAX, B_SCALE = 1.0, 5.0, 500.0, 100.0
HID, NL = 100, 4  # four tanh hidden layers of width 100 per trunk (n_hidden_layers = 3 plus the input layer)
BUDGET = 50.0


def make_weights(N, seed, ckpt=None):
    """Weights of MFGRegretNetScaled(N, 1, hidden 100, n_hidden_layers 3, separate=True)."""
    torch.manual_seed(seed)
    in_size = 2 * N * 3 + 1

    def mlp(out):
        sizes = [in_size] + [HID] * NL + [out]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            lin = torch.nn.Linear(a, b)
            torch.nn.init.xavier_uniform_(lin.weight)
            layers.append([lin.weight.data.clone(), lin.bias.data.clone()])
        return layers

    A, P = mlp(2 * N), mlp(N)
    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu")["state_dict"]
        for head, layers in (("allocation_head", A), ("payment_head", P)):
            for k, idx in enumerate((0, 2, 4, 6, 8)):
                layers[k][0] = sd[f"{head}.{idx}.weight"].float()
                layers[k][1] = sd[f"{head}.{idx}.bias"].float()
    return A, P


def plain_forward(bids, B, A, P):
    N = bids.shape[0]
    r = bids / torch.tensor([VBAR, EPS_MAX, SIZE_MAX])
    m = r.mean(0, keepdim=True).repeat(N, 1)
    x = torch.cat([r.reshape(-1), m.reshape(-1), torch.tensor([B / B_SCALE])])

    def trunk(layers):
        h = x
        for W, b in layers[:-1]:
            h = torch.tanh(h @ W.t() + b)
        W, b = layers[-1]
        return h @ W.t() + b

    la = trunk(A).view(N, 2)
    z = torch.softmax(la, dim=1)[:, 0]
    p = B * torch.sigmoid(trunk(P))
    pbar = p * torch.clamp(B / p.sum(), max=1.0)
    return z, pbar


def secure_forward(bids_enc, N, B, A, P):
    scale = torch.tensor([1.0 / VBAR, 1.0 / EPS_MAX, 1.0 / SIZE_MAX])
    r = bids_enc * scale                                   # public scaling, local
    m = r.sum(0) * (1.0 / N)                               # mean-field statistic, local
    mrep = crypten.cat([m.unsqueeze(0)] * N, dim=0)
    xs = crypten.cat([r.flatten(), mrep.flatten()])        # secret part of the AHN input

    def trunk(layers):
        W, b = layers[0]
        h = (xs.matmul(W[:, :-1].t()) + (b + W[:, -1] * (B / B_SCALE))).tanh()  # budget column is public
        for W, b in layers[1:-1]:
            h = (h.matmul(W.t()) + b).tanh()
        W, b = layers[-1]
        return h.matmul(W.t()) + b

    la = trunk(A).view(N, 2)
    z = (la[:, 0] - la[:, 1]).sigmoid()                    # two-way softmax = sigmoid of the logit gap
    p = trunk(P).sigmoid() * B
    ratio = p.sum() * (1.0 / B)                            # sum_i p_i / B in (0, N)
    scale_b = ratio.reciprocal()                           # B / sum_i p_i
    scale_b = scale_b - (scale_b - 1.0).relu()             # min(1, B / sum_i p_i)
    pbar = p * scale_b
    return z, pbar


def bitonic_sort(x):
    n = x.size(0)
    idx = torch.arange(n)
    k = 2
    stages = 0
    while k <= n:
        j = k // 2
        while j >= 1:
            partner = idx ^ j
            sel = partner > idx
            i1, i2 = idx[sel], partner[sel]
            asc = ((i1 & k) == 0).float()
            a, b = x[i1], x[i2]
            lt = a.lt(b)                                    # secure comparison
            mn = b + (a - b) * lt
            mx = a + b - mn
            first = mn * asc + mx * (1 - asc)               # public multiplexing, local
            second = mx * asc + mn * (1 - asc)
            order = torch.cat([i1, i2])
            inv = torch.empty_like(order)
            inv[order] = torch.arange(n)
            x = crypten.cat([first, second])[inv]
            stages += 1
            j //= 2
        k *= 2
    return x, stages


def bench_party(N, reps, ckpt, seed):
    rank = comm.get().get_rank()
    torch.set_num_threads(4)
    A, P = make_weights(N, seed, ckpt if N == 10 else None)
    g = torch.Generator().manual_seed(1000 + N)
    out = {"N": N, "mfg": [], "sort": []}
    for rep in range(reps + 1):  # first repetition is a warm-up
        bids = torch.stack([torch.rand(N, generator=g) * VBAR,
                            0.1 + torch.rand(N, generator=g) * (EPS_MAX - 0.1),
                            torch.randint(10, 501, (N,), generator=g).float()], 1)
        # MFG-RegretNet round
        comm.get().barrier() if hasattr(comm.get(), "barrier") else None
        comm.get().reset_communication_stats()
        t0 = time.perf_counter()
        enc = crypten.cryptensor(bids, src=0)
        z, pbar = secure_forward(enc, N, BUDGET, A, P)
        z_p, p_p = z.get_plain_text(), pbar.get_plain_text()
        t1 = time.perf_counter()
        st = comm.get().get_communication_stats()
        zr, pr = plain_forward(bids, BUDGET, A, P)
        tp0 = time.perf_counter()
        for _ in range(20):
            plain_forward(bids, BUDGET, A, P)
        tp = (time.perf_counter() - tp0) / 20
        if rep > 0:
            out["mfg"].append({"s": t1 - t0, "rounds": st["rounds"], "bytes": st["bytes"], "plain_s": tp,
                               "err_z": float((z_p - zr).abs().max()), "err_p": float((p_p - pr).abs().max()),
                               "err_p_rel_budget": float((p_p - pr).abs().max() / BUDGET),
                               "sum_p_secure": float(p_p.sum()), "sum_p_plain": float(pr.sum())})
        # oblivious sort of the N bid keys (v_i * eps_i), padded to a power of two
        n2 = 1 << math.ceil(math.log2(N))
        keys = torch.cat([bids[:, 0] * bids[:, 1], torch.full((n2 - N,), 1e3)])
        comm.get().reset_communication_stats()
        t0 = time.perf_counter()
        enc = crypten.cryptensor(keys, src=0)
        srt, stages = bitonic_sort(enc)
        s_p = srt.get_plain_text()
        t1 = time.perf_counter()
        st = comm.get().get_communication_stats()
        ok = bool(torch.allclose(s_p, torch.sort(keys).values, atol=1e-2))
        if rep > 0:
            out["sort"].append({"s": t1 - t0, "rounds": st["rounds"], "bytes": st["bytes"], "stages": stages,
                                "padded_n": n2, "sorted_ok": ok})
    out["rank"] = rank
    return out


def main():
    out_json, n_list, reps = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3])
    ckpt = sys.argv[4] if len(sys.argv) > 4 else None
    crypten.cfg.communicator.verbose = True
    results = []
    for N in n_list:
        runner = mpc.run_multiprocess(world_size=2)(bench_party)
        per_rank = runner(N, reps, ckpt, 0)
        if per_rank is None:
            print("FAILED", N, flush=True)
            continue
        agg = {"N": N}
        for key in ("mfg", "sort"):
            reps_ = []
            for r in range(len(per_rank[0][key])):
                rows = [pr[key][r] for pr in per_rank]
                row = dict(rows[0])
                row["s"] = max(x["s"] for x in rows)
                row["bytes_total"] = sum(x["bytes"] for x in rows) / 2  # CrypTen convention
                row["rounds"] = max(x["rounds"] for x in rows)
                reps_.append(row)
            agg[key] = reps_
        s = lambda key, f: [r[f] for r in agg[key]]
        agg["summary"] = {
            "mfg_s_mean": sum(s("mfg", "s")) / len(s("mfg", "s")), "mfg_s_min": min(s("mfg", "s")),
            "mfg_s_max": max(s("mfg", "s")),
            "mfg_rounds": agg["mfg"][0]["rounds"], "mfg_MB": agg["mfg"][0]["bytes_total"] / 1e6,
            "mfg_plain_s": sum(s("mfg", "plain_s")) / len(s("mfg", "plain_s")),
            "mfg_err_z_max": max(s("mfg", "err_z")), "mfg_err_p_rel_max": max(s("mfg", "err_p_rel_budget")),
            "sort_s_mean": sum(s("sort", "s")) / len(s("sort", "s")), "sort_s_min": min(s("sort", "s")),
            "sort_s_max": max(s("sort", "s")),
            "sort_rounds": agg["sort"][0]["rounds"], "sort_MB": agg["sort"][0]["bytes_total"] / 1e6,
            "sort_stages": agg["sort"][0]["stages"], "sort_ok": all(s("sort", "sorted_ok"))}
        print(json.dumps({**agg["summary"], "N": N}), flush=True)
        results.append(agg)
        with open(out_json, "w") as fh:
            json.dump({"setting": "CrypTen 2PC semi-honest, trusted-dealer randomness (online phase), loopback gloo, "
                                  "public mechanism weights, secret bids; 4 threads per party",
                       "budget": BUDGET, "reps": reps, "results": results}, fh, indent=1)


if __name__ == "__main__":
    main()
