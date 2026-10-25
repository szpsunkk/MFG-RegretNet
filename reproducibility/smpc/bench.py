"""Two-party timing and numerical screening. No changes to CrypTen precision."""
import argparse
import json
import os
import resource
import time
from pathlib import Path

import crypten
import crypten.communicator as comm
import crypten.mpc as mpc
import torch

from secure_mechanism import mechanism, plaintext_purchase, public_context
from support import ERROR_LIMITS, REVISION, atomic_json, environment, now, reference


def heartbeat(path, **data):
    if comm.get().get_rank() == 0:
        atomic_json(path, dict(time=now(), unix=time.time(), **data))


def party(n, repetitions, output, seed, screening):
    rank = comm.get().get_rank()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model, common, timing = reference()
    b, k = torch.tensor([[50.]], dtype=torch.float64), torch.tensor([[3.]], dtype=torch.float64)
    records = []
    for rep in range(repetitions + 1):
        reports = common.generate_privacy_paper_bids(n, 1, seed + rep).double()
        ctx = public_context(model, reports[0])
        with torch.no_grad():
            expected = timing.ours_cached(model, reports, b, k)
            expected_s = plaintext_purchase(model, reports, ctx)
        diagnostics = {}
        comm.get().barrier()
        heartbeat(output + ".heartbeat.json", status="running", N=n, repetition=rep, start_unix=time.time())
        comm.get().reset_communication_stats()
        start = time.perf_counter()
        encrypted = crypten.cryptensor(reports[0, :, :2], src=0)
        outcomes, recipe = mechanism(encrypted, ctx, diagnostics=diagnostics)
        revealed = [x.get_plain_text() for x in outcomes]
        elapsed = time.perf_counter() - start
        stats = dict(comm.get().get_communication_stats())
        # A2 diagnostic on synthetic inputs only, after the benchmark snapshot.
        diagnostic_start = time.perf_counter()
        revealed_s = diagnostics["S"].get_plain_text()
        diagnostic_time = time.perf_counter() - diagnostic_start
        diagnostic_stats = comm.get().get_communication_stats()
        s_deviation = float((revealed_s.double() - expected_s).item())
        errors = {name: float((a.double() - e[0]).abs().max())
                  for name, a, e in zip(("e", "p", "h"), revealed, expected)}
        finite = all(bool(torch.isfinite(x).all()) for x in revealed + [revealed_s])
        abnormal = not finite or any(errors[name] > limit for name, limit in ERROR_LIMITS.items())
        row = dict(rep=rep, profile_seed=seed + rep, warmup=rep == 0, time_s=elapsed, rounds=stats["rounds"],
                   bytes=stats["bytes"], error_max_abs=errors, finite=finite,
                   abnormal=abnormal, numerical_limits=ERROR_LIMITS,
                   S_plain=float(expected_s.item()), S_secure=float(revealed_s.item()),
                   S_deviation=s_deviation, S_error_abs=abs(s_deviation),
                   diagnostic_reveal_excluded_from_cost=dict(
                       time_s=diagnostic_time, rounds=diagnostic_stats["rounds"] - stats["rounds"],
                       bytes=diagnostic_stats["bytes"] - stats["bytes"]),
                   sum_payment_plus_dividend=float((revealed[1] + revealed[2]).sum()),
                   cap_overshoot=float((revealed[0] - reports[0, :, 1]).clamp_min(0).max()),
                   rss_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
        records.append(row)
        atomic_json(output + ".rank%d.json" % rank, dict(N=n, rank=rank, records=records))
        heartbeat(output + ".heartbeat.json", status="correctness_failed" if abnormal else "repetition_done",
                  N=n, repetition=rep, start_unix=time.time())
        if rank == 0:
            print(json.dumps(dict(N=n, rep=rep, time_s=elapsed, errors=errors,
                                  S_deviation=s_deviation, abnormal=abnormal)), flush=True)
        if abnormal:
            break
    return dict(rank=rank, N=n, records=records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--screening", action="store_true")
    args = ap.parse_args()
    crypten.cfg.communicator.verbose = True
    env = environment()
    assert crypten.__version__ == "0.4.1" and torch.__version__ == "1.13.1+cpu"
    assert crypten.cfg.encoder.precision_bits == 16
    assert crypten.cfg.functions.sigmoid_tanh_method == "reciprocal"
    assert crypten.cfg.mpc.provider == "TFP"
    rows = mpc.run_multiprocess(world_size=2)(party)(args.n, args.repetitions, args.out, args.seed, args.screening)
    if rows is None or len(rows) != 2:
        raise SystemExit("two-party execution failed")
    records = []
    for a, b in zip(rows[0]["records"], rows[1]["records"]):
        assert a["rep"] == b["rep"]
        da, db = a["diagnostic_reveal_excluded_from_cost"], b["diagnostic_reveal_excluded_from_cost"]
        records.append(dict(a, time_s=max(a["time_s"], b["time_s"]),
                            rounds=max(a["rounds"], b["rounds"]),
                            bytes=(a["bytes"] + b["bytes"]) / 2.,
                            error_max_abs={name: max(a["error_max_abs"][name], b["error_max_abs"][name])
                                           for name in ERROR_LIMITS},
                            S_error_abs=max(a["S_error_abs"], b["S_error_abs"]),
                            abnormal=a["abnormal"] or b["abnormal"], finite=a["finite"] and b["finite"],
                            diagnostic_reveal_excluded_from_cost=dict(
                                time_s=max(da["time_s"], db["time_s"]),
                                rounds=max(da["rounds"], db["rounds"]), bytes=(da["bytes"] + db["bytes"]) / 2.),
                            rss_peak_mib=max(a["rss_peak_mib"], b["rss_peak_mib"])))
    valid = [r for r in records if not r["warmup"]]
    failed = any(r["abnormal"] for r in records)
    result = dict(N=args.n, measured=True, status="correctness_failed" if failed else "complete",
                  repetitions=args.repetitions, records=records, environment=env,
                  completed_at=now(), seed=args.seed, revision=REVISION,
                  formal_repetitions_completed=len(valid), numerical_limits=ERROR_LIMITS,
                  screening_only=args.screening, summary_excludes_warmup=True,
                  S_definition="Truthful purchase after the projection's cap-sum clamp, before allocation and 0.01 floor",
                  diagnostic_scope="S revealed only for synthetic validation, after e/p/h timing and communication snapshot")
    if valid:
        result["summary"] = dict(time_s=sum(r["time_s"] for r in valid) / len(valid),
                                 rounds=sum(r["rounds"] for r in valid) / len(valid),
                                 MB=sum(r["bytes"] for r in valid) / len(valid) / 1e6,
                                 error_max_abs={name: max(r["error_max_abs"][name] for r in valid)
                                                for name in ERROR_LIMITS},
                                 S_error_max_abs=max(r["S_error_abs"] for r in valid),
                                 formal_repetitions_completed=len(valid))
    atomic_json(args.out, result)
    if failed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
