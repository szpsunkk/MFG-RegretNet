"""Independent unit-cost measurements, at the declared public tensor shapes."""
import argparse
import json
import resource
import time

import crypten
import crypten.communicator as comm
import crypten.mpc as mpc
import torch

from secure_mechanism import bitonic_sort, comparison, raw_minimum, raw_relu
from support import atomic_json, environment, now


def party(spec, repeats, output):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    rank = comm.get().get_rank()
    shape, kind = spec["shape"], spec["kind"]
    g = torch.Generator().manual_seed(6401)
    x = torch.rand(shape, generator=g, dtype=torch.float64) * 2 - 1
    y = torch.rand(shape, generator=g, dtype=torch.float64) * 2 - 1
    enc = crypten.cryptensor(x, src=0)
    enc_y = crypten.cryptensor(y, src=0)
    if kind == "public_linear":
        weight = torch.rand((shape[-1], spec["output_width"]), generator=g, dtype=torch.float64) / shape[-1]
    rows = []
    for rep in range(repeats + 1):
        comm.get().barrier()
        if rank == 0:
            atomic_json(output + ".heartbeat.json", dict(status="running", start_unix=time.time(),
                                                         repetition=rep, operation=spec, time=now()))
        comm.get().reset_communication_stats()
        start = time.perf_counter()
        if kind == "tanh":
            result = enc.tanh()
        elif kind == "sigmoid":
            result = enc.sigmoid()
        elif kind == "bitonic_sort":
            result = bitonic_sort(enc)
        elif kind == "public_linear":
            result = enc.matmul(weight) + .1
        elif kind == "public_scale":
            result = enc * .25
        elif kind == "secret_multiply":
            result = enc * enc_y
        elif kind == "minimum":
            result = raw_minimum(enc, enc_y if spec["rhs"] == "secret" else 0.)
        elif kind == "maximum_zero":
            result = raw_relu(enc)
        elif kind.startswith("compare_"):
            result = comparison(enc, enc_y if spec["rhs"] == "secret" else 0., kind[8:])
        elif kind == "input_sharing":
            result = crypten.cryptensor(x, src=0)
        elif kind == "output_reveal":
            result = enc.get_plain_text()
        else:
            raise ValueError(kind)
        elapsed = time.perf_counter() - start
        stats = comm.get().get_communication_stats()
        row = dict(rep=rep, warmup=rep == 0, time_s=elapsed, rounds=stats["rounds"], bytes=stats["bytes"],
                   rss_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
        rows.append(row)
        # Materialized eager tensors: retain until after timing, then release.
        del result
        atomic_json(output + ".rank%d.json" % rank, dict(operation=spec, rank=rank, records=rows))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repetitions", type=int, default=5)
    args = ap.parse_args()
    spec = json.loads(args.spec)
    crypten.cfg.communicator.verbose = True
    results = mpc.run_multiprocess(world_size=2)(party)(spec, args.repetitions, args.out)
    if results is None or len(results) != 2:
        raise SystemExit("primitive measurement failed")
    rows = []
    for a, b in zip(*results):
        rows.append(dict(a, time_s=max(a["time_s"], b["time_s"]), rounds=max(a["rounds"], b["rounds"]),
                         bytes=(a["bytes"] + b["bytes"]) / 2., rss_peak_mib=max(a["rss_peak_mib"], b["rss_peak_mib"])))
    valid = rows[1:]
    atomic_json(args.out, dict(operation=spec, status="complete", environment=environment(), records=rows,
                              summary=dict(time_s=sum(r["time_s"] for r in valid) / len(valid),
                                           rounds=sum(r["rounds"] for r in valid) / len(valid),
                                           MB=sum(r["bytes"] for r in valid) / len(valid) / 1e6)))


if __name__ == "__main__":
    main()
