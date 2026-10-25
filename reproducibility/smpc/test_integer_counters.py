"""Two-party regression for the exact counter semantics used by A1.

Reveals synthetic public test vectors only. It is not a cost measurement.
"""
import argparse
import crypten
import crypten.communicator as comm
import crypten.mpc as mpc
import torch

from support import atomic_json, environment, now


def party():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    bits = torch.tensor([[0., 1., 1., 1.], [0., 0., 0., 0.],
                         [1., 1., 0., 1.], [0., 0., 1., 0.]], dtype=torch.float64)
    secret_bits = crypten.cryptensor(bits, src=0).gt(.5)
    assert secret_bits.encoder.scale == 1
    preceding = secret_bits.cumsum(-1) - secret_bits
    expected_count = bits.cumsum(-1) - bits
    expected_zero = (expected_count == 0).double()
    old = preceding.eq(0).get_plain_text().double()
    fixed = preceding.lt(1).get_plain_text().double()
    assert torch.equal(fixed, expected_zero), (fixed, expected_zero)
    first = secret_bits * preceding.lt(1)
    absent = secret_bits.sum(-1, keepdim=True).lt(1)
    last = torch.tensor([0., 0., 0., 1.], dtype=torch.float64)
    selected = (first + absent * last).get_plain_text().double()
    wanted = torch.tensor([[0., 1., 0., 0.], [0., 0., 0., 1.],
                           [1., 0., 0., 0.], [0., 0., 1., 0.]], dtype=torch.float64)
    assert torch.equal(selected, wanted), (selected, wanted)
    # The equivalent comparison also works with the default fixed-point scale.
    default = crypten.cryptensor(expected_count, src=0).lt(1).get_plain_text().double()
    assert torch.equal(default, expected_zero)
    return dict(rank=comm.get().get_rank(), passed=True,
                comparison_scale=int(secret_bits.encoder.scale),
                old_eq_decoded=old.tolist(), expected_zero=expected_zero.tolist(),
                fixed_lt1_decoded=fixed.tolist(), selected_positions=selected.tolist(),
                old_eq_bug_reproduced=not torch.equal(old, expected_zero))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = mpc.run_multiprocess(world_size=2)(party)()
    assert rows is not None and len(rows) == 2
    assert all(row["passed"] for row in rows)
    atomic_json(args.out, dict(time=now(), passed=True, per_rank=rows, environment=environment()))
    print("A1 two-party counter regression passed; old eq bug reproduced:",
          [row["old_eq_bug_reproduced"] for row in rows], flush=True)


if __name__ == "__main__":
    main()
