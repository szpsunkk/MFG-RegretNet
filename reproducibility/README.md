# Reproducing the October 25, 2026 manuscript

This package corresponds to the P2-v2 / QV-sel revision of **MFG-RegretNet**. It preserves the trained parameters, numerical sources and recorded results. Paths inside `snapshot/` intentionally mirror the original research workspace so the mechanism's existing source/checkpoint hash checks continue to work.

Run the commands below from the repository root. Use Python 3.10 and an isolated environment. The FL campaign recorded PyTorch **2.4.1+cu121**, NumPy **1.24.4**, Opacus **1.5.3**, and CUDA on an RTX A5000. CPU execution is supported for validation but need not be bitwise identical to GPU training. Install `requirements.txt`; for the original GPU wheels install PyTorch 2.4.1 and torchvision 0.19.1 from the CUDA 12.1 wheel index. The SMPC experiment uses a separate older environment described in [smpc/README.md](smpc/README.md).

## 1. Recompute the reported evidence

```bash
python reproducibility/reproduce.py verify
python reproducibility/reproduce.py tables
python reproducibility/reproduce.py figures
python reproducibility/reproduce_control.py
```

The first command verifies `MANIFEST.json`. The others unpack the original JSON bytes into `outputs/records/`, validate their original hashes, and run the original analysis functions. They independently reconstruct all 880 endpoints from confusion matrices, apply the recorded missing-target rules, compare the decisions and values to the saved paper results, and recompute cumulative privacy statistics from the original ledgers. Approximately 500 MB of temporary unpacked storage is needed.

The figure command recreates the unsmoothed RQ4 curves, the main four-panel figures and smoothed all-setting figures, plus collusion, accounting and scalability plots and their LaTeX table snippets. The output is `outputs/paper/`. Font and renderer versions can change PDF/PNG bytes without changing the plotted values.

The main/supplement curve presentation uses a symmetric ±10-round display window on the five-round sampling grid; the last point remains the mean of rounds 96–100. This presentation was selected after results were available. The original unsmoothed trajectories are preserved in `evidence/rq4_curves_raw.csv` and the per-run records. No endpoint statistic is smoothed.

## 2. Inspect or retrain the mechanism

```bash
python reproducibility/test_release.py
python reproducibility/train.py --family mfg_p2v2 --out reproducibility/outputs/train-mfg
# Optional matched architecture ablations:
python reproducibility/train.py --family nob_p2v2 --out reproducibility/outputs/train-no-pop
python reproducibility/train.py --family profile_p2v2 --out reproducibility/outputs/train-profile
```

The portable trainer replays the final schedule: seed 42, CPU float64, 12,000 updates, batch 64, Adam, learning rate 0.001 for 10,000 updates then 0.00003 for 2,000 updates, gradient norm 1, and log-uniform κ in [0.003, 300]. It uses the original profile sampler and objective. The output records the maximum parameter difference from the included checkpoint. `--steps 100` is an explicitly marked smoke run, not a replacement checkpoint.

The final threshold table is supplied with SHA-256:

```text
777b80c70115950ba1cb968bc35e9e3e66cf10c93ab6295175eb8b891d7dd3c7
```

The main checkpoint SHA-256 is:

```text
baddeed7201f2d4dfe82bce9aeb65a0fed86d29759618cab4a7130f57a2b382b
```

The table calibration source is `snapshot/worktrees/codex-p2v2-20261020/a7flow/tau.py`: 4,096 synthetic profiles (seed 4242), 11 κ values, 9 amount values, 51 thresholds, a 400-point total grid, log-coordinate interpolation and explicit tie breaking. The historical calibration/selection orchestration is retained for inspection; the portable trainer consumes the frozen table and does not repeat model selection on the confirmation data.

## 3. Rerun an FL comparison

First obtain the public datasets:

```bash
python reproducibility/download_data.py --out data
python reproducibility/replay_fl.py --list
```

Each listed `run_XXXX` identifies an original setting, seed and arm. For example, `run_0000` is the proposed mechanism on MNIST, α=0.5, B=10, seed 231.

```bash
# Verify the complete split, market sequence and allocations without training.
python reproducibility/replay_fl.py --job run_0000 --data-root data --check-only

# A full 100-round rerun; choose a fresh output directory.
python reproducibility/replay_fl.py --job run_0000 --data-root data --device cuda \
  --out reproducibility/outputs/fl

# One real-data round for execution/accountant validation; marked partial.
python reproducibility/replay_fl.py --job run_0000 --data-root data --device cpu \
  --max-rounds 1 --out reproducibility/outputs/fl-smoke
```

To run another mechanism, use its listed job key. No calibration parameter is selected from test accuracy: each run consumes the original frozen, per-round allocation and payment arrays. This replays the FL evaluation conditional on those arrays. The original allocation/calibration and baseline implementation sources are also in the snapshot; regenerating the complete historical model-selection campaign is a separate task from replaying the reported confirmation experiments.

`replay_fl.py` retains the original data split, held-out public training subset, model factory, local optimizer, augmentation/noise streams, aggregation weights, release checks and external/Opacus ledger parity. It verifies the split, reports, allocation and payment hashes before training and stops if an actual release differs. It replaces historical scheduler gates and absolute paths with release/input verification. It writes to a fresh directory and does not overwrite or silently resume another run. `--max-rounds <100` does not produce a final-accuracy endpoint.

The historical A3 comparator and P2-v2 `common.py` files differ only in their output-directory constants; the shared runner has the same source hash. All 880 result records retain their actual source and environment metadata.

The final paper's **data dividend was added after these FL allocations were frozen**. It does not change allocations or training. FL payment fields remain the original compensation records; auction-side experiments and the web app include compensation plus dividend. Do not relabel the old FL payment field as the full final contract.

## 4. Auction-side experiments

These wrappers invoke the unchanged scripts with the included checkpoints. Defaults reproduce the declared sample sizes; reduced flags are for smoke checks only.

```bash
# RQ1: unilateral deviations, 10,000 profiles, several budgets/κ values.
python reproducibility/run_auction.py rq1 --family mfg_p2v2 \
  --out reproducibility/outputs/rq1

# RQ3: budget use, welfare and related economic statistics.
python reproducibility/run_auction.py rq3 --family mfg_p2v2 \
  --out reproducibility/outputs/rq3.json

# Collusion and long-horizon accounting.
python reproducibility/run_auction.py collusion --out reproducibility/outputs/collusion.json
python reproducibility/run_auction.py accountant --noise gaussian \
  --out reproducibility/outputs/accountant-gaussian.json
python reproducibility/run_auction.py accountant --noise laplace \
  --out reproducibility/outputs/accountant-laplace.json

# One of ten adaptive-attack trajectories; repeat --traj 0 through 9.
python reproducibility/run_auction.py adaptive --traj 0 --out reproducibility/outputs/adaptive

# RQ2: timing and incremental GPU memory at N=10,50,100,200,500.
python reproducibility/run_auction.py rq2_timing --device cuda \
  --out reproducibility/outputs/rq2.json
```

Auction-side accounting curves are the paper's synthetic accounting diagnostic; they do not replace the actual DP-SGD ledgers from RQ4. Timing compares the mechanism's computation graphs, with freshly instantiated neural baseline architectures at each N; it is not an accuracy experiment with an N=500 trained baseline. Timings depend on hardware, thread count and load.

## Evidence and figure map

| Paper component | Source / stored evidence | Reproduction route |
| --- | --- | --- |
| RQ1 and attacks | `p2v2_auction_side_20261022/exp/`; `evidence/paper_data.json` | `run_auction.py rq1`, `collusion`, `adaptive`; `reproduce.py figures` |
| RQ2 scalability | `rq2_timing_p2v2.py`; `evidence/rq2_timing.json` | `run_auction.py rq2_timing`; `reproduce.py figures` |
| RQ3 economics | `rq3_p2v2.py`; `evidence/paper_data.json` | `run_auction.py rq3`; `reproduce.py figures` |
| RQ4 accuracy and curves | `evidence/rq4_records.zip`, index and exclusions | `reproduce.py tables/figures`; `replay_fl.py` |
| RQ4 cumulative privacy | `evidence/rq4_privacy_ledgers.zip` | Recomputed and checked by `reproduce.py tables/figures` |
| Aggregation control | `evidence/agg2x2_records.zip`, registration and analysis source | `reproduce_control.py` recomputes all 32 runs; FL control training is separate from the 880-run wrapper |
| Secure computation | `smpc/secure_mechanism.py`, `smpc/smpc_p2v2.json` | Separate environment and commands in `smpc/README.md` |

Paths in the first three rows are relative to `snapshot/main_final2/revisions/`. The package includes the current mechanism and baseline source code, all three final neural-family checkpoints and historical neural baseline checkpoints. It does not include per-round FL model states, the ~2 GB of execution-event traces, private deployment keys, or operating-system/SSH orchestration. Result records preserve hashes of the original event traces, and fresh FL runs write new event traces.

## Release integrity and portability

`SOURCE_PROVENANCE.json` maps copied files to their original workspace-relative paths and source hashes. `MANIFEST.json` records the actual public release bytes, including portability edits. Historical absolute paths inside recorded metadata are provenance strings; launchers do not require those directories. The frozen numerical sources and learned parameters are unchanged. The selected source definitions loaded by the training/parity helpers are verified before execution; no scientific expression is rewritten.

`snapshot/exp`, `next` and `a7flow` also contain historical campaign scripts that rely on archived scheduling state. Use the documented portable entrypoints, rather than running an arbitrary historical coordinator. The frozen checkpoint and table are intentionally loaded as trusted, hash-checked artifacts; do not replace them with untrusted pickle files.

See [VALIDATION.md](VALIDATION.md) for the checks actually performed on the public package. A packaged reproduction workflow is not a claim that all 880 long FL runs or the full secure benchmark were rerun for this release.
