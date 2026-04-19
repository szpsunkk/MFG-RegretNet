# FL-Market

Code, data, and models for two related papers on privacy-aware federated learning marketplaces.

---

## Papers

### 1. FL-Market: Trading Private Models in Federated Learning
**IEEE BigData 2022**

- IEEE Xplore: https://ieeexplore.ieee.org/document/10020232
- arXiv: https://arxiv.org/abs/2106.04384

### 2. Privacy as Commodity: MFG-RegretNet for Privacy-Aware Auctions
See `main_m.pdf` for the full paper.

---

## Repository Structure

```
FL-Market/
├── Core FL & Auction Code
│   ├── FL.py                    # Federated learning core
│   ├── client.py                # FL client
│   ├── aggregation.py           # Aggregation algorithms
│   ├── experiments.py           # Experiment runner
│   ├── regretnet.py             # RegretNet architecture
│   ├── ibp.py                   # Interval bound propagation
│   ├── singleminded.py          # Single-minded auction utilities
│   ├── datasets.py              # Dataset loaders (FL-Market)
│   ├── datasets_fl_benchmark.py # Dataset loaders (privacy paper)
│   └── utils.py                 # Shared utilities
│
├── baselines/                   # Baseline auction mechanisms
│   ├── pac.py                   # PAC mechanism
│   ├── vcg.py                   # VCG mechanism
│   ├── csra.py                  # CSRA mechanism
│   └── mfg_pricing.py           # MFG pricing baseline
│
├── exp_rq/                      # Per-RQ experiment scripts (privacy paper)
│   ├── rq1_incentive_compatibility.py
│   ├── rq2_paper_benchmark.py
│   ├── rq3_paper_complete.py
│   ├── rq4_fl_convergence.py
│   ├── rq5_privacy_utility.py
│   ├── rq6_robustness.py
│   └── ablation_study.py
│
├── scripts/                     # Shell scripts to reproduce paper figures
├── docs/                        # Detailed experiment design docs
├── data/                        # Datasets (MNIST, CIFAR-10, Bank, NSL-KDD)
├── model/                       # Pre-trained RegretNet checkpoints
├── result/                      # Pre-computed experiment results
├── config.yaml                  # Experiment hyperparameters
└── main_m.pdf                   # Privacy paper
```

---

## Installation

```bash
pip install torch torchvision numpy cvxpy matplotlib
```

- Python 3.6+
- `cvxpy` is required for Phase 2/4/5 and all auction experiments
- `matplotlib` is required for figure generation

---

## Reproducing Results (Privacy Paper — MFG-RegretNet)

### Quick Test (30–60 min)
```bash
QUICK=1 bash run_all_rq.sh
```

### Full Reproduction (~6–10 hours)

**Step 1 — Train models:**
```bash
# MFG-RegretNet (required for RQ1/2/3)
python train_mfg_regretnet.py --num-epochs 200 --num-examples 102400 --n-agents 10

# RegretNet and DM-RegretNet (RQ1 comparison)
python train_regretnet_privacy.py --num-epochs 200 --n-agents 10
python train_dm_regretnet_privacy.py --num-epochs 200 --n-agents 10
```

**Step 2 — Run all RQs:**
```bash
bash run_all_rq.sh
```

**Step 3 — View results:**
```bash
ls run/privacy_paper/*/figures/*.png   # figures
ls run/privacy_paper/*/table_*.md      # tables
```

### Individual RQs

| RQ | Description | Command |
|----|-------------|---------|
| RQ1 | Incentive compatibility (regret/IR) | `bash run_rq1.sh` |
| RQ2 | Scalability (N vs. time) | `bash scripts/run_rq2_paper.sh` |
| RQ3 | Revenue & social welfare | `bash run_rq3.sh` |
| RQ4 | FL convergence (MNIST/CIFAR) | `bash scripts/run_rq4_paper.sh` |
| RQ5 | Privacy–utility tradeoff | `bash scripts/run_rq5_paper.sh` |
| RQ6 | Robustness to false bids | `python exp_rq/rq6_robustness.py` |

See `RUN_EXPERIMENTS.md` and `QUICK_REFERENCE.md` for full CLI options.

---

## Reproducing Results (FL-Market Paper)

```bash
# Reproduce Figure 9 (error bounds)
python reproduce_fig9.py

# Reproduce FL accuracy results
python reproduce_fl_accuracy.py

# Run invalid gradient experiment
python run_invalid_gradient_experiment.py
```

Pre-trained models are in `model/`. Datasets are in `data/`.

---

## Phase-by-Phase Pipeline

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `python run_phase1_full_check.py` | Validate data pipeline |
| 2 | `python run_phase2_verify.py` | Verify PAC/VCG baselines |
| 3 | `python train_mfg_regretnet.py` | Train MFG-RegretNet |
| 4 | `python run_phase4_eval.py` | Evaluate RQ1/2/3 |
| 5 | `python run_phase5_tables_figures.py` | Generate tables & figures |

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{flmarket2022,
  title     = {FL-Market: Trading Private Models in Federated Learning},
  booktitle = {IEEE International Conference on Big Data (BigData)},
  year      = {2022},
  doi       = {10.1109/BigData55660.2022.10020232}
}
```
