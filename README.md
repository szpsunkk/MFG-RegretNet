# Privacy as Commodity

Research code and a deployed, multi-user demonstration of **MFG-RegretNet** for privacy procurement in federated learning.

**[Open the platform](https://privacy-as-commodity.pages.dev/)** ·
**[Compare algorithms](https://privacy-as-commodity.pages.dev/results)** ·
**[Join an auction](https://privacy-as-commodity.pages.dev/auction)** ·
**[Paper (PDF)](docs/papers/main_m_tdsc.pdf)** ·
**[Supplement (PDF)](docs/papers/main_m_supp.pdf)**

This repository accompanies *Dependable and Secure Privacy Trading in Federated Learning via MFG-RegretNet* by Kangkang Sun, Jianhua Li, Xiuzhen Chen, Weizhi Meng and Minyi Guo. The October 25, 2026 release contains the frozen **P2-v2 / QV-sel** implementation and evidence used in the current manuscript. Earlier implementations remain available at the repository root; use [`reproducibility/`](reproducibility/README.md) for the current results.

The latest manuscript PDFs, updated **October 26, 2026**, are available directly in this repository: [main paper, 17 pages](docs/papers/main_m_tdsc.pdf) and [supplement, 12 pages](docs/papers/main_m_supp.pdf).

## What the project does

An institution can contribute to a shared model while limiting its privacy exposure. A coordinator has a finite procurement budget. MFG-RegretNet conditions the purchase on reported costs, privacy capacities, verified data sizes and a learned population embedding, then assigns privacy budgets and compensation.

The current implementation uses a monotone total-purchase network, a calibrated QV-sel participation rule, capped allocation, numerical integral payments and a data dividend. The accompanying DP-SGD experiments evaluate the resulting allocations with client privacy accounting. The population embedding is a mechanism input; it is not, by itself, a computed dynamic Nash equilibrium.

## A working public platform

[Privacy as Commodity](https://privacy-as-commodity.pages.dev/) is deployed on **Cloudflare Pages with server-side computation and D1 persistence**. It operates independently of the authors' computers or private network.

| Page | What a reviewer can explore |
| --- | --- |
| [Results](https://privacy-as-commodity.pages.dev/results) | MNIST/CIFAR-10 comparisons, fixed-budget and matched-purchased-privacy settings, seed variation, privacy and secure computation costs |
| [Method](https://privacy-as-commodity.pages.dev/method) | Registration, population representation, allocation, payment and settlement |
| [Baselines](https://privacy-as-commodity.pages.dev/baselines) | Individual explanations of the comparison mechanisms |
| [Live auction](https://privacy-as-commodity.pages.dev/auction) | Shared rooms for 2–10 participants, private bids, contracts and visible rewards |
| [Resources](https://privacy-as-commodity.pages.dev/paper) | Manuscript, supplement, downloadable data and provenance |

To try the auction, create a room and share its invite link. Participants register an institution and a synthetic record count; the host verifies registration and opens bidding. Each participant submits a cost and privacy capacity. The server applies the frozen mechanism, issues a contract, and records demo credits after acceptance. Rejection settles to zero. Receipts can be downloaded, and repeat acceptance cannot credit twice.

This demonstrates an implemented path from the paper's mechanism to persistent, multi-user procurement and settlement. The application scenario is **cross-institution intrusion detection**; the reported model-accuracy experiments use MNIST and CIFAR-10. Rewards are demo credits. Connecting real institutions requires identity verification, a client DP/FL agent, verified release evidence, cumulative privacy ledgers and a payment connector. The web service processes bids on the server over HTTPS; its execution is separate from the paper's SMPC benchmark.

The [platform source and deployment guide](platform/README.md) are included. A [Chinese walkthrough](platform/GUIDE_zh.md) explains a demonstration using two devices.

## Reproduce the current paper

The release includes original result JSON, allocation specifications and cumulative privacy ledgers for **880 completed runs**, the frozen checkpoints and threshold table, numerical implementations, experiment drivers, and plotting code. SHA-256 manifests connect the release to the original artifacts.

```bash
git clone https://github.com/szpsunkk/MFG-RegretNet.git
cd MFG-RegretNet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r reproducibility/requirements.txt

# Verify files and reconstruct endpoints, privacy statistics and tables.
python reproducibility/reproduce.py verify
python reproducibility/reproduce.py tables
python reproducibility/reproduce_control.py

# Recreate the main/supplementary RQ4 curves and RQ1–RQ3 plot assets.
python reproducibility/reproduce.py figures

# Check original-selector parity, budget feasibility and payment invariants.
python reproducibility/test_release.py
```

Outputs go to `reproducibility/outputs/`. These commands recompute saved experimental evidence; they do not retrain 880 FL runs. The [reproduction guide](reproducibility/README.md) provides separate commands for model training, FL reruns, attacks, scalability and SMPC. It also documents the historical scripts and portability adapters.

For a lightweight download without historical datasets, use:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/szpsunkk/MFG-RegretNet.git
cd MFG-RegretNet
git sparse-checkout set reproducibility platform docs
```

## What the accuracy evidence shows

The confirmation protocol has **18 settings**, seeds **231/232/233**, **10 clients**, and **100 rounds** per run. Final accuracy is the mean over rounds 96–100, recomputed from saved confusion matrices. The comparator in each setting is the best external mechanism with complete, feasible evidence.

| Dataset | Settings ahead of the best external comparator | Mean paired margin across settings |
| --- | ---: | ---: |
| CIFAR-10 | 8 / 8 | +1.701 percentage points |
| MNIST | 5 / 10 | +0.037 percentage points |

All settings, losses and unreachable comparisons are retained. An aggregation control shows that the advantage depends on the evaluated aggregation rule. Matching purchased privacy per round does not imply equal cumulative privacy for every client; the release includes the client ledgers and distribution summaries.

The main and supplementary curves apply disclosed display smoothing. Unsmoothed per-round values and the original confusion matrices are included; endpoint tables are not smoothed.

## Run the platform locally

Requires Node.js 22 or later.

```bash
cd platform
npm ci
npm test
npm run db:local
npm run dev
```

Open `http://127.0.0.1:8091`. For your own cloud deployment, create a Pages project and D1 database, replace the placeholder in `wrangler.toml`, and follow [the deployment instructions](platform/README.md). No author account credentials or private participant access files are included.

## Release layout

| Location | Purpose |
| --- | --- |
| [`docs/papers/`](docs/papers/) | Latest main paper and supplement PDFs (October 26, 2026) |
| [`reproducibility/`](reproducibility/README.md) | Current experiment entrypoints and reproduction protocol |
| [`reproducibility/snapshot/`](reproducibility/snapshot/) | Original numerical source files and frozen model artifacts, retaining historical relative paths |
| [`reproducibility/evidence/`](reproducibility/evidence/) | Original RQ4 records, allocation specifications, privacy ledgers and paper summaries |
| [`reproducibility/smpc/`](reproducibility/smpc/README.md) | Separate secure computation benchmark and retained failure history |
| [`platform/`](platform/README.md) | Deployable research website and multi-user auction application |
| [`docs/README_legacy.md`](docs/README_legacy.md) | Previous README and historical experiment instructions |

The original root-level scripts, datasets and checkpoints are preserved for historical use. Their commands and timings refer to an earlier manuscript, rather than the current frozen protocol.

## Citation

For the version supplied with this release:

```bibtex
@misc{sun2026dependable,
  title = {Dependable and Secure Privacy Trading in Federated Learning via MFG-RegretNet},
  author = {Sun, Kangkang and Li, Jianhua and Chen, Xiuzhen and Meng, Weizhi and Guo, Minyi},
  year = {2026},
  note = {Author manuscript, October 26 revision},
  url = {https://github.com/szpsunkk/MFG-RegretNet}
}
```

The manuscript link does not imply a publication or acceptance status. See the [validation record](reproducibility/VALIDATION.md) for checks performed on this release.
