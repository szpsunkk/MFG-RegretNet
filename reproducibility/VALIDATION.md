# Public-package validation — 2026-10-25

The release was prepared in an isolated checkout. Existing paper sources, experiment outputs and historical repository files were preserved. No new FL result was substituted for a published measurement.

| Check | Observed result |
| --- | --- |
| Original RQ4 records | All 880 original result files and allocation specifications retain their SHA-256 hashes |
| Endpoint recomputation | All 880 endpoints reconstructed from confusion matrices; all 18 decision cells and stored rows match exactly |
| Cumulative privacy | All 880 original privacy files match the hashes in their result records; all 185 complete setting/arm summaries match the paper's privacy JSON |
| Aggregation control | All 32 original runs recomputed from confusion matrices; the three analysis blocks exactly match the final control report |
| Figures and tables | Original plotting/analysis code generates RQ4 main/supplementary curves, unsmoothed curves, collusion/accounting/scalability figures and LaTeX snippets |
| Original selector parity | Three neural families, budgets 10/30/100, κ=3/30: allocation and compensation agree within 1e-11; capacity, total payment and reported-cost IR checks pass |
| Mechanism training | Full 12,000-update main-family replay completed on CPU in 27.29 s; maximum absolute parameter difference from the frozen checkpoint: **8.2742e-9** |
| FL execution | Original MNIST B=10, α=0.5, seed 231 partition, reports and allocations match hashes; one full-data CPU round completes in 93.15 s, with 10 releases and independent accountant parity |
| RQ1 entrypoint | One-profile reduced grid/PGA smoke completes; this is an execution check, not a new regret result |
| Secure computation code | Plaintext graph/reference parity passes for N=10 and N=50, plus projection boundary and sorting checks; no new secure timing claimed |
| Web mechanism | 36 Python/JavaScript reference profiles: maximum absolute difference **9.3259e-15**; invalid-input checks pass |
| Web evidence | Public source/asset hashes and all 18 displayed best-comparator margins pass verification |

The small parameter discrepancy in the CPU retraining is floating-point reproducibility evidence, not byte-identical checkpoint reproduction. The distributed checkpoint remains the original, hash-checked one. The FL smoke run has `partial_smoke: true` and no final-accuracy endpoint. The GPU was occupied by other work, so the packaging checks used CPU without interrupting that work.

Validation used Python 3.8.3, PyTorch 2.4.1+cu121, NumPy 1.24.4 and Opacus 1.5.3 in the existing scientific environment. The public installation recipe targets Python 3.10 and uses a newer compatible Matplotlib release; a clean installation of that entire environment was not performed during packaging. Rendering bytes can differ across fonts/library versions. Node.js 22 was used for the web tests.

The deployed application's earlier integration acceptance included concurrent registration, private bids, authorization, idempotent clearing and settlement, a complete 10-participant room, and desktop/mobile checks. See [the deployment record](../platform/DEPLOYMENT.md). That earlier deployment validation is separate from the code-package checks above.

The full 880-run FL campaign and the expensive two-party secure timing campaign were not rerun for this GitHub update. Their original evidence is distributed and its analysis is reproduced. The secure failure history and the post hoc A2 numerical screening limits remain disclosed.
