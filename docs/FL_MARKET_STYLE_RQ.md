# FL-market–style figures → privacy paper RQ mapping

This bundle reproduces the **axes and figure types** from `FL-market.pdf` using:

- **Synthetic privacy-paper bids** (`generate_privacy_paper_bids`) for mechanism-side metrics (same spirit as RQ4/RQ5).
- **RQ4 raw JSON** for federated test accuracy vs round.

| FL-market–style figure | RQ / role | Script | Default output |
|------------------------|-----------|--------|----------------|
| Effect of financial budget on error bound | Mechanism finite-sample bound (`mse_eval` → `error_bound_by_plosses_weights_batch`) | `exp_rq/fl_market_style_figures.py` | `run/.../figures/figure_flm_error_bound_vs_budget.png` |
| Invalid gradient rate vs budget | Same setting as `experiments.invalid_rate_budget` (product of “no loss” probs) | same | `figure_flm_invalid_grad_rate_vs_budget.png` |
| Model accuracy over FL training rounds | **RQ4** FL benchmark | same (reads `--rq4-dir`) | `figure_flm_RQ4_model_accuracy_vs_round.png` |
| Effect of parameter M | Regret / IR violation vs **number of items M** (checkpoints per `M` via `resolve_mfg_regretnet_ckpt`) | same (`--m-items ...`) | `figure_flm_effect_of_M.png` |

Raw numbers: `run/privacy_paper/fl_market_style/raw/budget_sweeps.json`, `guarantees_vs_m.json`.

## Commands

```bash
# Full pipeline (slow: loads nets, sweeps budget, optional misreport search for Fig. M)
bash scripts/run_fl_market_style.sh

# Re-plot from saved JSON + RQ4 only
python exp_rq/fl_market_style_figures.py --out-dir run/privacy_paper/fl_market_style \
  --only-plot --rq4-dir run/privacy_paper/rq4/raw
```

**Checkpoints:** resolve `MFG-RegretNet` / `RegretNet` via `exp_rq/rq1_ckpt_resolve.py` (e.g. `result/mfg_regretnet_privacy_*_checkpoint.pt`).  
For **Fig. M**, you need one trained checkpoint per `M` in `--m-items`; missing `M` are skipped.

## Relation to existing RQ scripts

- **RQ4** (`exp_rq/rq4_fl_benchmark.py`, `rq4_plot_paper_figures.py`): primary FL accuracy curves; this bundle adds FL-market–labeled export for the same data.
- **RQ5** / privacy–utility: budget sweeps here are **mechanism-side** error bounds, not DP–accuracy curves; combine both in the paper if needed.
