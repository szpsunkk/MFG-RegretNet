# RQ1 Main table

| Method | $\bar{\mathrm{rgt}}$ | $\mathcal{V}_{\mathrm{IR}}$ (%) | Truthful (%) | Bid CV |
|--------|----------------------|-----------------------------------|--------------|--------|
| PAC | 0.0000 | 0.0000 | 100.0000 | 0.3514 ± 0.0000 |
| VCG | 0.0000 | 0.0000 | 100.0000 | 0.3514 ± 0.0000 |
| CSRA | 60.7450 ± 9.5361 | 2.1872 ± 0.0951 | 2.8068 ± 0.2687 | 0.9751 ± 0.0056 |
| RegretNet | 1.2155 ± 0.5153 | 0.0295 ± 0.0120 | 92.3411 ± 0.1832 | 0.1979 ± 0.0050 |
| DM-RegretNet | 0.9065 ± 0.6730 | 0.0182 ± 0.0179 | 92.3198 ± 0.3049 | 0.1981 ± 0.0050 |
| MFG-Pricing | 196.0848 ± 18.1167 | 0.0000 | 0.1339 ± 0.0430 | 0.5646 ± 0.0015 |
| Ours | 0.5505 ± 0.0072 | 0.5125 ± 0.0621 | 2.4725 ± 0.1313 | 0.1996 ± 0.0050 |

*5 seeds: 42,43,44,45,46. Paired t-test (Ours vs method) on mean regret per seed:*

- **Ours vs CSRA**: p = 0.0001 *
- **Ours vs DM-RegretNet**: p = 0.3011
- **Ours vs MFG-Pricing**: p = 0.0000 *
- **Ours vs PAC**: p = 0.0000 *
- **Ours vs RegretNet**: p = 0.0436 *
- **Ours vs VCG**: p = 0.0000 *

* p<0.05.

MFG-Pricing: posted budget split pay_i=B·ε_i/Σε_j, pl_i=ε_i (mean-field style). IR violation % = fraction of agent-slots with u_i=p_i−v_i·ε_alloc<0. Truthful % = normalized regret < 0.02 (neural after PGA; baselines grid). Bid CV = CV of strategic v (neural) or payments (baselines) across agents per profile.
