# RQ1: Incentive compatibility

| Mechanism | Mean Regret | Std | Mean IR | Std IR | Honesty proxy | Note |
|-----------|-------------|-----|---------|--------|---------------|------|
| PAC | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | grid best single-agent dev |
| VCG | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | grid best single-agent dev |
| CSRA | 60.745011 | 9.536130 | 0.007681 | 0.000236 | 0.016196 | grid best single-agent dev |
| MFG-Pricing | 196.084778 | 18.116674 | 0.000000 | 0.000000 | 0.005074 | grid best single-agent dev |
| RegretNet | 1.215529 | 0.515262 | 1.046068 | 0.479205 | 0.451360 | strategy_stability=std(regret) across se |
| DM-RegretNet | 0.906453 | 0.673006 | 0.643633 | 0.626677 | 0.524534 | strategy_stability=std(regret) across se |
| MFG-RegretNet | 0.550496 | 0.007219 | 0.000434 | 0.000060 | 0.644955 | strategy_stability=std(regret) across se |

**RegretNet vs MFG-RegretNet (Welch t-test on per-seed mean regret):** p = 0.0447
