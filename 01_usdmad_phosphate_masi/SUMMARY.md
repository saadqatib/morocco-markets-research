# USD/MAD vs Phosphate vs MASI: Connectedness

Research conducted with Dr. Amine Ben Amar, Professor of Economics at UM6P (Mohammed VI Polytechnic University).

**Data:** monthly log returns, Mar 2002 to Dec 2025 (285 months)
USD/MAD, EUR/USD, MASI: Investing.com · Phosphate rock, DAP: World Bank Pink Sheet

| Test | Result |
|---|---|
| Correlation USD/MAD-phosphate | 0.20 |
| Correlation USD/MAD-EUR/USD | −0.97 |
| USD/MAD ~ EUR/USD | β = −0.78, R² = 0.93 |
| + phosphate (lags 0-3), EUR/USD controlled | not significant (p > 0.2) |
| + MASI, EUR/USD controlled | n.s. full sample; weakly negative post-2015 (β ≈ −0.08, p ≈ 0.05) |
| Granger causality (VAR) | none (all p > 0.2) |
| Diebold-Yilmaz total connectedness (USD/MAD, phos, MASI) | 3.4% (2.7% excl. Dec-2023 outlier) |
| Same incl. EUR/USD | 27%, almost all USD/MAD ↔ EUR/USD |
| Rolling 60m connectedness | min 2.5% (2018), max 19.7% (Mar 2020), now 15.5% |

## Takeaways
- The 60/40 EUR/USD peg basket drives USD/MAD; phosphate and MASI add no significant explanatory power.
- The raw phosphate correlation is a dollar effect, not a phosphate effect.
- Post-2015 phosphate significance comes only from the Dec-2023 −82% break; disappears without it or with DAP.
- World Bank phosphate rock is a contract price: 57% zero monthly changes, flat since 2024, so it is weak for monthly analysis.

## Next steps
- MASI vs OCP share price / Moroccan export revenues
- Daily DAP/TSP futures instead of the contract price

## Files
`analysis.py` (run from this folder) · `results.json` · `returns_monthly.csv` · `rolling_connectedness.csv` · `charts.png` · `data/`

## Breitung-Candelon frequency-domain causality (Ben Amar method)
Script: `bc_usdmad.py` (same `bc_test` as BenAmar_replication). Output: `bc_results.json`, `bc_usdmad_phos_masi.png`. 5% crit = 5.99.

| Direction | VAR(3), full | VAR(3), excl. Dec-23 | VAR(9) | Post-2015 |
|---|---|---|---|---|
| Phosphate → USD/MAD | sig: >10.6m & 2-4m (W 11.0) | sig: >7m & 2-4m (W 14.2) | sig: 6-13m & 3m | sig: >5m |
| Phosphate → USD/MAD (EUR/USD-adjusted) | none | none | none | none |
| Phosphate → MASI | none | none | sig 3-4m | none |
| MASI → phosphate | none | none | sig >10m | none |
| MASI ↔ USD/MAD | none | none | none | none |

**Read:** phosphate "causes" USD/MAD at long (>6-10m) and short (2-4m) cycles in every spec, but the effect vanishes once EUR/USD is removed → it runs through the dollar, not a Moroccan phosphate channel. Phosphate ↔ MASI appears only with 9 lags → fragile.
