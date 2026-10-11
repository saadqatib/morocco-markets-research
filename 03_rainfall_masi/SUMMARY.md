# Rain & the Moroccan stock market (2002-2026)

**Question:** Morocco's harvest depends on rain. Does the stock market price it?
**Data:** ERA5 daily rainfall (Open-Meteo), mean of 8 cereal-belt sites; MASI daily/monthly and 67 listed stocks (Investing.com); World Bank macro; Casablanca daily cloud cover.
**Rain measure:** season-to-date (Sep→) cumulative rainfall vs its 2001-2025 normal.

## 1. Rain drives the real economy
| | ρ | p |
|---|---|---|
| Season rain vs cereal production | 0.789 | 0.0 |
| Season rain vs agricultural value-added growth | 0.614 | 0.0009 |
| Season rain vs real GDP growth | 0.249 | 0.22 |

## 2. …but the stock market ignores it
| Test | Result |
|---|---|
| Season rain vs MASI return Sep-Mar (24 seasons) | ρ = -0.286 (p = 0.17, wrong sign) |
| Season rain vs MASI Apr-Dec (after rain is known) | ρ = -0.117 |
| Monthly: rain z-score → next-month MASI (crop months) | b = -0.3484, p = 0.295 |
| Agri-food & consumer stocks → next month | p = 0.294 |
| Banks → next month | p = 0.163 |
| Rain-timed MASI backtest | Sharpe 0.222 vs 0.434 buy-and-hold |

Only mining & energy co-move with rain (b = 1.4116, p = 0.014), with no plausible mechanism, treated as coincidence with commodity cycles; the pre-specified agri-vs-mining trade loses money (Sharpe -0.287).

## 3. Mood channel (Casablanca weather, Hirshleifer-Shumway 2003)
Cloud cover vs daily MASI return: b = -0.037% per 100 pp, p = 0.332; clear vs overcast days p = 0.799. No sunshine effect.

## Takeaway
Rain explains ~62% of the variance in Moroccan cereal output (ρ = 0.79), yet it has no measurable link to MASI or sector returns. The listed market (banks, telecom, real estate, cement) is largely disconnected from the rain-fed real economy: agriculture is almost absent from the exchange.

Files: `rain.py`, `rain_results.json`, `rain_index_monthly.csv`, `rain_masi_annual.csv`, `rain_masi.png`, `data/`
