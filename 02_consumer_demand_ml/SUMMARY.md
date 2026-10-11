# What Moroccans want before Eid: event-driven consumer demand & ML forecasting

**Data:** weekly Google Trends, Morocco, Jan 2010 to Oct 2026 (875 weeks), stitched from four overlapping 5-year windows. 10 spending categories kept (sheep "mouton", caftan, jellaba, gold, dates, Jumia, television, flights, credit, sales); 4 dropped as too sparse (sheep in Arabic, chebakia, Eid clothes, jersey; jersey used only for tournament comparisons). Hijri event dates: Umm al-Qura calendar (Morocco may differ by one day).

## 1. Anticipation around events (search ÷ trailing 52-week median, avg over 13-16 events)
| Category → event | Peak | Weeks ≥1.5× before event |
|---|---|---|
| Sheep → Eid al-Adha | **16.3× normal** (event week) | 6 |
| Jellaba → Eid al-Fitr | 2.5× (3 weeks before) | 10 |
| Dates → start of Ramadan | 1.9× (Ramadan week) | 1 |
| Flights → Eid al-Adha | 1.6× (week after) | 1 |
| Morocco jersey, AFCON 2025 (home) | 5.7× same weeks a year earlier | n/a |

## 2. Natural experiment: the 2025 royal call to skip the Eid al-Adha sacrifice (26 Feb 2025)
LightGBM counterfactual trained on 2010-2024 calendar/event features only.
| | Actual | ML counterfactual | Gap | Placebo 2024 gap |
|---|---|---|---|---|
| Sheep searches, 6 weeks to Eid 2025 (avg) | 7.8 | 23.1 | **−66%** | +7% |
| Sheep searches, peak week 2025 | 16.9 | 65.2 | **−74%** | −5% |
| Sheep searches, 6 weeks to Eid 2026 | 25.5 | 23.3 | +9% (full rebound) | |
Spillovers were within placebo noise (gold −11%, caftan −19% vs placebo −26%).

## 3. Forecasting a full year ahead (walk-forward 2016-2026, 10 categories, MAE in index points)
| Model | All weeks | Ramadan & Eid weeks |
|---|---|---|
| Gregorian seasonal (Fourier regression) | 17.6 | 18.4 |
| LASSO + Hijri event features | 15.1 | 15.7 |
| Seasonal naive (same week last year) | 7.0 | 8.7 |
| **LightGBM + Hijri event features** | **5.8** | **6.9** |
LightGBM cuts error 67% vs the Gregorian seasonal model and 16-21% vs seasonal naive; beats seasonal naive in 88% of category-years. Top features: "≤4 weeks before Eid al-Adha" (sheep), Ramadan timing (dates), Mawlid timing (gold).

## Takeaways
- Moroccan demand runs on the lunar calendar: a model that knows the Hijri calendar forecasts far better than one using only Gregorian seasons, because Ramadan and the Eids move ~11 days a year.
- Sheep demand is the most event-driven category in the data (16× in Eid week), and the 2025 cancellation wiped out about three-quarters of it before a full rebound in 2026.
- Caveats: searches measure intent, not spending; stitching adds noise; Hijri dates approximate Moroccan moon-sighting by ±1 day.

Files: `consumer.py`, `consumer_results.json`, `gtrends_weekly_stitched.csv`, `calendar_features.csv`, `forecast_horserace.csv`, `consumer_events.png`, `data/`
