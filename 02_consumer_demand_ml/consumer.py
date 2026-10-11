"""What Moroccans want before Eid: event-driven consumer demand and ML forecasting.
Weekly Google Trends (Morocco), 2010-2026, stitched from four overlapping 5-year windows.
1) Event-time study: search uplift around Ramadan, Eid al-Fitr, Eid al-Adha, Black Friday, football tournaments.
2) Natural experiment: the 2025 royal call to skip the Eid al-Adha sacrifice. ML counterfactual for sheep demand.
3) Forecast horse race (walk-forward, a full year ahead): seasonal naive, Gregorian-seasonal regression (Fourier),
   LASSO and LightGBM with Hijri-calendar event features.
"""
import json, numpy as np, pandas as pd, warnings
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

raw = json.load(open("data/gtrends_raw.json"))
LABEL = {"sheep_ar": "Sheep (Arabic)", "sheep_fr": "Sheep (mouton)", "chebakia": "Chebakia", "caftan": "Caftan",
         "jellaba": "Jellaba", "gold_ar": "Gold", "jumia": "Jumia", "television": "Television", "flight": "Flights (billet avion)",
         "credit": "Credit (crédit)", "jersey": "Morocco jersey", "soldes": "Sales (soldes)", "eid_clothes": "Eid clothes", "dates_ar": "Dates"}

def stitch(windows):
    keys = sorted(windows)
    S = None
    for k in keys:
        s = pd.Series({pd.Timestamp(int(t), unit="s"): float(v) for t, v, part in windows[k] if not part})
        s = s[~s.index.duplicated()]
        if S is None: S = s; continue
        ov = S.index.intersection(s.index)
        a, b = S.loc[ov].mean(), s.loc[ov].mean()
        f = a / b if b > 0 else 1.0
        S = pd.concat([S[S.index < s.index.min()], s * f])
    S = S.sort_index(); return S / S.max() * 100
W = pd.DataFrame({k: stitch(v) for k, v in raw.items() if len(v) == 4})
W.index = W.index.normalize(); W = W[W.index >= "2010-01-03"]
W.to_csv("gtrends_weekly_stitched.csv")
zero_share = (W == 0).mean()
SPARSE = zero_share[zero_share > 0.30].index.tolist()          # too few searches for weekly analysis
W_all = W.copy(); W = W.drop(columns=SPARSE)
out = {"weeks": len(W), "terms": list(W.columns), "dropped_sparse": SPARSE, "start": str(W.index[0].date()), "end": str(W.index[-1].date())}

# ---------------- calendar ----------------
ev = pd.read_csv("data/events_hijri.csv", parse_dates=["date"])
gdates = {e: sorted(ev[ev.event == e].date) for e in ev.event.unique()}
bf = [pd.Timestamp(y, 11, 1) + pd.offsets.Week(weekday=4) * 4 for y in range(2009, 2028)]   # 4th Friday of November
bf = [d if d.month == 11 else d - pd.Timedelta(days=7) for d in bf]
gdates["black_friday"] = bf
tourn = {"AFCON 2012": ("2012-01-21", "2012-02-12"), "AFCON 2013": ("2013-01-19", "2013-02-10"), "AFCON 2017": ("2017-01-14", "2017-02-05"),
         "WC 2018": ("2018-06-14", "2018-07-15"), "AFCON 2019": ("2019-06-21", "2019-07-19"), "AFCON 2021": ("2022-01-09", "2022-02-06"),
         "WC 2022": ("2022-11-20", "2022-12-18"), "AFCON 2023": ("2024-01-13", "2024-02-11"), "AFCON 2025 (home)": ("2025-12-21", "2026-01-18"),
         "WC 2026": ("2026-06-11", "2026-07-19")}

def weeks_to(idx, dates):
    """signed weeks from each week-start to the nearest event (negative = before)."""
    d = np.array(dates, dtype="datetime64[ns]"); out_ = []
    for t in idx.values:
        diff = (d - t) / np.timedelta64(7, "D")
        j = np.argmin(np.abs(diff + 0.5)); out_.append(-diff[j])   # week containing the event -> ~0
    return np.round(np.array(out_), 1)

X = pd.DataFrame(index=W.index)
for e in ["ramadan_start", "eid_fitr", "eid_adha", "mawlid", "achoura", "black_friday"]:
    w = weeks_to(W.index, gdates[e]); X[f"wk_{e}"] = np.clip(w, -12, 12)
    X[f"in_{e}_m4_0"] = ((w >= -4) & (w <= 0.5)).astype(float)
X["in_ramadan"] = 0.0
for s, e in zip(gdates["ramadan_start"], gdates["eid_fitr"]):
    X.loc[(X.index >= s - pd.Timedelta(days=6)) & (X.index < e), "in_ramadan"] = 1.0
X["tournament"] = 0.0
for k, (a, b) in tourn.items():
    X.loc[(X.index >= pd.Timestamp(a) - pd.Timedelta(days=6)) & (X.index <= b), "tournament"] = 1.0
X["home_tournament"] = ((X.index >= "2025-12-15") & (X.index <= "2026-01-18")).astype(float)
X["woy"] = X.index.isocalendar().week.astype(int).values
X["month"] = X.index.month
for k in (1, 2, 3):
    X[f"sin{k}"] = np.sin(2 * np.pi * k * X.index.dayofyear / 365.25); X[f"cos{k}"] = np.cos(2 * np.pi * k * X.index.dayofyear / 365.25)
X["trend"] = (X.index - X.index[0]).days / 365.25
X["covid"] = ((X.index >= "2020-03-15") & (X.index <= "2020-07-01")).astype(float)
X["no_sacrifice_2025"] = ((X.index >= "2025-02-26") & (X.index <= "2025-07-01")).astype(float)
X.to_csv("calendar_features.csv")

# ---------------- 1) event-time uplift ----------------
def uplift(series, dates, pre=10, post=4, skip_years=()):
    base = series.rolling(52, min_periods=26).median().shift(1)
    rel = series / base
    rows = []
    for d in dates:
        if d.year in skip_years: continue
        i = series.index.searchsorted(d) - 1          # week containing d
        if i - pre < 0 or i + post >= len(series): continue
        seg = rel.iloc[i - pre:i + post + 1].values
        if np.isnan(seg).any(): continue
        rows.append(seg)
    A = np.array(rows)
    return pd.Series(np.nanmean(A, axis=0), index=range(-pre, post + 1)), len(A)
U = {}
for e in ["ramadan_start", "eid_fitr", "eid_adha", "black_friday"]:
    U[e] = {}
    for c in W.columns:
        s, n = uplift(W[c], gdates[e], skip_years=(2025,) if e == "eid_adha" else ())
        U[e][c] = {"peak_x": round(s.max(), 2), "peak_week": int(s.idxmax()), "n_events": n,
                   "lead_weeks_above_1.5x": int((s.loc[:0] > 1.5).sum()), "curve": s.round(3).tolist()}
out["uplift"] = {e: {c: {k: v for k, v in d.items() if k != "curve"} for c, d in U[e].items()} for e in U}
# tournaments: mean of series in tournament weeks vs same weeks prior year
tu = {}
for c in ["jersey", "television"]:
    if c not in W_all: continue
    r = {}
    for k, (a, b) in tourn.items():
        cur = W_all[c].loc[a:b].mean(); prev = W_all[c].loc[pd.Timestamp(a) - pd.DateOffset(years=1):pd.Timestamp(b) - pd.DateOffset(years=1)].mean()
        r[k] = round(cur / prev, 2) if prev > 0 else None
    tu[c] = r
out["tournament_vs_prior_year"] = tu

# ---------------- 2) 2025 natural experiment: ML counterfactual ----------------
FEAT_CAL = [c for c in X if c not in ("no_sacrifice_2025",)]
def cf_model():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=5, subsample=0.9, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0)
cf = {}
for c in [c for c in ["sheep_fr", "gold_ar", "caftan", "jellaba", "flight"] if c in W]:
    y = np.log1p(W[c])
    tr = X.index < "2025-01-01"
    m = cf_model().fit(X.loc[tr, FEAT_CAL], y[tr])
    pred = np.expm1(pd.Series(m.predict(X[FEAT_CAL]), index=X.index))
    win = (X.index >= "2025-04-27") & (X.index <= "2025-06-08")          # six weeks up to Eid al-Adha 2025 (6 Jun)
    # placebo: same procedure for 2024 (train < 2024, test 2024 Eid window) to size normal forecast error
    tr24 = X.index < "2024-01-01"; m24 = cf_model().fit(X.loc[tr24, FEAT_CAL], y[tr24])
    p24 = np.expm1(pd.Series(m24.predict(X[FEAT_CAL]), index=X.index))
    win24 = (X.index >= "2024-05-05") & (X.index <= "2024-06-16")
    win26 = (X.index >= "2026-04-19") & (X.index <= "2026-05-31")
    cf[c] = {"actual_2025_eid_window": round(W[c][win].mean(), 1), "counterfactual_2025": round(pred[win].mean(), 1),
             "gap_2025_%": round((W[c][win].mean() / pred[win].mean() - 1) * 100, 1),
             "peak_week_actual_2025": round(W[c][win].max(), 1), "peak_week_counterfactual_2025": round(pred[win].max(), 1),
             "peak_gap_2025_%": round((W[c][win].max() / pred[win].max() - 1) * 100, 1),
             "placebo_peak_gap_2024_%": round((W[c][win24].max() / p24[win24].max() - 1) * 100, 1),
             "placebo_gap_2024_%": round((W[c][win24].mean() / p24[win24].mean() - 1) * 100, 1),
             "actual_2026_eid_window": round(W[c][win26].mean(), 1) if win26.any() else None,
             "counterfactual_2026": round(pred[win26].mean(), 1) if win26.any() else None,
             "gap_2026_%": round((W[c][win26].mean() / pred[win26].mean() - 1) * 100, 1) if win26.any() else None}
    if c == "sheep_fr": CF_PLOT = (pred, W[c])
out["eid2025_counterfactual"] = cf

# ---------------- 3) forecast horse race: walk-forward, one full year ahead ----------------
def fourier_reg():
    return make_pipeline(StandardScaler(), LinearRegression())
F_GREG = ["sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "trend", "covid"]
F_ALL = [c for c in X if c not in ("woy",)]
def mdl_lasso(): return make_pipeline(StandardScaler(), LassoCV(cv=5, alphas=50, max_iter=20000))
def mdl_gbm(): return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=5, subsample=0.9,
                                       subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
HR = []
for c in W.columns:
    y = np.log1p(W[c])
    for Y in range(2016, 2027):
        tr = X.index.year < Y; te = X.index.year == Y
        if te.sum() < 20: continue
        act = W[c][te]
        # seasonal naive: same ISO week last year
        sn = W[c].shift(52)[te]
        preds = {"Seasonal naive": sn,
                 "Gregorian seasonal (Fourier)": np.expm1(fourier_reg().fit(X.loc[tr, F_GREG], y[tr]).predict(X.loc[te, F_GREG])),
                 "LASSO + Hijri events": np.expm1(mdl_lasso().fit(X.loc[tr, F_ALL], y[tr]).predict(X.loc[te, F_ALL])),
                 "LightGBM + Hijri events": np.expm1(mdl_gbm().fit(X.loc[tr, F_ALL], y[tr]).predict(X.loc[te, F_ALL]))}
        evw = (X.loc[te, ["in_eid_adha_m4_0", "in_eid_fitr_m4_0", "in_ramadan_start_m4_0", "in_ramadan"]].sum(axis=1) > 0).values
        for k, p in preds.items():
            p = np.asarray(p, float); a = act.values
            ok = ~np.isnan(p)
            HR.append({"term": c, "year": Y, "model": k, "mae": np.mean(np.abs(a[ok] - p[ok])),
                       "mae_event": np.mean(np.abs(a[ok & evw] - p[ok & evw])) if (ok & evw).any() else np.nan})
HR = pd.DataFrame(HR)
HR.to_csv("forecast_horserace.csv", index=False)
HRx = HR[~((HR.year == 2025) & HR.term.isin(["sheep_fr"]))]        # exclude the shock year for sheep
summ = HRx.groupby("model")[["mae", "mae_event"]].mean()
ref = summ.loc["Gregorian seasonal (Fourier)"]
out["horserace_mean_MAE"] = summ.round(2).to_dict("index")
out["horserace_error_reduction_vs_gregorian_%"] = ((1 - summ / ref) * 100).round(1).to_dict("index")
wins = HRx.pivot_table(index=["term", "year"], columns="model", values="mae")
out["share_of_term_years_gbm_beats_gregorian"] = round((wins["LightGBM + Hijri events"] < wins["Gregorian seasonal (Fourier)"]).mean(), 2)
out["share_of_term_years_gbm_beats_naive"] = round((wins["LightGBM + Hijri events"] < wins["Seasonal naive"]).mean(), 2)
per_term = HRx.groupby(["term", "model"]).mae_event.mean().unstack()
out["event_window_MAE_by_term"] = per_term.round(1).to_dict("index")

# permutation importance, LightGBM trained to 2024, scored on 2023-24 (in-sample-ish) for sheep / chebakia / jersey
imp = {}
for c in [c for c in ["sheep_fr", "dates_ar", "caftan", "jumia", "gold_ar", "flight"] if c in W]:
    y = np.log1p(W[c]); tr = X.index < "2023-01-01"; te = (X.index >= "2023-01-01") & (X.index < "2025-01-01")
    m = mdl_gbm().fit(X.loc[tr, F_ALL], y[tr])
    pi = permutation_importance(m, X.loc[te, F_ALL], y[te], n_repeats=15, random_state=0)
    imp[c] = pd.Series(pi.importances_mean, index=F_ALL).sort_values(ascending=False).head(5).round(3).to_dict()
out["top_features"] = imp
json.dump(out, open("consumer_results.json", "w"), indent=1, default=str)

# ---------------- charts ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
a = ax[0, 0]
for c, col in [("sheep_fr", "#17624f"), ("dates_ar", "#a85a2a"), ("jellaba", "#6b4fa0"), ("flight", "#3b6fb6")]:
    if c not in W: continue
    e = {"sheep_fr": "eid_adha", "dates_ar": "ramadan_start", "jellaba": "eid_fitr", "flight": "eid_adha"}[c]
    cur = U[e][c]["curve"]; a.plot(range(-10, 5), cur, color=col, lw=1.8, label=f"{LABEL[c]} → {e.replace('_',' ')}")
a.axvline(0, color="k", lw=.8, ls=":"); a.axhline(1, color="0.6", lw=.8); a.set_yscale("log"); a.set_yticks([0.5,1,2,4,8,16]); a.set_yticklabels(["0.5×","1×","2×","4×","8×","16×"])
a.set_title("Anticipation: search vs 52-week median, weeks around the event", fontweight="bold"); a.set_xlabel("Weeks relative to event (0 = event week)")
a.set_ylabel("× normal level (log scale)"); a.legend(frameon=False, fontsize=8); a.grid(color="0.92")
a = ax[0, 1]
pred, act = CF_PLOT
seg = (act.index >= "2022-01-01")
a.plot(act.index[seg], act[seg], color="k", lw=1.2, label="Actual searches for 'mouton' (sheep)")
a.plot(pred.index[seg], pred[seg], color="#17624f", lw=1.4, ls="--", label="ML counterfactual (trained to 2024)")
a.axvspan(pd.Timestamp("2025-02-26"), pd.Timestamp("2025-06-08"), color="#a85a2a", alpha=.12)
a.text(pd.Timestamp("2025-03-05"), a.get_ylim()[1] * 0.92, "Royal call to skip\nthe sacrifice (26 Feb 2025)", fontsize=8, color="#a85a2a")
a.set_title("Natural experiment: Eid al-Adha 2025", fontweight="bold"); a.legend(frameon=False, fontsize=8, loc="upper left"); a.grid(color="0.92")
a = ax[1, 0]
order = ["Seasonal naive", "Gregorian seasonal (Fourier)", "LASSO + Hijri events", "LightGBM + Hijri events"]
vals = [summ.loc[o, "mae_event"] for o in order]; vals2 = [summ.loc[o, "mae"] for o in order]
x = np.arange(4); a.bar(x - .18, vals2, .36, color="0.65", label="All weeks"); a.bar(x + .18, vals, .36, color="#17624f", label="Ramadan & Eid weeks")
a.set_xticks(x); a.set_xticklabels(["Seasonal\nnaive", "Gregorian\nseasonal", "LASSO +\nHijri", "LightGBM +\nHijri"])
a.set_ylabel("Mean absolute error (index points)"); a.set_title("Forecasting a full year ahead, walk-forward 2016-2026", fontweight="bold")
a.legend(frameon=False); a.grid(axis="y", color="0.92")
a = ax[1, 1]
if "jersey" in W_all:
    s = W_all["jersey"].loc["2021-06-01":]
    a.plot(s.index, s, color="#a85a2a", lw=1.2, label="Morocco jersey")
for k, (b0, b1) in tourn.items():
    if pd.Timestamp(b0) >= pd.Timestamp("2021-06-01"):
        a.axvspan(pd.Timestamp(b0), pd.Timestamp(b1), color="#17624f", alpha=.10); a.text(pd.Timestamp(b0), a.get_ylim()[1] * .95 if a.get_ylim()[1] > 1 else 95, k, fontsize=7, rotation=90, va="top")
a.set_title("Football: Morocco jersey searches (shaded = tournaments)", fontweight="bold"); a.legend(frameon=False, fontsize=8); a.grid(color="0.92")
fig.text(0.5, -0.01, "Weekly Google Trends, Morocco, stitched from overlapping 5-year windows (index, peak = 100). Hijri dates: Umm al-Qura calendar (Morocco may differ by a day).",
         ha="center", fontsize=8, color="0.3")
plt.tight_layout(); plt.savefig("consumer_events.png", dpi=200, bbox_inches="tight")
print(json.dumps({k: v for k, v in out.items() if k not in ("uplift", "event_window_MAE_by_term")}, indent=1, default=str, ensure_ascii=False))
print(json.dumps(out["uplift"], ensure_ascii=False)[:3000])
