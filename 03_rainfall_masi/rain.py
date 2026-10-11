"""Rain and the Moroccan stock market.
Channel 1 (fundamental): rainfall over the cereal belt -> harvest -> GDP/consumption -> MASI and rain-sensitive sectors.
Channel 2 (mood): Casablanca cloud cover on the trading day (Hirshleifer & Shumway 2003 'sunshine effect').
Rainfall: Open-Meteo ERA5 reanalysis, daily, 8 agricultural sites. Prices: Investing.com.
"""
import numpy as np, pandas as pd, json, statsmodels.api as sm
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = {}

# ---------- rainfall index ----------
rd = pd.read_csv("data/morocco_rain_daily.csv", parse_dates=["date"]).set_index("date")
rain_d = rd.mean(axis=1)                                         # mm/day, average of 8 sites
rain_m = rain_d.resample("ME").sum(); rain_m.index = rain_m.index.to_period("M")
rain_m = rain_m[rain_m.index <= "2026-09"]
# crop season: Sep(t-1)..Aug(t); season-to-date cumulative rain vs its 2000-2025 climatology for the same month
season = np.where(rain_m.index.month >= 9, rain_m.index.year + 1, rain_m.index.year)
rm = pd.DataFrame({"rain": rain_m.values, "season": season}, index=rain_m.index)
rm["std_cum"] = rm.groupby("season").rain.cumsum()
rm["m"] = rm.index.month
clim = rm[rm.season.between(2001, 2025)].groupby("m").std_cum.agg(["mean", "std"])
rm = rm.join(clim, on="m")
rm["anom"] = (rm.std_cum - rm["mean"]) / rm["mean"] * 100       # % above/below normal, season-to-date
rm["z"] = (rm.std_cum - rm["mean"]) / rm["std"]
seas = rm[rm.m == 3].set_index("season")[["std_cum", "anom"]].rename(columns={"std_cum": "rain_sep_mar_mm", "anom": "anom_%"})
out["season_rain_sep_mar"] = seas.round(1).to_dict("index")

# ---------- macro check: does rain actually move the real economy? ----------
wb = pd.read_csv("data/morocco_wb_macro.csv").set_index("year")
mac = seas.join(wb, how="inner")
def cr(a, b):
    x = mac[[a, b]].dropna(); rr, p = stats.pearsonr(x[a], x[b]); return {"rho": round(rr, 3), "p": round(p, 4), "n": len(x)}
out["macro"] = {"rain vs cereal production": cr("anom_%", "cereal_prod_t"), "rain vs agri value-added growth": cr("anom_%", "agri_va_growth"),
                "rain vs real GDP growth": cr("anom_%", "gdp_growth")}
mx = mac[["anom_%", "gdp_growth"]].dropna(); mx = mx[mx.index != 2020]
out["macro"]["rain vs real GDP growth (ex-2020)"] = {"rho": round(mx.corr().iloc[0, 1], 3), "n": len(mx)}
g = sm.OLS(mx.gdp_growth, sm.add_constant(mx["anom_%"])).fit()
out["macro"]["GDP growth pp per +10% rain (ex-2020)"] = round(g.params["anom_%"] * 10, 3)

# ---------- MASI monthly ----------
md = pd.read_csv("data/masi_daily.csv", parse_dates=["date"]).set_index("date")["masi"]
masi_m = md.resample("ME").last(); masi_m.index = masi_m.index.to_period("M")
r = (np.log(masi_m).diff() * 100).rename("masi")
D = rm.join(r, how="inner").dropna(subset=["masi"])
D["masi_next"] = D.masi.shift(-1)
D["masi_next3"] = D.masi.shift(-1) + D.masi.shift(-2) + D.masi.shift(-3)
crop = D[D.m.isin([11, 12, 1, 2, 3, 4])]                         # months when the harvest is being decided

def ols(y, X, d, lags=3):
    d = d[[y] + X].dropna(); m = sm.OLS(d[y], sm.add_constant(d[X])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"b": round(m.params[X[0]], 4), "t": round(m.tvalues[X[0]], 2), "p": round(m.pvalues[X[0]], 3), "R2": round(m.rsquared, 3), "n": int(m.nobs)}
out["monthly"] = {
    "masi_t ~ rain_z_t (crop months)": ols("masi", ["z"], crop),
    "masi_t+1 ~ rain_z_t (crop months)": ols("masi_next", ["z"], crop),
    "masi_t+1..t+3 ~ rain_z_t (crop months)": ols("masi_next3", ["z"], crop, 4),
    "masi_t+1 ~ rain_z_t (all months)": ols("masi_next", ["z"], D),
}

# ---------- annual: season rain vs MASI over the following Apr-Dec and over the season ----------
ann = []
for s_ in range(2003, 2027):
    try:
        a = masi_m.loc[f"{s_}-03"]; b = masi_m.loc[f"{s_}-12"] if f"{s_}-12" in masi_m.index.astype(str) else masi_m.iloc[-1]
        a0 = masi_m.loc[f"{s_-1}-09"]
        ann.append({"season": s_, "rain_anom_%": seas.loc[s_, "anom_%"], "masi_sep_mar_%": (a / a0 - 1) * 100,
                    "masi_apr_dec_%": (b / a - 1) * 100})
    except KeyError:
        pass
A = pd.DataFrame(ann).set_index("season")
out["annual_corr"] = {"rain vs MASI Sep-Mar (contemporaneous)": round(A["rain_anom_%"].corr(A["masi_sep_mar_%"]), 3),
                      "rain vs MASI Apr-Dec (after rain is known)": round(A["rain_anom_%"].corr(A["masi_apr_dec_%"]), 3),
                      "n_seasons": len(A)}
dry = A[A["rain_anom_%"] < -20]; wet = A[A["rain_anom_%"] > 20]
out["annual_dry_vs_wet"] = {"dry_seasons": dry.index.tolist(), "wet_seasons": wet.index.tolist(),
                            "MASI Apr-Dec after dry (mean %)": round(dry["masi_apr_dec_%"].mean(), 2),
                            "MASI Apr-Dec after wet (mean %)": round(wet["masi_apr_dec_%"].mean(), 2),
                            "MASI Sep-Mar in dry": round(dry["masi_sep_mar_%"].mean(), 2),
                            "MASI Sep-Mar in wet": round(wet["masi_sep_mar_%"].mean(), 2)}
A.round(2).to_csv("rain_masi_annual.csv")

# ---------- sector portfolios (if stock file present) ----------
try:
    st = pd.read_csv("data/stocks_monthly.csv"); st["date"] = pd.PeriodIndex(st.date, freq="M")
    P = st.pivot_table(index="date", columns="sym", values="close")
    R = np.log(P).diff() * 100
    R = R.where(R.abs() < 60)                                    # glitches
    groups = {"Agri-food & consumer": ["CSR", "LES", "DRI", "OUL", "UMR", "MUT", "CRS", "LBV", "ATH", "NEJ", "EQD", "SLF"],
              "Banks": ["ATW", "BCP", "BOA", "BCI", "CDM", "CIH", "CFG"],
              "Building & real estate": ["LHM", "CMA", "ADH", "RDS", "SID", "ARD", "JET"],
              "Mining & energy": ["MNG", "SMI", "CMT", "GAZ", "TQM", "TMA"]}
    G = pd.DataFrame({k: R[[c for c in v if c in R]].mean(axis=1, skipna=True) for k, v in groups.items()})
    G["Agri − Mining"] = G["Agri-food & consumer"] - G["Mining & energy"]
    GD = rm.join(G, how="inner"); GD = GD[GD.index >= "2003-01"]
    for c in list(groups) + ["Agri − Mining"]:
        GD[c + "_next"] = GD[c].shift(-1)
    gc = GD[GD.m.isin([11, 12, 1, 2, 3, 4])]
    out["sectors"] = {c: {"same_month": ols(c, ["z"], gc), "next_month": ols(c + "_next", ["z"], gc)} for c in list(groups) + ["Agri − Mining"]}
    out["sector_members_found"] = {k: [c for c in v if c in R] for k, v in groups.items()}
    have_sectors = True
except FileNotFoundError:
    have_sectors = False; G = None

# ---------- backtest: rain-timed MASI (monthly) ----------
bt = D[["masi", "z", "m"]].copy()
bt["signal"] = np.where(bt.m.isin([11, 12, 1, 2, 3, 4]), (bt.z > 0).astype(float), 1.0)  # outside crop months: hold
bt["strat"] = bt.signal.shift(1) * bt.masi
bt = bt.dropna()
def perf(x):
    ann = x.mean() * 12; vol = x.std() * np.sqrt(12)
    eq = x.cumsum(); dd = (eq - eq.cummax()).min()
    return {"ann_ret_%": round(ann, 2), "ann_vol_%": round(vol, 2), "sharpe": round(ann / vol, 3), "max_dd_logpts": round(dd, 1)}
out["backtest_masi_timing"] = {"buy_and_hold": perf(bt.masi), "rain_timed": perf(bt.strat),
                               "months_in_cash": int((bt.signal.shift(1) == 0).sum())}
if have_sectors:
    g = rm[["z", "m"]].join(G, how="inner").dropna(subset=["Agri-food & consumer", "Mining & energy"])
    g["sig"] = np.where(g.m.isin([11, 12, 1, 2, 3, 4]), np.sign(g.z), 0).astype(float)
    g["ls"] = g.sig.shift(1) * g["Agri − Mining"]
    out["backtest_agri_vs_mining_ls"] = perf(g.ls.dropna()[g.sig.shift(1).dropna() != 0])

# ---------- channel 2: Casablanca weather mood effect (daily) ----------
cw = pd.read_csv("data/casablanca_weather_daily.csv", parse_dates=["date"]).set_index("date")
rd_ = (np.log(md).diff() * 100).dropna(); rd_ = rd_[rd_.abs() < 15]
dd = pd.DataFrame({"r": rd_}).join(cw)
dd["cloud_dev"] = dd.cloud - dd.groupby(dd.index.month).cloud.transform("mean")   # deseasonalised (as in H&S)
dd["rainy"] = (dd.precip > 1).astype(float)
for k in range(1, 4): dd[f"l{k}"] = dd.r.shift(k)
dd = dd.dropna()
m1 = sm.OLS(dd.r, sm.add_constant(dd[["cloud_dev", "l1", "l2", "l3"]])).fit(cov_type="HC1")
m2 = sm.OLS(dd.r, sm.add_constant(dd[["rainy", "l1", "l2", "l3"]])).fit(cov_type="HC1")
sunny = dd[dd.cloud <= 10].r; overc = dd[dd.cloud >= 90].r
lg = sm.Logit((dd.r > 0).astype(int), sm.add_constant(dd[["cloud_dev"]])).fit(disp=0)
out["mood_daily"] = {"n_days": len(dd),
    "r ~ cloud_dev (per 100 pp cloud)": {"b_%": round(m1.params.cloud_dev * 100, 3), "t": round(m1.tvalues.cloud_dev, 2), "p": round(m1.pvalues.cloud_dev, 3)},
    "r ~ rainy day": {"b_%": round(m2.params.rainy, 3), "t": round(m2.tvalues.rainy, 2), "p": round(m2.pvalues.rainy, 3), "n_rainy": int(dd.rainy.sum())},
    "logit up-day ~ cloud_dev": {"b": round(lg.params.cloud_dev, 4), "p": round(lg.pvalues.cloud_dev, 3)},
    "mean r clear days (cloud<=10%)": round(sunny.mean(), 4), "n_clear": len(sunny),
    "mean r overcast (cloud>=90%)": round(overc.mean(), 4), "n_overcast": len(overc),
    "annualised gap clear-minus-overcast_%": round((sunny.mean() - overc.mean()) * 250, 1),
    "welch_p": round(stats.ttest_ind(sunny, overc, equal_var=False).pvalue, 3)}

json.dump(out, open("rain_results.json", "w"), indent=1, default=str)
rm.round(3).to_csv("rain_index_monthly.csv")

# ---------- charts ----------
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
ax[0].bar(seas.index, seas["anom_%"], color=np.where(seas["anom_%"] >= 0, "#2a6fb0", "#c0703a"))
ax[0].axhline(0, color="k", lw=.8); ax[0].set_title("Cereal-belt rainfall, Sep-Mar (% vs normal)", fontweight="bold"); ax[0].grid(axis="y", color="0.9")
ax[1].scatter(A["rain_anom_%"], A["masi_sep_mar_%"], color="k", s=18)
for s_, row in A.iterrows(): ax[1].annotate(str(s_)[2:], (row["rain_anom_%"], row["masi_sep_mar_%"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
b = np.polyfit(A["rain_anom_%"], A["masi_sep_mar_%"], 1); xs = np.linspace(A["rain_anom_%"].min(), A["rain_anom_%"].max(), 10)
ax[1].plot(xs, np.polyval(b, xs), color="#2a6fb0", lw=1.4)
ax[1].set_xlabel("Season rainfall, % vs normal"); ax[1].set_ylabel("MASI return Sep-Mar (%)")
ax[1].set_title(f"The market ignores rain (ρ = {A['rain_anom_%'].corr(A['masi_sep_mar_%']):.2f})", fontweight="bold") if False else ax[1].set_title(f"Rain vs MASI by crop season (ρ = {A['rain_anom_%'].corr(A['masi_sep_mar_%']):.2f})", fontweight="bold"); ax[1].grid(color="0.9")
mc = mac[["anom_%", "cereal_prod_t"]].dropna()
ax[2].scatter(mc["anom_%"], mc.cereal_prod_t / 1e6, color="#2a6fb0", s=18)
for y_, row in mc.iterrows(): ax[2].annotate(str(y_)[2:], (row["anom_%"], row.cereal_prod_t / 1e6), fontsize=7, xytext=(3, 2), textcoords="offset points")
b2 = np.polyfit(mc["anom_%"], mc.cereal_prod_t / 1e6, 1); xs2 = np.linspace(mc["anom_%"].min(), mc["anom_%"].max(), 10)
ax[2].plot(xs2, np.polyval(b2, xs2), color="k", lw=1.2)
ax[2].set_xlabel("Season rainfall, % vs normal"); ax[2].set_ylabel("Cereal production (million t)")
ax[2].set_title(f"…but rain drives the harvest (ρ = {mc.corr().iloc[0, 1]:.2f})", fontweight="bold"); ax[2].grid(color="0.9")
eq = bt[["masi", "strat"]].cumsum(); eq.index = eq.index.to_timestamp()
ax[3].plot(eq.index, eq.masi, color="0.5", lw=1.2, label="MASI buy & hold"); ax[3].plot(eq.index, eq.strat, color="#2a6fb0", lw=1.4, label="Rain-timed MASI")
ax[3].legend(frameon=False); ax[3].set_title("Backtest: cash in crop months with below-normal rain", fontweight="bold"); ax[3].set_ylabel("Cumulative log return (%)"); ax[3].grid(color="0.9")
fig.text(0.5, -0.03, "Rainfall: Open-Meteo ERA5, mean of 8 cereal-belt sites (Settat, Meknès, Saïss, Gharb, Tadla, Doukkala, Haouz, Oujda). MASI: Investing.com.",
         ha="center", fontsize=8, color="0.3")
plt.tight_layout(); plt.savefig("rain_masi.png", dpi=200, bbox_inches="tight")
print(json.dumps({k: v for k, v in out.items() if k != "season_rain_sep_mar"}, indent=1, default=str))
