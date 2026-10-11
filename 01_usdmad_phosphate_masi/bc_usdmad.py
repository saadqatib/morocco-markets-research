"""Breitung-Candelon (2006) frequency-domain causality, same implementation as BenAmar_replication/bc_test.py.
Series: USD/MAD, phosphate rock (World Bank), MASI. Monthly log returns (%)."""
import numpy as np, pandas as pd, json, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import chi2
from statsmodels.tsa.api import VAR

def bc_test(y, x, p, omegas):
    n = len(y); Y = y[p:]
    X = [np.ones(n - p)] + [y[p - j:n - j] for j in range(1, p + 1)] + [x[p - j:n - j] for j in range(1, p + 1)]
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None); resid = Y - X @ beta
    s2 = resid @ resid / (len(Y) - X.shape[1]); V = s2 * np.linalg.inv(X.T @ X)
    idx = np.arange(1 + p, 1 + 2 * p); b, Vb = beta[idx], V[np.ix_(idx, idx)]; j = np.arange(1, p + 1)
    out = []
    for w in omegas:
        R = np.vstack([np.cos(j * w), np.sin(j * w)]); Rb = R @ b
        out.append(float(Rb @ np.linalg.solve(R @ Vb @ R.T, Rb)))
    return np.array(out)

omegas = np.linspace(0.01, 3.13, 313); crit = chi2.ppf(0.95, 2)
def bands(stat):
    b, on = [], None
    for w, s in zip(omegas, stat):
        if s > crit and on is None: on = w
        if s <= crit and on is not None: b.append((on, w)); on = None
    if on is not None: b.append((on, omegas[-1]))
    return b
def fmt(b): return ", ".join(f"{2*np.pi/c:.1f}-{2*np.pi/a:.0f}m" for a, c in b) or "none"

r = pd.read_csv("returns_monthly.csv", index_col=0); r.index = pd.PeriodIndex(r.index, freq="M")
r["usdmad_adj"] = sm.OLS(r.usdmad, sm.add_constant(r.eurusd)).fit().resid
specs = {"main": r, "excl_2023-12": r.drop(pd.Period("2023-12", "M"))}
pairs = [("phos","usdmad"),("usdmad","phos"),("phos","masi"),("masi","phos"),("masi","usdmad"),("usdmad","masi"),
         ("phos","usdmad_adj"),("masi","usdmad_adj"),("dap","usdmad_adj"),("dap","masi")]
res = {}
for name, d in specs.items():
    sel = VAR(d[["usdmad","phos","masi"]]).select_order(maxlags=12); p = max(sel.selected_orders["aic"], 3)
    res[name] = {"lag": int(p)}
    for x, y in pairs:
        s = bc_test(d[y].values, d[x].values, p, omegas)
        res[name][f"{x} -> {y}"] = {"max_stat": round(float(s.max()), 2), "sig_cycles": fmt(bands(s))}
        if name == "main": res[name][f"{x} -> {y}"]["_stat"] = s
    print(f"\n[{name}] VAR lag {p} (AIC {sel.selected_orders['aic']})")
    for k, v in res[name].items():
        if k != "lag": print(f"  {k:22s} max W {v['max_stat']:6.2f}  significant: {v['sig_cycles']}")

# chart (main spec, 6 core directions), same style as Brent/TASI figure
plt.rcParams.update({"font.family": "serif", "font.size": 9})
titles = {("phos","usdmad"):"Phosphate → USD/MAD",("usdmad","phos"):"USD/MAD → Phosphate",("phos","masi"):"Phosphate → MASI",
          ("masi","phos"):"MASI → Phosphate",("masi","usdmad"):"MASI → USD/MAD",("usdmad","masi"):"USD/MAD → MASI"}
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
for ax, (x, y) in zip(axes.flat, titles):
    s = res["main"][f"{x} -> {y}"]["_stat"]
    ax.plot(omegas, s, color="black", lw=1.6); ax.axhline(crit, color="red", ls=":", lw=2, label="5% critical value")
    for a, c in bands(s): ax.axvspan(a, c, color="red", alpha=0.08)
    ax.set_title(titles[(x, y)], fontweight="bold"); ax.set_xlim(0, np.pi); ax.set_ylim(0, max(s.max(), crit) * 1.15)
    ax.set_xlabel("Frequency ω   (long run ← → short run)"); ax.set_ylabel("Wald statistic"); ax.grid(color="0.9")
    top = ax.secondary_xaxis("top"); t = [2*np.pi/T for T in (24,12,6,4,3,2)]
    top.set_xticks(t); top.set_xticklabels(["24m","12m","6m","4m","3m","2m"], fontsize=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
fig.suptitle(f"Breitung-Candelon frequency-domain causality: USD/MAD, phosphate, MASI "
             f"(monthly log returns, {r.index[0]}-{r.index[-1]}, VAR({res['main']['lag']}))", fontsize=11)
fig.text(0.5, -0.01, "Dotted line: 5% critical value (χ²(2) = 5.99). Shaded: significant frequencies. "
         "Data: Investing.com (USD/MAD, MASI), World Bank Pink Sheet (phosphate rock).", ha="center", fontsize=8, color="0.3")
plt.tight_layout(); plt.savefig("bc_usdmad_phos_masi.png", dpi=200, bbox_inches="tight")
for v in res["main"].values():
    if isinstance(v, dict): v.pop("_stat", None)
json.dump(res, open("bc_results.json", "w"), indent=1)
