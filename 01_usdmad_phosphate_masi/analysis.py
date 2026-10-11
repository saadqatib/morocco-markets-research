"""USD/MAD vs phosphate vs MASI: monthly log returns.
Correlation, OLS w/ EUR/USD control (HAC), Granger, Diebold-Yilmaz (2012) connectedness."""
import numpy as np, pandas as pd, openpyxl, warnings, json
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
warnings.filterwarnings("ignore")
D="data/"
fx=pd.read_csv(D+"fx_monthly.csv"); fx.index=pd.PeriodIndex(fx.date,freq="M")
ma=pd.read_csv(D+"masi_monthly.csv"); ma.index=pd.PeriodIndex(ma.date,freq="M")
ws=openpyxl.load_workbook(D+"CMO-Historical-Data-Monthly.xlsx",read_only=True)["Monthly Prices"]
rows=[r for r in ws.iter_rows(min_row=7,values_only=True) if r[0] and str(r[0])[:2] in("19","20")]
wb=pd.DataFrame({"phos":[r[57] for r in rows],"dap":[r[58] for r in rows],"brent":[r[2] for r in rows]},
                index=pd.PeriodIndex([str(r[0]).replace("M","-") for r in rows],freq="M"))
lv=pd.concat([fx[["usdmad","eurusd"]],ma[["masi"]],wb.apply(pd.to_numeric,errors="coerce")],axis=1).dropna()
r=(np.log(lv).diff()*100).dropna()
out={"sample":f"{r.index[0]}-{r.index[-1]}","T":len(r)}
print(out)

# 1 correlations
out["corr_full"]=r[["usdmad","phos","dap","masi","eurusd","brent"]].corr().round(3).to_dict()
post=r.loc["2015-04":]; out["corr_post2015"]=post[["usdmad","phos","masi","eurusd"]].corr().round(3).to_dict()
print(r[["usdmad","phos","dap","masi","eurusd"]].corr().round(3))

# 2 OLS with HAC
def ols(y,X,df):
    m=sm.OLS(df[y],sm.add_constant(df[X])).fit(cov_type="HAC",cov_kwds={"maxlags":3}); return m
d=r.copy(); 
for k in (1,2,3): d[f"phos_l{k}"]=d.phos.shift(k); d[f"masi_l{k}"]=d.masi.shift(k)
d=d.dropna()
models={
 "A: usdmad ~ phos":ols("usdmad",["phos"],d),
 "B: usdmad ~ eurusd":ols("usdmad",["eurusd"],d),
 "C: usdmad ~ eurusd + phos + masi":ols("usdmad",["eurusd","phos","masi"],d),
 "D: usdmad ~ eurusd + phos lags 0-3":ols("usdmad",["eurusd","phos","phos_l1","phos_l2","phos_l3"],d),
 "E: masi ~ phos + usdmad":ols("masi",["phos","usdmad"],d),
 "F: masi ~ phos lags 0-3":ols("masi",["phos","phos_l1","phos_l2","phos_l3"],d),
}
out["ols"]={}
for k,m in models.items():
    out["ols"][k]={"coef":m.params.round(4).to_dict(),"p":m.pvalues.round(4).to_dict(),"R2":round(m.rsquared,3),"n":int(m.nobs)}
    print(k,"R2",round(m.rsquared,3)); print(pd.DataFrame({"b":m.params,"p":m.pvalues}).round(4))

# residual USD/MAD (strip EUR/USD basket mechanics) for VAR
mB=models["B: usdmad ~ eurusd"]; d["usdmad_resid"]=mB.resid
# 3 Granger (VAR, AIC lag)
V=r[["usdmad","phos","masi"]]
var=VAR(V); p=max(1,var.select_order(12).selected_orders["aic"]); res=var.fit(p)
out["var_lag"]=int(p); out["granger"]={}
for c in V.columns:
    for e in V.columns:
        if c!=e:
            t=res.test_causality(e,[c],kind="f"); out["granger"][f"{c} -> {e}"]=round(float(t.pvalue),4)
print("VAR lag",p, out["granger"])

# 4 Diebold-Yilmaz generalized FEVD connectedness
def dy(df,p,H=10):
    m=VAR(df).fit(p); S=m.sigma_u.values; A=m.ma_rep(H-1); k=df.shape[1]
    th=np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            num=sum((A[h][i]@S[:,j])**2 for h in range(H))/S[j,j]
            den=sum(A[h][i]@S@A[h][i] for h in range(H))
            th[i,j]=num/den
    th=th/th.sum(1,keepdims=True)*100
    T=pd.DataFrame(th,index=df.columns,columns=df.columns)
    tot=(th.sum()-np.trace(th))/k
    return T,tot
T,tot=dy(V,p)
frm=T.sum(0)-np.diag(T); to=T.sum(1)-np.diag(T)
print(T.round(1)); print("TOTAL",round(tot,1))
out["dy_table"]=T.round(2).to_dict(); out["dy_total"]=round(tot,2)
out["dy_to_others"]=frm.round(2).to_dict(); out["dy_from_others"]=to.round(2).to_dict(); out["dy_net"]=(frm-to).round(2).to_dict()
# same with EUR/USD-adjusted USD/MAD
V2=pd.concat([d.usdmad_resid.rename("usdmad_adj"),d.phos,d.masi],axis=1)
T2,tot2=dy(V2,p); out["dy_adj_total"]=round(tot2,2); out["dy_adj_table"]=T2.round(2).to_dict(); print(T2.round(1),"TOTAL adj",round(tot2,1))
# 4-var system incl EUR/USD
V4=r[["usdmad","eurusd","phos","masi"]]; T4,tot4=dy(V4,p); out["dy4_total"]=round(tot4,2); out["dy4_table"]=T4.round(2).to_dict(); print(T4.round(1),"TOTAL4",round(tot4,1))
# rolling total connectedness (60m)
roll=[]
for i in range(60,len(V)+1):
    w=V.iloc[i-60:i]; _,t=dy(w,1); roll.append((str(V.index[i-1]),round(t,2)))
pd.DataFrame(roll,columns=["end","total_conn"]).to_csv("rolling_connectedness.csv",index=False)
rc=pd.DataFrame(roll,columns=["end","t"]); out["rolling_minmax"]={"min":rc.loc[rc.t.idxmin()].tolist(),"max":rc.loc[rc.t.idxmax()].tolist(),"last":rc.iloc[-1].tolist()}
# rolling corr usdmad-phos 36m
rcorr=r.usdmad.rolling(36).corr(r.phos).dropna(); out["rollcorr_phos"]={"min":[str(rcorr.idxmin()),round(rcorr.min(),3)],"max":[str(rcorr.idxmax()),round(rcorr.max(),3)],"last":round(rcorr.iloc[-1],3)}
# 2008 & 2022 phosphate spikes
out["episodes"]={k:{"phos_%":round((lv.phos[b]/lv.phos[a]-1)*100,1),"usdmad_%":round((lv.usdmad[b]/lv.usdmad[a]-1)*100,1),"masi_%":round((lv.masi[b]/lv.masi[a]-1)*100,1),"eurusd_%":round((lv.eurusd[b]/lv.eurusd[a]-1)*100,1)}
  for k,(a,b) in {"2007-12→2008-08 boom":("2007-12","2008-08"),"2008-08→2009-06 bust":("2008-08","2009-06"),"2021-06→2022-06 boom":("2021-06","2022-06"),"2022-06→2023-06 bust":("2022-06","2023-06")}.items()}
json.dump(out,open("results.json","w"),indent=1,default=str)
r.to_csv("returns_monthly.csv")
# charts
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,1,figsize=(10,7))
(lv[["usdmad","phos","masi"]]/lv[["usdmad","phos","masi"]].iloc[0]*100).plot(ax=ax[0],lw=1.3); ax[0].set_title("Levels rebased (=100)"); ax[0].grid(alpha=.3)
rc.assign(end=pd.PeriodIndex(rc.end,freq="M").to_timestamp()).set_index("end").t.plot(ax=ax[1],color="k",lw=1.3); ax[1].set_title("Total connectedness, USD/MAD-phosphate-MASI (60m rolling, %)"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("charts.png",dpi=160)
print(json.dumps({k:out[k] for k in ["rolling_minmax","rollcorr_phos","episodes"]},indent=1))
