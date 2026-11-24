"""
Revised analysis (2026) for the NEOMA M.Sc dissertation
"Mirrors and Markets: Reflexivity Evidence of Retail Traders' Influence in Meme Stocks"
(Yann Simons, August 2024).

Inputs
- daily_reddit_sentiment.csv: daily mean VADER compound score of post titles and
  post counts for r/GME, r/DDintoGME and r/GMEJungle (GameStop Reddit Dataset,
  Han & Iliadis, public domain), as published in the dissertation repository.
- GME daily close and volume and the VIX, Yahoo Finance (yfinance), downloaded once
  and cached in gme_vix_daily.csv.

Outputs: figures (fig/), LaTeX tables (tab/) and key numbers (results.json).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycwt
import statsmodels.api as sm
from arch.unitroot import ADF, KPSS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

HERE = Path(__file__).resolve().parent
SENT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "daily_reddit_sentiment.csv"
PRICES = HERE / "gme_vix_daily.csv"
FIG, TAB = HERE / "fig", HERE / "tab"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 200})
np.random.seed(20240831)  # Monte Carlo significance of the wavelet coherence
R = {}
EVENT = pd.Timestamp("2021-01-28")   # trading restrictions by Robinhood and other brokers
RGME_END = pd.Timestamp("2021-08-26")  # last day covered by the r/GME file

# --------------------------------------------------------------------------- data
if not PRICES.exists():
    import yfinance as yf
    gme = yf.download("GME", start="2020-01-01", end="2021-12-31", auto_adjust=True, progress=False)
    vix = yf.download("^VIX", start="2020-01-01", end="2021-12-31", auto_adjust=True, progress=False)
    px = pd.DataFrame({"Close": gme["Close"].squeeze(), "Volume": gme["Volume"].squeeze(),
                       "VIX": vix["Close"].squeeze()})
    px.index.name = "date"
    px.to_csv(PRICES, float_format="%.6f")
px = pd.read_csv(PRICES, parse_dates=["date"], index_col="date")
s = pd.read_csv(SENT, parse_dates=["date"], index_col="date")
R["n_sentiment_days"] = int(s["sentiment_combined"].notna().sum())
R["sentiment_first_day"] = s.index.min().strftime("%d %B %Y")
R["sentiment_first_value"] = float(s["sentiment_combined"].iloc[0])
R["sentiment_first_posts"] = int(s["r_gme_posts"].iloc[0])

# --------------------------------------------------------------------------- 1. replication
rep = px[["Close", "Volume", "VIX"]].copy()
rep["Sentiment"] = s["sentiment_combined"].reindex(rep.index).bfill()
rep = rep.dropna()
m_rep = sm.OLS(rep["Close"], sm.add_constant(rep[["Volume", "Sentiment", "VIX"]])).fit()
R["rep"] = {"n": int(m_rep.nobs), "r2": float(m_rep.rsquared), "F": float(m_rep.fvalue),
            "coef": {k: float(v) for k, v in m_rep.params.items()},
            "p": {k: float(v) for k, v in m_rep.pvalues.items()},
            "cond": float(np.linalg.cond(sm.add_constant(rep[["Volume", "Sentiment", "VIX"]]).values)),
            "corr_close_sent": float(rep["Close"].corr(rep["Sentiment"])),
            "corr_close_vix": float(rep["Close"].corr(rep["VIX"])),
            "corr_close_vol": float(rep["Close"].corr(rep["Volume"])),
            "desc": rep.describe().to_dict()}
imputed = rep.index < s.index.min()
R["rep"]["share_imputed"] = float(imputed.mean())
R["rep"]["n_imputed"] = int(imputed.sum())
R["rep"]["corr_close_sent_2021_only"] = float(rep.loc[~imputed, "Close"].corr(rep.loc[~imputed, "Sentiment"]))
R["rep"]["share_at_first_value"] = float((rep["Sentiment"].round(4) == round(R["sentiment_first_value"], 4)).mean())

# Figure 1: the imputation problem
fig, ax = plt.subplots(2, 1, figsize=(6.3, 3.8), sharex=True)
ax[0].plot(rep.index, rep["Close"], color="k", lw=0.9)
ax[0].set_ylabel("GME close (US\\$)")
ax[1].plot(rep.index[imputed], rep["Sentiment"][imputed], color="0.6", lw=1.2, label="back-filled (2020)")
ax[1].plot(rep.index[~imputed], rep["Sentiment"][~imputed], color="k", lw=0.7, label="observed (2021)")
ax[1].set_ylabel("Sentiment")
ax[1].legend(frameon=False, loc="center left")
for a in ax:
    a.axvline(EVENT, color="0.4", ls=":", lw=0.8)
fig.align_ylabels()
ax[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.savefig(FIG / "fig1_replication_series.pdf")
plt.close(fig)

# --------------------------------------------------------------------------- 2. revised daily panel
# map every calendar day's posts to the next trading session (posts after the close of
# day t-1 and up to day t belong to session t), posts-weighted sentiment and total posts
subs = ["r_gme", "r_ddintogme", "r_gmejungle"]
cal = pd.DataFrame(index=s.index)
num = sum(s[f"{k}_mean"].fillna(0) * s[f"{k}_posts"].fillna(0) for k in subs)
den = sum(s[f"{k}_posts"].fillna(0) for k in subs)
cal["num"], cal["posts"] = num, den
tdays = px.index
pos = tdays.searchsorted(cal.index)          # next trading day (same day if trading day)
cal = cal[pos < len(tdays)]
cal["session"] = tdays[pos[pos < len(tdays)]]
agg = cal.groupby("session")[["num", "posts"]].sum()
panel = px.copy()
panel["S"] = agg["num"] / agg["posts"]
panel["posts"] = agg["posts"]
# data artefact: 603,691 r/GME posts dated 16 March 2021 (60% of the file) and no posts on
# 18-26 March -> the attention measure is set to missing on 16 March
ART = pd.Timestamp("2021-03-16")
R["artefact_posts"] = int(s.loc[ART, "r_gme_posts"])
R["artefact_share"] = float(s.loc[ART, "r_gme_posts"] / s["r_gme_posts"].sum())
panel.loc[ART, "posts"] = np.nan
R["early_jan_posts_median"] = float(s.loc["2021-01-04":"2021-01-26", "r_gme_posts"].median())
panel["r"] = 100 * np.log(panel["Close"]).diff()
panel["absr"] = panel["r"].abs()
panel["dV"] = np.log(panel["Volume"]).diff()
panel["dVIX"] = 100 * np.log(panel["VIX"]).diff()
panel["A"] = np.log(panel["posts"])
panel["dA"] = panel["A"].diff()
main = panel.loc["2021-01-05":RGME_END].dropna(subset=["r", "S", "A", "dV", "dVIX"])
full21 = panel.loc["2021-01-05":"2021-12-30"].dropna(subset=["r", "S", "dV", "dVIX"])
R["main_n"], R["main_span"] = len(main), [main.index.min().strftime("%d %B %Y"), main.index.max().strftime("%d %B %Y")]
R["full21_n"] = len(full21)
R["posts_total_main"] = int(main["posts"].sum())

# Figure 2: revised panel
fig, ax = plt.subplots(3, 1, figsize=(6.3, 4.2), sharex=True)
ax[0].plot(main.index, main["r"], color="k", lw=0.7)
ax[0].set_ylabel("GME return (%)")
ax[1].plot(main.index, main["S"], color="k", lw=0.7)
ax[1].set_ylabel("Sentiment")
ax[2].plot(main.index, main["posts"] / 1000, color="k", lw=0.7)
ax[2].set_ylabel("Posts (thousands)")
for a in ax:
    a.axvline(EVENT, color="0.4", ls=":", lw=0.8)
fig.align_ylabels()
ax[2].xaxis.set_major_locator(mdates.MonthLocator())
ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
fig.savefig(FIG / "fig2_revised_panel.pdf")
plt.close(fig)

# --------------------------------------------------------------------------- Table 1: descriptive statistics
cols = {"Return (\\%)": main["r"], "Sentiment": main["S"], "Posts": main["posts"],
        "$\\Delta\\ln$ volume": main["dV"], "VIX change (\\%)": main["dVIX"]}
t1 = pd.DataFrame({k: [v.mean(), v.median(), v.std(), v.min(), v.max(), v.skew(), v.kurt() + 3]
                   for k, v in cols.items()},
                  index=["Mean", "Median", "Std. dev.", "Min", "Max", "Skewness", "Kurtosis"])
R["desc_main"] = {k: dict(zip(t1.index, map(float, t1[k]))) for k in t1.columns}


def f3(x):
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3f}"


with open(TAB / "tab1_descriptive.tex", "w") as f:
    f.write("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n & " + " & ".join(t1.columns) + " \\\\\n\\midrule\n")
    for i, row in t1.iterrows():
        f.write(f"{i} & " + " & ".join(f"{v:,.0f}" if (c == "Posts" and i not in ("Skewness", "Kurtosis"))
                                       else f3(v) for c, v in row.items()) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# --------------------------------------------------------------------------- Table 2: stationarity
rows, R["unitroot"] = [], {}
for name, ser in [("GME close (level)", np.log(panel.loc["2020-01-02":"2021-12-30", "Close"]).dropna()),
                  ("Return", main["r"]), ("Sentiment", main["S"]), ("Attention (log posts)", main["A"]),
                  ("$\\Delta\\ln$ volume", main["dV"]), ("VIX change", main["dVIX"])]:
    a, k = ADF(ser, trend="c", method="aic"), KPSS(ser, trend="c")
    rows.append((name, a.stat, a.pvalue, k.stat, k.pvalue))
    R["unitroot"][name] = {"adf_p": float(a.pvalue), "kpss_p": float(k.pvalue)}
with open(TAB / "tab2_unitroot.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\n & \\multicolumn{2}{c}{ADF} & \\multicolumn{2}{c}{KPSS} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nSeries & stat. & $p$ & stat. & $p$ \\\\\n\\midrule\n")
    for r_ in rows:
        f.write(f"{r_[0]} & {r_[1]:.2f} & {r_[2]:.3f} & {r_[3]:.3f} & {r_[4]:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# --------------------------------------------------------------------------- Table 3: regressions
def hac(y, X):
    return sm.OLS(y, sm.add_constant(X), missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 5})


d = main.copy()
for c in ["r", "S", "A", "dVIX", "dV"]:
    d[c + "_l1"] = d[c].shift(1)
m1 = hac(d["r"], d[["S", "dA", "dV", "dVIX"]])                       # contemporaneous
m2 = hac(d["r"], d[["S_l1", "A_l1", "r_l1", "dV_l1", "dVIX_l1"]])     # predictive
m3 = hac(d["absr"], d[["A", "S", "dVIX"]])                           # attention and volatility
X1 = sm.add_constant(d[["S", "dA", "dV", "dVIX"]].dropna())
R["vif"] = {c: float(variance_inflation_factor(X1.values, i)) for i, c in enumerate(X1.columns) if c != "const"}
R["cond_revised"] = float(np.linalg.cond(((X1 - X1.mean()) / X1.std()).fillna(1).values))
for nm, m in [("m1", m1), ("m2", m2), ("m3", m3)]:
    R[nm] = {"n": int(m.nobs), "r2": float(m.rsquared),
             "coef": {k: [float(m.params[k]), float(m.bse[k]), float(m.pvalues[k])] for k in m.params.index}}
# robustness of the predictive regression
sq = (d.index >= "2021-01-11") & (d.index <= "2021-02-05")          # short-squeeze window
m2b = hac(d.loc[~sq, "r"], d.loc[~sq, ["S_l1", "A_l1", "r_l1", "dV_l1", "dVIX_l1"]])
f21 = full21.copy()
for c in ["r", "S", "dVIX", "dV"]:
    f21[c + "_l1"] = f21[c].shift(1)
m2c = hac(f21["r"], f21[["S_l1", "r_l1", "dV_l1", "dVIX_l1"]])    # whole of 2021, no attention
for nm, m in [("m2b_ex_squeeze", m2b), ("m2c_full2021", m2c)]:
    R[nm] = {"n": int(m.nobs), "r2": float(m.rsquared),
             "coef": {k: [float(m.params[k]), float(m.bse[k]), float(m.pvalues[k])] for k in m.params.index}}
R["sd_S_main"] = float(d["S"].std())
# event split: does the sentiment-return link change after 28 January?
pre, post = d.loc[:EVENT], d.loc[EVENT + pd.Timedelta(days=1):]
R["corr_pre_post"] = {"pre_n": len(pre), "post_n": len(post),
                      "S_r_pre": float(pre["S"].corr(pre["r"])), "S_r_post": float(post["S"].corr(post["r"])),
                      "A_absr_pre": float(pre["A"].corr(pre["absr"])), "A_absr_post": float(post["A"].corr(post["absr"]))}
R["corr_main"] = {"S_r": float(d["S"].corr(d["r"])), "S_lag_r": float(d["S_l1"].corr(d["r"])),
                  "A_absr": float(d["A"].corr(d["absr"])), "S_A": float(d["S"].corr(d["A"])),
                  "S_level_close": float(d["S"].corr(d["Close"]))}


def stars(p):
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.10 else ""


labels = [("const", "Constant"), ("S", "Sentiment$_t$"), ("dA", "$\\Delta$ attention$_t$"),
          ("dV", "$\\Delta\\ln$ volume$_t$"), ("dVIX", "VIX change$_t$"), ("S_l1", "Sentiment$_{t-1}$"),
          ("A_l1", "Attention$_{t-1}$"), ("r_l1", "Return$_{t-1}$"), ("dV_l1", "$\\Delta\\ln$ volume$_{t-1}$"),
          ("dVIX_l1", "VIX change$_{t-1}$"), ("A", "Attention$_t$ (log posts)")]
with open(TAB / "tab3_regressions.tex", "w") as f:
    f.write("\\begin{tabular}{lccc}\n\\toprule\n & (1) Return$_t$ & (2) Return$_t$ & (3) $|$Return$_t|$ \\\\\n"
            " & contemporaneous & predictive & attention \\\\\n\\midrule\n")
    for k, lbl in labels:
        cells, ses = [], []
        for m in (m1, m2, m3):
            if k in m.params.index:
                cells.append(f"${m.params[k]:.3f}{stars(m.pvalues[k])}$")
                ses.append(f"$({m.bse[k]:.3f})$")
            else:
                cells.append("")
                ses.append("")
        if any(cells):
            f.write(f"{lbl} & " + " & ".join(cells) + " \\\\\n & " + " & ".join(ses) + " \\\\\n")
    f.write("\\midrule\n$R^2$ & " + " & ".join(f"{m.rsquared:.3f}" for m in (m1, m2, m3)) + " \\\\\n")
    f.write("Observations & " + " & ".join(str(int(m.nobs)) for m in (m1, m2, m3)) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# --------------------------------------------------------------------------- Table 4: Granger (feedback)
def granger(y, x, lag):
    t = grangercausalitytests(pd.concat([y, x], axis=1).dropna(), maxlag=[lag])[lag][0]["ssr_ftest"]
    return float(t[0]), float(t[1])


pairs = [("S", "r", "Sentiment", "Return"), ("r", "S", "Return", "Sentiment"),
         ("A", "r", "Attention", "Return"), ("r", "A", "Return", "Attention"),
         ("A", "absr", "Attention", "$|$Return$|$"), ("absr", "A", "$|$Return$|$", "Attention")]
R["granger"] = {}
with open(TAB / "tab4_granger.tex", "w") as f:
    f.write("\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{Lag selected by AIC}"
            " & \\multicolumn{2}{c}{Lag selected by BIC} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nNull hypothesis & lag, $F$ & $p$ & lag, $F$ & $p$ \\\\\n\\midrule\n")
    for x, y, xl, yl in pairs:
        sel = VAR(d[[y, x]].dropna()).select_order(maxlags=10)
        la, lb = max(1, int(sel.aic)), max(1, int(sel.bic))
        ga, gb = granger(d[y], d[x], la), granger(d[y], d[x], lb)
        R["granger"][f"{x}->{y}"] = {"aic": [la, *ga], "bic": [lb, *gb]}
        f.write(f"{xl} $\\nrightarrow$ {yl} & {la}, {ga[0]:.2f} & {ga[1]:.3f} & {lb}, {gb[0]:.2f} & {gb[1]:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# robustness: Granger tests without the short-squeeze window
R["granger_ex_squeeze"] = {}
dx = d.loc[(d.index < "2021-01-11") | (d.index > "2021-02-05")]
for x, y in [("S", "r"), ("r", "S"), ("A", "absr"), ("absr", "A")]:
    sel = VAR(dx[[y, x]].dropna()).select_order(maxlags=10)
    la, lb = max(1, int(sel.aic)), max(1, int(sel.bic))
    R["granger_ex_squeeze"][f"{x}->{y}"] = {"aic": [la, *granger(dx[y], dx[x], la)], "bic": [lb, *granger(dx[y], dx[x], lb)]}
# robustness: whole of 2021 for the sentiment-return pair
R["granger_full2021"] = {}
for x, y in [("S", "r"), ("r", "S")]:
    sel = VAR(full21[[y, x]].dropna()).select_order(maxlags=10)
    lb = max(1, int(sel.bic))
    R["granger_full2021"][f"{x}->{y}"] = [lb, *granger(full21[y], full21[x], lb)]

# --------------------------------------------------------------------------- wavelet coherence
def wtc_figure(x, y, idx, fname, title_note):
    dt, dj, s0 = 1.0, 1 / 12, 2.0
    J = int(np.log2(32 / s0) / dj)
    WCT, aWCT, coi, freq, sig = pycwt.wct(x, y, dt, dj=dj, s0=s0, J=J, sig=True, significance_level=0.95,
                                          wavelet="morlet", normalize=True, cache=False)
    period = 1 / freq
    t = np.arange(len(x))
    sig95 = WCT / (np.ones([1, len(t)]) * sig[:, None])
    inside = period[:, None] <= coi[None, :]
    signif = (sig95 > 1) & inside
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    cf = ax.contourf(t, np.log2(period), WCT, np.linspace(0, 1, 11), cmap="Greys")
    ax.contour(t, np.log2(period), sig95, [1], colors="k", linewidths=1.0)
    ax.fill_between(t, np.log2(np.maximum(coi, 2)), np.log2(period.max()), color="white", alpha=0.55,
                    hatch="//", edgecolor="0.6", lw=0)
    for i in range(0, len(period), 5):
        for j in range(0, len(t), 4):
            if signif[i, j]:
                ax.quiver(t[j], np.log2(period[i]), np.cos(aWCT[i, j]), np.sin(aWCT[i, j]), scale=40,
                          width=0.0025, headwidth=4, headlength=4, color="k", pivot="middle")
    yt = [2, 4, 8, 16, 32]
    ax.set_yticks(np.log2(yt))
    ax.set_yticklabels(yt)
    ax.set_ylim(np.log2(2), np.log2(32))
    ax.invert_yaxis()
    ticks = [i for i, dd in enumerate(idx) if i == 0 or idx[i - 1].month != dd.month]
    ax.set_xticks(ticks)
    ax.set_xticklabels([idx[i].strftime("%b") for i in ticks])
    ev = np.searchsorted(idx, EVENT)
    ax.axvline(ev, color="0.4", ls=":", lw=0.8)
    ax.set_ylabel("Period (trading days)")
    fig.colorbar(cf, ax=ax, pad=0.02).set_label("Coherence")
    fig.savefig(FIG / fname)
    plt.close(fig)

    def x_leads(a):
        return ((a > 0) & (a < np.pi / 2)) | (a < -np.pi / 2)
    ph = aWCT[signif]
    out = {"share_sig": float(signif.sum() / inside.sum()), "n_sig": int(signif.sum()),
           "inphase": float(np.mean(np.abs(ph) < np.pi / 2)) if ph.size else None,
           "first_leads": float(np.mean(x_leads(ph))) if ph.size else None, "months": {}}
    for mth in sorted(set(idx.month)):
        mt = np.asarray(idx.month == mth)
        cells = signif[:, mt]
        sub = aWCT[:, mt][cells]
        out["months"][int(mth)] = {"share": float(cells.sum() / max(1, inside[:, mt].sum())), "n": int(sub.size),
                                   "inphase": float(np.mean(np.abs(sub) < np.pi / 2)) if sub.size else None,
                                   "first_leads": float(np.mean(x_leads(sub))) if sub.size else None}
    bands = {"2-4": (2, 4), "4-8": (4, 8), "8-16": (8, 16), "16-32": (16, 32)}
    for bn, (lo, hi) in bands.items():
        m = (period >= lo) & (period < hi)
        sub = aWCT[m][signif[m]]
        out[f"band_{bn}"] = {"share": float(signif[m].sum() / max(1, inside[m].sum())), "n": int(sub.size),
                             "inphase": float(np.mean(np.abs(sub) < np.pi / 2)) if sub.size else None,
                             "first_leads": float(np.mean(x_leads(sub))) if sub.size else None}
    return out


R["wtc_sent_ret"] = wtc_figure(main["S"].values, main["r"].values, main.index, "fig3_wtc_sent_ret.pdf", "")
R["wtc_att_vol"] = wtc_figure(main["A"].values, main["absr"].values, main.index, "fig4_wtc_att_vol.pdf", "")

json.dump(R, open(HERE / "results.json", "w"), indent=1, default=str)
print("done")
