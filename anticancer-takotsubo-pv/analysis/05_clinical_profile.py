"""Clinical profile of anticancer-associated Takotsubo syndrome (TTS) in JADER and FAERS:
anticancer vs other TTS, class-level profiles with time to onset (median and Weibull shape),
severity, cancer type, medical history (JADER), logistic regression for death, sensitivity analyses,
and exploratory analyses (ICI and myocarditis, annual trend, 5-FU cardiac PTs in JADER).
Writes CSV tables to RESULTS_DIR.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from common import WORK_DIR, RESULTS_DIR, CLASS_NAMES, ror

W = lambda name: os.path.join(WORK_DIR, name)
R = lambda name: os.path.join(RESULTS_DIR, name)
J = pd.read_parquet(W("cases_jader.parquet"))
F = pd.read_parquet(W("cases_faers.parquet"))
DBS = {"JADER": J, "FAERS": F, "FAERS excluding Japan": F[F.jp == 0]}
BINARY = ["female", "death", "myocarditis", "vasospasm", "cardiogenic_shock", "cardiac_arrest", "severe_composite",
          "life_threatening", "hospitalised", "hypersensitivity"]


def p_binary(a, b):
    t = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
    return stats.fisher_exact(t)[1] if t.min() < 5 else stats.chi2_contingency(t)[1]


def weibull_shape(t, n_boot=300, seed=1):
    t = t.dropna()
    t = t[t >= 0] + 0.5
    if len(t) < 10:
        return np.nan, np.nan, np.nan
    c = stats.weibull_min.fit(t, floc=0)[0]
    rng = np.random.default_rng(seed)
    bs = [stats.weibull_min.fit(rng.choice(t, len(t)), floc=0)[0] for _ in range(n_boot)]
    return c, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# 1. anticancer vs other TTS
rows = []
for db, x in DBS.items():
    A, O = x[x.anticancer], x[~x.anticancer]
    rows.append(dict(db=db, item="n", anticancer=len(A), other=len(O), p=np.nan))
    a, o = A.age.dropna(), O.age.dropna()
    rows.append(dict(db=db, item="age median (IQR)", anticancer=f"{a.median():.0f} ({a.quantile(.25):.0f}-{a.quantile(.75):.0f})",
                     other=f"{o.median():.0f} ({o.quantile(.25):.0f}-{o.quantile(.75):.0f})", p=stats.mannwhitneyu(a, o)[1]))
    for c in BINARY:
        if c not in x:
            continue
        a, o = A[c].dropna(), O[c].dropna()
        rows.append(dict(db=db, item=c, anticancer=f"{int(a.sum())}/{len(a)} ({100 * a.mean():.1f}%)",
                         other=f"{int(o.sum())}/{len(o)} ({100 * o.mean():.1f}%)", p=p_binary(a, o)))
pd.DataFrame(rows).to_csv(R("table_anticancer_vs_other.csv"), index=False)

# 2. class profiles incl. time to onset
rows = []
for db, x in DBS.items():
    for k in CLASS_NAMES + ["All anticancer"]:
        s = x[x.anticancer] if k == "All anticancer" else x[x[k]]
        t = s.tto.dropna()
        c, l, u = weibull_shape(t)
        d = dict(db=db, cls=k, n=len(s), female_pct=100 * s.female.mean(), age_median=s.age.median(),
                 tto_n=len(t), tto_median=t.median(), tto_q1=t.quantile(.25), tto_q3=t.quantile(.75),
                 weibull_beta=c, weibull_low=l, weibull_high=u)
        for col in ["death", "myocarditis", "cardiogenic_shock", "cardiac_arrest", "severe_composite", "life_threatening", "hospitalised", "hypersensitivity"]:
            if col in s:
                d[col + "_pct"] = 100 * s[col].mean()
        rows.append(d)
pd.DataFrame(rows).round(2).to_csv(R("table_class_profiles.csv"), index=False)

# 3. sensitivity: single class only; catecholamine-free
rows = []
for db, x in [("JADER", J), ("FAERS", F)]:
    for label, sub in [("single class only", x[x.anticancer & (x.nclass == 1)]), ("no catecholamine", x[x.anticancer & ~x.catechol])]:
        for k in CLASS_NAMES:
            s = sub[sub[k]]
            rows.append(dict(db=db, analysis=label, cls=k, n=len(s), female_pct=100 * s.female.mean(),
                             death_pct=100 * s.death.mean(), myocarditis_pct=100 * s.myocarditis.mean()))
pd.DataFrame(rows).round(1).to_csv(R("table_sensitivity_profiles.csv"), index=False)

# 4. cancer type and medical history (JADER)
ct = pd.concat({db: x[x.anticancer].cancer_type.fillna("Other/unknown").value_counts() for db, x in [("JADER", J), ("FAERS", F)]}, axis=1).fillna(0)
ct.to_csv(R("table_cancer_type.csv"))
HX = {"Hypertension": "高血圧", "Diabetes": "糖尿病", "Dyslipidaemia": "脂質異常|高脂血症|高コレステロール",
      "Psychiatric disorder": "うつ|不安|統合失調|双極|パニック|精神", "Neurological disorder": "てんかん|脳梗塞|脳出血|パーキンソン|認知症",
      "Coronary disease": "狭心症|心筋梗塞|冠動脈", "Asthma/COPD": "喘息|閉塞性肺疾患|ＣＯＰＤ|COPD"}
rows = []
for k, pat in HX.items():
    a = J[J.anticancer].history.str.contains(pat)
    o = J[~J.anticancer].history.str.contains(pat)
    rows.append(dict(history=k, anticancer=f"{a.sum()} ({100 * a.mean():.0f}%)", other=f"{o.sum()} ({100 * o.mean():.0f}%)", p=p_binary(a, o)))
pd.DataFrame(rows).to_csv(R("table_history_jader.csv"), index=False)

# 5. logistic regression for death among anticancer TTS
for db, x in [("FAERS", F), ("JADER", J)]:
    A = x[x.anticancer].dropna(subset=["age", "female"]).copy()
    A["age_per_10y"] = A.age / 10
    cols = ["age_per_10y", "female", "myocarditis"] + [k for k in CLASS_NAMES if A[k].sum() >= 10]
    Z = sm.add_constant(A[cols].astype(float))
    fit = sm.Logit(A.death.astype(float), Z).fit(disp=0)
    ci = fit.conf_int()
    pd.DataFrame({"OR": np.exp(fit.params), "CI_low": np.exp(ci[0]), "CI_high": np.exp(ci[1]), "p": fit.pvalues}).drop("const") \
        .round(3).to_csv(R(f"table_logistic_death_{db}.csv"))

# 6. exploratory: ICI-associated TTS with/without myocarditis; annual trend
rows = []
for db, x in [("JADER", J), ("FAERS", F)]:
    s = x[x.ICI]
    for m_ in [1, 0]:
        t = s[s.myocarditis == m_]
        rows.append(dict(db=db, myocarditis=m_, n=len(t), death=int(t.death.sum()), death_pct=100 * t.death.mean(),
                         myositis_or_mg=int(t.myositis_mg.sum()), tto_median=t.tto.median(), female_pct=100 * t.female.mean()))
pd.DataFrame(rows).round(1).to_csv(R("table_ici_myocarditis.csv"), index=False)
pd.concat({"JADER": J[J.anticancer].groupby("year").size(), "FAERS": F[F.anticancer].groupby("year").size(),
           "JADER_ICI": J[J.anticancer & J.ICI].groupby("year").size(), "FAERS_ICI": F[F.anticancer & F.ICI].groupby("year").size()},
          axis=1).fillna(0).astype(int).to_csv(R("table_annual_trend.csv"))

# 7. exploratory: cardiac PTs reported with intravenous 5-FU in JADER
ID, GN = "識別番号", "医薬品（一般名）"
jd = pd.read_parquet(W("jader_drug.parquet"))
jr = pd.read_parquet(W("jader_reac.parquet"))
N = jr[ID].nunique()
fu = set(jd[(jd["医薬品の関与"] == "被疑薬") & jd[GN].str.contains("^フルオロウラシル$", na=False)][ID])
rows = []
for pt in ["ストレス心筋症", "プリンツメタル狭心症", "冠動脈攣縮", "心筋虚血", "狭心症", "急性心筋梗塞", "心筋梗塞", "急性冠動脈症候群",
           "心停止", "心室細動", "心不全", "心筋症", "心筋炎"]:
    ev = set(jr[jr["有害事象"] == pt][ID])
    o, l, u = ror(len(fu & ev), len(fu), len(ev), N)
    rows.append(dict(pt=pt, cases=len(fu & ev), ROR=o, CI_low=l, CI_high=u))
pd.DataFrame(rows).round(2).to_csv(R("table_5fu_cardiac_pts_jader.csv"), index=False)
print("done")
