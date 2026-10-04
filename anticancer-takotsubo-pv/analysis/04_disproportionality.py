"""Disproportionality analyses (reporting odds ratios) in JADER and FAERS:
  main analysis by class and representative drug; FAERS sensitivity analyses (Japan excluded, PS only);
  exploratory regimen-level and two-class combination analyses; fluoropyrimidine comparison.
Writes CSV tables to RESULTS_DIR.
"""
import os
import itertools
import numpy as np
import pandas as pd
from common import (WORK_DIR, RESULTS_DIR, CLASSES, CLASS_NAMES, SINGLE_DRUGS, TTS_PT_JA, ror)

W = lambda name: os.path.join(WORK_DIR, name)
R = lambda name: os.path.join(RESULTS_DIR, name)
ID, GN = "識別番号", "医薬品（一般名）"

# Drugs used to define regimens: (FAERS regex, JADER regex)
DRUGS = {"5FU": ("FLUOROURACIL", "^フルオロウラシル$"), "CAPE": ("CAPECITABINE|XELODA", "カペシタビン"),
         "S1": ("GIMERACIL|OTERACIL|TS-1|TEYSUNO", "テガフール・ギメラシル"), "OX": ("OXALIPLATIN|ELOXATIN", "オキサリプラチン"),
         "CIS": ("CISPLATIN", "^シスプラチン"), "CARBO": ("CARBOPLATIN|PARAPLATIN", "カルボプラチン"),
         "IRI": ("IRINOTECAN|CAMPTOSAR", "イリノテカン"), "PTX": ("PACLITAXEL|TAXOL|ABRAXANE", "パクリタキセル"),
         "DTX": ("DOCETAXEL|TAXOTERE", "ドセタキセル"), "PEM": ("PEMETREXED|ALIMTA", "ペメトレキセド"),
         "GEM": ("GEMCITABINE|GEMZAR", "ゲムシタビン"), "BEV": ("BEVACIZUMAB|AVASTIN", "ベバシズマブ"),
         "TRA": ("TRASTUZUMAB|HERCEPTIN", "トラスツズマブ"), "PER": ("PERTUZUMAB|PERJETA", "ペルツズマブ"), "DOX": ("DOXORUBICIN|ADRIAMYCIN|DOXIL|CAELYX", "ドキソルビシン"),
         "CPA": ("CYCLOPHOSPHAMIDE", "シクロホスファミド"), "RIT": ("RITUXIMAB|RITUXAN|MABTHERA", "リツキシマブ"),
         "NIVO": ("NIVOLUMAB|OPDIVO", "ニボルマブ"), "IPI": ("IPILIMUMAB|YERVOY", "イピリムマブ"),
         "PEMB": ("PEMBROLIZUMAB|KEYTRUDA", "ペムブロリズマブ"), "ATEZ": ("ATEZOLIZUMAB|TECENTRIQ", "アテゾリズマブ"),
         "DURV": ("DURVALUMAB|IMFINZI", "デュルバルマブ"), "OSI": ("OSIMERTINIB|TAGRISSO", "オシメルチニブ"),
         "CETUX": ("CETUXIMAB|ERBITUX", "セツキシマブ"), "PANI": ("PANITUMUMAB|VECTIBIX", "パニツムマブ"),
         "ETO": ("ETOPOSIDE", "エトポシド"), "RAM": ("RAMUCIRUMAB|CYRAMZA", "ラムシルマブ")}
FP = {"5FU", "CAPE", "S1"}
ICI1 = {"PEMB", "ATEZ", "NIVO", "DURV"}
REGIMENS = [
    ("FP + oxaliplatin (FOLFOX/CAPOX/SOX)", lambda s: bool(s & FP) and "OX" in s and "IRI" not in s),
    ("FP + irinotecan (FOLFIRI)", lambda s: bool(s & FP) and "IRI" in s and "OX" not in s),
    ("FP + oxaliplatin/irinotecan + anti-VEGF/EGFR mAb", lambda s: bool(s & FP) and bool(s & {"OX", "IRI"}) and bool(s & {"BEV", "CETUX", "PANI"})),
    ("Cisplatin + 5-FU", lambda s: "CIS" in s and "5FU" in s),
    ("5-FU (intravenous) monotherapy", lambda s: s == {"5FU"}),
    ("Capecitabine monotherapy", lambda s: s == {"CAPE"}),
    ("S-1 monotherapy", lambda s: s == {"S1"}),
    ("Carboplatin + paclitaxel", lambda s: {"CARBO", "PTX"} <= s),
    ("Carboplatin + paclitaxel + ICI", lambda s: {"CARBO", "PTX"} <= s and bool(s & ICI1)),
    ("Platinum + pemetrexed (+/- ICI)", lambda s: "PEM" in s and bool(s & {"CIS", "CARBO"})),
    ("Platinum + etoposide", lambda s: "ETO" in s and bool(s & {"CIS", "CARBO"})),
    ("Trastuzumab + taxane (+/- pertuzumab)", lambda s: "TRA" in s and bool(s & {"PTX", "DTX"})),
    ("Trastuzumab monotherapy", lambda s: s == {"TRA"}),
    ("Doxorubicin + cyclophosphamide (AC)", lambda s: {"DOX", "CPA"} <= s and "RIT" not in s),
    ("Rituximab + cyclophosphamide + doxorubicin (R-CHOP-like)", lambda s: {"RIT", "CPA", "DOX"} <= s),
    ("Gemcitabine + cisplatin (+/- durvalumab)", lambda s: {"GEM", "CIS"} <= s),
    ("Nivolumab + ipilimumab", lambda s: {"NIVO", "IPI"} <= s),
    ("PD-1/PD-L1 inhibitor monotherapy", lambda s: len(s) == 1 and bool(s & ICI1)),
    ("Atezolizumab + bevacizumab", lambda s: {"ATEZ", "BEV"} <= s),
    ("Osimertinib monotherapy", lambda s: s == {"OSI"}),
    ("Any S-1", lambda s: "S1" in s),
    ("Any intravenous 5-FU", lambda s: "5FU" in s),
]


def drug_sets(df, id_col, name_col, lang):
    out = {}
    for k, pats in DRUGS.items():
        for i in df.loc[df[name_col].str.contains(pats[lang], na=False, regex=True), id_col].unique():
            out.setdefault(i, set()).add(k)
    return out


def row(label, analysis, db, a, n, T, N):
    o, l, u = ror(a, n, T, N)
    return dict(db=db, analysis=analysis, exposure=label, cases=a, exposed_reports=n, ROR=o, CI_low=l, CI_high=u)


out = []
# ---------------- JADER ----------------
jd = pd.read_parquet(W("jader_drug.parquet"))
jr = pd.read_parquet(W("jader_reac.parquet"))
NJ = jr[ID].nunique()
tj = set(jr.loc[jr["有害事象"] == TTS_PT_JA, ID])
js = jd[jd["医薬品の関与"] == "被疑薬"]
for k, (_, ja) in list(CLASSES.items()) + list(SINGLE_DRUGS.items()):
    e = set(js[js[GN].str.contains(ja, na=False)][ID])
    out.append(row(k, "main", "JADER", len(e & tj), len(e), len(tj), NJ))
jsets = drug_sets(js, ID, GN, 1)
jt = {i: s for i, s in jsets.items() if i in tj}

# ---------------- FAERS ----------------
rep = pd.read_parquet(W("faers_reports.parquet"))
cases = pd.read_parquet(W("cases_faers.parquet"))
ac = pd.read_parquet(W("faers_ac_suspect.parquet"))
DR = pd.read_parquet(W("faers_tts_drug.parquet"))
for analysis, mask, ps_only in [("main", ~rep.deleted, False), ("excluding Japan", ~rep.deleted & ~rep.jp, False), ("PS only", ~rep.deleted, True)]:
    valid = pd.Index(rep.id[mask.values])
    N = len(valid)
    T = cases[cases.index.isin(valid)]
    acb = ac[ac.id.isin(valid) & (ac.role.eq("PS") if ps_only else True)]
    for k, (en, _) in list(CLASSES.items()) + list(SINGLE_DRUGS.items()):
        e = acb[acb.name.str.contains(en, regex=True)].id.nunique()
        if k in CLASSES:
            a = int(T[k + "_PS"].sum()) if ps_only else int(T[k].sum())
        else:
            dsel = DR[DR.role.isin(["PS"] if ps_only else ["PS", "SS"]) & DR.name.str.contains(en, na=False)]
            a = int(T.index.isin(dsel.id).sum())
        out.append(row(k, analysis, "FAERS", a, e, len(T), N))
pd.DataFrame(out).to_csv(R("table_ror_classes.csv"), index=False)

# ---------------- regimens (exploratory) ----------------
valid = pd.Index(rep.id[(~rep.deleted).values])
N = len(valid)
T = cases[cases.index.isin(valid)]
fsets = drug_sets(ac[ac.id.isin(valid)], "id", "name", 0)
dsus = DR[DR.role.isin(["PS", "SS"]) & DR.id.isin(T.index)]
ft = drug_sets(dsus, "id", "name", 0)
reg = []
for name, f in REGIMENS:
    for db, bg, tt, n_tts, n_all in [("FAERS", fsets, ft, len(T), N), ("JADER", jsets, jt, len(tj), NJ)]:
        e = sum(1 for s in bg.values() if f(s))
        a = sum(1 for s in tt.values() if f(s))
        reg.append(row(name, "regimen", db, a, e, n_tts, n_all))
pd.DataFrame(reg).to_csv(R("table_ror_regimens.csv"), index=False)

# ---------------- two-class combinations, FAERS (exploratory) ----------------
acv = ac[ac.id.isin(valid)].copy()
for k, (en, _) in CLASSES.items():
    acv[k] = acv.name.str.contains(en, regex=True)
B = acv.groupby("id")[CLASS_NAMES].any()
TA = T[T.anticancer]
comb = []
for a_, b_ in itertools.combinations(CLASS_NAMES, 2):
    if int((TA[a_] & TA[b_]).sum()) < 3:
        continue
    for label, eb, tb in [(f"{a_} without {b_}", B[a_] & ~B[b_], T[a_] & ~T[b_]),
                          (f"{b_} without {a_}", B[b_] & ~B[a_], T[b_] & ~T[a_]),
                          (f"{a_} + {b_}", B[a_] & B[b_], T[a_] & T[b_])]:
        r_ = row(label, f"{a_} x {b_}", "FAERS", int(tb.sum()), int(eb.sum()), len(T), N)
        comb.append(r_)
pd.DataFrame(comb).to_csv(R("table_ror_class_combinations.csv"), index=False)
print(pd.DataFrame(out).round(2).to_string(index=False))
