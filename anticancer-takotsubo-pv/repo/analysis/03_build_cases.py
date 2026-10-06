"""Build harmonised case-level tables of Takotsubo reports for JADER and FAERS.

FAERS: reports of FDA-deleted cases are removed, and a secondary deduplication drops reports with an
identical event date, age, sex and suspect-drug set under a different CASEID (same patient reported
by several senders).
"""
import os
import re
import numpy as np
import pandas as pd
from common import (WORK_DIR, CLASSES, CLASS_NAMES, OTHER_AC, ALL_AC, CATECHOLAMINES, TTS_PT_JA, ymd)

W = lambda name: os.path.join(WORK_DIR, name)
ID, GN = "識別番号", "医薬品（一般名）"


def cancer_type_ja(s):
    for k, p in [("Lung", "肺"), ("Breast", "乳"), ("Colorectal", "結腸|直腸|大腸"), ("Gastric", "胃"), ("Esophageal", "食道"),
                 ("Pancreatic", "膵"), ("Biliary", "胆"), ("Hepatocellular", "肝細胞|肝癌"), ("Head and neck", "頭頸|咽頭|喉頭|口腔|舌"),
                 ("Ovarian/uterine", "卵巣|子宮|体癌"), ("Urological", "膀胱|尿路|腎細胞|前立腺"), ("Melanoma", "黒色腫|メラノーマ"),
                 ("Hematologic", "リンパ腫|白血病|骨髄腫")]:
        if re.search(p, s):
            return k
    return "Other/unknown"


def cancer_type_en(s):
    s = s.lower()
    for k, p in [("Lung", "lung|nsclc|small cell"), ("Breast", "breast"), ("Colorectal", "colo|rectal|rectum|bowel"), ("Gastric", "gastric|stomach"),
                 ("Esophageal", "oesophag|esophag"), ("Pancreatic", "pancrea"), ("Biliary", "biliary|bile duct|cholangio|gallbladder"),
                 ("Hepatocellular", "hepatocellular|liver"), ("Head and neck", "head and neck|pharyn|laryn|oral|tongue|tonsil"),
                 ("Ovarian/uterine", "ovar|uter|endometri|cervi"), ("Urological", "bladder|urothel|renal|prostat"),
                 ("Melanoma", "melanoma"), ("Hematologic", "lymphoma|leuk|myeloma")]:
        if re.search(p, s):
            return k
    return "Other/unknown"


def flag(ptsets, index, test):
    return np.array([int(any(test(p) for p in ptsets.get(i, set()))) for i in index])


# ======================= JADER =======================
d = pd.read_parquet(W("jader_drug.parquet"))
r = pd.read_parquet(W("jader_reac.parquet"))
m = pd.read_parquet(W("jader_demo.parquet")).drop_duplicates(ID).set_index(ID)
h = pd.read_parquet(W("jader_hist.parquet"))
tr = r[r["有害事象"] == TTS_PT_JA].groupby(ID).first()
ids = tr.index
dd = d[d[ID].isin(ids)]
sus = dd[dd["医薬品の関与"] == "被疑薬"]
X = pd.DataFrame(index=ids)
for k, (_, ja) in CLASSES.items():
    X[k] = X.index.isin(sus[sus[GN].str.contains(ja, na=False)][ID])
X["anticancer"] = X[CLASS_NAMES].any(axis=1) | X.index.isin(sus[sus[GN].str.contains(OTHER_AC[1], na=False)][ID])
X["nclass"] = X[CLASS_NAMES].sum(axis=1)
X["catechol"] = X.index.isin(dd[dd[GN].str.contains(CATECHOLAMINES[1], na=False)][ID])
mm = m.reindex(ids)
X["female"] = np.where(mm["性別"] == "女性", 1.0, np.where(mm["性別"] == "男性", 0.0, np.nan))
X["age"] = mm["年齢"].str.extract(r"(\d+)")[0].astype(float) + 5  # midpoint of 10-year band
X.loc[mm["年齢"].str.contains("未満", na=False).values, "age"] = 5
X["year"] = mm["報告年度・四半期"].str[:4].astype(float)
X["death"] = (tr["転帰"] == "死亡").astype(int)
pts = r[r[ID].isin(ids)].groupby(ID)["有害事象"].apply(set)
X["myocarditis"] = flag(pts, X.index, lambda p: "心筋炎" in p)
X["vasospasm"] = flag(pts, X.index, lambda p: ("冠動脈攣縮" in p) or ("血管攣縮" in p) or ("プリンツメタル" in p))
X["cardiogenic_shock"] = flag(pts, X.index, lambda p: p == "心原性ショック")
X["cardiac_arrest"] = flag(pts, X.index, lambda p: p in ("心停止", "心肺停止"))
X["myositis_mg"] = flag(pts, X.index, lambda p: ("筋炎" in p and "心筋炎" not in p) or ("重症筋無力症" in p))
X["severe_composite"] = ((X.cardiogenic_shock + X.cardiac_arrest + X.death) > 0).astype(int)
acs = sus[sus[GN].str.contains(ALL_AC[1], na=False)].copy()
acs["st"] = ymd(acs["投与開始日"].fillna("").str.split(";").str[0])
X["tto"] = (ymd(tr["有害事象の発現日"]) - acs.groupby(ID).st.min().reindex(ids)).dt.days
X.loc[X.tto < 0, "tto"] = np.nan
X["cancer_type"] = acs.groupby(ID)["使用理由"].apply(lambda s: cancer_type_ja(" ".join(s.fillna("")))).reindex(ids)
X["history"] = h[h[ID].isin(ids)].groupby(ID)["原疾患等"].apply(lambda s: " ".join(s.fillna(""))).reindex(ids).fillna("")
X["jp"] = 1
X.to_parquet(W("cases_jader.parquet"))
print("JADER Takotsubo", len(X), "anticancer", int(X.anticancer.sum()))

# ======================= FAERS =======================
DM = pd.read_parquet(W("faers_tts_demo.parquet")).set_index("id")
DR = pd.read_parquet(W("faers_tts_drug.parquet"))
RE = pd.read_parquet(W("faers_tts_reac.parquet"))
TH = pd.read_parquet(W("faers_tts_ther.parquet"))
IN = pd.read_parquet(W("faers_tts_indi.parquet"))
OC = pd.read_parquet(W("faers_tts_outc.parquet"))
REP = pd.read_parquet(W("faers_reports.parquet")).set_index("id")

F = pd.DataFrame(index=DM.index)
dsus = DR[DR.role.isin(["PS", "SS"])]
dps = DR[DR.role == "PS"]
for k, (en, _) in CLASSES.items():
    F[k] = F.index.isin(dsus[dsus.name.str.contains(en, na=False)].id)
    F[k + "_PS"] = F.index.isin(dps[dps.name.str.contains(en, na=False)].id)
F["anticancer"] = F[CLASS_NAMES].any(axis=1) | F.index.isin(dsus[dsus.name.str.contains(OTHER_AC[0], na=False)].id)
F["nclass"] = F[CLASS_NAMES].sum(axis=1)
F["catechol"] = F.index.isin(DR[DR.name.str.contains(CATECHOLAMINES[0], na=False)].id)
sx = DM.sex.str.upper()
F["female"] = np.where(sx == "F", 1.0, np.where(sx == "M", 0.0, np.nan))
a = pd.to_numeric(DM.age, errors="coerce")
ac_ = DM.age_cod.str.upper()
F["age"] = np.select([ac_.isin(["YR", "Y"]), ac_ == "DEC", ac_ == "MON", ac_ == "WK", ac_ == "DY"], [a, a * 10, a / 12, a / 52, a / 365], np.nan)
F.loc[(F.age < 0) | (F.age > 110), "age"] = np.nan
F["year"] = pd.to_numeric(DM.fda.str[:4], errors="coerce")
oc = OC.groupby("id").outc.apply(set)
F["death"] = [int("DE" in oc.get(i, set())) for i in F.index]
F["life_threatening"] = [int("LT" in oc.get(i, set())) for i in F.index]
F["hospitalised"] = [int("HO" in oc.get(i, set())) for i in F.index]
pts = RE.groupby("id").pt.apply(set)
F["myocarditis"] = flag(pts, F.index, lambda p: "myocarditis" in p)
F["vasospasm"] = flag(pts, F.index, lambda p: any(k in p for k in ["coronary artery spasm", "vasospasm", "prinzmetal"]))
F["cardiogenic_shock"] = flag(pts, F.index, lambda p: "cardiogenic shock" in p)
F["cardiac_arrest"] = flag(pts, F.index, lambda p: p in ("cardiac arrest", "cardio-respiratory arrest"))
F["myositis_mg"] = flag(pts, F.index, lambda p: ("myositis" in p) or ("myasthenia" in p))
F["hypersensitivity"] = flag(pts, F.index, lambda p: any(k in p for k in ["anaphyla", "hypersensitivity", "infusion related reaction", "infusion-related"]))
F["severe_composite"] = ((F.cardiogenic_shock + F.cardiac_arrest + F.death + F.life_threatening) > 0).astype(int)
acd = dsus[dsus.name.str.contains(ALL_AC[0], na=False)][["id", "seq"]]
th = TH.merge(acd, on=["id", "seq"])
th["st"] = ymd(th.start)
F["tto"] = (ymd(DM.event_dt) - th.groupby("id").st.min().reindex(F.index)).dt.days
F.loc[F.tto < 0, "tto"] = np.nan
ii = IN.merge(acd, on=["id", "seq"])
F["cancer_type"] = ii.groupby("id").indi.apply(lambda s: cancer_type_en(" ".join(s.fillna("")))).reindex(F.index)
F["jp"] = REP.jp.reindex(F.index).fillna(False).astype(int)
F["deleted"] = REP.deleted.reindex(F.index).fillna(False)

# secondary deduplication
sus2 = dsus.copy()
sus2["ing"] = sus2.name.str.replace(r"[^A-Z]", "", regex=True).str[:8]
dset = sus2.groupby("id").ing.apply(lambda s: "|".join(sorted(set(s))))
key_df = pd.DataFrame({"ev": DM.event_dt.reindex(F.index), "age": F.age.round(), "sex": F.female, "drugs": dset.reindex(F.index)})
ok = key_df.ev.str.len().eq(8) & key_df.age.notna() & key_df.sex.notna()
key = key_df.ev + "_" + key_df.age.astype(str) + "_" + key_df.sex.astype(str) + "_" + key_df.drugs.fillna("")
dup = ok & key.duplicated(keep="first")
print("FAERS secondary duplicates removed", int(dup.sum()), "| deleted-case reports removed", int((F.deleted & ~dup).sum()))
F = F[~dup & ~F.deleted].drop(columns="deleted")
F.to_parquet(W("cases_faers.parquet"))
print("FAERS Takotsubo", len(F), "anticancer", int(F.anticancer.sum()))
