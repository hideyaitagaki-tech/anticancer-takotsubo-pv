"""Prepare FAERS: retained reports (case-level dedup + FDA deleted cases), reporter country,
anticancer suspect-drug table for the whole database, and the Takotsubo report tables.

Inputs (FAERS_DIR): keep_ids.txt, all_demo_slim.txt.gz, all_demo_country.txt.gz,
                    deleted_cases.txt, all_ac_drug.txt.gz, tts_*.txt
FAERS file layouts differ by era (legacy AERS 2004Q1-2012Q3 vs FAERS from 2012Q4, with further
column additions in 2014Q3); each row is parsed according to its layout.
"""
import os
import numpy as np
import pandas as pd
from common import FAERS_DIR, WORK_DIR

F = lambda name: os.path.join(FAERS_DIR, name)
W = lambda name: os.path.join(WORK_DIR, name)
ROLES = {"PS", "SS", "C", "I"}
AERS_MAX_ID = 30_000_000  # legacy AERS ISRs are < 3e7; FAERS primary ids are larger


def is_aers(i):
    return int(i) < AERS_MAX_ID


# ---- 1. retained reports, deleted cases, reporter country ----
keep = np.loadtxt(F("keep_ids.txt"), dtype=np.int64)
keep.sort()
dl = pd.read_csv(F("deleted_cases.txt"), sep="$", header=None, names=["q", "case"], dtype=str)
deleted_cases = np.unique(pd.to_numeric(dl.case.str.strip(), errors="coerce").dropna().astype(np.int64))

parts = []
for ch in pd.read_csv(F("all_demo_slim.txt.gz"), sep="$", header=None, names=["q", "id", "case", "fda"],
                      dtype=str, chunksize=3_000_000):
    ch = ch[ch.id.str.isdigit().fillna(False)]
    i = ch.id.astype(np.int64).values
    m = np.isin(i, keep)
    parts.append(pd.DataFrame({"id": i[m], "case": pd.to_numeric(ch.case.values[m], errors="coerce")}))
K = pd.concat(parts).drop_duplicates("id")
K["deleted"] = np.isin(K.case.fillna(-1).astype(np.int64).values, deleted_cases)

parts = []
for ch in pd.read_csv(F("all_demo_country.txt.gz"), sep="$", header=None, names=["q", "id", "a", "b"],
                      dtype=str, chunksize=3_000_000):
    ch = ch[ch.id.str.isdigit().fillna(False)]
    i = ch.id.astype(np.int64).values
    m = np.isin(i, keep)
    parts.append(pd.DataFrame({"id": i[m], "a": ch.a.values[m], "b": ch.b.values[m]}))
C = pd.concat(parts).drop_duplicates("id")
K = K.merge(C, on="id", how="left")
up = lambda s: s.fillna("").str.strip().str.upper()
K["jp"] = up(K.a).isin(["JP", "JAPAN"]) | up(K.b).isin(["JP", "JAPAN"])
K[["id", "deleted", "jp"]].to_parquet(W("faers_reports.parquet"))
print("retained reports", len(K), "deleted", int(K.deleted.sum()), "Japan", int(K.jp.sum()))

# ---- 2. anticancer suspect drugs in the whole database ----
del K, C, parts
import gc; gc.collect()
parts = []
for ch in pd.read_csv(F("all_ac_drug.txt.gz"), sep="$", header=None, names=["q", "id", "c3", "c4", "c5", "c6"],
                      dtype=str, quoting=3, on_bad_lines="skip", chunksize=1_000_000):
    ch = ch[ch.id.str.isdigit().fillna(False)]
    aers = ch.c3.isin(ROLES)  # legacy: ISR, ROLE, DRUGNAME ; FAERS: primaryid, drug_seq, role, drugname, prod_ai
    role = np.where(aers, ch.c3, ch.c4)
    name = np.where(aers, ch.c4.fillna(""), ch.c5.fillna("") + " " + ch.c6.fillna("")).astype(str)
    i = ch.id.astype(np.int64).values
    m = np.isin(i, keep) & np.isin(role, ["PS", "SS"])
    parts.append(pd.DataFrame({"id": i[m], "role": role[m], "name": pd.Series(name[m]).str.upper().values}))
ac = pd.concat(parts)
ac.to_parquet(W("faers_ac_suspect.parquet"))
print("anticancer suspect rows", len(ac))


# ---- 3. Takotsubo report tables ----
def rows(t):
    with open(F(f"tts_{t}.txt"), encoding="latin-1") as fh:
        return [l.rstrip("\r\n").split("$") for l in fh]


dem = []
for r in rows("demo"):
    n = len(r)
    if n == 25:    # AERS: ISR CASE I_F FOLL IMAGE EVENT_DT MFR_DT FDA_DT ... AGE AGE_COD GNDR ... COUNTRY
        dem.append((r[1], r[2], r[6], r[8], r[12], r[13], r[14], r[23]))
    elif n == 23:  # FAERS 2012Q4-2014Q2
        dem.append((r[1], r[2], r[5], r[8], r[12], r[13], r[14], r[21]))
    else:          # FAERS from 2014Q3
        dem.append((r[1], r[2], r[5], r[8], r[14], r[15], r[17], r[24]))
DM = pd.DataFrame(dem, columns=["id", "case", "event_dt", "fda", "age", "age_cod", "sex", "country"]).drop_duplicates("id")
DM["id"] = DM.id.astype(np.int64)
DM = DM[DM.id.isin(set(keep.tolist()))]

dr = []
for r in rows("drug"):
    if len(r) == 14:
        dr.append((r[1], r[2], r[3], r[4].upper()))
    else:
        dr.append((r[1], r[3], r[4], (r[5] + " " + r[6]).upper()))
DR = pd.DataFrame(dr, columns=["id", "seq", "role", "name"]).drop_duplicates()
RE = pd.DataFrame([(r[1], (r[2] if is_aers(r[1]) else r[3]).strip().lower()) for r in rows("reac")],
                  columns=["id", "pt"]).drop_duplicates()
TH = pd.DataFrame([(r[1], r[2], r[3]) if is_aers(r[1]) else (r[1], r[3], r[4]) for r in rows("ther")],
                  columns=["id", "seq", "start"])
IN = pd.DataFrame([(r[1], r[2], r[3].lower()) if len(r) == 4 else (r[1], r[3], r[4].lower()) for r in rows("indi")],
                  columns=["id", "seq", "indi"])
OC = pd.DataFrame([(r[1], (r[2] if is_aers(r[1]) else r[3]).strip()) for r in rows("outc")], columns=["id", "outc"])
for x in (DR, RE, TH, IN, OC):
    x["id"] = x.id.astype(np.int64)

tts = set(RE[RE.pt == "stress cardiomyopathy"].id) & set(DM.id)
for name, x in [("demo", DM), ("drug", DR), ("reac", RE), ("ther", TH), ("indi", IN), ("outc", OC)]:
    x[x.id.isin(tts)].to_parquet(W(f"faers_tts_{name}.parquet"))
print("Takotsubo reports after case-level dedup", len(tts))
