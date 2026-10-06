"""Exploratory analysis of co-reported potential triggers of Takotsubo syndrome (TTS):
anticancer-associated vs other TTS (JADER, FAERS) and by drug class (FAERS).
A trigger is counted when any preferred term of the report contains one of the listed strings
(the TTS term itself and cardiomyopathy/myocarditis terms are excluded).
Categories that may reflect symptoms of TTS itself (dyspnoea/respiratory failure, nausea/vomiting/diarrhoea,
pain including chest pain) are reported separately and are not counted as potential precipitants.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from common import WORK_DIR, RESULTS_DIR, CLASS_NAMES

W = lambda name: os.path.join(WORK_DIR, name)
R = lambda name: os.path.join(RESULTS_DIR, name)
ID = "識別番号"

TRIGGERS = {  # name: (FAERS PT substrings, JADER PT substrings)
    "Infection/sepsis": (["sepsis", "septic", "pneumonia", "infection", "bacteraemia", "febrile neutropenia"],
                         ["敗血症", "肺炎", "感染", "菌血症", "発熱性好中球減少"]),
    "Hypersensitivity/anaphylaxis": (["anaphyla", "hypersensitivity", "infusion related reaction"],
                                     ["アナフィラキシー", "過敏症", "注入に伴う反応", "インフュージョンリアクション"]),
    "Respiratory failure/distress": (["respiratory failure", "respiratory distress", "hypoxia", "dyspnoea"],
                                     ["呼吸不全", "呼吸窮迫", "低酸素", "呼吸困難"]),
    "Vomiting/diarrhoea/dehydration": (["vomiting", "diarrhoea", "dehydration", "nausea"], ["嘔吐", "下痢", "脱水", "悪心"]),
    "Electrolyte/metabolic disturbance": (["hyponatraemia", "hypokalaemia", "hypoglycaemia", "hypomagnesaemia", "hypocalcaemia"],
                                          ["低ナトリウム", "低カリウム", "低血糖", "低マグネシウム", "低カルシウム"]),
    "Bleeding/anaemia": (["haemorrhage", "anaemia", "bleeding"], ["出血", "貧血"]),
    "Pain": (["pain"], ["疼痛", "痛"]),
    "Neurological event": (["seizure", "convulsion", "cerebral infarction", "cerebrovascular accident", "encephalopathy"],
                           ["痙攣", "てんかん", "脳梗塞", "脳出血", "脳症"]),
    "Psychological stress": (["anxiety", "stress", "depress", "panic", "emotional distress"], ["不安", "ストレス", "うつ", "パニック"]),
    "Myositis/myasthenia gravis": (["myositis", "myasthenia"], ["筋炎", "重症筋無力症"]),
    "Endocrine (thyroid/adrenal)": (["thyro", "adrenal insufficiency", "adrenocortical"], ["甲状腺", "副腎"]),
    "Surgery/procedure": (["procedural", "postoperative", "post procedural"], ["処置", "術後"]),
}
EXCL = (["cardiomyopath", "myocarditis"], ["心筋症", "心筋炎"])
SYMPTOMS = {"Respiratory failure/distress", "Vomiting/diarrhoea/dehydration", "Pain"}
PRECIPITANTS = [k for k in TRIGGERS if k not in SYMPTOMS]

F = pd.read_parquet(W("cases_faers.parquet"))
J = pd.read_parquet(W("cases_jader.parquet"))
fp = pd.read_parquet(W("faers_tts_reac.parquet")).groupby("id").pt.apply(set)
jr = pd.read_parquet(W("jader_reac.parquet"))
jp = jr[jr[ID].isin(J.index)].groupby(ID)["有害事象"].apply(set)


def flag(sets, index, keys, excl):
    return pd.Series([int(any(any(k in p for k in keys) and not any(e in p for e in excl) for p in sets.get(i, set())))
                      for i in index], index=index)


rows, flags = [], {"FAERS": {}, "JADER": {}}
for name, (en, ja) in TRIGGERS.items():
    for db, x, s, keys, ex in [("FAERS", F, fp, en, EXCL[0]), ("JADER", J, jp, ja, EXCL[1])]:
        v = flag(s, x.index, keys, ex)
        flags[db][name] = v
        a, o = v[x.anticancer], v[~x.anticancer]
        t = np.array([[a.sum(), len(a) - a.sum()], [o.sum(), len(o) - o.sum()]])
        p = stats.fisher_exact(t)[1] if t.min() < 5 else stats.chi2_contingency(t)[1]
        rows.append(dict(trigger=name, db=db, anticancer=f"{a.sum()}/{len(a)} ({100 * a.mean():.1f}%)",
                         other=f"{o.sum()}/{len(o)} ({100 * o.mean():.1f}%)", p=p))
for db, x in [("FAERS", F), ("JADER", J)]:
    anyv = pd.concat({k: flags[db][k] for k in PRECIPITANTS}, axis=1).max(axis=1)
    flags[db]["Any potential precipitant"] = anyv
    a, o = anyv[x.anticancer], anyv[~x.anticancer]
    t = np.array([[a.sum(), len(a) - a.sum()], [o.sum(), len(o) - o.sum()]])
    rows.append(dict(trigger="Any potential precipitant", db=db, anticancer=f"{a.sum()}/{len(a)} ({100 * a.mean():.1f}%)",
                     other=f"{o.sum()}/{len(o)} ({100 * o.mean():.1f}%)", p=stats.chi2_contingency(t)[1]))
out = pd.DataFrame(rows)
out["group"] = np.where(out.trigger.isin(SYMPTOMS), "possible TTS symptom", "potential precipitant")
out.to_csv(R("table_triggers_anticancer_vs_other.csv"), index=False)

byclass = pd.DataFrame({name: [100 * flags["FAERS"][name][F[k]].mean() for k in CLASS_NAMES]
                        for name in ["Any potential precipitant"] + list(TRIGGERS)},
                       index=CLASS_NAMES).T.round(1)
byclass.loc["n"] = [int(F[k].sum()) for k in CLASS_NAMES]
byclass.to_csv(R("table_triggers_by_class_faers.csv"))
print(out.to_string(index=False))
print(byclass.to_string())
