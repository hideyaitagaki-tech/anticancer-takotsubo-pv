"""Shared settings, drug definitions and helpers.

Paths are taken from environment variables so the scripts can be run anywhere:
  JADER_DIR   folder with the JADER CSV files (demoYYYYMM.csv, drugYYYYMM.csv, reacYYYYMM.csv, histYYYYMM.csv)
  FAERS_DIR   output folder of extraction/01_extract_faers.sh (and keep_ids.txt from 02_deduplicate_faers.sh)
  WORK_DIR    folder for intermediate files (default: ./work)
  RESULTS_DIR folder for result tables (default: ./results)
"""
import os
import glob
import numpy as np
import pandas as pd

JADER_DIR = os.environ.get("JADER_DIR", "data/jader")
FAERS_DIR = os.environ.get("FAERS_DIR", "data/faers")
WORK_DIR = os.environ.get("WORK_DIR", "work")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
for _d in (WORK_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

TTS_PT_JA = "ストレス心筋症"
TTS_PT_EN = "stress cardiomyopathy"

# Anticancer classes: (FAERS regex on upper-case drug name + active ingredient, JADER regex on generic name)
CLASSES = {
    "EGFR-TKI": ("OSIMERTINIB|TAGRISSO|GEFITINIB|IRESSA|ERLOTINIB|TARCEVA|AFATINIB|GILOTRIF|GIOTRIF|DACOMITINIB",
                 "オシメルチニブ|ゲフィチニブ|エルロチニブ|アファチニブ|ダコミチニブ"),
    "Anti-HER2": ("TRASTUZUMAB|HERCEPTIN|KADCYLA|ENHERTU|PERTUZUMAB|PERJETA|LAPATINIB|TYKERB|NERATINIB|TUCATINIB",
                  "トラスツズマブ|ペルツズマブ|ラパチニブ|ネラチニブ|ツカチニブ"),
    "Taxane": ("PACLITAXEL|TAXOL|ABRAXANE|DOCETAXEL|TAXOTERE|CABAZITAXEL",
               "パクリタキセル|ドセタキセル|カバジタキセル"),
    "Platinum": ("CARBOPLATIN|PARAPLATIN|CISPLATIN|OXALIPLATIN|ELOXATIN|NEDAPLATIN",
                 "カルボプラチン|シスプラチン|オキサリプラチン|ネダプラチン"),
    "Fluoropyrimidine": ("FLUOROURACIL|CAPECITABINE|XELODA|TEGAFUR|TRIFLURIDINE|LONSURF",
                         "フルオロウラシル|カペシタビン|テガフール|トリフルリジン"),
    "ICI": ("NIVOLUMAB|OPDIVO|PEMBROLIZUMAB|KEYTRUDA|ATEZOLIZUMAB|TECENTRIQ|DURVALUMAB|IMFINZI|IPILIMUMAB|YERVOY|AVELUMAB|BAVENCIO|TREMELIMUMAB|CEMIPLIMAB|LIBTAYO|DOSTARLIMAB",
            "ニボルマブ|ペムブロリズマブ|アテゾリズマブ|デュルバルマブ|イピリムマブ|アベルマブ|トレメリムマブ|セミプリマブ|ドスタルリマブ"),
    "Anti-VEGF": ("BEVACIZUMAB|AVASTIN|RAMUCIRUMAB|CYRAMZA|AFLIBERCEPT|ZALTRAP|SORAFENIB|NEXAVAR|LENVATINIB|LENVIMA|AXITINIB|INLYTA|SUNITINIB|SUTENT|PAZOPANIB|VOTRIENT|REGORAFENIB|STIVARGA|CABOZANTINIB|CABOMETYX",
                  "ベバシズマブ|ラムシルマブ|アフリベルセプト|ソラフェニブ|レンバチニブ|アキシチニブ|スニチニブ|パゾパニブ|レゴラフェニブ|カボザンチニブ"),
    "Anti-EGFR mAb": ("CETUXIMAB|ERBITUX|PANITUMUMAB|VECTIBIX", "セツキシマブ|パニツムマブ"),
    "Anthracycline": ("DOXORUBICIN|ADRIAMYCIN|DOXIL|CAELYX|EPIRUBICIN|DAUNORUBICIN|IDARUBICIN|AMRUBICIN|PIRARUBICIN",
                      "ドキソルビシン|エピルビシン|ダウノルビシン|イダルビシン|アムルビシン|ピラルビシン"),
}
CLASS_NAMES = list(CLASSES)
# Other anticancer drugs that define "anticancer-associated TTS" but are not analysed as classes
OTHER_AC = ("GEMCITABINE|GEMZAR|PEMETREXED|ALIMTA|IRINOTECAN|CAMPTOSAR|ETOPOSIDE|CYCLOPHOSPHAMIDE|VINCRISTINE|VINORELBINE|RITUXIMAB|RITUXAN|MABTHERA",
            "ゲムシタビン|ペメトレキセド|イリノテカン|エトポシド|シクロホスファミド|ビンクリスチン|ビノレルビン|リツキシマブ")
ALL_AC = ("|".join(v[0] for v in CLASSES.values()) + "|" + OTHER_AC[0],
          "|".join(v[1] for v in CLASSES.values()) + "|" + OTHER_AC[1])
SINGLE_DRUGS = {"Osimertinib": ("OSIMERTINIB|TAGRISSO", "オシメルチニブ"),
                "Trastuzumab": ("TRASTUZUMAB|HERCEPTIN|KADCYLA|ENHERTU", "トラスツズマブ"),
                "5-FU (intravenous)": ("FLUOROURACIL", "^フルオロウラシル$")}
CATECHOLAMINES = ("EPINEPHRINE|ADRENALINE|NOREPINEPHRINE|NORADRENALINE|DOPAMINE|DOBUTAMINE|ISOPROTERENOL|ISOPRENALINE",
                  "アドレナリン|ドパミン|ドブタミン|イソプレナリン")


def ror(a, n_exposed, n_events, n_total, min_a=3):
    """Reporting odds ratio with 95% CI. a = exposed cases with the event."""
    b, c = n_exposed - a, n_events - a
    d = n_total - a - b - c
    if a < min_a or min(b, c, d) <= 0:
        return np.nan, np.nan, np.nan
    o = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return o, o * np.exp(-1.96 * se), o * np.exp(1.96 * se)


def ymd(s):
    """Parse YYYYMMDD strings (first 8 characters); shorter dates become NaT."""
    s = s.fillna("").astype(str)
    return pd.to_datetime(s.where(s.str.len() >= 8).str[:8], format="%Y%m%d", errors="coerce")


def jader_file(kind):
    files = sorted(glob.glob(os.path.join(JADER_DIR, f"{kind}*.csv")))
    if not files:
        raise FileNotFoundError(f"{kind}*.csv not found in {JADER_DIR}")
    return files[-1]
