# Anticancer drug-associated Takotsubo syndrome: JADER/FAERS pharmacovigilance study

Code for data extraction and analysis in:

> Itagaki H. Agent-specific clinical profiles of anticancer drug-associated Takotsubo syndrome: a dual-database pharmacovigilance study using JADER and FAERS. *(manuscript in preparation)*

The study evaluates disproportionality (reporting odds ratios) and clinical profiles of Takotsubo syndrome
(MedDRA PT *Stress cardiomyopathy*) reported with nine anticancer drug classes in the Japanese Adverse Drug
Event Report database (JADER) and the US FDA Adverse Event Reporting System (FAERS), following the
READUS-PV reporting guideline.

## Data

The raw data are publicly available and are **not** included in this repository.

| Database | Source | Version used |
| --- | --- | --- |
| JADER | Pharmaceuticals and Medical Devices Agency (PMDA), https://www.pmda.go.jp/ | Release of August 2026 (`demo/drug/reac/hist202608.csv`) |
| FAERS | US FDA quarterly ASCII files, https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html | 2004 Q1 – 2026 Q2 |

## Requirements

- bash, unzip, awk, sort, gzip (macOS or Linux)
- Python 3.13 with the packages in `requirements.txt` (pandas 3.0, NumPy 2.5, SciPy 1.18, statsmodels 0.15, pyarrow)

## How to reproduce

```bash
# 1. Extract the FAERS subsets (FAERS zip files grouped by year, e.g. 2004.zip ... 2026.zip)
bash extraction/01_extract_faers.sh /path/to/FAERS_zips data/faers

# 2. Case-level deduplication of FAERS
bash extraction/02_deduplicate_faers.sh data/faers

# 3. Analysis (place the JADER CSV files in data/jader)
export JADER_DIR=data/jader FAERS_DIR=data/faers WORK_DIR=work RESULTS_DIR=results
cd analysis
python 01_prepare_jader.py        # JADER CSV (CP932) -> parquet
python 02_prepare_faers.py        # retained reports, deleted cases, reporter country, Takotsubo tables
python 03_build_cases.py          # harmonised case-level tables, secondary deduplication
python 04_disproportionality.py   # RORs: classes, drugs, sensitivity analyses, regimens, combinations
python 05_clinical_profile.py     # clinical profiles, time to onset, severity, regression, exploratory analyses
```

Result tables are written to `results/`.

## Processing steps

**JADER** – reports are linked across tables by identification number; exposure = drug classified as a *suspected* drug.

**FAERS**
1. Case-level deduplication: for each CASEID, the report with the latest FDA receipt date is kept (ties: largest PRIMARYID; legacy AERS: ISR).
2. Cases on the FDA deleted-case lists are removed.
3. Secondary deduplication of Takotsubo reports: reports with identical event date, age, sex and suspect-drug set under different CASEIDs are treated as duplicates of the same patient.
4. Exposure = drug reported as primary suspect (PS) or secondary suspect (SS).
5. Differences in file layout between legacy AERS (2004 Q1–2012 Q3) and FAERS (from 2012 Q4; columns added in 2014 Q3) are handled row by row.

**Definitions** – drug classes, other anticancer drugs, regimens and catecholamines are defined in `analysis/common.py` and `analysis/04_disproportionality.py`.

**Signal criterion** – at least 3 cases and lower limit of the 95% CI of the ROR > 1.

## Output tables

| File | Content |
| --- | --- |
| `table_ror_classes.csv` | ROR by class and representative drug (JADER; FAERS main, excluding Japan, PS only) |
| `table_ror_regimens.csv` | ROR by treatment regimen (exploratory) |
| `table_ror_class_combinations.csv` | ROR for two-class combinations vs each class alone (FAERS, exploratory) |
| `table_anticancer_vs_other.csv` | Anticancer-associated vs other Takotsubo syndrome |
| `table_class_profiles.csv` | Profiles by class: sex, age, death, myocarditis, severity, time to onset, Weibull shape |
| `table_sensitivity_profiles.csv` | Single-class-only and catecholamine-free profiles |
| `table_cancer_type.csv`, `table_history_jader.csv` | Cancer type; medical history (JADER only) |
| `table_logistic_death_*.csv` | Logistic regression for death |
| `table_ici_myocarditis.csv`, `table_annual_trend.csv`, `table_5fu_cardiac_pts_jader.csv` | Exploratory analyses |

## License

MIT License (see `LICENSE`).
