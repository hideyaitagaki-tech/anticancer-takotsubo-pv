#!/usr/bin/env bash
# Extract the FAERS subsets used in this study from the quarterly ASCII zip files.
#
# Usage:
#   bash 01_extract_faers.sh <directory containing FAERS zip files> <output directory>
#
# The zip files are the FDA quarterly ASCII releases (2004Q1-2026Q2), grouped by year
# (e.g. 2004.zip ... 2026.zip, each containing faers_ascii_YYYYqN/ASCII/*.txt).
# Any zip that contains the DEMO/DRUG/REAC/THER/INDI/OUTC text files works.
#
# Outputs (in <output directory>):
#   tts_{drug,reac,demo,ther,indi,outc}.txt  all rows of reports with a Takotsubo PT
#   all_demo_slim.txt.gz      primary id, case id, FDA receipt date of every report
#   all_demo_country.txt.gz   the last two DEMO fields (reporter / occurrence country)
#   all_ac_drug.txt.gz        DRUG rows of every report that mentions a study anticancer drug
#   deleted_cases.txt         FDA deleted-case lists shipped with the quarterly files
# Every output row is prefixed with the source zip name (e.g. "2024$").

set -euo pipefail
SRC="${1:?FAERS zip directory}"
OUT="${2:?output directory}"
mkdir -p "$OUT"
: > "$OUT/deleted_cases.txt"
for t in drug reac demo ther indi outc; do : > "$OUT/tts_${t}.txt"; done
rm -f "$OUT/all_demo_slim.txt.gz" "$OUT/all_demo_country.txt.gz" "$OUT/all_ac_drug.txt.gz"

AC='osimertinib|tagrisso|gefitinib|iressa|erlotinib|tarceva|afatinib|gilotrif|giotrif|dacomitinib|trastuzumab|herceptin|kadcyla|enhertu|pertuzumab|perjeta|lapatinib|tykerb|neratinib|tucatinib|paclitaxel|taxol|abraxane|docetaxel|taxotere|cabazitaxel|carboplatin|paraplatin|cisplatin|oxaliplatin|eloxatin|nedaplatin|fluorouracil|capecitabine|xeloda|tegafur|trifluridine|lonsurf|nivolumab|opdivo|pembrolizumab|keytruda|atezolizumab|tecentriq|durvalumab|imfinzi|ipilimumab|yervoy|avelumab|bavencio|tremelimumab|cemiplimab|libtayo|dostarlimab|bevacizumab|avastin|ramucirumab|cyramza|aflibercept|zaltrap|sorafenib|nexavar|lenvatinib|lenvima|axitinib|inlyta|sunitinib|sutent|pazopanib|votrient|regorafenib|stivarga|cabozantinib|cabometyx|cetuximab|erbitux|panitumumab|vectibix|doxorubicin|adriamycin|doxil|caelyx|epirubicin|daunorubicin|idarubicin|amrubicin|pirarubicin|gemcitabine|gemzar|pemetrexed|alimta|irinotecan|camptosar|etoposide|cyclophosphamide|vincristine|vinorelbine|rituximab|rituxan|mabthera'

for z in "$SRC"/*.zip; do
  echo "$z" >&2
  q=$(basename "$z" .zip)
  # 1) primary ids of reports with a Takotsubo PT
  unzip -p -C "$z" '*reac*.txt' -x '__MACOSX/*' \
    | awk -F'$' 'tolower($0) ~ /stress cardiomyopathy|takotsubo/ {print $1}' | sort -u > "$OUT/ids_tmp.txt"
  # 2) all rows of those reports from the six tables
  for t in drug reac demo ther indi outc; do
    unzip -p -C "$z" "*${t}*.txt" -x '__MACOSX/*' \
      | awk -F'$' -v q="$q" 'NR==FNR{id[$1]=1;next} ($1 in id){print q"$"$0}' "$OUT/ids_tmp.txt" - >> "$OUT/tts_${t}.txt"
  done
  # 3) background tables for deduplication and disproportionality
  unzip -p -C "$z" '*demo*.txt' -x '__MACOSX/*' \
    | awk -F'$' -v q="$q" '{print q"$"$1"$"$2"$"$8}' | gzip >> "$OUT/all_demo_slim.txt.gz"
  unzip -p -C "$z" '*demo*.txt' -x '__MACOSX/*' \
    | awk -F'$' -v q="$q" '{print q"$"$1"$"$(NF-1)"$"$NF}' | gzip >> "$OUT/all_demo_country.txt.gz"
  unzip -p -C "$z" '*drug*.txt' -x '__MACOSX/*' \
    | awk -F'$' -v q="$q" -v ac="$AC" 'tolower($0) ~ ac {print q"$"$1"$"$3"$"$4"$"$5"$"$6}' | gzip >> "$OUT/all_ac_drug.txt.gz"
  # 4) FDA deleted-case lists
  unzip -l "$z" | grep -i 'deleted/.*\.txt' | grep -v __MACOSX | awk '{print $NF}' | while read -r f; do
    unzip -p "$z" "$f" | awk -v q="$q" '{print q"$"$0}' >> "$OUT/deleted_cases.txt"
  done
done
rm -f "$OUT/ids_tmp.txt"
echo "done" >&2
