#!/usr/bin/env bash
# Case-level deduplication of FAERS (latest FDA receipt date per CASEID; ties broken by the
# largest primary id / ISR). Uses sort on disk so that ~25 million rows fit in modest memory.
#
# Usage: bash 02_deduplicate_faers.sh <output directory of 01_extract_faers.sh>
# Output: keep_ids.txt (one retained primary id / ISR per line)

set -euo pipefail
OUT="${1:?output directory}"
zcat "$OUT/all_demo_slim.txt.gz" \
  | awk -F'$' '$2!="ISR" && tolower($2)!="primaryid" {printf "%s\t%s\t%012d\n",$3,$4,$2}' \
  | LC_ALL=C sort -S 1G -T "$OUT" -t$'\t' -k1,1 -k2,2 -k3,3 \
  | awk -F'\t' '{if($1!=p && NR>1) print last; p=$1; last=$3+0} END{print last}' > "$OUT/keep_ids.txt"
wc -l "$OUT/keep_ids.txt" >&2
