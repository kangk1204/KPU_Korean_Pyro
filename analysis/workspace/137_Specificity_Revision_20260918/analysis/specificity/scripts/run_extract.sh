#!/bin/bash
R="${KPU_PROJECT_ROOT:-$HOME/mnt/03_KPU_ML_Final}"   # folder holding 000_*_beta/processed_beta.txt
W="${KPU_SPECIFICITY_WORK:-$(cd "$(dirname "$0")/.." && pwd)}"
REG="$W/registry"; PRV="$W/private"; LOG="$W/logs"
mkdir -p "$PRV" "$LOG"
for spec in CMCBSN:000_CMCBSN_catholic_beta SNUH:000_SNUH_seoul_beta ASAN:000_ASAN_seoul_beta; do
  c=${spec%%:*}; d=${spec##*:}
  echo "[$(date +%T)] $c start" >> "$LOG/extract.log"
  awk -v KEEP="$REG/rows_to_extract.txt" \
      -v OPENSEA="$REG/purity_opensea_probes.txt" \
      -v ISLAND="$REG/purity_island_probes.txt" \
      -v KEEPOUT="$PRV/${c}_keep_beta.tsv" \
      -v MEANOUT="$PRV/${c}_class_means.tsv" \
      -f "$W/scripts/extract_pass.awk" "$R/$d/processed_beta.txt" \
      2>> "$LOG/extract.log"
  echo "[$(date +%T)] $c done  keep=$(wc -l < "$PRV/${c}_keep_beta.tsv")" >> "$LOG/extract.log"
done
echo "[$(date +%T)] ALL DONE" >> "$LOG/extract.log"
