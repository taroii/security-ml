#!/usr/bin/env bash
set -e
mkdir -p logs results cache

# Run sequentially: k=1 and k=5 first (small mixed datasets, c²=10201 rows)
# then k=0 last (huge raw frame pool, ~150k rows)
for k in 1 5 0; do
    log="logs/mia_k${k}.log"
    echo "$(date): Running k=${k}, logging to ${log}"
    python membership_inference.py --k "${k}" > "${log}" 2>&1
    echo "$(date): k=${k} done"
done

echo
echo "All k values complete."
echo "Merging results..."
python merge_results.py
echo "Generating Figure 2..."
python figure2_mia_robustness.py
echo "Generating Figure 3..."
python figure3_integer_vs_half.py