#!/usr/bin/env bash
#
# run_all.sh
#
# Launches the three MIA jobs in parallel (one per k value). Each job
# iterates over all sigma values internally, writing one CSV per
# (k, sigma) cell to ./results/.
#
# Logs go to ./logs/mia_k{k}.log so you can tail them while running.
#
# After all three finish:
#   python merge_results.py
#   python figure2_mia_robustness.py
#
# Usage:
#   bash run_all.sh                  # background, returns immediately
#   bash run_all.sh --wait           # foreground, waits for all to finish

set -e

mkdir -p logs results cache

K_VALUES=(0 1 5)

PIDS=()
for k in "${K_VALUES[@]}"; do
    log="logs/mia_k${k}.log"
    echo "Launching k=${k}, logging to ${log}"
    python membership_inference.py --k "${k}" \
        > "${log}" 2>&1 &
    PIDS+=($!)
done

echo
echo "Launched ${#PIDS[@]} jobs with PIDs: ${PIDS[*]}"
echo "Tail logs with:  tail -f logs/mia_k*.log"

if [[ "${1:-}" == "--wait" ]]; then
    echo "Waiting for all jobs to finish..."
    for pid in "${PIDS[@]}"; do
        wait "${pid}"
    done
    echo "All jobs finished."
    echo
    echo "Merging results..."
    python merge_results.py
    echo
    echo "Generating Figure 2..."
    python figure2_mia_robustness.py
fi