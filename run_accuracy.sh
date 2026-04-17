#!/usr/bin/env bash
#
# run_accuracy.sh
#
# Launches 12 accuracy jobs (3 k values * 4 sigma values), one process
# per (k, sigma) cell. Logs to logs/acc_k{k}_sigma{sigma}.log.
#
# The first job to run will train and cache the no-obfuscation baseline
# (accuracy_results/baseline.csv); subsequent jobs reuse it.
#
# IMPORTANT: by default this launches all 12 jobs in parallel. On a
# single GPU that may be too much. Edit MAX_PARALLEL below to throttle,
# or run sequentially by setting it to 1.
#
# After all jobs finish:
#   python merge_accuracy.py
#   python figure4_pareto.py
#
# Usage:
#   bash run_accuracy.sh                  # launches in background
#   bash run_accuracy.sh --wait           # waits for all to finish, then plots

set -e

mkdir -p logs accuracy_results

K_VALUES=(0 1 5)
SIGMAS=(0.01 0.05 0.10 0.50)
MAX_PARALLEL=4   # adjust to fit your GPU memory; set to 1 for sequential

# Run baseline-bearing job first so the cache exists before others start
FIRST_K="${K_VALUES[0]}"
FIRST_SIGMA="${SIGMAS[0]}"
echo "Running baseline-bearing job first: k=${FIRST_K}, sigma=${FIRST_SIGMA}"
python main_video.py --k "${FIRST_K}" --sigma "${FIRST_SIGMA}" \
    > "logs/acc_k${FIRST_K}_sigma${FIRST_SIGMA}.log" 2>&1
echo "First job done."

# Now launch the rest with limited parallelism
PIDS=()
for k in "${K_VALUES[@]}"; do
    for sigma in "${SIGMAS[@]}"; do
        if [[ "${k}" == "${FIRST_K}" && "${sigma}" == "${FIRST_SIGMA}" ]]; then
            continue  # already done
        fi

        # throttle: wait if at parallel limit
        while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
            sleep 5
        done

        log="logs/acc_k${k}_sigma${sigma}.log"
        echo "Launching k=${k}, sigma=${sigma} -> ${log}"
        python main_video.py --k "${k}" --sigma "${sigma}" \
            > "${log}" 2>&1 &
        PIDS+=($!)
    done
done

echo
echo "Launched ${#PIDS[@]} background jobs, MAX_PARALLEL=${MAX_PARALLEL}"
echo "Tail logs with:  tail -f logs/acc_k*.log"

if [[ "${1:-}" == "--wait" ]]; then
    echo "Waiting for all jobs to finish..."
    for pid in "${PIDS[@]}"; do
        wait "${pid}"
    done
    echo "All jobs finished."
    echo
    echo "Merging accuracy results..."
    python merge_accuracy.py
    echo
    echo "Generating Figure 4 (requires merged_results.csv from MIA runs)..."
    python figure4_pareto.py
fi