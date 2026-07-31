#!/usr/bin/env bash
#
# run_temporal_mia.sh
#
# End-to-end pipeline for the temporal MIA experiments under the LiRA-style
# scoring redesign (branch: temporal-membership-inference).
#
# Defaults are intentionally SMALL so the full sweep finishes fast and we
# can iterate. Once the figures look right, scale the knobs up via env
# vars (see "Compute knobs" below).
#
# Usage:
#   bash run_temporal_mia.sh                # full pipeline (MIA + accuracy + plots)
#   bash run_temporal_mia.sh mia            # MIA sweep only
#   bash run_temporal_mia.sh accuracy       # accuracy sweep only
#   bash run_temporal_mia.sh plots          # plots only (assumes CSVs exist)
#   bash run_temporal_mia.sh smoke          # 1-cell, 2-trial smoke test
#
# Compute knobs (override via env var):
#   N_TARGETS              MIA targets per cell             (default 10)
#   N_TRIALS               MIA MC trials per target         (default 5)
#   N_CLIP_CANDIDATES      Attack 2 same-class candidates   (default 50)
#   N_FRAME_CANDIDATES_A3  Attack 3 universe subsample      (default 2000)
#   K_VALUES               space-separated k list           (default "0 1 5")
#   SIGMAS                 space-separated sigma list       (default "0.01 0.05 0.10 0.50")
#   ACC_PARALLEL           accuracy jobs in parallel        (default 1)
#   SKIP_ACCURACY          if set, skip the downstream rerun (default unset)
#
# Examples:
#   N_TARGETS=50 N_TRIALS=10 bash run_temporal_mia.sh mia
#   K_VALUES="0 1" SIGMAS="0.01 0.10" bash run_temporal_mia.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Conda env wrapper. If $CONDA_DEFAULT_ENV already names the right env we
# call python directly; otherwise we route through `conda run -n <env>`.
# Override CONDA_ENV=foo to use a different env name.
: "${CONDA_ENV:=security}"
if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    PY=(python)
else
    PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
fi

# Compute knobs (small defaults for fast iteration)
: "${N_TARGETS:=10}"
: "${N_TRIALS:=5}"
: "${N_CLIP_CANDIDATES:=50}"
: "${N_FRAME_CANDIDATES_A3:=2000}"
: "${K_VALUES:=0 1 5}"
: "${SIGMAS:=0.01 0.05 0.10 0.50}"
: "${ACC_PARALLEL:=1}"

# Output dirs
RESULTS_DIR="./results"
ACC_DIR="./accuracy_results"
LOG_DIR="./logs"
IMG_DIR="./images"

mkdir -p "$RESULTS_DIR" "$ACC_DIR" "$LOG_DIR" "$IMG_DIR"

PHASE="${1:-all}"

echo "========================================================="
echo "Temporal MIA pipeline — phase: $PHASE"
echo "  N_TARGETS=$N_TARGETS  N_TRIALS=$N_TRIALS"
echo "  N_CLIP_CANDIDATES=$N_CLIP_CANDIDATES"
echo "  N_FRAME_CANDIDATES_A3=$N_FRAME_CANDIDATES_A3"
echo "  K_VALUES='$K_VALUES'  SIGMAS='$SIGMAS'"
echo "========================================================="

run_smoke() {
    echo
    echo "--- Smoke test: k=0, sigma=0.10, n_targets=2, n_trials=2 ---"
    "${PY[@]}" src/membership_inference.py \
        --k 0 --sigmas 0.10 \
        --n-targets 2 --n-trials 2 \
        --n-clip-candidates 10 \
        --n-frame-candidates-a3 200 \
        --results-dir ./_smoke_results
    echo "Smoke result CSV:"
    ls -la ./_smoke_results/
    rm -rf ./_smoke_results
    echo "--- Smoke OK ---"
}

run_mia() {
    echo
    echo "--- MIA sweep ---"
    for k in $K_VALUES; do
        log="$LOG_DIR/mia_k${k}.log"
        echo "Running k=$k -> $log"
        "${PY[@]}" src/membership_inference.py \
            --k "$k" \
            --sigmas $SIGMAS \
            --n-targets "$N_TARGETS" \
            --n-trials "$N_TRIALS" \
            --n-clip-candidates "$N_CLIP_CANDIDATES" \
            --n-frame-candidates-a3 "$N_FRAME_CANDIDATES_A3" \
            --results-dir "$RESULTS_DIR" \
            > "$log" 2>&1
        echo "  k=$k done."
    done
    "${PY[@]}" src/aggregate.py --mode attacks
}

run_accuracy() {
    if [[ -n "${SKIP_ACCURACY:-}" ]]; then
        echo "Skipping accuracy sweep (SKIP_ACCURACY set)."
        return
    fi
    echo
    echo "--- Accuracy sweep ---"
    # First cell builds the embedding cache (slow); subsequent cells reuse it
    FIRST_K=""
    FIRST_S=""
    for k in $K_VALUES; do for s in $SIGMAS; do FIRST_K="$k"; FIRST_S="$s"; break 2; done; done
    log="$LOG_DIR/acc_k${FIRST_K}_sigma${FIRST_S}.log"
    echo "First cell (builds embedding cache): k=$FIRST_K sigma=$FIRST_S -> $log"
    "${PY[@]}" src/main_video.py --k "$FIRST_K" --sigma "$FIRST_S" \
        --results-dir "$ACC_DIR" > "$log" 2>&1

    PIDS=()
    for k in $K_VALUES; do
        for s in $SIGMAS; do
            if [[ "$k" == "$FIRST_K" && "$s" == "$FIRST_S" ]]; then continue; fi
            while (( $(jobs -rp | wc -l) >= ACC_PARALLEL )); do sleep 5; done
            log="$LOG_DIR/acc_k${k}_sigma${s}.log"
            echo "Launching k=$k sigma=$s -> $log"
            "${PY[@]}" src/main_video.py --k "$k" --sigma "$s" \
                --results-dir "$ACC_DIR" > "$log" 2>&1 &
            PIDS+=($!)
        done
    done
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    echo "All accuracy cells done."
    "${PY[@]}" src/aggregate.py --mode accuracy
}

run_plots() {
    echo
    echo "--- Tables and figures ---"
    # Refresh the merged CSVs in case results-dir / accuracy_results changed
    # since the last aggregation, then build the privacy-utility frontier
    # (supplement Table 4 + Figure 6) from them.
    if compgen -G "$RESULTS_DIR/attack_k*_sigma*.csv" > /dev/null; then
        "${PY[@]}" src/aggregate.py --mode attacks
    fi
    if compgen -G "$ACC_DIR/acc_k*_sigma*.csv" > /dev/null; then
        "${PY[@]}" src/aggregate.py --mode accuracy
    fi
    if compgen -G "$RESULTS_DIR/attack_k*_sigma*.csv" > /dev/null && \
       compgen -G "$ACC_DIR/acc_k*_sigma*.csv" > /dev/null; then
        "${PY[@]}" src/aggregate.py --mode pareto
    fi
    # Closed-form numerics: main Table 1, main Figure 1, supplement Table 7.
    "${PY[@]}" src/theory_numerics.py
}

case "$PHASE" in
    smoke)    run_smoke ;;
    mia)      run_mia ;;
    accuracy) run_accuracy ;;
    plots)    run_plots ;;
    all)
        run_smoke
        run_mia
        run_accuracy
        run_plots
        ;;
    *) echo "Unknown phase: $PHASE"; exit 1 ;;
esac

echo
echo "Pipeline phase '$PHASE' complete."
