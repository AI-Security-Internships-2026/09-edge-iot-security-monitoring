#!/usr/bin/env bash
# check_stage5_completeness.sh
#
# Run from experiments/Current model/ (same cwd the sweep scripts expect).
# Reports, for Table A (MAD-k) and Table B (Byzantine-f), which
# (condition, seed) combos have a real, non-empty per_client_krum_scores
# CSV vs. which are missing -- and separately flags any log file that
# contains the sweep scripts' own "CSV not found" warning, since that
# means main.py finished but the CSV never made it into results-dir
# (a silently incomplete run that won't error the sweep loop).
#
# Usage: bash check_stage5_completeness.sh

set -uo pipefail   # NOT -e: we want to keep checking even if some files are missing

SEEDS=(42 123 456 789 2024)
MODEL="network"

TABLE_A_DIR="experiments/results/task5_mad_k_sweep"
K_VALUES=(2.0 2.5 3.0 3.5 4.0)

TABLE_B_DIR="experiments/results/task5_f_sweep"
F_VALUES=(1 2 3)

check_csv() {
    local dir="$1" fname="$2"
    local path="$dir/$fname"
    if [[ ! -f "$path" ]]; then
        echo "MISSING"
        return
    fi
    local nlines
    nlines=$(wc -l < "$path" 2>/dev/null || echo 0)
    if [[ "$nlines" -le 1 ]]; then
        echo "EMPTY_OR_HEADER_ONLY(${nlines}L)"
        return
    fi
    echo "OK(${nlines}L)"
}

report_table() {
    local table_name="$1" dir="$2"
    shift 2
    local -a conditions=("$@")

    echo "=================================================================="
    echo " $table_name  -- dir: $dir"
    echo "=================================================================="

    if [[ ! -d "$dir" ]]; then
        echo "  Directory does not exist yet -- nothing has been aggregated here."
        echo ""
        return
    fi

    local total=0 ok=0 missing=0 empty=0
    local -a missing_list=() empty_list=()

    for cond_pair in "${conditions[@]}"; do
        # cond_pair is "tag_plain|tag_cal"
        IFS='|' read -r tag_plain tag_cal <<< "$cond_pair"
        for tag in "$tag_plain" "$tag_cal"; do
            for seed in "${SEEDS[@]}"; do
                total=$((total+1))
                fname="per_client_krum_scores_${MODEL}_${tag}_seed${seed}.csv"
                status=$(check_csv "$dir" "$fname")
                case "$status" in
                    OK*) ok=$((ok+1)) ;;
                    MISSING) missing=$((missing+1)); missing_list+=("$fname") ;;
                    EMPTY*) empty=$((empty+1)); empty_list+=("$fname ($status)") ;;
                esac
            done
        done
    done

    echo "  Total expected: $total | OK: $ok | Missing: $missing | Empty/header-only: $empty"
    if [[ $missing -gt 0 ]]; then
        echo ""
        echo "  --- MISSING files ---"
        for f in "${missing_list[@]}"; do echo "    $f"; done
    fi
    if [[ $empty -gt 0 ]]; then
        echo ""
        echo "  --- EMPTY / header-only files (ran but produced no rows) ---"
        for f in "${empty_list[@]}"; do echo "    $f"; done
    fi

    echo ""
    echo "  --- Logs containing the sweep script's own 'CSV not found' warning ---"
    echo "      (main.py finished, but its output CSV never landed in $dir --"
    echo "      these runs LOOK complete in the log but have no data file to aggregate)"
    local warn_count=0
    if compgen -G "$dir/log_*.txt" > /dev/null; then
        for log in "$dir"/log_*.txt; do
            if grep -q "WARNING: expected per_client_krum_scores CSV not found" "$log"; then
                echo "    $(basename "$log")"
                warn_count=$((warn_count+1))
            fi
        done
    fi
    if [[ $warn_count -eq 0 ]]; then
        echo "    (none found)"
    fi

    echo ""
    echo "  --- Logs with signs of a crash (Traceback / Error / CUDA / killed) ---"
    local crash_count=0
    if compgen -G "$dir/log_*.txt" > /dev/null; then
        for log in "$dir"/log_*.txt; do
            if grep -qiE "Traceback \(most recent call last\)|CUDA error|Killed|MemoryError|OutOfMemoryError" "$log"; then
                echo "    $(basename "$log")"
                crash_count=$((crash_count+1))
            fi
        done
    fi
    if [[ $crash_count -eq 0 ]]; then
        echo "    (none found)"
    fi
    echo ""
}

TABLE_A_CONDITIONS=()
for k in "${K_VALUES[@]}"; do
    TABLE_A_CONDITIONS+=("task5_madk_plain_k${k}|task5_madk_calibrated_k${k}")
done

TABLE_B_CONDITIONS=()
for f in "${F_VALUES[@]}"; do
    TABLE_B_CONDITIONS+=("task5_fsweep_plain_f${f}|task5_fsweep_calibrated_f${f}")
done

report_table "TABLE A (MAD-k sweep)" "$TABLE_A_DIR" "${TABLE_A_CONDITIONS[@]}"
report_table "TABLE B (Byzantine-f sweep)" "$TABLE_B_DIR" "${TABLE_B_CONDITIONS[@]}"

echo "=================================================================="
echo " Once both tables show 0 Missing / 0 Empty / 0 warnings, run:"
echo "   python scripts/aggregate_task5_results.py --table A --results-dir $TABLE_A_DIR"
echo "   python scripts/aggregate_task5_results.py --table B --results-dir $TABLE_B_DIR"
echo "=================================================================="
