#!/usr/bin/env bash
# Run the full sampler benchmark tour: edits calibration.sampler in
# params.yml for each sampler and launches the pipeline sequentially.
#
# Usage:
#   cd ~/mcmc-bench
#   export NXF_VER=25.04.6        # pin the Nextflow version (see below)
#   chmod +x run_all_samplers.sh  # only needed once
#   ./run_all_samplers.sh
#
# Logs go to benchmark_logs/<sampler>.log; params.yml is restored on exit.
set -u

SAMPLERS=(rwmcmc emcee pymc_slice pymc_smc dynesty)
PARAMS=params.yml
LOGDIR=benchmark_logs
mkdir -p "$LOGDIR"

# Safety rails ------------------------------------------------------------
if [ ! -f "$PARAMS" ]; then
    echo "ERROR: $PARAMS not found — run this from the repo root."; exit 1
fi
if [ -z "${NXF_VER:-}" ]; then
    echo "WARNING: NXF_VER is not exported; using whatever nextflow resolves to."
    read -r -p "Continue anyway? [y/N] " ans
    [ "$ans" = "y" ] || exit 1
fi

# Preserve the original params.yml no matter how we exit
cp "$PARAMS" "${PARAMS}.tour_backup"
trap 'cp "${PARAMS}.tour_backup" "$PARAMS"; echo "params.yml restored from backup."' EXIT

# Tour --------------------------------------------------------------------
declare -A STATUS DURATION

for s in "${SAMPLERS[@]}"; do
    sed -i -E "s/^([[:space:]]*sampler:)[[:space:]]*[[:alnum:]_]+/\1 ${s}/" "$PARAMS"

    # Verify the edit actually took before burning compute
    if ! grep -qE "^[[:space:]]*sampler:[[:space:]]*${s}\b" "$PARAMS"; then
        echo "[$s] ERROR: failed to set sampler in $PARAMS, skipping."
        STATUS[$s]="EDIT_FAIL"; continue
    fi

    echo "=== [$s] started $(date +%H:%M:%S), log: $LOGDIR/${s}.log ==="
    t0=$SECONDS
    if nextflow run main.nf -params-file "$PARAMS" --config_file "$PARAMS" \
            > "$LOGDIR/${s}.log" 2>&1; then
        STATUS[$s]="OK"
    else
        STATUS[$s]="FAIL"
    fi
    DURATION[$s]=$(( SECONDS - t0 ))
    echo "=== [$s] ${STATUS[$s]} in ${DURATION[$s]}s ==="
done

# Summary -----------------------------------------------------------------
echo
echo "================ TOUR SUMMARY ================"
for s in "${SAMPLERS[@]}"; do
    printf "%-12s %-10s %ss\n" "$s" "${STATUS[$s]:-?}" "${DURATION[$s]:-–}"
done
echo
echo "Newest bundles:"
ls -t outputs/ | head -n "${#SAMPLERS[@]}"