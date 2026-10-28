#!/bin/bash
# Retry a stage once before reporting failure to its dependent jobs.
run_stage() {
    local timing="$1"; shift
    local attempt status report delay="${AUTOBEHAVER_RETRY_DELAY:-30}"
    for attempt in 1 2; do
        report="$timing"
        (( attempt == 1 )) || report="${timing%.txt}_retry.txt"
        printf '[autobehaver] Stage attempt %s/2: %s\n' "$attempt" "$*" >&2
        if /usr/bin/time -v -o "$report" "$@"; then
            return 0
        else
            status=$?
        fi
        # Signals, cancellation and killed allocations are not command retries.
        if (( status >= 128 || attempt == 2 )); then
            printf '[autobehaver] Stage failed with exit code %s after attempt %s.\n' "$status" "$attempt" >&2
            return "$status"
        fi
        printf '[autobehaver] Retrying the stage in %s seconds; dependent jobs remain waiting.\n' "$delay" >&2
        sleep "$delay"
    done
}
