#!/usr/bin/env bash
set -euo pipefail

gcc -O3 bench_zgemm.c zgemm.c -o zgemm_test -lm -fopenmp

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-38}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

run_case() {
    if command -v numactl >/dev/null 2>&1; then
        numactl -N "${NUMA_NODE:-1}" ./zgemm_test "$@"
    else
        ./zgemm_test "$@"
    fi
}

run_case 7427 7427 256 1
run_case 14848 14848 256 1
run_case 37360 8192 512 1
