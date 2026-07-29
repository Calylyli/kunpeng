#!/usr/bin/env bash
set -euo pipefail

gcc -O3 bench_conv.c conv2d.c -o conv2d_test -lm -fopenmp

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-38}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

run_case() {
    if command -v numactl >/dev/null 2>&1; then
        numactl -N "${NUMA_NODE:-1}" ./conv2d_test "$@"
    else
        ./conv2d_test "$@"
    fi
}

run_case 4096 6144 39 39 1
run_case 6144 4096 41 41 1
run_case 4256 6390 55 55 1
run_case 6390 4256 81 81 1
