#!/usr/bin/env bash
set -euo pipefail

module use /home/HPC/HPCKit/latest/modulefiles
module load gcc/compiler12.3.1/gccmodule
module load gcc/kml25.2.0/kblas/multi

gcc -O3 bench_trsm.c trsm.c -o trsm_test -lm -lkblas -fopenmp

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-38}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

run_case() {
    if command -v numactl >/dev/null 2>&1; then
        numactl -N "${NUMA_NODE:-1}" ./trsm_test "$@"
    else
        ./trsm_test "$@"
    fi
}

run_case 512 19968 1
run_case 2432 17024 1
run_case 17024 512 1
