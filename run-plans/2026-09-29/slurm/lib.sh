# Sourced by every job in this directory, and not a job itself.
#
# What it does for each job: stop on the first error (`set -euo pipefail`), put the rustup
# toolchain on PATH, and (via `banner`) write the node, CPU model, AVX-512, commit and GPU at
# the top of the log, refusing to start if the build or venv is missing. The banner is the
# record of which hardware produced which numbers.
#
# A job starts in the directory it was submitted from, so submit from the repository root:
# the main checkout for the CPU jobs, and the `rnad` checkout for the GPU jobs. Each job
# begins with
#
#     cd "$SLURM_SUBMIT_DIR" && source run-plans/2026-09-29/slurm/lib.sh
#
# (`$0` inside a batch job is slurm's spooled copy of the script, so it cannot locate this file.)

set -euo pipefail

# Rust, if it was installed with rustup into the home directory.
if [ -f "$HOME/.cargo/env" ]; then
    . "$HOME/.cargo/env"
fi

# Uncomment and adjust if the cluster needs modules for Python >= 3.11 or CUDA:
# module load python/3.11
# module load cuda

banner() {
    echo "=================================================================="
    echo "job        ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
    echo "node       $(hostname)   cpus allotted ${SLURM_CPUS_PER_TASK:-?}   visible $(nproc)"
    echo "cpu        $(lscpu | sed -n 's/^Model name: *//p')"
    if grep -q avx512f /proc/cpuinfo 2>/dev/null; then echo "avx-512    yes"; else echo "avx-512    no"; fi
    echo "checkout   $(pwd)   $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
    echo "started    $(date -Is)"
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/gpu        /'
    fi
    echo "=================================================================="
    [ -x ./target/release/duel52 ] || { echo "no ./target/release/duel52 — run 00-build-cpu (or -gpu) first"; exit 1; }
    [ -x ./.venv/bin/python ] || { echo "no ./.venv — see the plan's setup section"; exit 1; }
}
