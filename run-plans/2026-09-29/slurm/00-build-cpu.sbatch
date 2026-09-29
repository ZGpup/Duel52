#!/bin/bash
# Build the engine and the Python extension ON A 32-CORE CPU NODE, then run the test suites.
# Submit from the MAIN checkout, and wait for "build OK" at the end of its log before
# submitting jobs 10–20.
#
# It builds with `-C target-cpu=native`, which is worth 20–40% of self-play throughput
# (`configs/train-24h-128c.toml`). But a native binary can die with "illegal instruction" on an
# older CPU, which is why the build runs on the kind of node that will run the jobs, not on the
# login node. If the CPU nodes are a mix of generations, set PORTABLE=1
# (`sbatch --export=ALL,PORTABLE=1 ...`) and pay the 20–40% rather than risk a crash.
# `00-build-gpu.sbatch` is the same thing for the `rnad` checkout on a GPU node.
#
# Needs no internet if the setup steps (`cargo fetch`, the pip installs) were done on the
# login node first.
#
#SBATCH --job-name=d52-build-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR" && source run-plans/2026-09-29/slurm/lib.sh

echo "build on $(hostname), $(date -Is)"
if [ "${PORTABLE:-0}" = "1" ]; then
    echo "PORTABLE=1: building for the generic target"
else
    export RUSTFLAGS="-C target-cpu=native"
fi

export CARGO_NET_OFFLINE=true
cargo build --release
.venv/bin/maturin develop --release

# The tests are the only check that this CPU reproduces the engine's results. "Same seed +
# same config → identical game" is a project invariant. `cargo test` is debug at -O1 on
# purpose, since the debug assertions are the state checks.
cargo test
.venv/bin/python -m pytest py/tests -q

./target/release/duel52 version
echo "build OK $(date -Is)"
