#!/bin/bash
# Build the engine and the Python extension ON A COMPUTE NODE, then run the test suites.
#
# Submit it twice: once from the main checkout onto a 32-core CPU node, and once from the
# `rnad` checkout onto a GPU node. It builds with `-C target-cpu=native`, which is worth 20–40%
# of self-play throughput (`configs/train-24h-128c.toml`). But a native binary can die with
# "illegal instruction" on an older CPU, which is why each checkout is built on the kind of
# node that will run it. If the CPU nodes are a mix of generations, set PORTABLE=1
# (`sbatch --export=ALL,PORTABLE=1 ...`) and pay the 20–40% rather than risk a crash.
#
# Needs no internet if the setup steps (`cargo fetch`, the pip installs) were done on the
# login node first.
#
#SBATCH --job-name=d52-build
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
