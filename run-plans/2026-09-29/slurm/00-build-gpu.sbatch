#!/bin/bash
# Build the engine and the Python extension ON A GPU NODE, then run the test suites.
# Submit from the `rnad` CHECKOUT, and wait for "build OK" at the end of its log before
# submitting jobs 30 and 31.
#
# The same as 00-build-cpu.sbatch, except where it runs. A GPU node's CPU may be a different
# generation from the 32-core nodes, and the build uses `-C target-cpu=native`, so the rnad
# checkout is built where it will run. PORTABLE=1 works here too.
#
# Adjust --gres to the cluster's name for a 40 GB card, e.g. --gres=gpu:a100:1. Keep it the
# same as in 30- and 31-, so the build lands on the node type that runs them.
#
#SBATCH --job-name=d52-build-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
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

cargo test
.venv/bin/python -m pytest py/tests -q

# Confirms the CUDA build of PyTorch sees the card, which the R-NaD jobs need.
.venv/bin/python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

./target/release/duel52 version
echo "build OK $(date -Is)"
