#!/bin/bash
# Item 3: R-NaD on the canonical (split-deck) rules, 24 h on one GPU with 40 GB.
# SUBMIT FROM THE `rnad` CHECKOUT: configs/rnad-24h.toml exists only on that branch.
#
# It runs `bench` first (~1 min), which times real steps on this GPU and prints how many
# regularisation updates 24 h buys. The config is sized for 8-15 of them. If the bench line
# says fewer than 6 or more than 20, stop the job, copy the entropy_schedule_size and
# target_network_avg it suggests into the config, and resubmit. The config header explains.
#
# **Resubmitting** is safe and needs nothing else. R-NaD's `hours` is cumulative across
# resumes (unlike the AlphaZero loop's), SIGTERM finishes the current step and saves
# state.pt, and the script adds --resume when state.pt exists.
#
# Adjust --gres to the cluster's name for a 40 GB card, e.g. --gres=gpu:a100:1.
#
#SBATCH --job-name=d52-rnad-canon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-03:00:00
#SBATCH --output=slurm-logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR" && source run-plans/2026-09-29/slurm/lib.sh
banner

CONFIG=configs/rnad-24h.toml
RUN_DIR=runs/rnad-24h-canonical

.venv/bin/python -m duel52.rnad check --config "$CONFIG"

EXTRA=()
if [ -f "$RUN_DIR/state.pt" ]; then
    echo "resuming $RUN_DIR"
    EXTRA+=(--resume)
else
    .venv/bin/python -m duel52.rnad bench --config "$CONFIG" --steps 5
fi

exec .venv/bin/python -m duel52.rnad run --config "$CONFIG" --run-dir "$RUN_DIR" "${EXTRA[@]}"
