#!/bin/bash
# Item 2: the base (shared-deck) rules, 24 h from scratch on 32 cores.
# configs/train-24h-32c-base.toml has the reasoning.
#
# **Resubmitting.** The loop saves each game as it finishes, and on SIGTERM (which slurm sends
# at the time limit, or on preemption) it stops at the next safe point. Resubmitting continues
# INSIDE the interrupted generation, bit-for-bit. The script adds --resume by itself when the
# run directory already holds a run.
#
# ⚠️ `hours` counts from the start of EACH invocation, so a resumed run gets another full 24 h
# unless told otherwise. On a resubmit, pass what is left:
#     sbatch --export=ALL,HOURS=9.5 run-plans/2026-09-29/slurm/20-train-base.sbatch
# Elapsed so far is the sum of the per-generation `seconds` in runs/base-32c/log.jsonl
# (the plan has a one-liner for it).
#
# The time limit is 30 h, not 24. The loop checks the clock only between generations, and the
# mod-traps run under the same config logged ~25.4 h of generations. Past 24 h it just finishes
# the one in flight.
#
#SBATCH --job-name=d52-train-base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-06:00:00
#SBATCH --output=slurm-logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR" && source run-plans/2026-09-29/slurm/lib.sh
banner

CONFIG=configs/train-24h-32c-base.toml
RUN_DIR=runs/base-32c

.venv/bin/python -m duel52.train check --config "$CONFIG"

EXTRA=()
# The trainer refuses a non-empty run directory without --resume, so that is the test.
if [ -d "$RUN_DIR" ] && [ -n "$(ls -A "$RUN_DIR")" ]; then
    echo "resuming $RUN_DIR"
    EXTRA+=(--resume)
fi
if [ -n "${HOURS:-}" ]; then
    EXTRA+=(--hours "$HOURS")
fi

exec .venv/bin/python -m duel52.train run --config "$CONFIG" --run-dir "$RUN_DIR" "${EXTRA[@]}"
