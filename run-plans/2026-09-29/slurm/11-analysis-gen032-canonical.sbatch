#!/bin/bash
# Item 1a: lane-gen032 playing itself at 4096 sims, canonical rules.
#
# lane 128x3 has half the trunk of the 128x6 nets, so it runs about twice as fast. That is
# ~0.13 games/sec, and 12,288 games is ~26 h in twelve ~2.2 h chunks. It shares deals with the
# other split-4096 columns up to seed 3,072, since every job starts at seed 1.
#
# Resumable: finished chunks are skipped. If the time limit hits, resubmit unchanged.
#
#SBATCH --job-name=d52-an-gen032-canon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR" && source run-plans/2026-09-29/slurm/lib.sh
banner

GAMES=${GAMES:-12288}

exec .venv/bin/python -m duel52.analysis \
    --dataset split-4096 \
    --agents netmcts:models/duel52-split-lane-gen032.d52nn@4096 \
    --games "$GAMES" --chunk 1024 --seed 1 \
    --threads 32 --eval-batch 32 --encoding-slots 21 \
    --card-value-games 0
