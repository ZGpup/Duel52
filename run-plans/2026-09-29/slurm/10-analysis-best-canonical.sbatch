#!/bin/bash
# Item 1a/1b: 32c-24h-best playing itself at 4096 sims, canonical rules. 1a and 1b share this corpus.
#
# Estimated ~0.064 games/sec on 32 cores, so 6,144 games is ~27 h. Six chunks of 1,024, each
# ~4.4 h, because every game of a chunk is in flight at once and they finish together.
# See the run plan for the derivation.
#
# Resumable: a finished chunk has a meta.json and is skipped on a re-run, and a killed chunk
# has none and is replayed. If the time limit hits, resubmit this file unchanged. For MORE
# games afterwards, raise GAMES and resubmit, and only the new chunks are played.
#
#SBATCH --job-name=d52-an-best-canon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR" && source run-plans/2026-09-29/slurm/lib.sh
banner

GAMES=${GAMES:-6144}

# --chunk 1024 = 32 threads x 32 in flight, so the eval batch is never clamped (CLAUDE.md,
# "Size the chunk so the batch is not clamped"). --card-value-games 0: the document is rendered
# on the laptop at the end, so this job renders nothing expensive.
exec .venv/bin/python -m duel52.analysis \
    --dataset split-4096 \
    --agents netmcts:models/duel52-32c-24h-best.d52nn@4096 \
    --games "$GAMES" --chunk 1024 --seed 1 \
    --threads 32 --eval-batch 32 --encoding-slots 21 \
    --card-value-games 0
