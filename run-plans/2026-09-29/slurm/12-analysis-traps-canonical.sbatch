#!/bin/bash
# Item 1b, part 1: the mod-traps agent (traps-32c-24h-gen063) playing itself at 4096 sims under
# the CANONICAL rules. This is the cross-ruleset control column.
#
# It was trained on two-blast-four-bomb, so `analyze` refuses it under canonical rules
# without --allow-cross-ruleset. With the flag, meta.json records `cross_ruleset` and the
# document flags the column. Same trunk as 32c-24h-best, so the same estimate: ~27 h for 6,144.
#
# Resumable: finished chunks are skipped. If the time limit hits, resubmit unchanged.
#
#SBATCH --job-name=d52-an-traps-canon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR" && source run-plans/2026-09-29/slurm/lib.sh
banner

GAMES=${GAMES:-6144}

exec .venv/bin/python -m duel52.analysis \
    --dataset split-4096 \
    --agents netmcts:models/duel52-traps-32c-24h-gen063.d52nn@4096 \
    --games "$GAMES" --chunk 1024 --seed 1 \
    --threads 32 --eval-batch 32 --encoding-slots 21 \
    --card-value-games 0 --allow-cross-ruleset
