#!/bin/bash
# Queue everything in the 2026-09-29 plan: two build jobs, then eight run jobs that wait on
# their build finishing cleanly (`afterok`, so a failed test suite stops everything behind it).
#
#   MAIN=/path/to/Duel52 RNAD=/path/to/Duel52-rnad \
#   ACCOUNT=<slurm account> CPU_PARTITION=<32-core partition> GPU_PARTITION=<gpu partition> \
#       bash run-plans/2026-09-29/slurm/submit.sh
#
# Any of ACCOUNT / CPU_PARTITION / GPU_PARTITION may be left unset if the cluster has
# defaults. Each job can also be submitted by hand; see the plan.

set -euo pipefail

MAIN=${MAIN:?set MAIN to the main-branch checkout}
RNAD=${RNAD:?set RNAD to the rnad-branch checkout}
S=run-plans/2026-09-29/slurm

common=()
[ -n "${ACCOUNT:-}" ] && common+=(--account "$ACCOUNT")
cpu=("${common[@]}")
[ -n "${CPU_PARTITION:-}" ] && cpu+=(--partition "$CPU_PARTITION")
gpu=("${common[@]}")
[ -n "${GPU_PARTITION:-}" ] && gpu+=(--partition "$GPU_PARTITION")

mkdir -p "$MAIN/slurm-logs" "$RNAD/slurm-logs"

cd "$MAIN"
build_cpu=$(sbatch --parsable "${cpu[@]}" "$S/00-build.sbatch")
echo "cpu build        $build_cpu"
for job in 10-analysis-best-canonical 11-analysis-gen032-canonical 12-analysis-traps-canonical \
           13-analysis-best-traps 14-analysis-traps-traps 20-train-base; do
    id=$(sbatch --parsable "${cpu[@]}" --dependency=afterok:"$build_cpu" "$S/$job.sbatch")
    echo "$job  $id"
done

cd "$RNAD"
# The GPU build: same script, on a GPU node, with fewer cores.
build_gpu=$(sbatch --parsable "${gpu[@]}" --gres=gpu:1 --cpus-per-task=16 "$S/00-build.sbatch")
echo "gpu build        $build_gpu"
for job in 30-rnad-canonical 31-rnad-base; do
    id=$(sbatch --parsable "${gpu[@]}" --dependency=afterok:"$build_gpu" "$S/$job.sbatch")
    echo "$job  $id"
done

echo
echo "squeue -u \$USER to watch. Logs land in slurm-logs/ of each checkout."
