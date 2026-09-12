#!/bin/bash
# Sequential g*=1 noise trajectories for the tier-3 dry run, subprocess-
# isolated (rare Aer-MPS segfaults cost one trajectory; retry bumps seed).
NTRAJ=${1:-8}
for i in $(seq 0 $((NTRAJ - 1))); do
    for attempt in 0 1 2; do
        seed=$((1000 + i + 100000 * attempt))
        [ -f "data/tmp_t3traj_${seed}.npz" ] && break
        OMP_NUM_THREADS=4 PYTHONPATH=. timeout 2400 .venv/bin/python \
            scripts/hw_t3_dryrun.py onetraj "$seed" \
            >> logs/t3_traj.log 2>&1 && break
        echo "traj $i attempt $attempt (seed $seed) FAILED, retrying" \
            >> logs/t3_traj.log
    done
done
echo "trajloop done: $(ls data/tmp_t3traj_*.npz 2>/dev/null | wc -l) files" \
    >> logs/t3_traj.log
