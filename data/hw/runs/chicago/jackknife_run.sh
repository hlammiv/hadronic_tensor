#!/bin/bash
# Delete-a-group jackknife over twirl draws for a pre-twirled drop (adopted
# 2026-11-19): G reductions with one group of instances left out
# (analyze_debias_jk.py through reduce_run.sh, maps at t_max 6), then the
# table of pooled asymmetry, shot and window errors and the jackknife error
# err^2 = (G-1)/G sum_g (A_-g - mean)^2.  Run the pooled reduction first.
#
#   data/hw/runs/chicago/jackknife_run.sh RUN SUB BASE [G] [BACKEND] [MAPFLAGS...]
#
#   BASE  the pooled reduction's tag under RUN/analysis (e.g. debias_w1);
#         the jackknife goes to RUN/analysis/BASE_jk/g0..g{G-1}
#   G     groups (default 8; capped at the instance count by the wrapper)
#
# Each group costs one full reduction (about 8 min and 3 GB per card on the
# 11 Nov drop); G runs sequentially.  Check memory first.
set -u
cd "$(dirname "$0")/../../../.."
RUN=${1:?run folder}; SUB=${2:?jobs/raw subfolder}; BASE=${3:?pooled analysis tag}
G=${4:-8}; DEV=${5:-ibm_chicago}; shift $(( $# < 5 ? $# : 5 ))
HERE=data/hw/runs/chicago
for g in $(seq 0 $((G - 1))); do
  HTQ_WRAP=$HERE/analyze_debias_jk.py HTQ_JK_GROUP=$g HTQ_JK_GROUPS=$G HTQ_TMAXES=6 \
    $HERE/reduce_run.sh "$RUN" "$SUB" "${BASE}_jk/g$g" "$DEV" "$@" > /dev/null
  echo "jackknife group $g of $G done $(date -u +%T)"
done
.venv/bin/python $HERE/jackknife_table.py "$RUN/analysis/$BASE" "$RUN/analysis/${BASE}_jk" $G \
  | tee "$RUN/analysis/${BASE}_jk/jackknife_table.txt"
