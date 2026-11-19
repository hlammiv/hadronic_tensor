#!/bin/bash
cd /home/hlamm/Desktop/QC/hadronic_tensor
S=$1; PY=.venv/bin/python
B=data/hw/runs/chicago/2026-11-06/analysis/2026-11-07_debias
declare -A SL=( [phoenix]=$B/phoenix_w1/slices [chi6]=$B/chicago_w1/slices [chi11]=data/hw/runs/chicago/2026-11-11/analysis/debias_w1/slices )
declare -A BASE=( [phoenix]="16 48" [chi6]="" [chi11]="" )
declare -A DROPS=( [none]="" [p1]="23 25" [p3]="21 27" [p4]="20 28" [odd]="21 23 25 27" [odd4]="20 21 23 25 27 28" [all4]="20 21 22 23 25 26 27 28" )
for dev in phoenix chi6 chi11; do for card in prod relA; do for d in none p1 p3 p4 odd odd4 all4; do
  case $card in prod) REF=prodR; C=prod_k1.26_s0.75_ns50; W="";; relA) REF=relA; C=relA_k1.26_s0.75_ns50; W="--wing-surrogate htq_hw/refs/wing_surrogate_relA.npz";; esac
  DS="${BASE[$dev]} ${DROPS[$d]}"; DS=$(echo $DS)
  OUT=$S/maps/${dev}_${card}_$d
  PYTHONPATH=. nice -n 10 $PY scripts/hw_w_tensor.py --j0 data/w_${REF}_ns50_j0_k1.26_s0.75.npz --j1 data/w_${REF}_ns50_j1_k1.26_s0.75.npz \
    --slices "${SL[$dev]}/$C/slice_{comp}_t{t:.2f}.npz" --ideal "htq_hw/cards/$C/ideal_{family}.npz" $W \
    --tmax 6 --parity --label x --kap-min 0.10 ${DS:+--drop-sites $DS} --out $OUT > $S/logs/${dev}_${card}_$d.log 2>&1 || echo "FAIL $OUT"
  PYTHONPATH=. $PY scripts/w_map_symmetry.py $OUT.npz 2>/dev/null | awk -v k="$dev $card $d" '/^  00:/{f=1} /^  10:/{f=0} f && /\[\+0.00, 2.50\]/{print k, $4, $5, $6, $7}'
done; done; done
