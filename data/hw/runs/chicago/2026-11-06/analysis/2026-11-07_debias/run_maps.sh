#!/bin/bash
# Maps, boost asymmetry and Ward test from the de-biased slices, with the
# phoenix 2026-10-28_postselect flags (kappa floor 0.10, parity plots) but
# no dropped sites: chicago's layout avoided the dead hardware by
# construction, so site drops are a data decision to be made from these maps.
# Usage: run_maps.sh <chicago_w0|chicago_w1|phoenix_w1> [more ...]
cd /home/hlamm/Desktop/QC/hadronic_tensor
PY=.venv/bin/python
BASE=data/hw/runs/chicago/2026-11-06/analysis/2026-11-07_debias
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $BASE/run.log; }
for D in "$@"; do
  case $D in chicago*) DEV=ibm_chicago;; phoenix*) DEV=ibm_phoenix;; esac
  OUT=$BASE/$D/kap0.10; mkdir -p $OUT/logs
  for tmax in 6 8; do
    log "$D maps prod_t$tmax"
    PYTHONPATH=. nice -n 10 $PY scripts/hw_w_tensor.py \
      --j0 data/w_prodR_ns50_j0_k1.26_s0.75.npz --j1 data/w_prodR_ns50_j1_k1.26_s0.75.npz \
      --slices "$BASE/$D/slices/prod_k1.26_s0.75_ns50/slice_{comp}_t{t:.2f}.npz" \
      --ideal 'htq_hw/cards/prod_k1.26_s0.75_ns50/ideal_{family}.npz' \
      --tmax $tmax --parity --label prodR/$DEV --kap-min 0.10 \
      --out $OUT/prod_t$tmax > $OUT/logs/map_prod_t$tmax.log 2>&1 || log "$D prod_t$tmax FAILED"
    log "$D maps relA_t$tmax"
    PYTHONPATH=. nice -n 10 $PY scripts/hw_w_tensor.py \
      --j0 data/w_relA_ns50_j0_k1.26_s0.75.npz --j1 data/w_relA_ns50_j1_k1.26_s0.75.npz \
      --slices "$BASE/$D/slices/relA_k1.26_s0.75_ns50/slice_{comp}_t{t:.2f}.npz" \
      --ideal 'htq_hw/cards/relA_k1.26_s0.75_ns50/ideal_{family}.npz' \
      --wing-surrogate htq_hw/refs/wing_surrogate_relA.npz \
      --tmax $tmax --parity --label relA/$DEV --kap-min 0.10 \
      --out $OUT/relA_t$tmax > $OUT/logs/map_relA_t$tmax.log 2>&1 || log "$D relA_t$tmax FAILED"
  done
  PYTHONPATH=. nice -n 10 $PY scripts/w_map_symmetry.py $OUT/prod_t6.npz $OUT/relA_t6.npz \
    $OUT/prod_t8.npz $OUT/relA_t8.npz > $OUT/asymmetry.txt 2>&1 || log "$D symmetry FAILED"
  PYTHONPATH=. nice -n 10 $PY scripts/ward_scale.py $OUT/prod_t6.npz $OUT/prod_t8.npz \
    $OUT/relA_t6.npz $OUT/relA_t8.npz --form both --derive > $OUT/ward.txt 2>&1 || log "$D ward FAILED"
  log "$D maps done"
done
