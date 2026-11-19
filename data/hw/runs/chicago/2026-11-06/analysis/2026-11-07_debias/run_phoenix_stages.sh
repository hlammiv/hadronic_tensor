#!/bin/bash
# The talk's mitigation ladder (slide 8b) under the de-bias rule: phoenix at
# Gauss windows 0 and 2 (window 1 already in phoenix_w1/) and the "no
# mitigation" stage (kappa = beta = 1, no wing anchor, no post-selection),
# each mapped with the talk's mask (kappa floor 0.10, dead sites 16 and 48
# dropped) at tmax 6, plus the boost asymmetry.  Everything else as
# run_analyze.sh / run_maps.sh.  2026-11-09.
cd /home/hlamm/Desktop/QC/hadronic_tensor
PY=.venv/bin/python
BASE=data/hw/runs/chicago/2026-11-06/analysis/2026-11-07_debias
CARDS="prod_k1.26_s0.75_ns50 relA_k1.26_s0.75_ns50"
PJ="data/hw/runs/phoenix/2026-10-06/jobs/htq_job_*.json"
PR="data/hw/runs/phoenix/2026-10-06/raw/htq_bits_{}.npz"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $BASE/run.log; }
files_for() {
  $PY - "$1" "$2" "$3" <<'PYEOF'
import json, glob, sys
jobs, raw, card = sys.argv[1:4]
out = []
for p in sorted(glob.glob(jobs)):
    j = json.load(open(p))
    if j.get("cancelled"):
        continue
    if any(n.split(".", 1)[1].split(":")[0] == card for n in j["pub_names"] if "." in n):
        out.append(raw.format(j["job_id"]))
print(" ".join(out))
PYEOF
}
analyze() {   # $1 dir-tag  $2 wrapper  $3 postselect window
  D=$BASE/$1; mkdir -p $D/logs $D/slices
  for card in $CARDS; do
    FILES=$(files_for "$PJ" "$PR" $card)
    log "$1 $card analyze start ($(echo $FILES | wc -w) files)"
    PYTHONPATH=. nice -n 10 $PY $BASE/$2 --ns 50 --card $card analyze $FILES --by-card \
      --components 00 10 01 11 00d 10d --backend ibm_phoenix --postselect $3 \
      --out "$D/slices/$card/slice_{comp}_t{t:.2f}.npz" > $D/logs/analyze_$card.log 2>&1 \
      || { log "$1 $card analyze FAILED"; tail -n 5 $D/logs/analyze_$card.log; }
    log "$1 $card analyze done: $(ls $D/slices/$card 2>/dev/null | wc -l) slices"
  done
}
maps() {   # $1 dir-tag
  D=$BASE/$1; OUT=$D/kap0.10_drop; mkdir -p $OUT/logs
  log "$1 maps prod_t6"
  PYTHONPATH=. nice -n 10 $PY scripts/hw_w_tensor.py \
    --j0 data/w_prodR_ns50_j0_k1.26_s0.75.npz --j1 data/w_prodR_ns50_j1_k1.26_s0.75.npz \
    --slices "$D/slices/prod_k1.26_s0.75_ns50/slice_{comp}_t{t:.2f}.npz" \
    --ideal 'htq_hw/cards/prod_k1.26_s0.75_ns50/ideal_{family}.npz' \
    --tmax 6 --parity --label prodR/ibm_phoenix --kap-min 0.10 --drop-sites 16 48 \
    --out $OUT/prod_t6 > $OUT/logs/map_prod_t6.log 2>&1 || log "$1 prod_t6 FAILED"
  log "$1 maps relA_t6"
  PYTHONPATH=. nice -n 10 $PY scripts/hw_w_tensor.py \
    --j0 data/w_relA_ns50_j0_k1.26_s0.75.npz --j1 data/w_relA_ns50_j1_k1.26_s0.75.npz \
    --slices "$D/slices/relA_k1.26_s0.75_ns50/slice_{comp}_t{t:.2f}.npz" \
    --ideal 'htq_hw/cards/relA_k1.26_s0.75_ns50/ideal_{family}.npz' \
    --wing-surrogate htq_hw/refs/wing_surrogate_relA.npz \
    --tmax 6 --parity --label relA/ibm_phoenix --kap-min 0.10 --drop-sites 16 48 \
    --out $OUT/relA_t6 > $OUT/logs/map_relA_t6.log 2>&1 || log "$1 relA_t6 FAILED"
  PYTHONPATH=. nice -n 10 $PY scripts/w_map_symmetry.py $OUT/prod_t6.npz $OUT/relA_t6.npz \
    > $OUT/asymmetry.txt 2>&1 || log "$1 symmetry FAILED"
  log "$1 maps done"
}
analyze phoenix_w0 analyze_debias.py 0
analyze phoenix_w2 analyze_debias.py 2
analyze phoenix_unmit analyze_debias_unmitigated.py 0
maps phoenix_w0
maps phoenix_w2
maps phoenix_unmit
log "PHOENIX STAGES DONE"
