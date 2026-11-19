#!/bin/bash
# Slices with the ancilla X bias removed per pub (analyze_debias.py), chicago at
# Gauss windows 0 and 1, and phoenix at window 1 so the rule is applied
# uniformly to both devices.  Everything else as 2026-11-07_baseline.
cd /home/hlamm/Desktop/QC/hadronic_tensor
PY=.venv/bin/python
BASE=data/hw/runs/chicago/2026-11-06/analysis/2026-11-07_debias
WRAP=$BASE/analyze_debias.py
CARDS="prod_k1.26_s0.75_ns50 relA_k1.26_s0.75_ns50"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $BASE/run.log; }
files_for() {   # $1 = jobs glob, $2 = raw template with {}, $3 = card
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
run() {   # $1 dev-tag $2 window $3 jobs-glob $4 raw-template $5 backend
  D=$BASE/$1_w$2; mkdir -p $D/logs $D/slices
  for card in $CARDS; do
    FILES=$(files_for "$3" "$4" $card)
    log "$1 w$2 $card analyze start ($(echo $FILES | wc -w) files)"
    PYTHONPATH=. nice -n 10 $PY $WRAP --ns 50 --card $card analyze $FILES --by-card \
      --components 00 10 01 11 00d 10d --backend $5 --postselect $2 \
      --out "$D/slices/$card/slice_{comp}_t{t:.2f}.npz" > $D/logs/analyze_$card.log 2>&1 \
      || { log "$1 w$2 $card analyze FAILED"; tail -n 5 $D/logs/analyze_$card.log; }
    log "$1 w$2 $card analyze done: $(ls $D/slices/$card 2>/dev/null | wc -l) slices"
  done
}
CJ="data/hw/runs/chicago/2026-11-06/jobs/main_rzz/htq_job_*.json"
CR="data/hw/runs/chicago/2026-11-06/raw/main_rzz/htq_bits_{}.npz"
PJ="data/hw/runs/phoenix/2026-10-06/jobs/htq_job_*.json"
PR="data/hw/runs/phoenix/2026-10-06/raw/htq_bits_{}.npz"
run chicago 0 "$CJ" "$CR" ibm_chicago
run chicago 1 "$CJ" "$CR" ibm_chicago
run phoenix 1 "$PJ" "$PR" ibm_phoenix
log "ANALYZE DONE"
