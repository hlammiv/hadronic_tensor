#!/bin/bash
# First reduction of the 6 November ibm_chicago bits (rzz basis): slices per
# card at Gauss windows 0 (unfiltered) and 1, same reducer and flags as the
# phoenix 2026-10-28_postselect pass, except the backend label and the file
# layout (jobs/<sub>/, raw/<sub>/).  Slice names use t:.2f because the
# quarter-step presets put slices at t = 0.25, 0.75, ...
cd /home/hlamm/Desktop/QC/hadronic_tensor
PY=.venv/bin/python
RUN=data/hw/runs/chicago/2026-11-06
BASE=$RUN/analysis/2026-11-07_baseline
CARDS="prod_k1.26_s0.75_ns50 relA_k1.26_s0.75_ns50"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $BASE/run.log; }
# the completed main_rzz jobs that carry each card's pubs
files_for() {
  $PY - "$1" <<'PYEOF'
import json, glob, sys
card = sys.argv[1]
out = []
for p in sorted(glob.glob("data/hw/runs/chicago/2026-11-06/jobs/main_rzz/htq_job_*.json")):
    j = json.load(open(p))
    if j.get("cancelled"):
        continue
    if any(n.split(".", 1)[1].split(":")[0] == card for n in j["pub_names"] if "." in n):
        out.append(f"data/hw/runs/chicago/2026-11-06/raw/main_rzz/htq_bits_{j['job_id']}.npz")
print(" ".join(out))
PYEOF
}
for W in 0 1; do
  D=$BASE/w$W; mkdir -p $D/logs $D/slices
  for card in $CARDS; do
    FILES=$(files_for $card)
    log "w$W $card analyze start ($(echo $FILES | wc -w) files)"
    PYTHONPATH=. nice -n 10 $PY -m htq_hw --ns 50 --card $card analyze $FILES --by-card \
      --components 00 10 01 11 00d 10d --backend ibm_chicago --postselect $W \
      --out "$D/slices/$card/slice_{comp}_t{t:.2f}.npz" > $D/logs/analyze_$card.log 2>&1 \
      || { log "w$W $card analyze FAILED"; tail -n 5 $D/logs/analyze_$card.log; }
    log "w$W $card analyze done: $(ls $D/slices/$card 2>/dev/null | wc -l) slices"
  done
done
log "ANALYZE DONE"
