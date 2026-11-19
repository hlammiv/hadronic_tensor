#!/bin/bash
# One-command reduction of a Chicago (or Phoenix) drop to slices, W maps,
# boost asymmetry and Ward test, under the adopted rules: ancilla X bias
# removed per pub (analyze_debias.py), Gauss post-selection window 1, kappa
# floor 0.10, parity plots, maps at tmax 6 and 8.  Twirl-aware: a job record
# that carries `twirl` maps to its group's merged bits file (htq_bits_<group
# id>.npz, written once per group by `htq_hw fetch`), so a --pre-twirl run
# reduces exactly like an untwirled one.  Cards with no slices are skipped.
#
#   data/hw/runs/chicago/reduce_run.sh RUN SUB OUT [BACKEND] [MAPFLAGS...]
#
#   RUN      run folder, e.g. data/hw/runs/chicago/2026-11-14
#   SUB      jobs/raw subfolder, e.g. main_rzz_twirl
#   OUT      analysis folder to create under RUN/analysis, e.g. debias_w1
#   BACKEND  ibm_chicago (default) or ibm_phoenix
#   MAPFLAGS extra scripts/hw_w_tensor.py flags, e.g.
#            --sum-rule-min-coverage 0.9 --unmeasured-error ref
#
# Example, Declan's pre-twirled production pass:
#   data/hw/runs/chicago/reduce_run.sh data/hw/runs/chicago/<date> main_rzz_twirl debias_w1
#   data/hw/runs/chicago/reduce_run.sh data/hw/runs/chicago/<date> main_rzz_twirl debias_w1_unmeas \
#       ibm_chicago --sum-rule-min-coverage 0.9 --unmeasured-error ref
#
# Environment: HTQ_WRAP replaces the analyze wrapper (jackknife_run.sh passes
# analyze_debias_jk.py), HTQ_TMAXES the map depths (default "6 8").
#
# Check memory first (free -g); each analyze holds one card's bits files.
set -u
cd "$(dirname "$0")/../../../.."
PY=.venv/bin/python
RUN=${1:?run folder}; SUB=${2:?jobs/raw subfolder}; OUTTAG=${3:?analysis tag}
DEV=${4:-ibm_chicago}; shift $(( $# < 4 ? $# : 4 )); MAPFLAGS=("$@")
WRAP=${HTQ_WRAP:-data/hw/runs/chicago/2026-11-06/analysis/2026-11-07_debias/analyze_debias.py}
CARDS="prod_k1.26_s0.75_ns50 relA_k1.26_s0.75_ns50"
WINDOW=1; KAP=0.10; TMAXES=${HTQ_TMAXES:-6 8}
OUT=$RUN/analysis/$OUTTAG; mkdir -p $OUT/logs $OUT/slices
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $OUT/run.log; }
[ -f $WRAP ] || { log "missing $WRAP"; exit 1; }

files_for() {   # $1 card -> the bits files (one per job or per twirl group) holding that card's pubs
  $PY - "$RUN/jobs/$SUB/htq_job_*.json" "$RUN/raw/$SUB/htq_bits_{}.npz" "$1" <<'PYEOF'
import glob, json, sys
jobs, raw, card = sys.argv[1:4]
seen, out = set(), []
for p in sorted(glob.glob(jobs)):
    j = json.load(open(p))
    if j.get("cancelled"):
        continue
    names = [n.split("#")[0] for n in j["pub_names"]]           # drop a twirl instance suffix
    if not any(n.split(".", 1)[1].split(":")[0] == card for n in names if "." in n):
        continue
    fid = (j.get("twirl") or {}).get("group") or j["job_id"]
    if fid in seen:
        continue
    seen.add(fid)
    out.append(raw.format(fid))
print(" ".join(out))
PYEOF
}

log "reduce $RUN/$SUB -> $OUT ($DEV, window $WINDOW, kappa floor $KAP, map flags: ${MAPFLAGS[*]:-none})"
for card in $CARDS; do
  FILES=$(files_for $card)
  if [ -z "$FILES" ]; then log "$card: no bits files, skipped"; continue; fi
  MISSING=0; for f in $FILES; do [ -f $f ] || { log "$card: missing raw file $f"; MISSING=1; }; done
  [ $MISSING = 0 ] || { log "$card: raw files missing, skipped"; continue; }
  log "$card analyze start ($(echo $FILES | wc -w) files)"
  PYTHONPATH=. nice -n 10 $PY $WRAP --ns 50 --card $card analyze $FILES --by-card \
    --components 00 10 01 11 00d 10d --backend $DEV --postselect $WINDOW \
    --out "$OUT/slices/$card/slice_{comp}_t{t:.2f}.npz" > $OUT/logs/analyze_$card.log 2>&1 \
    || { log "$card analyze FAILED"; tail -n 5 $OUT/logs/analyze_$card.log; continue; }
  log "$card analyze done: $(ls $OUT/slices/$card 2>/dev/null | wc -l) slices"
done

map() {   # $1 tag (prod|relA)  $2 card  $3 tmax  [$4 extra]
  case $1 in prod) REF=prodR;; relA) REF=relA;; esac      # MPS reference file tags
  PYTHONPATH=. nice -n 10 $PY scripts/hw_w_tensor.py \
    --j0 data/w_${REF}_ns50_j0_k1.26_s0.75.npz --j1 data/w_${REF}_ns50_j1_k1.26_s0.75.npz \
    --slices "$OUT/slices/$2/slice_{comp}_t{t:.2f}.npz" \
    --ideal "htq_hw/cards/$2/ideal_{family}.npz" ${4:-} \
    --tmax $3 --parity --label $REF/$DEV --kap-min $KAP "${MAPFLAGS[@]}" \
    --out $OUT/maps/${1}_t$3 > $OUT/logs/map_${1}_t$3.log 2>&1 || log "map ${1}_t$3 FAILED"
}
mkdir -p $OUT/maps; MAPS=()
if ls $OUT/slices/prod_k1.26_s0.75_ns50/slice_00_t*.npz >/dev/null 2>&1; then
  for tmax in $TMAXES; do log "map prod_t$tmax"; map prod prod_k1.26_s0.75_ns50 $tmax; MAPS+=($OUT/maps/prod_t$tmax.npz); done
fi
if ls $OUT/slices/relA_k1.26_s0.75_ns50/slice_00_t*.npz >/dev/null 2>&1; then
  for tmax in $TMAXES; do log "map relA_t$tmax"; map relA relA_k1.26_s0.75_ns50 $tmax "--wing-surrogate htq_hw/refs/wing_surrogate_relA.npz"; MAPS+=($OUT/maps/relA_t$tmax.npz); done
fi
EXIST=(); for m in "${MAPS[@]}"; do [ -f $m ] && EXIST+=($m); done
if [ ${#EXIST[@]} -gt 0 ]; then
  PYTHONPATH=. nice -n 10 $PY scripts/w_map_symmetry.py "${EXIST[@]}" > $OUT/asymmetry.txt 2>&1 || log "symmetry FAILED"
  PYTHONPATH=. nice -n 10 $PY scripts/ward_scale.py "${EXIST[@]}" --form both --derive > $OUT/ward.txt 2>&1 || log "ward FAILED"
fi
log "DONE: $(ls $OUT/maps/*.npz 2>/dev/null | wc -l) maps; asymmetry in $OUT/asymmetry.txt"
