#!/bin/bash
# Stronger Gauss post-selection as a software fix for the 11 Nov W00?
# Windows 2, 3, 4 impose the checks at sites 22-26, 21-27, 20-28 (window 4
# reaches the bad sites 27/28).  Component 00 only (the J1 components are
# XY-readout pubs and carry no Gauss checks).  Same de-bias wrapper, kappa
# floor 0.10 and t_max 6 as the adopted reduction; Phoenix with its dead
# sites 16 and 48 dropped, as in the talk.  The test is run on Phoenix too:
# a correction that moves the reference device's answer is not a fix.
# 2026-11-19.  Sequential and niced; one analyze holds one card's bits.
cd /home/hlamm/Desktop/QC/hadronic_tensor
PY=.venv/bin/python
WRAP=data/hw/runs/chicago/2026-11-06/analysis/2026-11-07_debias/analyze_debias.py
O=data/hw/runs/chicago/2026-11-11/analysis/postselect
files_for() {   # $1 jobs glob, $2 raw template, $3 card
  $PY - "$1" "$2" "$3" <<'PYEOF'
import glob, json, sys
jobs, raw, card = sys.argv[1:4]
seen, out = set(), []
for p in sorted(glob.glob(jobs)):
    j = json.load(open(p))
    if j.get("cancelled"):
        continue
    names = [n.split("#")[0] for n in j["pub_names"]]
    if not any(n.split(".", 1)[1].split(":")[0] == card for n in names if "." in n):
        continue
    fid = (j.get("twirl") or {}).get("group") or j["job_id"]
    if fid not in seen:
        seen.add(fid); out.append(raw.format(fid))
print(" ".join(out))
PYEOF
}
for dev in chi11 phoenix; do
  case $dev in
    chi11) J="data/hw/runs/chicago/2026-11-11/jobs/main_rzz_twirl/htq_job_*.json"; R="data/hw/runs/chicago/2026-11-11/raw/main_rzz_twirl/htq_bits_{}.npz"; BE=ibm_chicago; DROP="";;
    phoenix) J="data/hw/runs/phoenix/2026-10-06/jobs/htq_job_*.json"; R="data/hw/runs/phoenix/2026-10-06/raw/htq_bits_{}.npz"; BE=ibm_phoenix; DROP="--drop-sites 16 48";;
  esac
  for w in 2 3 4; do
    D=$O/${dev}_w$w; mkdir -p $D/logs
    for c in prod relA; do
      case $c in prod) REF=prodR; C=prod_k1.26_s0.75_ns50; WS="";; relA) REF=relA; C=relA_k1.26_s0.75_ns50; WS="--wing-surrogate htq_hw/refs/wing_surrogate_relA.npz";; esac
      PYTHONPATH=. nice -n 10 $PY $WRAP --ns 50 --card $C analyze $(files_for "$J" "$R" $C) --by-card \
        --components 00 --backend $BE --postselect $w \
        --out "$D/slices/$C/slice_{comp}_t{t:.2f}.npz" > $D/logs/analyze_$c.log 2>&1 || echo "analyze FAIL $dev w$w $c"
      PYTHONPATH=. nice -n 10 $PY scripts/hw_w_tensor.py --j0 data/w_${REF}_ns50_j0_k1.26_s0.75.npz --j1 data/w_${REF}_ns50_j1_k1.26_s0.75.npz \
        --slices "$D/slices/$C/slice_{comp}_t{t:.2f}.npz" --ideal "htq_hw/cards/$C/ideal_{family}.npz" $WS \
        --tmax 6 --parity --label $REF/$BE --kap-min 0.10 $DROP --out $D/maps/${c}_t6 > $D/logs/map_$c.log 2>&1 || echo "map FAIL $dev w$w $c"
    done
    PYTHONPATH=. $PY scripts/w_map_symmetry.py $D/maps/prod_t6.npz $D/maps/relA_t6.npz > $D/asymmetry.txt 2>&1
    echo "$dev w$w done $(date -u +%T)"
  done
done
