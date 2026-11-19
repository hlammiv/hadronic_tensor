#!/bin/bash
cd /home/hlamm/Desktop/QC/hadronic_tensor
R=data/hw/runs/chicago/2026-11-11; O=$R/analysis/jackknife; PY=.venv/bin/python
files_for() { $PY - "$R/jobs/main_rzz_twirl/htq_job_*.json" "$R/raw/main_rzz_twirl/htq_bits_{}.npz" "$1" <<'PYEOF'
import glob, json, sys
jobs, raw, card = sys.argv[1:4]
seen, out = set(), []
for p in sorted(glob.glob(jobs)):
    j = json.load(open(p))
    if j.get("cancelled"): continue
    names = [n.split("#")[0] for n in j["pub_names"]]
    if not any(n.split(".", 1)[1].split(":")[0] == card for n in names if "." in n): continue
    fid = (j.get("twirl") or {}).get("group") or j["job_id"]
    if fid in seen: continue
    seen.add(fid); out.append(raw.format(fid))
print(" ".join(out))
PYEOF
}
for i in 0 1 2 3; do
  mkdir -p $O/d$i/logs $O/d$i/maps
  for c in prod relA; do
    case $c in prod) REF=prodR; C=prod_k1.26_s0.75_ns50; W="";; relA) REF=relA; C=relA_k1.26_s0.75_ns50; W="--wing-surrogate htq_hw/refs/wing_surrogate_relA.npz";; esac
    HTQ_DROP=$i PYTHONPATH=. nice -n 10 $PY $R/analysis/analyze_debias_drop.py --ns 50 --card $C analyze $(files_for $C) --by-card \
      --components 00 10 01 11 --backend ibm_chicago --postselect 1 --out "$O/d$i/slices/$C/slice_{comp}_t{t:.2f}.npz" > $O/d$i/logs/analyze_$c.log 2>&1 || echo "analyze FAIL i$i $c"
    PYTHONPATH=. nice -n 10 $PY scripts/hw_w_tensor.py --j0 data/w_${REF}_ns50_j0_k1.26_s0.75.npz --j1 data/w_${REF}_ns50_j1_k1.26_s0.75.npz \
      --slices "$O/d$i/slices/$C/slice_{comp}_t{t:.2f}.npz" --ideal "htq_hw/cards/$C/ideal_{family}.npz" $W \
      --tmax 6 --parity --label $REF/ibm_chicago --kap-min 0.10 --out $O/d$i/maps/${c}_t6 > $O/d$i/logs/map_$c.log 2>&1 || echo "map FAIL i$i $c"
  done
  PYTHONPATH=. $PY scripts/w_map_symmetry.py $O/d$i/maps/prod_t6.npz $O/d$i/maps/relA_t6.npz > $O/d$i/asymmetry.txt 2>&1
  echo "drop $i done $(date -u +%T)"
done
