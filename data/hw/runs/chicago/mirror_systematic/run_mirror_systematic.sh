#!/bin/bash
# Mirror-understatement systematic (2026-11-19): reduce each run nominally and
# with kappa x r(t) (analyze_debias_kscale.py), t_max 6, same flags; the
# systematic is the shift in each asymmetry.  The 11 Nov nominal reduction is
# analysis/debias_w1.  Sequential and niced.
cd /home/hlamm/Desktop/QC/hadronic_tensor
C=data/hw/runs/chicago; M=$C/mirror_systematic
HTQ_WRAP=$C/analyze_debias_kscale.py HTQ_KAPPA_SCALE=$M/mirror_ratio_chicago_2026-11-11.npz HTQ_TMAXES=6 \
  $C/reduce_run.sh $C/2026-11-11 main_rzz_twirl mirror_sys_kscale ibm_chicago > /dev/null; echo "chicago scaled done $(date -u +%T)"
P=data/hw/runs/phoenix/2026-10-06
HTQ_TMAXES=6 $C/reduce_run.sh $P "" debias_w1_nominal ibm_phoenix --drop-sites 16 48 > /dev/null; echo "phoenix nominal done $(date -u +%T)"
HTQ_WRAP=$C/analyze_debias_kscale.py HTQ_KAPPA_SCALE=$M/mirror_ratio_phoenix_2026-10-06.npz HTQ_TMAXES=6 \
  $C/reduce_run.sh $P "" mirror_sys_kscale ibm_phoenix --drop-sites 16 48 > /dev/null; echo "phoenix scaled done $(date -u +%T)"
