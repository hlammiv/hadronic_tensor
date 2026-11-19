#!/usr/bin/env bash
# Run with bash. --dry-run validates local inputs and prints the command only.
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
  printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
  exit 2
fi

HTQ_RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HTQ_REPO_DIR="$(git -C "$HTQ_RUN_DIR" rev-parse --show-toplevel)"
cd "$HTQ_REPO_DIR"
HTQ_PYTHON="$HTQ_REPO_DIR/.venv/bin/python"

# This reads local files only. The CLI checks the live layout before submission.
"$HTQ_PYTHON" - "$HTQ_RUN_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
r = json.loads((run / "acceptance_main_rzz_twirl.json").read_text())
if (r.get("kind") != "acceptance" or r.get("level") != "full"
        or r.get("status") not in ("PASS", "WARN") or r.get("failures") != []):
    raise SystemExit("A successful full acceptance record is required.")
expected = {
    "ns": 50,
    "basis": "rzz",
    "presets": ["prod-bridge", "relA-core", "prod-quarter", "relA-quarter"],
    "options": {"mirror_eps": 1e-8, "twirling_gates": False,
                "twirling_measure": True, "dd": "XY4", "pre_twirl": 4},
}
if any(r.get(k) != v for k, v in expected.items()):
    raise SystemExit("Acceptance settings differ from this submission command.")
if r["target"].get("name") != "ibm_chicago" or r["target"].get("is_real") is not True:
    raise SystemExit("Acceptance must use live ibm_chicago.")
e = r["embedding"]
if e.get("kind") != "ladder" or e.get("cuts") != {
    "max_readout_error": 0.05, "max_2q_error": 0.02,
    "min_core_t2_us": 100.0, "basis": "rzz",
}:
    raise SystemExit("Acceptance embedding or calibration cuts differ.")
changed = [p for p, digest in r["files"].items() if p.endswith(".py")
           and hashlib.sha256((Path("htq_hw") / p).read_bytes()).hexdigest() != digest]
if changed:
    raise SystemExit(f"Source changed since acceptance: {changed}")
if any((run / "jobs/main_rzz_twirl").glob("htq_job_*.json")):
    raise SystemExit("Job records already exist. Inspect the existing submission before retrying.")
PY

HTQ_SUBMIT=(
  "$HTQ_PYTHON" -u -m htq_hw
  --ns 50 --max-readout-error 0.05 --max-2q-error 0.02
  --min-t2 100 --twirl-gates off --mirror-eps 1e-8 --pre-twirl 4
  submit --target ibm_chicago --basis rzz --mode ladder
  --preset prod-bridge relA-core prod-quarter relA-quarter
  --rep-time 6.2e-5 --real --confirm ibm_chicago --no-strict-guard
  --require-layout "$HTQ_RUN_DIR/acceptance_main_rzz_twirl.json"
  --tag main-rzz-twirl --out "$HTQ_RUN_DIR/jobs/main_rzz_twirl"
)

if [[ "${1-}" == --dry-run ]]; then
  printf '%q ' "${HTQ_SUBMIT[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$HTQ_RUN_DIR/logs"
git rev-parse HEAD > "$HTQ_RUN_DIR/source_commit.txt"
git diff --binary -- htq_hw > "$HTQ_RUN_DIR/configuration_changes.patch"
"${HTQ_SUBMIT[@]}" 2>&1 | tee -a "$HTQ_RUN_DIR/logs/submit_main_rzz_twirl.log"
