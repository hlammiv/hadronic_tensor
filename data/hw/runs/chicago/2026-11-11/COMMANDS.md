# Chicago production run with four pre-twirled instances

The preparation and submission commands below are retained for provenance.
Full acceptance completed on November 11 with no failures; its only warning
concerned the then-uncommitted CLI/test changes. All 508 job records are now
present. Use the collection section below for this submitted campaign.

Acceptance ran at `607c3fe`, which includes Hank's requested commit `247fe7b`.
The additional upstream changes affect analysis-map scripts. The existing
local `--require-layout` guard and incomplete-submission check are retained.
The current Python sources match the acceptance record's hashes.

Site 24 is on q53 (T2 164 us), the Hadamard ancilla is on q63 (315 us), and
the minimum core T2 is 134.72 us on q54. The six-qubit core mapping is unchanged
from November 10, but 13 wire assignments elsewhere in the full layout differ.

The four presets contain exactly 817 original PUBs and 46,380,000 total shots,
matching the November 6 main-RZZ records. Four twirled instances produce 508
jobs (127 original groups), without increasing the total shot count. The
`relA-core` preset includes its t = 6.5–8 extension in these totals.

Run the blocks in order in the same bash or zsh session. Change the run date
if executing on another day. Stop if acceptance or submission fails. If a
submission is interrupted, inspect the saved job records before retrying;
rerunning the whole submission could duplicate jobs.

## Setup and full acceptance

```sh
cd /Users/declanmillar/Projects/fermilab/hadronic-tensor/hadronic_tensor
set -o pipefail
HTQ_RUN=data/hw/runs/chicago/2026-11-11
HTQ_GLOBAL=(--ns 50 --max-readout-error 0.05 --max-2q-error 0.02
            --min-t2 100 --twirl-gates off --mirror-eps 1e-8 --pre-twirl 4)
HTQ_EXPERIMENT=(--target ibm_chicago --basis rzz --mode ladder
                --preset prod-bridge relA-core prod-quarter relA-quarter
                --rep-time 6.2e-5)
mkdir -p "$HTQ_RUN/logs"
git rev-parse HEAD > "$HTQ_RUN/source_commit.txt"
git diff --binary -- htq_hw > "$HTQ_RUN/configuration_changes.patch"

.venv/bin/python -m htq_hw "${HTQ_GLOBAL[@]}" acceptance \
  "${HTQ_EXPERIMENT[@]}" --level full --require-real \
  --record "$HTQ_RUN/acceptance_main_rzz_twirl.json" \
  2>&1 | tee "$HTQ_RUN/logs/acceptance_main_rzz_twirl.log"
```

Before submitting, send Hank the selected site-24 and Hadamard-ancilla qubits
and the core T2 values. The following prints them and saves the embedding.
If the T2 cut fails, send the acceptance failure; do not lower the floor.

```sh
.venv/bin/python - "$HTQ_RUN" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
r = json.loads((run / "acceptance_main_rzz_twirl.json").read_text())
if r.get("status") not in ("PASS", "WARN") or r.get("failures") != []:
    raise SystemExit("Acceptance failed; send Hank the failure and do not submit.")
e = r["embedding"]
core = e["core_t2_us"]
print(f"Site 24: q{e['layout'][48]}; Hadamard ancilla: q{e['layout'][100]}; "
      f"minimum core T2: {core['min']:.1f} us.")
print("Core T2: " + ", ".join(f"q{q}: {v:.1f} us" for q, v in core["qubits"].items()))
(run / "embedding.json").write_text(json.dumps(e, indent=2) + "\n")
PY
```

## Submit after sharing the layout

Use the standalone script below. It contains every option explicitly, sets
up logging and checks the local acceptance settings. It does not depend on
the terminal's HTQ variables or copying line-continuation backslashes.
Add `--dry-run` to validate local inputs and print its command without
contacting hardware. The script refuses to start if production job records
already exist.

```sh
bash data/hw/runs/chicago/2026-11-11/submit_main_rzz_twirl.sh
```

The earlier attempt lost its line-continuation backslashes, so only its first
line ran. It created 76 local `fake_boston`/CZ job records in `data/hw/`, and
then zsh rejected the remaining option lines as separate commands. Its memory
errors came from local simulation. No Chicago jobs or QPU usage resulted from
that attempt, and the acceptance record was unchanged.

The layout guard compares the newly selected full layout against acceptance
before any job is submitted. If it differs, rerun acceptance and report the
new layout before submitting.

`--no-strict-guard` avoids waiting or cancellation solely because usage
estimates are missing, as agreed. Known allocation checks remain active.
Batch submission is the default. `--rep-time` is an estimate for planning,
not a hardware repetition-delay setting.

The initial console summary reports 127 original job groups; expect 508
instance-job records. Verify the local submission records before sending the
submitted email. This checks submission records, not hardware completion.

```sh
.venv/bin/python - "$HTQ_RUN" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

folder = Path(sys.argv[1]) / "jobs/main_rzz_twirl"
records = [json.loads(p.read_text()) for p in folder.glob("htq_job_*.json")]
if len(records) != 508 or len({r["job_id"] for r in records}) != 508:
    raise SystemExit(f"Expected 508 distinct job records; found {len(records)}. Inspect before retrying.")
groups = defaultdict(list)
for r in records:
    if r.get("cancelled") is not False or r.get("guard") not in ("ok", "unverifiable"):
        raise SystemExit(f"Unresolved/cancelled submission: {r['job_id']}")
    if r.get("twirl", {}).get("instances") != 4:
        raise SystemExit(f"Unexpected twirl configuration: {r['job_id']}")
    groups[r["twirl"]["group"]].append(r["twirl"]["instance"])
if len(groups) != 127 or any(sorted(v) != [0, 1, 2, 3] for v in groups.values()):
    raise SystemExit("Missing or duplicate twirl instances; inspect the job records.")
shots = sum(sum(r["shots"].values()) for r in records)
if shots != 46_380_000:
    raise SystemExit(f"Expected 46,380,000 shots; found {shots:,}.")
print(f"Submission records complete: {len(records)} jobs, {shots:,} shots.")
PY
```

## Fetch after all jobs finish

The run-specific collector saves the raw bits, job metrics and execution
spans, then verifies the full campaign. It only reads existing Runtime jobs.
Run from the repository root:

```sh
.venv/bin/python -u data/hw/runs/chicago/2026-11-11/collect_results.py \
  --workers 3 \
  2>&1 | tee data/hw/runs/chicago/2026-11-11/logs/collect_results.log
```

It merges each group of four instances into one raw file, producing 127
files while preserving all 508 instance job records. Completed downloads
are reused on a retry. The collector checks the actual 15,000/30,000/60,000
merged-shot allocations, rather than assuming every PUB has 60,000 shots.

Repeat all saved-data checks without contacting Runtime:

```sh
.venv/bin/python data/hw/runs/chicago/2026-11-11/collect_results.py --check-only
```

`job_usage.csv` contains both job-status timestamps and the start/stop of
the result execution spans. Use the latter for shot-collection timing.
`completion_summary.json` and `raw_sha256.txt` are written only after every
group passes. See `README.md` for the verified handoff contents and timing.
