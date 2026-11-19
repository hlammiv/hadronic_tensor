# Chicago mirror diagnostic commands

Both variants have now completed and their results have been collected. The
submission commands below are retained as provenance; do not resubmit them.
Use `collect_results.py --check-only` to verify the saved results offline.
See `README.md` and `completion_summary.json` for the completed run.

Prepared against local `tier3-calibration-checks` at `60eb786`, with two local
changes: preset numerical checks now retain the twirling flag, and
`submit --require-layout RECORD` checks the newly selected full embedding
against a successful full live acceptance record before building or submitting.
The layout flag does not validate other experiment options. A further local
fix makes incomplete submissions exit nonzero, so the `&&` between the
submission commands below stops the sequence if the control is incomplete.

On 10 November, both full acceptances completed without failures. Their only
warning concerns uncommitted code. Both used the 09:49:05 UTC calibration and
selected site 24 on q53 (T2 164 us), the Hadamard ancilla on q63 (315 us), and
a minimum core T2 of 135 us. No QPU jobs were submitted during this review.

The user's first submission attempts at 10:22–10:25 UTC were cancelled by the
local budget guard after Runtime supplied no estimate within 60 seconds.
Runtime subsequently confirmed both jobs as CANCELLED, with no running
timestamp and zero QPU usage. The two records are preserved under
`cancelled_jobs/`; `budget_guard_cancellation.json` saves their remote metrics.
No jobs were retained from those attempts. Both complete variants were
subsequently resubmitted by the user and completed successfully. At retry
preparation, the calibration remained 09:49:05 UTC and the remaining
allocation was verified as 432,097 seconds. The assistant submitted no jobs.

Declan has confirmed that his runs are not constrained by Hank's QPU budget.
Usage estimates are informational for this workflow: do not wait for missing
estimates or cancel jobs because those estimates are unavailable. Retain
measured usage for the experiment records, not as a reason to delay the run.

The commands below therefore use the existing `--no-strict-guard` option. This
checks each job estimate once and proceeds if the estimate or remaining
allocation is unavailable. Known over-budget values still stop submission;
missing estimates are not revisited later. The flag leaves layout checks,
Runtime input validation, and circuit settings unchanged.

The recommended addition to Hank's configuration is `--min-t2 100` on every
acceptance and submit command. This filters the six core qubits, not the whole
device. `--mode ladder` prevents a change of embedding family. The mirror
angle stays at 1e-8, with Runtime gate twirling off, measurement twirling on,
and XY4 dynamical decoupling on. Each variant has 420,000 planned shots:
60,000 per pub, split into four 15,000-shot instances in the twirled variant.
The 62 us repetition-time argument is a budgeting assumption, not a hardware
repetition-delay setting. The two variants estimate 52.08 QPU seconds in total,
before any additional job overhead.

Chicago reported 990 pending jobs during the review. Separate submissions
create separate batches; back-to-back submission does not guarantee adjacent
execution. Preserve execution timestamps with the metrics. Also, this RZZ
twirl uses only Paulis commuting with ZZ, so it cannot average away a pure
coherent ZZ overrotation. Four instances are a small randomization ensemble;
pooled shot errors do not quantify variation across that ensemble.

Run the following in the same bash or zsh session, using this local checkout.
Change `HTQ_RUN` if running on another date. Stop if an acceptance fails, the
layout check fails, or submission reports a cancelled/partial campaign. Do
not rerun a completed submission command.

```sh
cd /Users/declanmillar/Projects/fermilab/hadronic-tensor/hadronic_tensor
HTQ_RUN=data/hw/runs/chicago/2026-11-10
HTQ_GLOBAL=(--ns 50 --max-readout-error 0.05 --max-2q-error 0.02
            --min-t2 100 --twirl-gates off --mirror-eps 1e-8)
HTQ_EXPERIMENT=(--target ibm_chicago --basis rzz --mode ladder
                --preset mirror-diag --rep-time 6.2e-5)
mkdir -p "$HTQ_RUN"
```

## Acceptance

These records already exist from the review. Rerun this stage if the calibration
or configuration changes, before spending shots on either variant.

```sh
.venv/bin/python -m htq_hw "${HTQ_GLOBAL[@]}" acceptance \
  "${HTQ_EXPERIMENT[@]}" --level full --require-real \
  --record "$HTQ_RUN/acceptance_mirror_diag_control2.json"

.venv/bin/python -m htq_hw "${HTQ_GLOBAL[@]}" --pre-twirl 4 acceptance \
  "${HTQ_EXPERIMENT[@]}" --level full --require-real \
  --record "$HTQ_RUN/acceptance_mirror_diag_twirl.json"
```

Check that both acceptances succeeded and share the entire layout, and save
the requested embedding. If site 24 is q108, stop and tell Hank before submitting.

```sh
.venv/bin/python - "$HTQ_RUN" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
records = [json.loads((run / f"acceptance_mirror_diag_{v}.json").read_text())
           for v in ("control2", "twirl")]
for record in records:
    if record["status"] not in ("PASS", "WARN") or record["failures"]:
        raise SystemExit("Acceptance failed; do not submit")
layouts = [r["embedding"]["layout"] for r in records]
if layouts[0] != layouts[1]:
    raise SystemExit("Acceptance layouts differ; do not submit")
layout = layouts[0]
if layout[48] == 108:
    raise SystemExit("Site 24 is q108; tell Hank before submitting")
embedding = {"backend": "ibm_chicago", "ns": 50, "site_24": layout[48],
             "hadamard_ancilla": layout[100], "layout": layout,
             "core_t2_us": records[0]["embedding"]["core_t2_us"]}
(run / "embedding.json").write_text(json.dumps(embedding, indent=2) + "\n")
print(f"Matched layout: site 24 -> q{layout[48]}, ancilla -> q{layout[100]}")
PY
```

## Submission

The next commands spend QPU allocation. Both compare their actual selected
layout to the same control acceptance. A changed layout is refused before
submission. Run the control first, then the twirled variant immediately after
the control submission completes; do not wait for its hardware results.

```sh
.venv/bin/python -m htq_hw "${HTQ_GLOBAL[@]}" submit \
  "${HTQ_EXPERIMENT[@]}" --real --confirm ibm_chicago --no-strict-guard \
  --require-layout "$HTQ_RUN/acceptance_mirror_diag_control2.json" \
  --tag mirror-diag-control2 \
  --out "$HTQ_RUN/jobs/mirror_diag_control2" &&
.venv/bin/python -m htq_hw "${HTQ_GLOBAL[@]}" --pre-twirl 4 submit \
  "${HTQ_EXPERIMENT[@]}" --real --confirm ibm_chicago --no-strict-guard \
  --require-layout "$HTQ_RUN/acceptance_mirror_diag_control2.json" \
  --tag mirror-diag-twirl \
  --out "$HTQ_RUN/jobs/mirror_diag_twirl"
```

Expect two control job records and eight twirled job records. The existing CLI
summary counts the two base groups even for twirling; the instance submission
logs and JSON records show the eight actual jobs. If a budget guard stops a
submission partway through, preserve its records and resolve the partial run
before continuing.

## Fetch and usage

After all jobs finish, fetch both variants. The twirled fetch needs all eight
job records together and writes two merged bits files. The old
`scripts/collect_hw_results.py` assumes unique job indices and untwirled pub
names; it does not support these twirled groups.

```sh
.venv/bin/python -m htq_hw fetch \
  "$HTQ_RUN"/jobs/mirror_diag_control2/htq_job_*.json \
  --out "$HTQ_RUN/raw/mirror_diag_control2"

.venv/bin/python -m htq_hw fetch \
  "$HTQ_RUN"/jobs/mirror_diag_twirl/htq_job_*.json \
  --out "$HTQ_RUN/raw/mirror_diag_twirl"
```

This separate read-only collection saves metrics, execution timestamps, and a
usage CSV for all ten jobs. It refuses incomplete or cancelled submissions.

```sh
.venv/bin/python - "$HTQ_RUN" <<'PY'
import csv
import json
import sys
from pathlib import Path
from qiskit_ibm_runtime import QiskitRuntimeService

run = Path(sys.argv[1])
records = []
for variant, count in (("mirror_diag_control2", 2), ("mirror_diag_twirl", 8)):
    paths = sorted((run / "jobs" / variant).glob("htq_job_*.json"))
    if len(paths) != count:
        raise SystemExit(f"{variant}: expected {count} job records, found {len(paths)}")
    for path in paths:
        meta = json.loads(path.read_text())
        if meta.get("cancelled") is not False or meta.get("guard") == "pending":
            raise SystemExit(f"Resolve incomplete/cancelled submission: {path}")
        records.append((variant, meta))
service = QiskitRuntimeService()
rows = []
for variant, meta in records:
    job = service.job(meta["job_id"])
    state = job.status()
    state = getattr(state, "name", str(state)).split(".")[-1].upper()
    if state != "DONE":
        raise SystemExit(f"{meta['job_id']}: {state}; collect after completion")
    metrics = job.metrics()
    folder = run / "metrics" / variant
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = {"job_id": meta["job_id"], "backend": meta["backend"],
                "status": state, "metrics": metrics}
    (folder / f"htq_metrics_{meta['job_id']}.json").write_text(
        json.dumps(snapshot, indent=2, default=str) + "\n")
    rows.append({"campaign": variant, "job_index": meta["job_index"],
                 "twirl_instance": meta.get("twirl", {}).get("instance", ""),
                 "job_id": meta["job_id"], "pubs": len(meta["pub_names"]),
                 "shots": sum(meta["shots"].values()),
                 "qpu_seconds": (metrics.get("usage") or {}).get("quantum_seconds")})
with (run / "job_usage.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved metrics and usage for {len(rows)} jobs")
PY
```

Return `raw/`, `jobs/`, `metrics/`, `job_usage.csv`, both acceptance records,
`embedding.json`, and the configuration provenance with the results. Preserve
`cancelled_jobs/` and `budget_guard_cancellation.json` as the attempt history.
Retain all instance records even though the raw bits are merged.
