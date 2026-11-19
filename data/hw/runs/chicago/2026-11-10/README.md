# Chicago mirror diagnostic — November 10, 2026

Both the fresh control and the `--pre-twirl 4` variant completed successfully.
All ten jobs report DONE. The four merged raw files contain 840,000 retained
shots in total, with 60,000 shots for every original PUB.

| Variant | Hardware jobs | Original PUBs | Retained shots | Measured QPU usage |
|---|---:|---:|---:|---:|
| `mirror_diag_control2` | 2 | 7 | 420,000 | 18 s |
| `mirror_diag_twirl` | 8 | 7 | 420,000 | 36 s |

The preset contains the production card's J0 Z-readout reference at t = 0 and
physics/mirror pairs at t = 2, 4, 6. The twirled variant uses four instances
per original PUB, each contributing 15,000 retained shots. Its eight jobs
have been merged into two raw files under the original PUB names. Instance
job IDs and their ordering are retained in the raw files and job records.

## Configuration and embedding

The runs use Ns = 50, the RZZ basis, a 5% readout-error cut, a 2% two-qubit
error cut, and a 100 us T2 floor on the six-qubit Hadamard-test core. No manual
qubit exclusions were added. Both full acceptance checks completed without
failures; their only warning concerned uncommitted local code changes.

All ten job records match both accepted 101-qubit ladder layouts exactly:

- Site 24: physical q53, T2 = 164.01 us.
- Hadamard ancilla: physical q63, T2 = 315.26 us.
- Lowest core T2: q54, 134.72 us.

The full mapping is in `embedding.json`. The calibration snapshot was updated
at 09:49:05 UTC. Mirror angle remains 1e-8. Runtime gate twirling is disabled;
measurement twirling and XY4 dynamical decoupling are enabled for both runs.
The four manually generated gate-twirled instances are separate from the 32
Runtime randomizations reported in the PUB result metadata.

## Timing

The jobs were submitted between 11:08:31 and 11:09:10 UTC. The recorded
data-collection spans place the control at approximately 12:07:53–12:08:07 UTC
and the twirled variant at 12:10:41–12:12:31 UTC. The gap between the control's
last span and the twirled variant's first span is approximately 2 minutes
34 seconds. All shot collection therefore occurred within about five minutes.

Use the result execution spans to compare the runs. The job-status
`metrics.timestamps.running` fields begin much earlier and do not identify
when the shots were collected. Execution spans bound data collection; their
envelopes do not imply continuous QPU activity throughout the interval.
The metrics files retain both forms of timing, and `job_usage.csv` includes
the data-collection start and stop times separately from the job-status times.

Runtime span sizes can include padding from measurement randomization, such
as 15,008 internal slots for a 15,000-shot PUB. The returned arrays and
per-PUB execution masks retain exactly the requested shots. Shot totals in
this package are based on the arrays, not the padded span size.

## Files and verification

- `raw/`: four compressed bits files, two per variant. Every PUB array is
  binary uint8 with shape `(60000, 101)` in logical-bit order.
- `jobs/`: all ten successful job records, including the eight twirl instances.
- `metrics/`: job status, measured usage, precise execution spans, native
  Runtime-encoded result metadata, and per-PUB result metadata.
- `job_usage.csv`: one row per successful job, including both kinds of timestamps.
- `completion_summary.json`: totals, timing, and raw-file verification.
- `raw_sha256.txt`: checksums of the four raw files.
- `MANIFEST.sha256`: checksums of all packaged files except the manifest itself.
- `acceptance_mirror_diag_*.json`, `embedding.json`, and `calibration/`:
  accepted configuration and the saved backend calibration.
- `configuration_changes.patch`: local code changes used in the workflow.
- `twirl_check_fix.patch`: the focused numerical-check fix and regression test,
  committed as `3752975` on `fix/preset-twirled-acceptance` in Declan's fork.

Independent verification found no discrepancies in PUB names/order, shot
counts, binary arrays, instance grouping, layouts, options, metrics, or hashes.
To repeat the saved-data checks without contacting Runtime, run from the
repository root:

```sh
.venv/bin/python data/hw/runs/chicago/2026-11-10/collect_results.py --check-only
```

The two initial submission attempts were cancelled locally when Runtime
did not supply an estimate within 60 seconds. Both have confirmed zero QPU
usage. Their records are preserved under `cancelled_jobs/`, with remote
status and metrics in `budget_guard_cancellation.json`. They are excluded
from the successful-result totals and raw files. The successful retry used
`--no-strict-guard`, as specified for Declan's workflow.

