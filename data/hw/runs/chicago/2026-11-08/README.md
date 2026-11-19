# Chicago mirror diagnostic — 8 November 2026

Hank's requested diagnostic, submitted from `tier3-calibration-checks` at
`03d820b2`. All three full acceptance checks passed without warnings.
The three variants selected the same 101-qubit ladder layout, including
matter site 24 on physical qubit 108 and the Hadamard ancilla on qubit 118.

The runs use `Ns=50`, RZZ, the `mirror-diag` preset, a 5% readout error cut,
a 2% two-qubit error cut, and no manual qubit exclusions. Measurement
twirling and XY4 dynamical decoupling remain enabled. Calibration-based
filtering removes three qubits and 24 couplers; this is distinct from
manually excluding qubits.

Each variant requests 60,000 shots for each of seven PUBs: the production
card's `j0` Z-readout reference at `t=0`, and physics/mirror pairs at
`t=2,4,6`. Each variant has two jobs and 420,000 planned shots.

| Variant | Gate twirling | Mirror angle | Hardware outcome |
|---|---|---:|---|
| `mirror_diag_control` | off | `1e-8` | Completed; 18 QPU seconds |
| `mirror_diag_twirl` | on | `1e-8` | Rejected, Runtime error 1519; zero QPU usage |
| `mirror_diag_eps05` | off | `0.05` | Completed; 18 QPU seconds |

All six jobs were submitted between 20:38:13 and 20:39:25 UTC. Their recorded
layouts and options agree with their corresponding acceptance records.
The submission commands ran consecutively between 20:37:38 and 20:39:33 UTC.

## Twirling rejection

Both twirling jobs were created successfully, then failed server validation:

> Error code 1519; Job not valid. "Gate twirling does not support fractional gates."

The variant was skipped as requested. Both jobs have measured
`usage.quantum_seconds=0`, and neither reached a running state.
`twirling_rejection.txt` preserves the complete server messages;
`twirling_rejection.json` includes the job IDs, status and metrics.
The original job metadata is preserved unchanged under
`failed_jobs/mirror_diag_twirl/`, outside the successful-results collector's
input directory. These are failed jobs, not cancelled jobs.

## Records

- `request_manifest.json`: exact acceptance/submission commands, commit and planned counts.
- `acceptance_mirror_diag_*.json`: full acceptance records, layouts, options and environment versions.
- `jobs/<variant>/htq_job_<id>.json`: supported job metadata and logical bit mappings.
- `failed_jobs/mirror_diag_twirl/`: original metadata for the two rejected jobs.
- `logs/`: acceptance and submission logs.
- `submission_timeline.json`: command timings and outcomes.
- `job_status.json`: latest status of the four supported jobs.
- `collection_state.json`: automatic collector status, completion or failure details.
- `collect_when_ready.py`: local background collector; never submits jobs.
- `provenance_checks.json`: verification of manifests, layouts and options at the submission-stage snapshot; completion is recorded separately in `completion_summary.json`.
- `calibration/`: saved Chicago calibration snapshot, updated at 19:50:22 UTC.
- `preflight_snapshot.json`: backend status and available allocation before submission.

## Collecting results

All four supported jobs completed. Their 14 PUBs contain 840,000 shots and
used 36 measured QPU seconds in total. The raw files passed the collector
checks; `raw_sha256.txt` records their hashes.
`completion_summary.json` records totals and execution timestamps.
From the repository root, collection and offline verification are repeatable:

```sh
.venv/bin/python scripts/collect_hw_results.py data/hw/runs/chicago/2026-11-08
.venv/bin/python scripts/collect_hw_results.py data/hw/runs/chicago/2026-11-08 --check-only
```

It writes `raw/<variant>/htq_bits_<id>.npz`, per-job JSON metrics under
`metrics/<variant>/`, and `job_usage.csv`. It checks identity, PUB order,
shot counts, bit-array shapes and logical mappings. The CSV covers completed
supported jobs; failed-twirl metrics are preserved separately as described above.

The local background collector has finished. It collected and verified all
four results, wrote `raw_sha256.txt` and `completion_summary.json`, and created
the sibling archive `../mirror_diag_chicago_2026-11-08.zip`. Its log is
`logs/collector.log`.
