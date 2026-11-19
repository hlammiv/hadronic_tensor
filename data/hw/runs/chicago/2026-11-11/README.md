# Chicago production campaign — November 11, 2026

All 508 hardware jobs report DONE. The downloaded results contain all 817
original PUBs and all 46,380,000 requested shots. The four pre-twirled
instances of each job group have been merged into 127 raw files, with
instance identities and ordering retained. No completion discrepancies
were found.

| Preset | Hardware jobs | Original PUBs | Retained shots |
|---|---:|---:|---:|
| `prod-bridge` | 100 | 183 | 9,735,000 |
| `relA-core` | 220 | 370 | 20,805,000 |
| `prod-quarter` | 40 | 75 | 4,500,000 |
| `relA-quarter` | 148 | 189 | 11,340,000 |
| Total | 508 | 817 | 46,380,000 |

The `relA-core` total includes its t = 6.5–8 extension. The quarter presets
add intermediate physical times. Four pre-twirled instances share the
original shot allocation; they do not multiply it by four.

## Configuration and provenance

The campaign uses Ns = 50, a 101-qubit ladder embedding, the RZZ basis,
`--pre-twirl 4`, and mirror angle 1e-8. Runtime gate twirling is off;
measurement twirling and XY4 dynamical decoupling are on. The 32 Runtime
measurement randomizations reported in PUB metadata are distinct from the
four manually generated gate-twirled instances.

Full acceptance completed without failures. All 508 job records and every
PUB layout match the accepted layout and options. The selection used a 5%
readout-error cut, a 2% two-qubit-error cut, and a 100 us T2 floor on the
six-qubit core:

- Site 24: physical q53, T2 = 164.01 us.
- Hadamard ancilla: physical q63, T2 = 315.26 us.
- Minimum core T2: q54, 134.72 us.

The six core qubits are q53, q63, q52, q43, q54 and q44. Their mapping is
unchanged from November 10; 13 assignments elsewhere in the layout differ.
The full mapping is in `embedding.json`.

The historical backend properties requested as of the acceptance time,
12:17:33 UTC, have update time 10:05:23 UTC. That timestamp and all six core
T2 values exactly match acceptance. `calibration/retrieval_provenance.json`
records the later retrieval and comparison. This is the acceptance-time
snapshot, not a measurement of calibration throughout execution.

`source_commit.txt` retains the original base commit, `607c3fe`, and
`configuration_changes.patch` retains the then-local changes. Those CLI
changes and their tests are now in commits `be53e50` and `407b042`.
All 63 acceptance-hashed files, including the two originally untracked
tests, match the collection checkout at `407b042`. The original patch does
not itself contain those untracked tests. `collection_provenance.json`
records the file comparisons, collector hashes and dependency versions.

## Execution timing

The result execution spans cover **15:15:52.786913–16:06:45.252835 UTC on
November 11**, an envelope of about 50 minutes 52 seconds. Preset execution
was interleaved. Every job has measured usage metadata; summed QPU usage
is 2,876 seconds (47 minutes 56 seconds).

`job_usage.csv` provides one row per instance job, with both job-status
timestamps and the start/stop of its shot-collection spans. Use the latter
to compare when data were collected. A job's RUNNING timestamp can precede
shot collection; the overall span envelope is not continuous QPU usage.

Runtime span sizes can include randomization padding. All per-PUB span
masks and returned arrays contain exactly the requested retained shots.

## Files and verification

- `raw/main_rzz_twirl/`: 127 compressed NPZ files, containing 817 original
  PUB arrays. Each array is binary `uint8`, shape `(shots, 101)`, with
  column i representing logical wire i. Site 24 is column 48 and the
  Hadamard ancilla is column 100.
- `jobs/main_rzz_twirl/`: all 508 original submission records, including
  instance IDs, seeds, layouts, options and per-instance shot allocations.
- `metrics/main_rzz_twirl/`: all 508 job statuses, measured usage records,
  job timestamps, native Runtime-encoded result metadata, per-PUB metadata,
  and normalized execution spans with per-PUB mask counts.
- `job_usage.csv`: timing and measured usage by job.
- `completion_summary.json`: verified totals, per-preset timing, raw-file
  inventory and hashes.
- `raw_sha256.txt`: checksums of all 127 raw files.
- `MANIFEST.sha256`: checksums of the run's data, provenance and handoff
  files; local logs, email drafts, Python caches and OS metadata are excluded.
- `acceptance_main_rzz_twirl.json`, `embedding.json`, `calibration/`,
  `source_commit.txt`, `configuration_changes.patch` and
  `collection_provenance.json`: acceptance and source/calibration provenance.
- `COMMANDS.md`, `submit_main_rzz_twirl.sh`, `collect_results.py` and
  `collect_calibration.py`: submission and collection procedures. Original
  submission and collection logs are retained locally under `logs/`.

Merged shot allocations are 60,000 for 747 PUBs, 15,000 for 36 plain PUBs,
and 30,000 for 34 plain mirrors. The four instance blocks appear in order
0, 1, 2, 3. Each contributes one quarter of its PUB's allocation; the
`twirl_jobs` and `twirl_instances` fields identify the source jobs.

To repeat the saved-data checks without contacting Runtime, run from the
repository root:

```sh
.venv/bin/python data/hw/runs/chicago/2026-11-11/collect_results.py --check-only
.venv/bin/python data/hw/runs/chicago/2026-11-11/collect_calibration.py --check-only
```

To check the handoff manifest, run `shasum -a 256 -c MANIFEST.sha256` from
this directory. Logs and email drafts are outside the manifest's scope.

This is a verified raw-data handoff. Completion checks do not establish
an improvement in the scientific observables. This campaign covers the
real-correlator presets above; it contains no new vacuum, qPDF, imaginary
tier or mid-circuit Gauss runs. The existing tensor reduction uses MPS
vacuum subtraction.
