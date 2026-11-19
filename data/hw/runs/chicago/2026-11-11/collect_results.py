"""Collect the November 11 campaign, or verify saved data with --check-only.

Only reads existing Runtime jobs. Never submits or cancels hardware jobs.
Downloads at most --workers groups concurrently and resumes completed files.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = next(p for p in RUN.parents if (p / "htq_hw").is_dir())
sys.path.insert(0, str(ROOT))
from htq_hw import campaign as CP

CAMPAIGN = "main_rzz_twirl"
EXPECTED = {
    "prod-bridge": (183, 9_735_000, 100),
    "relA-core": (370, 20_805_000, 220),
    "prod-quarter": (75, 4_500_000, 40),
    "relA-quarter": (189, 11_340_000, 148),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def utc_stamp(value):
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def records():
    acceptance = json.loads((RUN / f"acceptance_{CAMPAIGN}.json").read_text())
    require(acceptance["status"] in ("PASS", "WARN") and not acceptance["failures"],
            "Acceptance failed")
    require(acceptance["presets"] == list(EXPECTED), "Accepted presets differ")
    layout = acceptance["embedding"]["layout"]
    require(len(layout) == len(set(layout)) == 101 and layout[48] == 53
            and layout[100] == 63, "Invalid accepted embedding")
    specs = {s.name: s for p in EXPECTED for s in CP.preset_specs(p)}
    expected_shots = {n: (30_000 if s.mirror else 15_000) if s.family == "plain"
                      else 60_000 for n, s in specs.items()}
    require(len(specs) == 817 and sum(expected_shots.values()) == 46_380_000,
            "Current preset definitions differ from this campaign")
    paths = sorted((RUN / "jobs" / CAMPAIGN).glob("htq_job_*.json"))
    require(len(paths) == 508, "Expected 508 job records")
    ids, groups = set(), {}
    for path in paths:
        meta = json.loads(path.read_text())
        require(meta["job_id"] not in ids, "Duplicate job ID")
        ids.add(meta["job_id"])
        require(meta.get("cancelled") is False and meta.get("guard") in ("ok", "unverifiable"),
                f"{path}: cancelled or unresolved submission")
        require(meta["backend"] == "ibm_chicago" and meta["basis"] == "rzz"
                and meta["ns"] == 50 and meta["n_logical"] == 101, f"{path}: configuration differs")
        require(meta["initial_layout"] == layout, f"{path}: accepted layout differs")
        require(meta["options"] == acceptance["options"]
                and meta["options"]["pre_twirl"] == 4, f"{path}: accepted options differ")
        groups.setdefault(meta["twirl"]["group"], []).append(meta)
    require(len(groups) == 127, "Expected 127 result groups")
    actual_shots, jobs_by_preset, indices = {}, Counter(), set()
    for key, metas in groups.items():
        metas.sort(key=lambda m: m["twirl"]["instance"])
        require([m["twirl"]["instance"] for m in metas] == [0, 1, 2, 3],
                f"{key}: incomplete twirl group")
        require(key == metas[0]["job_id"], f"{key}: wrong group ID")
        require(len({m["job_index"] for m in metas}) == 1, "Mixed job indices")
        require(metas[0]["job_index"] not in indices, "Repeated group index")
        indices.add(metas[0]["job_index"])
        bases = metas[0]["twirl"]["of"]
        require(len(set(bases)) == len(bases) and not (set(bases) & set(actual_shots)),
                "Duplicate original PUBs")
        counts = dict.fromkeys(bases, 0)
        presets = {specs[n].preset for n in bases}
        require(len(presets) == 1, "Mixed presets in job group")
        jobs_by_preset[next(iter(presets))] += len(metas)
        for meta in metas:
            twirl = meta["twirl"]
            require(twirl["instances"] == 4 and twirl["of"] == bases, "Inconsistent twirl metadata")
            names = [CP.twirl_name(n, twirl["instance"]) for n in bases]
            require(meta["pub_names"] == names and set(meta["shots"]) == set(names)
                    and set(meta["pubs"]) == set(names), "Instance PUB names or order differ")
            for name, base in zip(names, bases):
                require(meta["shots"][name] == expected_shots[base] // 4,
                        f"{name}: requested instance shots differ")
                counts[base] += meta["shots"][name]
                info = meta["pubs"][name]
                require(info["layout"] == layout and info["clbit_to_logical"] == list(range(101))
                        and info["n_clbits"] == 101, f"{name}: bit mapping differs")
        actual_shots.update(counts)
    require(actual_shots == expected_shots, "Missing or incorrect original PUBs/shots")
    for preset, (n_pubs, shots, jobs) in EXPECTED.items():
        subset = {n: s for n, s in actual_shots.items() if specs[n].preset == preset}
        require(len(subset) == n_pubs and sum(subset.values()) == shots
                and jobs_by_preset[preset] == jobs, f"{preset}: totals differ")
    return dict(sorted(groups.items(), key=lambda kv: kv[1][0]["job_index"])), layout


def validate_bits(path, key, metas):
    bases = metas[0]["twirl"]["of"]
    counts = {n: sum(m["shots"][CP.twirl_name(n, m["twirl"]["instance"])] for m in metas)
              for n in bases}
    with np.load(path, allow_pickle=False) as bits:
        require(set(bits.files) == set(bases) | {"job_id", "backend", "pub_names",
                                              "twirl_jobs", "twirl_instances"},
                f"{path}: unexpected raw keys")
        require(bits["job_id"].item() == key and bits["backend"].item() == "ibm_chicago",
                f"{path}: identity differs")
        require(bits["pub_names"].tolist() == bases, f"{path}: PUB order differs")
        require(bits["twirl_jobs"].tolist() == [m["job_id"] for m in metas]
                and bits["twirl_instances"].item() == 4, f"{path}: instance order/count differs")
        for base in bases:
            array = bits[base]
            require(array.shape == (counts[base], 101) and array.dtype == np.uint8
                    and not np.any(array > 1), f"{path}: invalid array for {base}")
    return counts


class CachedResult:
    """Pass an already fetched result to campaign.fetch without another request."""

    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


def remote_call(function):
    for attempt in range(3):
        try:
            return function()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def metrics_row(snapshot, meta):
    jid = meta["job_id"]
    require(snapshot["job_id"] == jid and snapshot["backend"] == meta["backend"]
            and snapshot["status"] == "DONE", f"{jid}: saved identity/status differs")
    metrics = snapshot["metrics"]
    seconds = (metrics.get("usage") or {}).get("quantum_seconds")
    require(seconds is None or (np.isfinite(seconds) and seconds >= 0), f"{jid}: invalid usage")
    stamps = metrics.get("timestamps") or {}
    spans = snapshot.get("execution_spans", [])
    require(spans, f"{jid}: missing execution spans")
    covered = {n: sum(s["pub_sizes"].get(n, 0) for s in spans) for n in meta["pub_names"]}
    require(covered == meta["shots"], f"{jid}: execution masks do not cover requested shots")
    require(all(datetime.fromisoformat(s["start"]) <= datetime.fromisoformat(s["stop"])
                for s in spans), f"{jid}: invalid execution span")
    return {"campaign": CAMPAIGN, "preset": next(iter(meta["pubs"].values()))["preset"],
            "job_index": meta["job_index"], "twirl_instance": meta["twirl"]["instance"],
            "job_id": jid, "backend": meta["backend"], "status": "DONE",
            "pubs": len(meta["pub_names"]), "shots": sum(meta["shots"].values()),
            "qpu_seconds": seconds, "created": stamps.get("created"),
            "running": stamps.get("running"), "finished": stamps.get("finished"),
            "data_collection_start": min(s["start"] for s in spans),
            "data_collection_stop": max(s["stop"] for s in spans)}


def collect_group(key, metas, service):
    raw = RUN / "raw" / CAMPAIGN / f"htq_bits_{key}.npz"
    rows, items = [], []
    for meta in metas:
        jid = meta["job_id"]
        path = RUN / "metrics" / CAMPAIGN / f"htq_metrics_{jid}.json"
        snapshot = json.loads(path.read_text()) if path.exists() else {}
        complete = raw.exists() and snapshot.get("status") == "DONE" and snapshot.get("execution_spans")
        if service is not None and not complete:
            job = remote_call(lambda: service.job(jid))
            state = remote_call(job.status)
            state = getattr(state, "name", str(state)).split(".")[-1].upper()
            snapshot.update(job_id=jid, backend=job.backend().name, status=state,
                            metrics=remote_call(job.metrics))
            write_json(path, snapshot)
            require(state == "DONE", f"{jid}: remote status is {state}")
            from qiskit_ibm_runtime.utils.json import RuntimeEncoder
            result = remote_call(job.result)
            require(len(result) == len(meta["pub_names"]), f"{jid}: result PUB count differs")
            for name, pub in zip(meta["pub_names"], result):
                require(pub.data.c.num_shots == meta["shots"][name] and pub.data.c.num_bits == 101,
                        f"{jid}:{name}: returned instance shots/width differ")
            snapshot["result_metadata"] = json.loads(json.dumps(result.metadata, cls=RuntimeEncoder))
            snapshot["pub_result_metadata"] = json.loads(json.dumps(
                {n: p.metadata for n, p in zip(meta["pub_names"], result)}, cls=RuntimeEncoder))
            spans = result.metadata.get("execution", {}).get("execution_spans", [])
            snapshot["execution_spans"] = [
                {"start": utc_stamp(s.start), "stop": utc_stamp(s.stop), "type": type(s).__name__,
                 "size": int(s.size), "pub_indices": list(s.pub_idxs),
                 "pub_sizes": {meta["pub_names"][i]: int(s.mask(i).sum()) for i in s.pub_idxs}}
                for s in spans]
            write_json(path, snapshot)
            items.append((CachedResult(result), meta))
        rows.append(metrics_row(snapshot, meta))
    if service is not None and not raw.exists():
        # Publish the merged file only once the compressed archive is complete.
        raw.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".collect-", dir=raw.parent) as folder:
            output = Path(CP.fetch(items[0][0], metas[0], folder, group=items))
            output.replace(raw)
    counts = validate_bits(raw, key, metas)
    return rows, str(raw.relative_to(RUN)), {"pubs": counts, "sha256": digest(raw)}


def totals(rows):
    usage = [r["qpu_seconds"] for r in rows if r["qpu_seconds"] is not None]
    return {"jobs": len(rows), "shots": sum(r["shots"] for r in rows),
            "qpu_seconds_known": sum(usage), "missing_usage_jobs": len(rows) - len(usage),
            "data_collection_start": min(r["data_collection_start"] for r in rows),
            "data_collection_stop": max(r["data_collection_stop"] for r in rows)}


def csv_text(rows):
    import io
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--records-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    require(1 <= args.workers <= 8, "Use 1 to 8 workers")
    groups, layout = records()
    print(f"Validated 508 job records, {len(groups)} groups, 817 PUBs and 46,380,000 shots.", flush=True)
    if args.records_only:
        return
    service = None
    if not args.check_only:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
    rows, raw_checks, errors = [], {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(collect_group, key, metas, service): key for key, metas in groups.items()}
        for index, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                group_rows, path, check = future.result()
                rows.extend(group_rows)
                raw_checks[path] = check
            except Exception as exc:
                errors[key] = f"{type(exc).__name__}: {exc}"
                print(f"Group {key} failed: {errors[key]}", flush=True)
            if index % 5 == 0 or index == len(groups):
                print(f"Verified {len(raw_checks)}/{len(groups)} groups ({len(rows)}/508 jobs); "
                      f"{len(errors)} errors.", flush=True)
    if errors:
        if not args.check_only:
            write_json(RUN / "collection_errors.json", errors)
        raise SystemExit(f"{len(errors)} groups need attention; completed files were retained.")
    rows.sort(key=lambda r: (r["job_index"], r["twirl_instance"]))
    raw_checks = dict(sorted(raw_checks.items()))
    counts = {n: shots for entry in raw_checks.values() for n, shots in entry["pubs"].items()}
    require(len(counts) == 817 and sum(counts.values()) == 46_380_000, "Final PUB/shot totals differ")
    summaries = {p: {**totals([r for r in rows if r["preset"] == p]), "merged_pubs": expected[0]}
                 for p, expected in EXPECTED.items()}
    summary = {"campaign": CAMPAIGN, **totals(rows), "merged_pubs": len(counts),
               "raw_file_count": len(raw_checks), "presets": summaries,
               "shots_per_merged_pub": dict(sorted(Counter(counts.values()).items())),
               "site_24": layout[48], "hadamard_ancilla": layout[100],
               "same_accepted_layout": True, "job_records_and_metrics_validated": True,
               "raw_files": raw_checks}
    hashes = "".join(f"{v['sha256']}  {k}\n" for k, v in raw_checks.items())
    usage_csv = csv_text(rows)
    if args.check_only:
        require((RUN / "raw_sha256.txt").read_text() == hashes, "Saved raw checksums differ")
        require((RUN / "job_usage.csv").read_bytes() == usage_csv.encode(), "Saved usage CSV differs")
        saved = json.loads((RUN / "completion_summary.json").read_text())
        saved.pop("verified_at_utc", None)
        require(saved == json.loads(json.dumps(summary)), "Saved completion summary differs")
    else:
        (RUN / "job_usage.csv").write_bytes(usage_csv.encode())
        (RUN / "raw_sha256.txt").write_text(hashes)
        write_json(RUN / "completion_summary.json",
                   {"verified_at_utc": datetime.now(timezone.utc).isoformat(), **summary})
        (RUN / "collection_errors.json").unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "raw_files"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
