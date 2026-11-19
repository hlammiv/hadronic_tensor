"""Collect this matched diagnostic, or verify its saved results with --check-only.

This script only reads existing Runtime jobs. It never submits or cancels jobs.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = next(p for p in RUN.parents if (p / "htq_hw").is_dir())
sys.path.insert(0, str(ROOT))
from htq_hw import campaign as CP


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def utc_stamp(value):
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def records():
    selected, ids = {}, set()
    accepted = []
    for variant, count, instances in (("control2", 2, 0), ("twirl", 8, 4)):
        acceptance = json.loads((RUN / f"acceptance_mirror_diag_{variant}.json").read_text())
        require(acceptance["status"] in ("PASS", "WARN") and not acceptance["failures"],
                f"{variant}: acceptance failed")
        layout = acceptance["embedding"]["layout"]
        require(len(layout) == 101 and layout[48] != 108, "Invalid accepted embedding")
        accepted.append(layout)
        paths = sorted((RUN / "jobs" / f"mirror_diag_{variant}").glob("htq_job_*.json"))
        require(len(paths) == count, f"{variant}: expected {count} job records")
        groups = {}
        for path in paths:
            meta = json.loads(path.read_text())
            require(meta["job_id"] not in ids, "Duplicate job ID")
            ids.add(meta["job_id"])
            require(meta.get("cancelled") is False and meta.get("guard") != "pending",
                    f"{path}: cancelled or unresolved submission")
            require(meta["backend"] == "ibm_chicago" and meta["basis"] == "rzz"
                    and meta["ns"] == 50 and meta["n_logical"] == 101, f"{path}: configuration differs")
            require(meta["initial_layout"] == layout, f"{path}: accepted layout differs")
            require(meta["options"] == acceptance["options"], f"{path}: accepted options differ")
            require(meta["options"]["pre_twirl"] == instances, f"{path}: twirl count differs")
            twirl = meta.get("twirl")
            key = twirl["group"] if twirl else meta["job_id"]
            groups.setdefault(key, []).append(meta)
        require(len(groups) == 2, f"{variant}: expected two result groups")
        for key, metas in groups.items():
            metas.sort(key=lambda m: m.get("twirl", {}).get("instance", 0))
            require(len({m["job_index"] for m in metas}) == 1, "Mixed job indices in group")
            if instances:
                require([m["twirl"]["instance"] for m in metas] == list(range(instances)),
                        f"{key}: incomplete twirl group")
                require(metas[0]["job_id"] == key, f"{key}: wrong group ID")
                for meta in metas:
                    require(meta["twirl"]["instances"] == instances, "Inconsistent twirl count")
            else:
                require(len(metas) == 1, "Unexpected control grouping")
        selected[variant] = groups
    require(accepted[0] == accepted[1], "Control and twirl accepted layouts differ")
    return selected, accepted[0]


def validate_bits(path, key, metas):
    twirled = "twirl" in metas[0]
    bases = metas[0]["twirl"]["of"] if twirled else metas[0]["pub_names"]
    require(len(set(bases)) == len(bases), "Duplicate base PUB names")
    counts = dict.fromkeys(bases, 0)
    for meta in metas:
        expected_names = ([CP.twirl_name(n, meta["twirl"]["instance"]) for n in bases]
                          if twirled else bases)
        require(meta["pub_names"] == expected_names, "Instance PUB names or order differ")
        for name, base in zip(expected_names, bases):
            counts[base] += meta["shots"][name]
            info = meta["pubs"][name]
            require(info.get("clbit_to_logical") == list(range(101)), "Logical bit mapping differs")
            require(info.get("n_clbits", 101) == 101, "Unexpected classical register width")
    require(set(counts.values()) == {60000}, "Expected 60,000 shots per merged PUB")
    with np.load(path, allow_pickle=False) as bits:
        extras = {"twirl_jobs", "twirl_instances"} if twirled else set()
        require(set(bits.files) == set(bases) | {"job_id", "backend", "pub_names"} | extras,
                f"{path}: unexpected raw keys")
        require(bits["job_id"].item() == key and bits["backend"].item() == "ibm_chicago",
                f"{path}: identity differs")
        require(bits["pub_names"].tolist() == bases, f"{path}: PUB order differs")
        if twirled:
            require(bits["twirl_jobs"].tolist() == [m["job_id"] for m in metas],
                    f"{path}: instance order differs")
            require(bits["twirl_instances"].item() == 4, f"{path}: wrong instance count")
        for base in bases:
            array = bits[base]
            require(array.shape == (counts[base], 101) and array.dtype == np.uint8
                    and not np.any(array > 1), f"{path}: invalid array for {base}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    selected, layout = records()
    service = None
    if not args.check_only:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
    rows, raw_checks, summaries = [], {}, {}
    for variant, groups in selected.items():
        pub_counts = {}
        for key, metas in groups.items():
            raw = RUN / "raw" / f"mirror_diag_{variant}" / f"htq_bits_{key}.npz"
            items = []
            for meta in metas:
                jid = meta["job_id"]
                metrics_path = RUN / "metrics" / f"mirror_diag_{variant}" / f"htq_metrics_{jid}.json"
                if service is not None:
                    job = service.job(jid)
                    state = job.status()
                    state = getattr(state, "name", str(state)).split(".")[-1].upper()
                    print(f"{variant}: {jid} {state}", flush=True)
                    snapshot = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
                    snapshot.update(job_id=jid, backend=job.backend().name,
                                    status=state, metrics=job.metrics())
                    write_json(metrics_path, snapshot)
                    require(state == "DONE", f"{jid}: not completed successfully")
                    if not raw.exists() or "execution_spans" not in snapshot:
                        from qiskit_ibm_runtime.utils.json import RuntimeEncoder
                        result = job.result()
                        require(len(result) == len(meta["pub_names"]), f"{jid}: result count differs")
                        snapshot["result_metadata"] = json.loads(json.dumps(result.metadata, cls=RuntimeEncoder))
                        snapshot["pub_result_metadata"] = {n: p.metadata for n, p in zip(meta["pub_names"], result)}
                        spans = result.metadata.get("execution", {}).get("execution_spans", [])
                        snapshot["execution_spans"] = [
                            {"start": utc_stamp(span.start), "stop": utc_stamp(span.stop),
                             "type": type(span).__name__, "size": int(span.size),
                             "pub_indices": list(span.pub_idxs),
                             "pub_sizes": {meta["pub_names"][i]: int(span.mask(i).sum())
                                           for i in span.pub_idxs}}
                            for span in spans]
                        write_json(metrics_path, snapshot)
                    items.append((job, meta))
                else:
                    snapshot = json.loads(metrics_path.read_text())
                require(snapshot["job_id"] == jid and snapshot["backend"] == meta["backend"]
                        and snapshot["status"] == "DONE", f"{metrics_path}: identity/status differs")
                metrics = snapshot["metrics"]
                seconds = (metrics.get("usage") or {}).get("quantum_seconds")
                require(seconds is None or (np.isfinite(seconds) and seconds >= 0), "Invalid usage")
                stamps = metrics.get("timestamps") or {}
                spans = snapshot.get("execution_spans", [])
                if spans:
                    covered = {name: sum(span["pub_sizes"].get(name, 0) for span in spans)
                               for name in meta["pub_names"]}
                    require(covered == meta["shots"], f"{jid}: execution masks do not cover requested shots")
                span_starts = [span["start"] for span in spans]
                span_stops = [span["stop"] for span in spans]
                rows.append({"campaign": f"mirror_diag_{variant}", "job_index": meta["job_index"],
                             "twirl_instance": meta.get("twirl", {}).get("instance", ""),
                             "job_id": jid, "backend": meta["backend"], "status": "DONE",
                             "pubs": len(meta["pub_names"]), "shots": sum(meta["shots"].values()),
                             "qpu_seconds": seconds, "created": stamps.get("created"),
                             "running": stamps.get("running"), "finished": stamps.get("finished"),
                             "data_collection_start": min(span_starts) if span_starts else None,
                             "data_collection_stop": max(span_stops) if span_stops else None})
            if service is not None and not raw.exists():
                CP.fetch(items[0][0], metas[0], str(raw.parent), group=items, log=print)
            counts = validate_bits(raw, key, metas)
            require(not (set(pub_counts) & set(counts)), "PUB repeated across result groups")
            pub_counts.update(counts)
            raw_checks[str(raw.relative_to(RUN))] = {"pubs": counts, "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
        expected = {s.name for s in CP.preset_specs("mirror-diag")}
        require(set(pub_counts) == expected and sum(pub_counts.values()) == 420000,
                f"{variant}: incomplete preset")
        subset = [r for r in rows if r["campaign"] == f"mirror_diag_{variant}"]
        usage = [r["qpu_seconds"] for r in subset if r["qpu_seconds"] is not None]
        starts = [r["running"] for r in subset if r["running"]]
        ends = [r["finished"] for r in subset if r["finished"]]
        data_starts = [r["data_collection_start"] for r in subset if r["data_collection_start"]]
        data_stops = [r["data_collection_stop"] for r in subset if r["data_collection_stop"]]
        summaries[variant] = {"jobs": len(subset), "merged_pubs": len(pub_counts),
                              "shots": sum(pub_counts.values()), "qpu_seconds_known": sum(usage),
                              "missing_usage_jobs": len(subset) - len(usage),
                              "first_running": min(starts) if starts else None,
                              "last_finished": max(ends) if ends else None,
                              "data_collection_start": min(data_starts) if data_starts else None,
                              "data_collection_stop": max(data_stops) if data_stops else None}
    if args.check_only:
        saved_hashes = {path: digest for digest, path in
                        (line.split("  ", 1) for line in (RUN / "raw_sha256.txt").read_text().splitlines())}
        require(saved_hashes == {path: value["sha256"] for path, value in raw_checks.items()},
                "Saved raw checksums differ")
    else:
        with (RUN / "job_usage.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (RUN / "raw_sha256.txt").write_text("".join(f"{v['sha256']}  {k}\n" for k, v in sorted(raw_checks.items())))
    summary = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "variants": summaries,
               "total_jobs": len(rows), "total_shots": sum(r["shots"] for r in rows),
               "qpu_seconds_known": sum(r["qpu_seconds"] for r in rows if r["qpu_seconds"] is not None),
               "site_24": layout[48], "hadamard_ancilla": layout[100], "same_accepted_layout": True,
               "raw_files": raw_checks, "job_records_and_metrics_validated": True}
    control_stop = summaries["control2"]["data_collection_stop"]
    twirl_start = summaries["twirl"]["data_collection_start"]
    if control_stop and twirl_start:
        summary["control_to_twirl_gap_seconds"] = (
            datetime.fromisoformat(twirl_start) - datetime.fromisoformat(control_stop)).total_seconds()
    if not args.check_only:
        write_json(RUN / "completion_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "raw_files"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
