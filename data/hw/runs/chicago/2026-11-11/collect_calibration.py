"""Retrieve historical acceptance-time calibration, or verify the saved snapshot.

This collector only reads Runtime. It never requests a current-properties
fallback and never submits jobs. Use --check-only to avoid all network access.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


RUN = Path(__file__).resolve().parent
REPO = RUN.parents[4]
ACCEPTANCE = RUN / "acceptance_main_rzz_twirl.json"
CALIBRATION = RUN / "calibration"
RETRIEVAL = CALIBRATION / "retrieval_provenance.json"
AS_OF = datetime(2026, 11, 11, 12, 17, 33, tzinfo=timezone.utc)


def stamp(value):
    if isinstance(value, dict) and value.get("__type__") == "datetime":
        value = value["__value__"]
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Calibration timestamp has no timezone")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def collect_source_provenance(acceptance):
    checks = {}
    for name, expected in acceptance["files"].items():
        path = REPO / "htq_hw" / name
        actual = sha256(path) if path.is_file() else None
        checks[name] = {"acceptance_sha256": expected, "current_sha256": actual,
                        "matches": actual == expected}
    dependencies = {}
    for name in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy", "scipy"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    collectors = {}
    for name in ("collect_results.py", "collect_calibration.py"):
        path = RUN / name
        if path.is_file():
            collectors[name] = sha256(path)
    write_json(RUN / "collection_provenance.json", {
        "captured_at_utc": stamp(datetime.now(timezone.utc)),
        "purpose": "Collection environment and current source comparison; historical submission provenance is retained separately.",
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependencies,
        "acceptance_record": ACCEPTANCE.name,
        "acceptance_sha256": sha256(ACCEPTANCE),
        "acceptance_git": acceptance["git"],
        "current_git": {"commit": git("rev-parse", "HEAD"),
                        "branch": git("branch", "--show-current"),
                        "htq_hw_status": git("status", "--porcelain", "--", "htq_hw").splitlines()},
        "historical_source_commit_file": "source_commit.txt",
        "historical_local_changes_file": "configuration_changes.patch",
        "acceptance_file_checks": checks,
        "all_acceptance_files_match": all(item["matches"] for item in checks.values()),
        "all_acceptance_python_files_match": all(item["matches"] for name, item in checks.items()
                                                   if name.endswith(".py")),
        "collector_sha256": collectors,
    })


def compare(snapshot, acceptance):
    expected = acceptance["embedding"]["core_t2_us"]["qubits"]
    values = {}
    factors = {"s": 1e6, "ms": 1e3, "us": 1.0, "µs": 1.0, "μs": 1.0,
               "ns": 1e-3, "ps": 1e-6}
    for qubit, expected_t2 in expected.items():
        properties = [item for item in snapshot["qubits"][int(qubit)] if item["name"] == "T2"]
        if len(properties) != 1:
            raise ValueError(f"q{qubit}: expected exactly one T2 property")
        item = properties[0]
        if item["unit"] not in factors:
            raise ValueError(f"q{qubit}: unsupported T2 unit {item['unit']!r}")
        actual = float(item["value"]) * factors[item["unit"]]
        values[qubit] = {"acceptance_t2_us": expected_t2, "returned_t2_us": actual,
                        "matches": math.isclose(actual, expected_t2, rel_tol=1e-10, abs_tol=1e-8)}
    return {
        "backend_matches": snapshot["backend_name"] == acceptance["target"]["name"],
        "acceptance_last_update_utc": stamp(acceptance["target"]["last_update_date"]),
        "returned_last_update_utc": stamp(snapshot["last_update_date"]),
        "last_update_matches_acceptance": stamp(snapshot["last_update_date"]) == stamp(acceptance["target"]["last_update_date"]),
        "snapshot_not_after_requested_time": datetime.fromisoformat(
            stamp(snapshot["last_update_date"]).replace("Z", "+00:00")) <= AS_OF,
        "core_t2_us": values,
        "all_core_t2_values_match": all(item["matches"] for item in values.values()),
        "returned_min_core_t2_us": min(item["returned_t2_us"] for item in values.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    acceptance = json.loads(ACCEPTANCE.read_text())
    if (acceptance["target"]["name"] != "ibm_chicago" or acceptance["basis"] != "rzz"
            or acceptance["written"] != "2026-11-11T12:17:33"):
        raise ValueError("Unexpected acceptance record; historical request must be reviewed")
    if args.check_only:
        record = json.loads(RETRIEVAL.read_text())
        if not record.get("snapshot"):
            raise SystemExit(f"No historical snapshot saved: {record.get('outcome')}")
        path = RUN / record["snapshot"]
        if sha256(path) != record["snapshot_sha256"]:
            raise ValueError("Saved calibration checksum differs")
        if sha256(ACCEPTANCE) != record["acceptance_sha256"]:
            raise ValueError("Acceptance checksum differs from retrieval provenance")
        comparison = compare(json.loads(path.read_text()), acceptance)
        if comparison != record["comparison"]:
            raise ValueError("Saved calibration comparison differs")
    else:
        collect_source_provenance(acceptance)
        record = {
            "requested_as_of_utc": stamp(AS_OF),
            "request_time_basis": "Acceptance written timestamp interpreted as UTC, as requested for this retrieval.",
            "retrieved_at_utc": stamp(datetime.now(timezone.utc)),
            "backend": "ibm_chicago", "use_fractional_gates": True,
            "api": "QiskitRuntimeService.backend('ibm_chicago', use_fractional_gates=True).properties(refresh=True, datetime=aware_UTC_datetime)",
            "acceptance_record": ACCEPTANCE.name,
            "acceptance_sha256": sha256(ACCEPTANCE),
            "snapshot_scope": "Historical properties requested as of acceptance, not a claim about calibration throughout hardware execution.",
            "current_properties_fallback_requested": False,
        }
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            from qiskit_ibm_runtime.utils.json import RuntimeEncoder

            backend = QiskitRuntimeService().backend("ibm_chicago", use_fractional_gates=True)
            properties = backend.properties(refresh=True, datetime=AS_OF)
            if properties is None:
                raise RuntimeError("Historical properties API returned no snapshot")
            snapshot = json.loads(json.dumps(properties.to_dict(), cls=RuntimeEncoder))
            actual_time = datetime.fromisoformat(stamp(properties.last_update_date).replace("Z", "+00:00"))
            path = CALIBRATION / f"backend_properties_{actual_time.strftime('%Y-%m-%dT%H-%M-%SZ')}.json"
            if path.exists() and json.loads(path.read_text()) != snapshot:
                raise ValueError("Existing timestamp-named calibration has different contents")
            write_json(path, snapshot)
            record.update(snapshot=path.relative_to(RUN).as_posix(), snapshot_sha256=sha256(path))
            comparison = compare(snapshot, acceptance)
            record["comparison"] = comparison
            matched = all(comparison[key] for key in ("backend_matches", "last_update_matches_acceptance",
                                                      "snapshot_not_after_requested_time", "all_core_t2_values_match"))
            record["outcome"] = "matches_acceptance" if matched else "differs_from_acceptance"
        except Exception as exc:
            record["outcome"] = "retrieval_or_verification_failed"
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            write_json(RETRIEVAL, record)
            print(json.dumps({"outcome": record["outcome"], "error": record["error"]}, indent=2))
            raise SystemExit(1) from None
        write_json(RETRIEVAL, record)
    print(json.dumps({"outcome": record["outcome"], "snapshot": record["snapshot"],
                      "comparison": comparison}, indent=2))
    if record["outcome"] != "matches_acceptance":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
