"""Monitor this submitted diagnostic and collect it; never submit hardware jobs."""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

from qiskit_ibm_runtime import QiskitRuntimeService


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
PYTHON = REPO / '.venv/bin/python'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(name, value):
    path = ROOT / name
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def finalize(rows, service):
    # A DONE job's accounting may arrive after its results. The existing
    # collector resumes from saved metrics, so refresh incomplete snapshots.
    for row in rows:
        path = ROOT / 'metrics' / row['variant'] / f"htq_metrics_{row['job_id']}.json"
        if not path.exists():
            continue
        snapshot = json.loads(path.read_text())
        seconds = (snapshot.get('metrics', {}).get('usage') or {}).get('quantum_seconds')
        if seconds is None:
            metrics = service.job(row['job_id']).metrics()
            seconds = (metrics.get('usage') or {}).get('quantum_seconds')
            if seconds is None:
                raise RuntimeError(f"{row['job_id']}: measured usage is still pending")
            save(str(path.relative_to(ROOT)), {**snapshot, 'metrics': metrics})
    command = [str(PYTHON), '-u', 'scripts/collect_hw_results.py', str(ROOT)]
    subprocess.run(command, cwd=REPO, check=True)
    subprocess.run(command + ['--check-only'], cwd=REPO, check=True)
    totals = defaultdict(lambda: {'jobs': 0, 'pubs': 0, 'shots': 0, 'qpu_seconds': 0.0})
    timestamps = []
    for row in rows:
        variant, jid = row['variant'], row['job_id']
        meta = json.loads((ROOT / 'jobs' / variant / f'htq_job_{jid}.json').read_text())
        metric = json.loads((ROOT / 'metrics' / variant / f'htq_metrics_{jid}.json').read_text())
        seconds = (metric.get('metrics', {}).get('usage') or {}).get('quantum_seconds')
        if seconds is None:
            raise ValueError(f'{jid}: measured QPU usage is not available yet')
        item = totals[variant]
        item['jobs'] += 1
        item['pubs'] += len(meta['pub_names'])
        item['shots'] += sum(meta['shots'].values())
        item['qpu_seconds'] += float(seconds)
        timestamps.append({'variant': variant, 'job_id': jid, **metric['metrics']['timestamps']})
    assert sum(v['jobs'] for v in totals.values()) == 4
    assert sum(v['pubs'] for v in totals.values()) == 14
    assert sum(v['shots'] for v in totals.values()) == 840000
    raw_files = sorted((ROOT / 'raw').glob('*/htq_bits_*.npz'))
    assert len(raw_files) == 4
    checksums = [f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}' for path in raw_files]
    (ROOT / 'raw_sha256.txt').write_text('\n'.join(checksums) + '\n')
    summary = {
        'status': 'complete', 'completed_utc': now(), 'variants': dict(totals),
        'total_jobs': 4, 'total_pubs': 14, 'total_shots': 840000,
        'total_qpu_seconds': sum(v['qpu_seconds'] for v in totals.values()),
        'twirling_outcome': 'Skipped after Runtime error 1519; zero QPU usage',
        'raw_validation': 'PASS', 'raw_sha256_files': 4, 'execution_timestamps': timestamps,
    }
    save('completion_summary.json', summary)
    readme = ROOT / 'README.md'
    prose = readme.read_text()
    for variant in ('mirror_diag_control', 'mirror_diag_eps05'):
        lines = prose.splitlines()
        for index, line in enumerate(lines):
            if line.startswith(f'| `{variant}` |'):
                lines[index] = line.replace('Queued; see `job_status.json`',
                    f"Completed; {totals[variant]['qpu_seconds']:g} QPU seconds")
        prose = '\n'.join(lines) + '\n'
    prose = prose.replace(
        'The four supported jobs are being monitored. Raw results and final usage are\npending. From the repository root, the existing collector can resume safely:',
        f"All four supported jobs completed. Their 14 PUBs contain 840,000 shots and\n"
        f"used {summary['total_qpu_seconds']:g} measured QPU seconds in total. The raw\n"
        'files passed the collector checks; `raw_sha256.txt` records their hashes.\n'
        '`completion_summary.json` records totals and execution timestamps.\n'
        'From the repository root, collection and offline verification are repeatable:')
    readme.write_text(prose)
    save('collection_state.json', {'status': 'complete', 'updated_utc': now(),
                                 'summary': 'completion_summary.json'})
    archive = ROOT.parent / 'mirror_diag_chicago_2026-11-08.zip'
    temporary = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(ROOT.rglob('*')):
            if not path.is_file() or path.name.endswith(('.tmp', '.pid')):
                continue
            if '__pycache__' in path.parts or path.name == 'collector.log':
                continue
            bundle.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    temporary.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('Archive integrity check failed')
    print(f'{now()} COMPLETE: {archive}', flush=True)


def main():
    (ROOT / 'collector.pid').write_text(str(os.getpid()) + '\n')
    metas = [(p.parent.name, json.loads(p.read_text()))
             for p in sorted((ROOT / 'jobs').glob('*/htq_job_*.json'))]
    assert len(metas) == 4
    assert all(m.get('cancelled') is False and m.get('guard') == 'ok' for _, m in metas)
    service = None
    previous = None
    while True:
        try:
            if service is None:
                service = QiskitRuntimeService()
            rows = []
            for variant, meta in metas:
                job = service.job(meta['job_id'])
                status = job.status()
                status = getattr(status, 'name', str(status)).split('.')[-1].upper()
                row = {'variant': variant, 'job_index': meta['job_index'],
                       'job_id': meta['job_id'], 'status': status}
                if status in {'ERROR', 'CANCELLED'}:
                    row.update(error_message=job.error_message(), metrics=job.metrics())
                rows.append(row)
        except Exception as exc:
            save('collection_state.json', {'status': 'retrying_connection',
                 'updated_utc': now(), 'error': f'{type(exc).__name__}: {exc}'})
            print(f'{now()} Connection failed; retrying: {type(exc).__name__}: {exc}', flush=True)
            service = None
            time.sleep(60)
            continue
        save('job_status.json', {'checked_utc': now(), 'jobs': rows,
                                'excluded_failure_record': 'twirling_rejection.json'})
        states = [(row['job_id'], row['status']) for row in rows]
        if states != previous:
            print(now(), states, flush=True)
            previous = states
        if any(row['status'] in {'ERROR', 'CANCELLED'} for row in rows):
            raise RuntimeError('A supported job failed; see job_status.json')
        if all(row['status'] == 'DONE' for row in rows):
            save('collection_state.json', {'status': 'collecting', 'updated_utc': now()})
            for attempt in range(1, 6):
                try:
                    finalize(rows, service)
                    return
                except Exception as exc:
                    if attempt == 5:
                        raise
                    save('collection_state.json', {'status': 'retrying_collection',
                         'updated_utc': now(), 'attempt': attempt,
                         'error': f'{type(exc).__name__}: {exc}'})
                    print(f'{now()} Collection attempt {attempt} failed; retrying: {exc}', flush=True)
                    time.sleep(60)
        save('collection_state.json', {'status': 'waiting_for_hardware', 'updated_utc': now()})
        time.sleep(60)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save('collection_state.json', {'status': 'failed', 'updated_utc': now(),
                                      'error': f'{type(exc).__name__}: {exc}'})
        raise
