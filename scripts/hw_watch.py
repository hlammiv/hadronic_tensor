"""Poll queued IBM jobs; run ibm_hardware.py analyze on each as it completes.

  PYTHONPATH=. .venv/bin/python scripts/hw_watch.py <jobfile> [<jobfile> ...]
"""

import json
import subprocess
import sys
import time

from qiskit_ibm_runtime import QiskitRuntimeService

TERMINAL = {"DONE", "CANCELLED", "ERROR"}
files = {f: None for f in sys.argv[1:]}
svc = QiskitRuntimeService()
t0 = time.time()
while time.time() - t0 < 12 * 3600 and any(v is None for v in files.values()):
    for f, state in files.items():
        if state is not None:
            continue
        jid = json.load(open(f))["job_id"]
        st = str(svc.job(jid).status())
        st = st.split(".")[-1].strip("'>").upper()
        print(f"[{time.time()-t0:6.0f}s] {jid}: {st}", flush=True)
        if any(t in st for t in TERMINAL):
            files[f] = st
            if "DONE" in st:
                subprocess.run([".venv/bin/python", "scripts/ibm_hardware.py",
                                "analyze", f], env={"PYTHONPATH": ".",
                                                    "PATH": "/usr/bin:/bin"})
    if any(v is None for v in files.values()):
        time.sleep(600)
print("watch done:", files, flush=True)
