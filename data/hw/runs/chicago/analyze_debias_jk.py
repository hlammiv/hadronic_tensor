"""htq_hw analyze, de-biased ancilla (analyze_debias.py), with one group of
twirl instances left out of every pre-twirled pub: the delete-a-group
jackknife over twirl draws, adopted 2026-11-19 as the standard error of any
pre-twirled reduction (the shot error misses the scatter between draws,
which was 2-4x larger on 2026-11-11).

    HTQ_JK_GROUP=g  HTQ_JK_GROUPS=G  (default G = 8, capped at the instance count K)

The K instances of a bits file (``twirl_instances``; fetch concatenates them
in instance order) are split into G contiguous groups and group g is
removed.  Per-instance row counts come from ``twirl_counts`` when the file
has them (fetch since 953731b) and from an equal split otherwise.  A file
without ``twirl_instances`` (untwirled, or Runtime-twirled like Phoenix) is
left whole, and that is reported: it has no draws to resample.

Run it through jackknife_run.sh, which reduces each g with reduce_run.sh
and tabulates err^2 = (G-1)/G sum_g (A_-g - mean)^2.
"""
import json
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_debias = os.path.join(_here, "2026-11-06", "analysis", "2026-11-07_debias", "analyze_debias.py")
_src = open(_debias).read().split("from htq_hw.__main__ import main")[0]
exec(compile(_src, _debias, "exec"))

_G_REQ = int(os.environ.get("HTQ_JK_GROUPS", "8"))
_g = int(os.environ["HTQ_JK_GROUP"])
_untwirled = []


def _drop_rows(arr, counts, groups, g):
    edges = np.concatenate([[0], np.cumsum(counts)])
    k = len(counts)
    bounds = np.linspace(0, k, groups + 1).round().astype(int)       # instance index bounds per group
    lo, hi = edges[bounds[g]], edges[bounds[g + 1]]
    return np.concatenate([arr[:lo], arr[hi:]], axis=0)


def load_job_bits(bits_path, lat, eta=A.ETA, prefix=None, card=None, postselect=0, center=None, rounds=False):
    z = np.load(bits_path, allow_pickle=True)
    K = int(z["twirl_instances"]) if "twirl_instances" in z.files else 0
    counts = json.loads(str(z["twirl_counts"])) if "twirl_counts" in z.files else {}
    G = min(_G_REQ, K) if K else 0
    if K and not 0 <= _g < G:
        raise SystemExit(f"HTQ_JK_GROUP={_g} outside 0..{G - 1} for {bits_path} (K = {K})")
    bits = {}
    for k in z.files:
        if z[k].dtype != np.uint8 or z[k].ndim != 2:
            continue
        pre, bare = A.split_prefix(k)
        keep = A.card_of(pre) == card if card else (prefix or "") == pre
        if not keep:
            continue
        arr = z[k]
        if K:
            c = counts.get(k) or [len(arr) // K] * K
            if sum(c) != len(arr):
                raise SystemExit(f"{bits_path}:{k}: instance counts {sum(c)} != rows {len(arr)}")
            arr = _drop_rows(arr, c, G, _g)
        bits[bare] = arr
    if not K:
        _untwirled.append(os.path.basename(bits_path))
    job_id = str(z["job_id"]) if "job_id" in z.files else os.path.basename(bits_path)
    return A.JobBits(bits, lat, job_id, eta, postselect=postselect, center=center, rounds=rounds)


A.load_job_bits = load_job_bits
import atexit  # noqa: E402
atexit.register(lambda: _untwirled and print(f"jackknife: {len(_untwirled)} file(s) without twirl "
                                             f"instances left whole: {_untwirled[:3]}", file=sys.stderr))
from htq_hw.__main__ import main  # noqa: E402
sys.exit(main(sys.argv[1:]))
