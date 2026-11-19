"""htq_hw analyze, de-biased ancilla, with ONE twirl instance of every pub
left out (env HTQ_DROP = 0..K-1, HTQ_K default 4): rows [i n/K, (i+1) n/K)
of each merged array are removed (htq_hw.campaign.fetch concatenates the
instances in order).  The K leave-one-out reductions give the delete-one
jackknife error of the pooled K-instance result,
    err^2 = (K-1)/K * sum_i (theta_-i - mean theta_-)^2,
which, unlike the shot error, includes the scatter between twirl draws and
handles the nonlinear map/asymmetry correctly.  2026-11-14."""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_debias = os.path.join(_here, "..", "..", "2026-11-06", "analysis", "2026-11-07_debias", "analyze_debias.py")
_src = open(_debias).read().split("from htq_hw.__main__ import main")[0]
exec(compile(_src, _debias, "exec"))
_D, _K = int(os.environ["HTQ_DROP"]), int(os.environ.get("HTQ_K", "4"))
_Base = A.JobBits


class _DropBits(_Base):
    def __init__(self, bits, *a, **k):
        cut = {}
        for name, arr in bits.items():
            n = len(arr) // _K
            cut[name] = np.concatenate([arr[:_D * n], arr[(_D + 1) * n:]], axis=0)
        super().__init__(cut, *a, **k)


A.JobBits = _DropBits
from htq_hw.__main__ import main   # noqa: E402
sys.exit(main(sys.argv[1:]))
