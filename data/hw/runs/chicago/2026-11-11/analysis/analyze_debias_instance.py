"""htq_hw analyze, de-biased ancilla, restricted to ONE twirl instance of every
pub (env HTQ_INSTANCE = 0..K-1, HTQ_K default 4): rows [i n/K, (i+1) n/K) of
each merged array (htq_hw.campaign.fetch concatenates instances in order).
Each single-instance reduction is one realization of the twirl; the scatter
of a result over the K reductions, over sqrt(K), is an empirical error of the
K-instance result that includes the twirl-sampling noise the shot errors
omit.  2026-11-12."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_debias = os.path.join(_here, "..", "..", "2026-11-06", "analysis", "2026-11-07_debias", "analyze_debias.py")
_src = open(_debias).read().split("from htq_hw.__main__ import main")[0]
exec(compile(_src, _debias, "exec"))
_I, _K = int(os.environ["HTQ_INSTANCE"]), int(os.environ.get("HTQ_K", "4"))
_Base = A.JobBits


class _InstanceBits(_Base):
    def __init__(self, bits, *a, **k):
        cut = {}
        for name, arr in bits.items():
            n = len(arr) // _K
            cut[name] = arr[_I * n:(_I + 1) * n]
        super().__init__(cut, *a, **k)


A.JobBits = _InstanceBits
from htq_hw.__main__ import main   # noqa: E402
sys.exit(main(sys.argv[1:]))
