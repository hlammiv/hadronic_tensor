"""htq_hw analyze with the mirror-signal guard raised from 0.02 to KAP_GUARD
(env, default 0.08): a site defines its own kappa only when the ideal t=0
connected signal exceeds the guard; the others take the median of those.
On ibm_chicago the ancilla reads <X> = +0.06 at t = 0, which leaks a
staggered offset of about 0.015 into every site; with the 0.02 guard the
sites with |sx0| = 0.03 to 0.04 then define negative kappas and drag the
median fill below the 0.05 mask."""
import os, sys
import htq_hw.analyze as A
A.KAP_GUARD = float(os.environ.get("KAP_GUARD", "0.08"))
from htq_hw.__main__ import main
sys.exit(main(sys.argv[1:]))
