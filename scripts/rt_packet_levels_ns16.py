"""Wavepacket real-time current correlators and two-meson levels at
N_s = 16 (N_x = 8): the N_s = 12 script with the volume changed.  See
scripts/rt_packet_levels_ns12.py for the method and the output format.

    PYTHONPATH=. python scripts/rt_packet_levels_ns16.py [--T 200 --dt 0.1]

Outputs: data/rt_packet_levels_ns16.{npz,json}, data/rt_packet_levels.pdf.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_packet_levels_ns12 import run, make_figure  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=200.0)
    ap.add_argument("--dt", type=float, default=0.1)
    a = ap.parse_args()
    run(16, a.T, a.dt)
    make_figure()
