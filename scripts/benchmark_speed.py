"""Throughput of the simulation: million photons per second against the number of worker processes.

    python scripts/benchmark_speed.py [--photons 6000000] [--workers 1,2,4,6]

Reference measurement before the optimisations (reconstruct over every photon, separate table lookups): 0.37 M photons/s on one core.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from czt_mc import runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photons", type=int, default=6_000_000)
    ap.add_argument("--workers", default="1,2,4,6")
    ap.add_argument("--energy", type=float, default=300.0)
    a = ap.parse_args()
    s = runner.Setup()
    print(f"{os.cpu_count()} CPUs visible; {a.photons} photons at {a.energy} keV")
    t1 = None
    for w in [int(v) for v in a.workers.split(",")]:
        t = time.time()
        r = runner.run(s, a.photons, a.energy, pf=1.0, pa_deg=20.0, seed=1, workers=w)
        dt_ = time.time() - t
        t1 = t1 or dt_
        print(f"  {w:2d} workers: {dt_:6.2f} s  {a.photons / dt_ / 1e6:5.2f} M photons/s  speedup {t1 / dt_:4.2f}x  ({r['n_double']} doubles)")


if __name__ == "__main__":
    main()
