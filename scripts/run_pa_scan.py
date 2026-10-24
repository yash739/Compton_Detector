"""Recovered polarisation angle against the injected angle for a 100% polarised beam, 8-bin cosine fit after the geometry correction.

    python scripts/run_pa_scan.py [--energy 300] [--photons 20000000] [--step 10] [--seed 5]

Tests the bias of the cosine fit to the pixelated 8-bin curve. The notebooks this replaces tested only 0, 45 and 90 degrees, which are symmetry axes of the square pixels
where the bias vanishes by symmetry. Writes results/pa_scan_E<energy>.csv and results/figures/pa_scan_E<energy>.png.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from czt_mc import analysis as an, runner  # noqa: E402

RES = os.path.join(HERE, "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", type=float, default=300.0)
    ap.add_argument("--photons", type=int, default=20_000_000)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    s = runner.Setup()
    unp = runner.run(s, 2 * a.photons, a.energy, pf=0.0, seed=a.seed)
    rows = []
    print(f"E = {a.energy} keV, {unp['n_double']} unpolarised doubles")
    for k, pa in enumerate(np.arange(0.0, 180.0, a.step)):
        pol = runner.run(s, a.photons, a.energy, pf=1.0, pa_deg=pa, seed=a.seed + 1 + k)
        fit = an.fit_modulation(*an.corrected_curve(pol["hist"], unp["hist"]))
        pa_rec = (fit["peak"] - 90.0) % 180.0
        d = ((pa_rec - pa + 90.0) % 180.0) - 90.0
        rows.append([pa, pa_rec, d, fit["mu"], pol["n_double"]])
        print(f"  injected PA {pa:6.1f}: recovered {pa_rec:6.1f}  offset {d:+5.1f}  modulation A/B {fit['mu']:.3f}  doubles {pol['n_double']}")
    path = os.path.join(RES, f"pa_scan_E{int(a.energy)}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pa_injected_deg", "pa_recovered_deg", "offset_deg", "mu_100", "n_doubles"])
        for r in rows:
            w.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in r])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    r = np.array(rows, float)
    ax[0].plot(r[:, 0], r[:, 2], "o-"); ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_xlabel("injected PA (deg)"); ax[0].set_ylabel("recovered - injected (deg)"); ax[0].set_title(f"cosine-fit PA bias, {a.energy:.0f} keV")
    ax[1].plot(r[:, 0], r[:, 3], "o-"); ax[1].set_xlabel("injected PA (deg)"); ax[1].set_ylabel("modulation A/B of the corrected curve"); ax[1].set_title("100% polarised")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "figures", f"pa_scan_E{int(a.energy)}.png"), dpi=130)
    print("wrote", path)


if __name__ == "__main__":
    main()
