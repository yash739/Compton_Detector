"""Normal-incidence modulation curves: raw and geometry-corrected 8-bin histograms for an unpolarised and 100% polarised beams (replaces the 8binsDetector_pref_*.ipynb notebooks).

    python scripts/run_modulation.py [--energy 300] [--pas 0,45,90] [--photons 20000000]

Writes results/modulation_E<energy>.csv (counts, corrected curve, fit per PA) and results/figures/modulation_E<energy>.png.
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
    ap.add_argument("--pas", default="0,45,90")
    ap.add_argument("--photons", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    pas = [float(p) for p in a.pas.split(",")]
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    s = runner.Setup()
    unp = runner.run(s, a.photons, a.energy, pf=0.0, seed=a.seed)
    print(f"E = {a.energy} keV: unpolarised: {unp['n_double']} doubles ({1e6 * unp['n_double'] / unp['n_fid']:.0f} per 10^6 photons on the fiducial block), {unp['n_single']} singles")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(pas), figsize=(4.2 * len(pas), 6.4), squeeze=False)
    rows = []
    phi = np.degrees(an.BIN_ANGLES)
    fine = np.linspace(0, 360, 400)
    for j, pa in enumerate(pas):
        pol = runner.run(s, a.photons, a.energy, pf=1.0, pa_deg=pa, seed=a.seed + 1 + j)
        c, sg = an.corrected_curve(pol["hist"], unp["hist"])
        fit = an.fit_modulation(c, sg)
        rec_pa = (fit["peak"] - 90.0) % 180.0
        print(f"  PA {pa:5.1f}: doubles {pol['n_double']}, raw {pol['hist'].astype(int)}, corrected modulation A/B = {fit['mu']:.3f}, recovered PA {rec_pa:.1f}, chi2/dof {fit['chi2']:.1f}/5")
        rows.append([pa, *pol["hist"].astype(int), *np.round(c, 2), fit["mu"], rec_pa, fit["chi2"]])
        ax[0, j].bar(phi, pol["hist"], width=40, alpha=0.6, label="100% polarised"); ax[0, j].bar(phi, unp["hist"] * pol["n_double"] / unp["n_double"], width=18, color="C1", label="unpolarised (scaled)")
        ax[0, j].set_title(f"raw, E-vector at {pa:.0f} deg"); ax[0, j].legend(fontsize=7)
        ax[1, j].errorbar(phi, c, sg, fmt="o", color="k")
        ax[1, j].plot(fine, fit["B"] + fit["beta"][1] * np.cos(2 * np.radians(fine)) + fit["beta"][2] * np.sin(2 * np.radians(fine)), "r-")
        ax[1, j].set_title(f"corrected: A/B = {fit['mu']:.3f}, peak {fit['peak']:.1f} deg"); ax[1, j].set_xlabel("direction of the double (deg)")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "figures", f"modulation_E{int(a.energy)}.png"), dpi=130)
    path = os.path.join(RES, f"modulation_E{int(a.energy)}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pa_deg"] + [f"raw_{int(p)}" for p in phi] + [f"corr_{int(p)}" for p in phi] + ["mu_A_over_B", "pa_recovered_deg", "chi2_5dof"])
        w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
