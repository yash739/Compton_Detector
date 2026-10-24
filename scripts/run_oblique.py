"""Oblique incidence: modulation, yield and recovered PA/PF at several polar angles (replaces 8binsDetector_pref_testing_oblique.ipynb).

    python scripts/run_oblique.py [--energy 300] [--thetas 0,30,45,60] [--phi 45] [--photons 20000000]

Unlike the notebooks, the unpolarised reference is truly unpolarised at every angle (the old generator gave 7, 17, 33% polarisation at 30, 45, 60 deg),
photons illuminate the whole array uniformly and the pixel/fiducial selection is the same as at normal incidence. For an oblique beam the E vector is the
projection of the PA direction on the plane perpendicular to the beam. Writes results/oblique_E<E>.csv and a figure of the corrected curves.
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
    ap.add_argument("--thetas", default="0,30,45,60")
    ap.add_argument("--phi", type=float, default=45.0)
    ap.add_argument("--pas", default="0,45")
    ap.add_argument("--photons", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    thetas = [float(v) for v in a.thetas.split(",")]; pas = [float(v) for v in a.pas.split(",")]
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    s = runner.Setup()
    rows = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(pas), len(thetas), figsize=(3.6 * len(thetas), 3.2 * len(pas)), squeeze=False)
    phi = np.degrees(an.BIN_ANGLES)
    print(f"E = {a.energy} keV, beam azimuth {a.phi} deg")
    print(f"{'theta':>6s} {'PA':>5s} {'doubles':>8s} {'per 1e6':>8s} {'mu_100 A/B':>11s} {'peak deg':>9s} {'PF(0.5) out':>13s} {'PA out':>12s}")
    k = 0
    for i, th in enumerate(thetas):
        kw = dict(e_kev=a.energy, theta_inc=th, phi_inc=a.phi)
        unp = runner.run(s, 2 * a.photons, pf=0.0, seed=a.seed + k, **kw); k += 1
        for j, pa in enumerate(pas):
            ref = runner.run(s, a.photons, pf=1.0, pa_deg=pa, seed=a.seed + k, **kw); k += 1
            half = runner.run(s, a.photons, pf=0.5, pa_deg=pa, seed=a.seed + k, **kw); k += 1
            o = an.recover_pf_pa(half["hist"], unp["hist"], ref["hist"], rng=np.random.default_rng(k))
            fit = an.fit_modulation(*an.corrected_curve(ref["hist"], unp["hist"]))
            rows.append([th, a.phi, pa, ref["n_double"], 1e6 * ref["n_double"] / ref["n_fid"], o["mu_100"], fit["peak"], o["pf"], o["pf_err"], o["pa"], o["pa_err"], o["chi2_100"]])
            print(f"{th:6.1f} {pa:5.1f} {ref['n_double']:8d} {1e6 * ref['n_double'] / ref['n_fid']:8.0f} {o['mu_100']:11.3f} {fit['peak']:9.1f} {o['pf']:7.3f}+-{o['pf_err']:.3f} {o['pa']:6.1f}+-{o['pa_err']:4.1f}")
            c, sg = an.corrected_curve(ref["hist"], unp["hist"])
            ax[j, i].errorbar(phi, c, sg, fmt="o", color="k"); fine = np.linspace(0, 360, 200)
            ax[j, i].plot(fine, fit["B"] + fit["beta"][1] * np.cos(2 * np.radians(fine)) + fit["beta"][2] * np.sin(2 * np.radians(fine)), "r-")
            ax[j, i].set_title(f"theta {th:g}, PA {pa:g}: A/B {fit['mu']:.2f}, chi2 {fit['chi2']:.0f}/5", fontsize=8)
    path = os.path.join(RES, f"oblique_E{int(a.energy)}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["theta_deg", "beam_azimuth_deg", "pa_deg", "n_doubles_100pct", "doubles_per_1e6", "mu_100", "peak_deg", "pf_out_for_0.5", "pf_err", "pa_out", "pa_err", "chi2_100pct_5dof"])
        for r in rows:
            w.writerow([f"{v:.5g}" if isinstance(v, float) else v for v in r])
    fig.tight_layout(); fig.savefig(os.path.join(RES, "figures", f"oblique_E{int(a.energy)}.png"), dpi=120)
    print("wrote", path)


if __name__ == "__main__":
    main()
