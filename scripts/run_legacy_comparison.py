"""What the old notebooks' conventions did, measured inside the corrected framework.

    python scripts/run_legacy_comparison.py [--photons 20000000]

(1) attenuation: the notebooks used 2.64 /cm at every energy and made every interaction a Compton scatter. Compare the event yield and the modulation A/B of the
    corrected 100% polarised curve at 200 and 300 keV with the real energy-dependent attenuation, photoabsorption and K fluorescence.
(2) histogram bins: np.histogram(range=(-pi/4, 8 pi/4), bins=8) has bins of 9 pi/32 and the fit used their centres (spacing 0.884 rad) instead of 0, 45, ... deg.
(3) ratio cut: the old E_high/E_low <= 6 selection against none.
Writes results/legacy_comparison.csv.
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from czt_mc import analysis as an, runner  # noqa: E402

RES = os.path.join(HERE, "..", "results")


def wrong_bin_fit(h_pol, h_unp):
    """the notebooks' fit: A cos(2 (x - phi)) + B at the centres of np.histogram(range=(-pi/4, 8pi/4)) bins"""
    edges = np.linspace(-np.pi / 4, 2 * np.pi, 9); x = 0.5 * (edges[1:] + edges[:-1])
    corr = h_pol / h_unp
    corr = corr / corr.mean()                                     # unit mean: well-scaled starting values (only A/B matters)
    best = None
    for p0 in (0.3, 1.5, 2.5):
        po, _ = curve_fit(lambda t, A, ph, B: A * np.cos(2 * (t - ph)) + B, x, corr, p0=[0.2, p0, 1.0], bounds=([1e-9, 0, 0.0], [10.0, np.pi, 10.0]), max_nfev=5000)
        r = np.sum((po[0] * np.cos(2 * (x - po[1])) + po[2] - corr) ** 2)
        if best is None or r < best[0]:
            best = (r, po)
    return best[1][0] / best[1][2], np.degrees(best[1][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photons", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    s = runner.Setup()
    rows = []
    print(f"{'E':>5s} {'PA':>4s} {'variant':<44s} {'doubles/1e6':>11s} {'A/B':>7s} {'peak':>7s}")
    for e in (200.0, 300.0):
        for pa in (0.0, 45.0):
            def sim(**kw):
                unp = runner.run(s, a.photons, e, pf=0.0, seed=a.seed, **kw)
                pol = runner.run(s, a.photons, e, pf=1.0, pa_deg=pa, seed=a.seed + 1, **kw)
                return unp, pol
            for label, kw in (("real attenuation + K fluorescence (this package)", {}),
                              ("same, no upper ratio limit vs E_high/E_low <= 6", {"ratio_max": 6.0}),
                              ("old attenuation 2.64 /cm, always Compton", {"mu_const": 2.64}),
                              ("real attenuation, fluorescence switched off", {"fluorescence": False})):
                unp, pol = sim(**kw)
                fit = an.fit_modulation(*an.corrected_curve(pol["hist"], unp["hist"]))
                rows.append([e, pa, label, 1e6 * pol["n_double"] / pol["n_fid"], fit["mu"], fit["peak"]])
                print(f"{e:5.0f} {pa:4.0f} {label:<44s} {rows[-1][3]:11.0f} {fit['mu']:7.3f} {fit['peak']:7.1f}")
                if not kw:
                    mu_w, pk_w = wrong_bin_fit(pol["hist"], unp["hist"])
                    rows.append([e, pa, "same histograms fitted at the notebooks' bin centres", rows[-1][3], mu_w, pk_w])
                    print(f"{e:5.0f} {pa:4.0f} {'same histograms fitted at the notebooks bin centres':<44s} {rows[-1][3]:11.0f} {mu_w:7.3f} {pk_w:7.1f}")
    path = os.path.join(RES, "legacy_comparison.csv")
    os.makedirs(RES, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["E_keV", "PA_deg", "variant", "doubles_per_1e6_photons", "mu_A_over_B", "fitted_peak_deg"])
        for r in rows:
            w.writerow([f"{v:.5g}" if isinstance(v, float) else v for v in r])
    print("wrote", path)


if __name__ == "__main__":
    main()
