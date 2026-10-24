"""Print the validation of the Klein-Nishina sampler: theta marginal chi^2 against the analytic cross section and the azimuthal modulation per theta bin.

    python scripts/validate_kn_sampling.py [--n 2000000]

(the same checks run as unit tests in tests/test_physics.py; this prints the numbers.) Replaces Testing Klein-Nishina Sampling.py, which only plotted histograms.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from czt_mc import physics as ph  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2_000_000); a = ap.parse_args()
    rng = np.random.default_rng(2)
    for energy in (60.0, 100.0, 200.0, 300.0, 600.0):
        n = a.n
        e = np.full(n, energy); k = np.tile([0.0, 0.0, 1.0], (n, 1)); pol = np.tile([1.0, 0.0, 0.0], (n, 1))
        mu, phi, *_ = ph.sample_compton(e, k, pol, rng)
        edges = np.linspace(-1, 1, 21)
        obs = np.histogram(mu, edges)[0]
        grid = np.linspace(-1, 1, 400001); pdf = ph.kn_unpolarised_pdf_cos(energy, grid)
        cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))]); cdf /= cdf[-1]
        exp = n * np.diff(np.interp(edges, grid, cdf))
        chi2 = np.sum((obs - exp) ** 2 / exp)
        line = f"E = {energy:5.0f} keV: cos(theta) marginal chi2 = {chi2:5.1f} (19 dof);  <cos2phi>_theta against -m/2: "
        for lo, hi in ((-0.9, -0.5), (-0.2, 0.2), (0.5, 0.9)):
            s = (mu > lo) & (mu < hi)
            meas = np.mean(np.cos(2 * phi[s])); err = np.std(np.cos(2 * phi[s])) / np.sqrt(s.sum()); pred = -0.5 * np.mean(ph.kn_modulation(energy, mu[s]))
            line += f"[{lo:+.1f},{hi:+.1f}] {meas:+.4f}+-{err:.4f} vs {pred:+.4f};  "
        print(line)
    # old rejection loop: acceptance on the first draw
    t = rng.uniform(0, np.pi, 400000); p = rng.uniform(0, 2 * np.pi, 400000)
    for energy in (100.0, 300.0):
        eps0 = energy / ph.ME_KEV
        eps = eps0 / (1 + eps0 * (1 - np.cos(t))); r = eps / eps0
        dsig = 0.5 * r ** 2 * (1 / r + r - 2 * np.sin(t) ** 2 * np.cos(p) ** 2) * np.sin(t)
        t1 = (1 + eps0) / eps0 ** 3; t2 = 2 * eps0 * (1 + eps0) / (1 + 2 * eps0); t3 = np.log(1 + 2 * eps0); t4 = (1 + 3 * eps0) / (1 + 2 * eps0) ** 2
        sig = 2 * np.pi * (t1 * (t2 - t3) + t3 / (2 * eps0) - t4)
        print(f"old notebooks: P(first draw accepted) at {energy:.0f} keV = {np.mean(rng.uniform(0, 1, t.size) < dsig / sig):.4f} (= 1/(2 pi^2) = {1 / (2 * np.pi ** 2):.4f}); they discarded the other 95% of photons")


if __name__ == "__main__":
    main()
