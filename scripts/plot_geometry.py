"""Geometry figure: the detector, the beam direction k, the E vector and the preferred scattering direction (replaces visualisation_of_scattering.py).

    python scripts/plot_geometry.py [--theta 30] [--phi 45] [--pa 20]

Conventions of czt_mc: the E vector is the projection of the PA direction (cos pa, sin pa, 0) on the plane perpendicular to the beam; Compton scattering prefers the direction
perpendicular to E within that plane, k x E. For normal incidence the preferred scattering azimuth is PA + 90 deg.
Writes results/figures/geometry.png.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from czt_mc import constants as C, physics as ph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=30.0); ap.add_argument("--phi", type=float, default=45.0); ap.add_argument("--pa", type=float, default=20.0)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, p = np.radians(a.theta), np.radians(a.phi)
    k = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    d = np.array([np.cos(np.radians(a.pa)), np.sin(np.radians(a.pa)), 0.0]); e = d - (d @ k) * k; e /= np.linalg.norm(e)
    pref = np.cross(k, e)
    assert abs(e @ k) < 1e-12 and abs(pref @ k) < 1e-12 and abs(pref @ e) < 1e-12
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
    h = 0.5 * C.N_PIX * C.PITCH_CM
    xs = np.linspace(-h, h, C.N_PIX + 1)
    for x in xs:
        ax.plot([x, x], [-h, h], [0, 0], color="0.7", lw=0.6); ax.plot([-h, h], [x, x], [0, 0], color="0.7", lw=0.6)
    for x in xs[[0, -1]]:
        pass
    scale = 1.2
    org = np.array([0.0, 0.0, 0.0])
    for v, c, lab in ((k, "tab:red", "beam k"), (e, "tab:green", "E vector"), (pref, "tab:blue", "preferred scattering k x E"), (d, "0.4", "PA direction in the detector plane")):
        ax.quiver(*org, *(scale * v), color=c, arrow_length_ratio=0.12, label=lab)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (depth)"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"beam theta {a.theta:g}, phi {a.phi:g}; PA {a.pa:g} deg")
    os.makedirs(os.path.join(HERE, "..", "results", "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "..", "results", "figures", "geometry.png"), dpi=130)
    print("E vector", np.round(e, 4), " preferred scattering", np.round(pref, 4), " (both perpendicular to k and to each other)")


if __name__ == "__main__":
    main()
