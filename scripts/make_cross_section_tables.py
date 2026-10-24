"""Build the tabulated cross sections and K-fluorescence data of Cd(0.9)Zn(0.1)Te used by czt_mc.

    python scripts/make_cross_section_tables.py        (needs xraydb: pip install xraydb)

Source: the Elam et al. tables as packaged in xraydb; validated against the NIST element tables to 0.1% (scripts/validate_cross_sections.py).
Writes czt_mc/data/czt_xs.csv        energy grid 5-800 keV (log grid plus points either side of the Zn, Cd, Te K edges):
                                     linear attenuation (1/cm, rho = 5.78 g/cm3) for photoabsorption, incoherent (Compton) and coherent scattering,
                                     the photoabsorption contribution of each element, and the K-shell fraction of that element's photoabsorption (1 - 1/jump ratio above the edge)
       czt_mc/data/czt_kfluor_lines.csv   K fluorescence lines of Cd, Zn, Te: energy, relative intensity, fluorescence yield, and CZT total attenuation at the line energy
"""
import csv
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import xraydb  # noqa: E402
from czt_mc.constants import ELEMENTS, FORMULA, MASS_FRACTIONS, RHO_G_CM3  # noqa: E402

OUT = os.path.join(HERE, "..", "czt_mc", "data")


def main():
    xraydb.add_material("czt_mc", FORMULA, RHO_G_CM3)
    edges = {el: xraydb.xray_edge(el, "K") for el in ELEMENTS}
    grid = list(np.geomspace(5.0, 800.0, 520))
    for el, ed in edges.items():
        e = ed.energy / 1e3
        grid += [e - 1e-3, e + 1e-3]
    grid = np.array(sorted(grid))

    cols = ["E_keV", "mu_photo", "mu_incoh", "mu_coh"] + [f"mu_photo_{el}" for el in ELEMENTS] + [f"fK_{el}" for el in ELEMENTS]
    rows = []
    for e in grid:
        ev = e * 1e3
        r = [e, *(float(xraydb.material_mu("czt_mc", ev, kind=k)) for k in ("photo", "incoh", "coh"))]
        r += [RHO_G_CM3 * MASS_FRACTIONS[el] * float(xraydb.mu_elam(el, ev, kind="photo")) for el in ELEMENTS]
        r += [(1.0 - 1.0 / edges[el].jump_ratio) if ev > edges[el].energy else 0.0 for el in ELEMENTS]
        rows.append(r)
    with open(os.path.join(OUT, "czt_xs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([f"{v:.8g}" for v in r])

    lrows = []
    for el in ELEMENTS:
        lines = xraydb.xray_lines(el, "K")
        tot = sum(v.intensity for v in lines.values())
        for name, v in lines.items():
            if v.intensity <= 0:
                continue
            ev = v.energy
            mu_tot = float(xraydb.material_mu("czt_mc", ev, kind="total"))
            mu_ph = float(xraydb.material_mu("czt_mc", ev, kind="photo"))
            lrows.append([el, name, ev / 1e3, v.intensity / tot, edges[el].fyield, edges[el].energy / 1e3, mu_tot, mu_ph / mu_tot])
    with open(os.path.join(OUT, "czt_kfluor_lines.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["element", "line", "E_keV", "rel_intensity", "fluor_yield_K", "K_edge_keV", "mu_total_cm", "photo_fraction"])
        for r in lrows:
            w.writerow([r[0], r[1]] + [f"{v:.8g}" for v in r[2:]])
    print(f"wrote {len(rows)} energies and {len(lrows)} K lines to {os.path.abspath(OUT)}")
    for el in ELEMENTS:
        print(f"  {el}: K edge {edges[el].energy / 1e3:.3f} keV, omega_K {edges[el].fyield:.3f}, jump ratio {edges[el].jump_ratio:.2f}")
    for r in lrows:
        print(f"  {r[0]} {r[1]:4s} {r[2]:7.3f} keV  I={r[3]:.3f}  mu_tot(CZT)={r[6]:8.1f} /cm  1/mu = {1e4 / r[6]:6.1f} um  photo fraction {r[7]:.3f}")


if __name__ == "__main__":
    main()
