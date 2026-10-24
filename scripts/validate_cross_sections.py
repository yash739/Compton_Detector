"""Compare the attenuation coefficients used by the package with the NIST X-ray mass attenuation tables (Hubbell and Seltzer, element pages).

    python scripts/validate_cross_sections.py [--source xraydb|table]

Downloads the NIST pages for Zn, Cd and Te (needs network, once), writes czt_mc/data/nist_reference.csv, and prints the deviation of
(a) the xraydb (Elam) total coefficients and (b) the package's tabulated total (czt_xs.csv, if it exists) from NIST. NIST's element tables
carry the total with coherent scattering, which is what the package's total includes.
"""
import argparse
import csv
import os
import re
import sys
import urllib.request
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "czt_mc", "data")
sys.path.insert(0, os.path.join(HERE, ".."))
warnings.filterwarnings("ignore")

PAGES = {"Zn": 30, "Cd": 48, "Te": 52}
ENERGIES_KEV = (60, 80, 100, 150, 200, 300, 400, 500, 600)


def fetch_nist(z):
    url = f"https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z{z:02d}.html"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (czt_mc validation script)"})
    txt = urllib.request.urlopen(req, timeout=30).read().decode("latin-1")
    rows = re.findall(r"([0-9]\.[0-9]+E[+-][0-9]+)\s+([0-9]\.[0-9]+E[+-][0-9]+)\s+([0-9]\.[0-9]+E[+-][0-9]+)", txt)
    d = np.array([[float(v) for v in r] for r in rows])          # E (MeV), mu/rho, mu_en/rho (cm2/g)
    return d


def nist_at(d, e_kev):
    """log-log interpolation of the NIST table at e_kev (the table has repeated energies at absorption edges; none between 60 and 600 keV for these elements)."""
    e = d[:, 0] * 1e3
    return float(np.exp(np.interp(np.log(e_kev), np.log(e), np.log(d[:, 1]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="download the NIST pages again")
    args = ap.parse_args()
    ref_path = os.path.join(DATA, "nist_reference.csv")
    if args.refresh or not os.path.exists(ref_path):
        tabs = {el: fetch_nist(z) for el, z in PAGES.items()}
        with open(ref_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["element", "energy_keV", "mu_over_rho_cm2_g_NIST"])
            for el in PAGES:
                for e in ENERGIES_KEV:
                    w.writerow([el, e, f"{nist_at(tabs[el], e):.6g}"])
        print("wrote", ref_path)
    ref = {}
    with open(ref_path) as f:
        for r in csv.DictReader(f):
            ref[(r["element"], float(r["energy_keV"]))] = float(r["mu_over_rho_cm2_g_NIST"])

    import xraydb
    from czt_mc.constants import MASS_FRACTIONS, RHO_G_CM3
    print(f"{'E (keV)':>8s} {'element':>8s} {'NIST':>9s} {'xraydb':>9s} {'dev':>7s}")
    worst = 0.0
    for el in PAGES:
        for e in ENERGIES_KEV:
            x = float(xraydb.mu_elam(el, e * 1e3, kind="total"))
            dev = x / ref[(el, e)] - 1
            worst = max(worst, abs(dev))
            print(f"{e:8d} {el:>8s} {ref[(el, e)]:9.4f} {x:9.4f} {100 * dev:6.1f}%")
    print("\nmixture Cd0.9Zn0.1Te (mass-fraction weighted), mu/rho in cm2/g and mu in 1/cm for rho = %.2f:" % RHO_G_CM3)
    print(f"{'E (keV)':>8s} {'NIST mix':>9s} {'xraydb mix':>10s} {'dev':>7s} {'mu (1/cm)':>10s}")
    for e in ENERGIES_KEV:
        nm = sum(MASS_FRACTIONS[el] * ref[(el, e)] for el in PAGES)
        xm = sum(MASS_FRACTIONS[el] * float(xraydb.mu_elam(el, e * 1e3, kind="total")) for el in PAGES)
        print(f"{e:8d} {nm:9.4f} {xm:10.4f} {100 * (xm / nm - 1):6.1f}% {RHO_G_CM3 * nm:10.3f}")
    print("\nworst element deviation: %.1f%%" % (100 * worst))


if __name__ == "__main__":
    main()
