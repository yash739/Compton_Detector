"""Modulation analysis of per-pixel energy tables (e.g. CZTPhysSim FITS output): selection, 8-bin histogram, geometry correction, cosine fit, PF/PA.

    python scripts/analyse_pixel_table.py unpolarised.fits polarised.fits [reference_100pct.fits] [--hdu 1] [--fiducial 2,14]

Each file needs a table with columns eventID, pixelID, edep_tot (pixelID = 16 * row + column, energies in keV). With a 100% polarised reference at the same PA the
polarisation fraction is recovered as well. NOTE: not yet run on a real CZTPhysSim output file (none was available when this was written); the selection code is
tested against the package's own Monte Carlo in tests/test_transport_analysis.py.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from czt_mc import analysis as an, pixel_io  # noqa: E402


def hist_of(path, hdu, fid):
    ev, px, ed = pixel_io.read_fits_table(path, hdu)
    d = pixel_io.pixel_table_doubles(ev, px, ed, fiducial=fid)
    return an.histogram(d["bin"]), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unpolarised"); ap.add_argument("measured"); ap.add_argument("reference", nargs="?")
    ap.add_argument("--hdu", type=int, default=1); ap.add_argument("--fiducial", default="2,14")
    a = ap.parse_args()
    fid = tuple(int(v) for v in a.fiducial.split(","))
    hu, du = hist_of(a.unpolarised, a.hdu, fid)
    hm, dm = hist_of(a.measured, a.hdu, fid)
    print("unpolarised doubles", int(hu.sum()), hu.astype(int)); print("measured doubles   ", int(hm.sum()), hm.astype(int))
    fit = an.fit_modulation(*an.corrected_curve(hm, hu))
    print(f"corrected curve: A/B = {fit['mu']:.3f}, peak {fit['peak']:.1f} deg (E-vector PA = {(fit['peak'] - 90) % 180:.1f} deg), chi2/5 = {fit['chi2']:.1f}")
    if a.reference:
        hr, _ = hist_of(a.reference, a.hdu, fid)
        o = an.recover_pf_pa(hm, hu, hr)
        print(f"PF = {o['pf']:.3f} +- {o['pf_err']:.3f}, PA = {o['pa']:.1f} +- {o['pa_err']:.1f} deg")


if __name__ == "__main__":
    main()
