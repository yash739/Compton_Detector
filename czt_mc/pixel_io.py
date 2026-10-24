"""Apply the double-event selection and the 8-bin analysis to a per-pixel energy table, e.g. the output of the Geant4 simulation CZTPhysSim (dakshasat/CZTPhysSim).

CZTPhysSim writes one row per (event, pixel) with the pixel's summed deposit (columns eventID, pixelID, edep_tot, ...), pixelID = 16 * row + column on a 16 x 16 module of 2.46 mm pixels.
pixel_table_doubles() reproduces the CZTI-style selection of czt_mc.detector.reconstruct on such a table, so a Geant4 table and the Monte Carlo of this package go through the
same selection and the same fits.
"""
import numpy as np

from . import constants as C


def pixel_table_doubles(event, pixel, edep, n_pix=16, e_thr=C.E_THRESHOLD_KEV, e_window=C.E_WINDOW_KEV, ratio_max=C.RATIO_MAX, fiducial=(2, 14), column_is_x=True):
    """Doubles from a (event, pixel, edep) table. Returns dict(bin, e_low, e_high, ix, iy, event, n_single) as in detector.reconstruct.

    Rows with edep below e_thr do not trigger. An event is a double if exactly two pixels trigger, they are adjacent (8 neighbours), their summed energy is inside e_window,
    (optionally E_high / E_low <= ratio_max) and the lower-energy pixel is inside the fiducial block [fiducial[0], fiducial[1]) in both indices.
    Direction bins are 0, 45, ... deg from the lower- to the higher-energy pixel with +x along increasing column index (column_is_x=True) or row index.
    """
    event = np.asarray(event); pixel = np.asarray(pixel); edep = np.asarray(edep, float)
    trig = edep >= e_thr
    event, pixel, edep = event[trig], pixel[trig], edep[trig]
    order = np.argsort(event, kind="stable")
    event, pixel, edep = event[order], pixel[order], edep[order]
    uniq, start, cnt = np.unique(event, return_index=True, return_counts=True)
    n_single = int(np.sum((cnt == 1) & (edep[start] >= e_window[0]) & (edep[start] <= e_window[1])))
    st = start[cnt == 2]
    p0, p1 = pixel[st], pixel[st + 1]
    e0, e1 = edep[st], edep[st + 1]
    ev = event[st]
    row0, col0, row1, col1 = p0 // n_pix, p0 % n_pix, p1 // n_pix, p1 % n_pix
    ix0, iy0, ix1, iy1 = (col0, row0, col1, row1) if column_is_x else (row0, col0, row1, col1)
    ok = (e0 + e1 >= e_window[0]) & (e0 + e1 <= e_window[1]) & (np.maximum(np.abs(ix1 - ix0), np.abs(iy1 - iy0)) == 1)
    lo0 = e0 <= e1
    lx, ly = np.where(lo0, ix0, ix1), np.where(lo0, iy0, iy1)
    hx, hy = np.where(lo0, ix1, ix0), np.where(lo0, iy1, iy0)
    elow, ehigh = np.minimum(e0, e1), np.maximum(e0, e1)
    if ratio_max is not None:
        ok &= ehigh <= ratio_max * elow
    ok &= (lx >= fiducial[0]) & (lx < fiducial[1]) & (ly >= fiducial[0]) & (ly < fiducial[1])
    ang = np.degrees(np.arctan2(hy - ly, hx - lx)) % 360.0
    bins = np.rint(ang / 45.0).astype(int) % 8
    k = np.where(ok)[0]
    return {"bin": bins[k], "e_low": elow[k], "e_high": ehigh[k], "ix": lx[k], "iy": ly[k], "event": ev[k], "n_single": n_single}


def read_fits_table(path, hdu=1, columns=("eventID", "pixelID", "edep_tot")):
    """read the per-pixel ntuple of a CZTPhysSim FITS file (needs astropy). Column names are matched case-insensitively."""
    from astropy.io import fits
    with fits.open(path) as h:
        d = h[hdu].data
        names = {n.lower(): n for n in d.columns.names}
        return tuple(np.asarray(d[names[c.lower()]]) for c in columns)
