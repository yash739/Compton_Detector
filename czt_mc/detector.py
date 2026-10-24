"""Detector geometry and the CZTI-style double-event selection."""
import numpy as np

from . import constants as C

CODE_CHARS = {1: "C", 2: "P", 3: "F"}      # Compton recoil, photoabsorption of a primary/scattered photon, absorption of a K fluorescence photon


class Detector:
    """An n x n array of pixels (pitch p) of thickness t. Photons enter at z = 0 (cathode side) travelling towards +z; the anode plane is z = t."""

    def __init__(self, n_pix=C.N_PIX, pitch=C.PITCH_CM, thickness=C.THICKNESS_CM, fiducial=C.FIDUCIAL):
        self.n, self.p, self.t = n_pix, pitch, thickness
        self.half = 0.5 * n_pix * pitch
        self.fid = fiducial

    def inside(self, pos):
        return (np.abs(pos[:, 0]) < self.half) & (np.abs(pos[:, 1]) < self.half) & (pos[:, 2] > 0.0) & (pos[:, 2] < self.t)

    def pixel(self, pos):
        ix = np.floor((pos[:, 0] + self.half) / self.p).astype(np.int16)
        iy = np.floor((pos[:, 1] + self.half) / self.p).astype(np.int16)
        return np.clip(ix, 0, self.n - 1), np.clip(iy, 0, self.n - 1)

    @property
    def fiducial_area(self):
        return ((self.fid[1] - self.fid[0]) * self.p) ** 2


def origin_labels(sig):
    """signature integer (base-5 digits = process codes in time order) -> string such as 'CP'"""
    out = []
    s = int(sig)
    while s:
        out.append(CODE_CHARS[s % 5]); s //= 5
    return "".join(out)


def _reconstruct_multi(dep_ix, dep_iy, dep_e, dep_code, det, e_thr, e_window, ratio_max):
    """The pixel-merging and selection logic on photons that have at least two deposits (the loops are O(M^2) per photon, so they are kept off the many one-deposit photons)."""
    valid = dep_code > 0
    n, m = dep_e.shape
    pid = np.where(valid, dep_ix.astype(np.int32) * det.n + dep_iy, -1)
    esum = np.zeros((n, m))
    content = np.zeros((n, m), np.int8)                 # bit 0: Compton recoil, bit 1: photoabsorption, bit 2: fluorescence photon absorbed, among the deposits of the same pixel
    for k in range(m):
        same = valid[:, k:k + 1] & (pid == pid[:, k:k + 1])
        esum += np.where(same, dep_e[:, k:k + 1], 0.0)
        content |= np.where(same, (1 << np.maximum(dep_code[:, k:k + 1].astype(np.int16) - 1, 0)).astype(np.int8), 0).astype(np.int8)
    first = valid.copy()
    for j in range(1, m):
        for k in range(j):
            first[:, j] &= ~(valid[:, k] & (pid[:, k] == pid[:, j]))
    trig = first & (esum >= e_thr)
    ntrig = trig.sum(axis=1)
    hist = np.bincount(np.minimum(ntrig, 8), minlength=9)

    # single-pixel events
    single = ntrig == 1
    e1 = np.where(trig, esum, 0.0).sum(axis=1)
    n_single = int(np.sum(single & (e1 >= e_window[0]) & (e1 <= e_window[1])))

    # double events
    sel = np.where(ntrig == 2)[0]
    order = np.argsort(~trig[sel], axis=1, kind="stable")[:, :2]
    r = np.arange(sel.size)[:, None]
    ix = dep_ix[sel][r, order].astype(np.int32); iy = dep_iy[sel][r, order].astype(np.int32)
    ee = esum[sel][r, order]
    cc = content[sel][r, order]
    etot = ee.sum(axis=1)
    ok = (etot >= e_window[0]) & (etot <= e_window[1])
    dx = ix[:, 1] - ix[:, 0]; dy = iy[:, 1] - iy[:, 0]
    ok &= (np.maximum(np.abs(dx), np.abs(dy)) == 1)
    lo = np.argmin(ee, axis=1); hi = 1 - lo
    rr = np.arange(sel.size)
    e_low, e_high = ee[rr, lo], ee[rr, hi]
    c_low, c_high = cc[rr, lo], cc[rr, hi]
    if ratio_max is not None:
        ok &= (e_high <= ratio_max * e_low)
    lx, ly = ix[rr, lo], iy[rr, lo]
    ok &= (lx >= det.fid[0]) & (lx < det.fid[1]) & (ly >= det.fid[0]) & (ly < det.fid[1])
    ddx = (ix[rr, hi] - lx); ddy = (iy[rr, hi] - ly)
    ang = np.degrees(np.arctan2(ddy, ddx)) % 360.0
    bins = (np.rint(ang / 45.0).astype(int)) % 8
    code = np.where(valid, dep_code, 0).astype(np.int64)
    sig = np.zeros(n, dtype=np.int64)
    for k in range(m):
        sig += code[:, k] * (5 ** k)
    sig = sig[sel]
    keep = np.where(ok)[0]
    return {
        "n_single": n_single,
        "n_triggered_pixels": hist,
        "double": {"bin": bins[keep], "e_low": e_low[keep], "e_high": e_high[keep], "sig": sig[keep], "ix": lx[keep], "iy": ly[keep], "event": sel[keep], "c_low": c_low[keep], "c_high": c_high[keep]},
    }


def reconstruct(dep_ix, dep_iy, dep_e, dep_code, det, e_thr=C.E_THRESHOLD_KEV, e_window=C.E_WINDOW_KEV, ratio_max=C.RATIO_MAX):
    """Turn energy deposits into detector events.

    dep_* have shape (n, M); unused entries have dep_code == 0. Deposits in the same pixel are summed; a pixel triggers when its summed energy is above e_thr.
    Returns a dict with
      n_single           number of events with exactly one triggered pixel and its energy inside e_window
      double             dict of arrays for events with exactly two triggered, adjacent pixels, summed energy in e_window, optional E_high/E_low <= ratio_max and
                         the lower-energy pixel inside the fiducial block: bin (0..7, direction from the lower- to the higher-energy pixel in steps of 45 deg),
                         e_low, e_high, sig (origin signature, see origin_labels), ix, iy (lower-energy pixel), event (row index), c_low, c_high (pixel contents, see double_class)
      n_triggered_pixels histogram of the number of triggered pixels per event (index 0..8)
    """
    valid = dep_code > 0
    nv = valid.sum(axis=1)
    hist = np.zeros(9, dtype=np.int64)
    hist[0] += int(np.sum(nv == 0))
    n_single = 0
    one = np.where(nv == 1)[0]
    if one.size:                                                    # one deposit: one pixel, no merging needed
        e1 = np.where(valid[one], dep_e[one], 0.0).sum(axis=1)
        trig1 = e1 >= e_thr
        hist[1] += int(trig1.sum()); hist[0] += int((~trig1).sum())
        n_single = int(np.sum(trig1 & (e1 >= e_window[0]) & (e1 <= e_window[1])))
    multi = np.where(nv >= 2)[0]
    res = _reconstruct_multi(dep_ix[multi], dep_iy[multi], dep_e[multi], dep_code[multi], det, e_thr, e_window, ratio_max)
    hist += res["n_triggered_pixels"]
    d = res["double"]
    d["event"] = multi[d["event"]]
    return {"n_single": n_single + res["n_single"], "n_triggered_pixels": hist, "double": d}


C_BIT, P_BIT, F_BIT = 1, 2, 4


def double_class(c_low, c_high):
    """Physical class of selected doubles from the contents (bit mask of Compton recoil / photoabsorption / fluorescence-photon absorption) of the lower- and higher-energy pixel.

    'genuine'      one pixel holds only a Compton recoil, the other the photoabsorption of the scattered photon (with any K X-ray absorbed in that same pixel)
    'multi'        more than one Compton scatter is involved and the final photon is absorbed (or two recoils, the photon escaping)
    'fluorescence' one pixel holds only the absorbed K fluorescence photon: the K X-ray left the pixel of the photoabsorption
    'other'        anything else
    """
    lo = np.asarray(c_low).astype(int); hi = np.asarray(c_high).astype(int)
    out = np.full(lo.shape, "other", dtype=object)
    fl = (lo == F_BIT) | (hi == F_BIT)
    gen = ((lo == C_BIT) & ((hi == P_BIT) | (hi == (P_BIT | F_BIT)))) | ((hi == C_BIT) & ((lo == P_BIT) | (lo == (P_BIT | F_BIT))))
    multi = ((lo & C_BIT) > 0) & ((hi & C_BIT) > 0)
    multi |= ((lo & C_BIT) > 0) & ((hi & P_BIT) > 0) & ~gen & ~fl
    multi |= ((hi & C_BIT) > 0) & ((lo & P_BIT) > 0) & ~gen & ~fl
    out[multi] = "multiple Compton"
    out[gen] = "Compton + photoabsorption"
    out[fl & ~gen] = "K fluorescence photon in the neighbour"
    return out


CLASS_NAMES = ["Compton + photoabsorption", "multiple Compton", "K fluorescence photon in the neighbour", "other"]
