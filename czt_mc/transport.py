"""Vectorised photon transport in the CZT array: photoabsorption, Compton scattering (exact polarised Klein-Nishina), K fluorescence.

Approximations (documented in the README): electrons deposit their energy at the interaction point (no electron range, no charge sharing), Compton scattering is on free
electrons at rest (no binding, no Doppler broadening), coherent scattering is treated as no interaction, L fluorescence and Auger electrons are absorbed locally,
the energy resolution is perfect, and the pixels are ideal rectangles with sharp boundaries.
"""
import numpy as np

from . import physics as ph
from .constants import ME_KEV
from .detector import Detector

MAX_STEPS = 10


def simulate(n_photons, e_kev, theta_inc_deg, phi_inc_deg, pf, pa_deg, det=None, xs=None, kf=None, rng=None, max_steps=MAX_STEPS, mu_const=None, fluorescence=True):
    """Transport n_photons monoenergetic photons.

    theta_inc_deg, phi_inc_deg : polar and azimuthal angle of the beam direction (0, 0 = along +z, normal incidence).
    pf, pa_deg                 : polarisation fraction and E-vector angle in the detector plane (for oblique beams: the projection onto the plane perpendicular to the beam).
    mu_const                   : if given (1/cm), use this energy-independent attenuation and always scatter (the old notebooks' behaviour, for comparison only).
    fluorescence               : switch K fluorescence off for comparisons.
    Returns a dict with deposit arrays (n, max_steps): ix, iy, e (keV), code (0 unused, 1 Compton recoil, 2 photoabsorption, 3 absorption of a fluorescence photon), the flags
    escaped / unfinished per photon, n_start (photons started), and fid_scale = (fiducial entrance area) / (area the photons were started over).
    """
    det = det or Detector()
    xs = xs or ph.CrossSections()
    kf = kf or ph.KFluorescence(xs)
    rng = rng or np.random.default_rng()
    n = int(n_photons)
    th, pp = np.radians(theta_inc_deg), np.radians(phi_inc_deg)
    k0 = np.array([np.sin(th) * np.cos(pp), np.sin(th) * np.sin(pp), np.cos(th)])
    margin = det.t * np.tan(th) if th < np.radians(89.0) else 0.0
    w = det.half + margin
    pos = np.zeros((n, 3))
    pos[:, 0] = rng.uniform(-w, w, n); pos[:, 1] = rng.uniform(-w, w, n)
    k = np.tile(k0, (n, 1))
    pol = ph.beam_polarisation(n, k0, pf, np.radians(pa_deg), rng)
    e = np.full(n, float(e_kev))
    alive = np.ones(n, bool); isfl = np.zeros(n, bool); escaped = np.zeros(n, bool)
    dep_ix = np.full((n, max_steps), -1, np.int16); dep_iy = np.full((n, max_steps), -1, np.int16)
    dep_e = np.zeros((n, max_steps), np.float32); dep_code = np.zeros((n, max_steps), np.int8)
    nd = np.zeros(n, np.int64)

    def record(idx, code, edep):
        ix, iy = det.pixel(pos[idx])
        j = nd[idx]
        ok = j < max_steps
        i2, j2 = idx[ok], j[ok]
        dep_ix[i2, j2] = ix[ok]; dep_iy[i2, j2] = iy[ok]; dep_e[i2, j2] = edep[ok]; dep_code[i2, j2] = code[ok] if np.ndim(code) else code
        nd[idx] += 1

    for step in range(max_steps):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        ei = e[idx]
        low = ei < 5.0                                                 # below the table: absorbed locally
        if np.any(low):
            record(idx[low], np.where(isfl[idx[low]], 3, 2), ei[low]); alive[idx[low]] = False
            idx = idx[~low]; ei = e[idx]
        if mu_const is None:
            mu, pph = xs.transport_and_photo_fraction(ei)
        else:
            mu, pph = np.full(idx.size, float(mu_const)), np.zeros(idx.size)
        d = rng.exponential(1.0 / mu)
        newpos = pos[idx] + d[:, None] * k[idx]
        inside = det.inside(newpos)
        out = idx[~inside]
        alive[out] = False; escaped[out] = True
        pph = pph[inside]
        idx = idx[inside]; pos[idx] = newpos[inside]; ei = e[idx]
        if idx.size == 0:
            continue
        is_pe = rng.random(idx.size) < pph                              # (mu_const mode, the old behaviour: pph = 0, every interaction is a Compton scatter)
        # photoabsorption
        pi = idx[is_pe]
        if pi.size:
            if fluorescence:
                emitted, eline, _ = kf.emit(e[pi], rng)
            else:
                emitted, eline = np.zeros(pi.size, bool), np.zeros(pi.size)
            record(pi, np.where(isfl[pi], 3, 2), e[pi] - eline)
            cont = pi[emitted]
            alive[pi[~emitted]] = False
            if cont.size:                                              # the K X-ray continues, isotropically
                v = rng.normal(size=(cont.size, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
                k[cont] = v; e[cont] = eline[emitted]; isfl[cont] = True
                u1, _ = ph.perpendicular_basis(v)
                pol[cont] = u1
        # Compton scattering
        ci = idx[~is_pe]
        if ci.size:
            mu_c, phi_c, e_new, k_new, pol_new = ph.sample_compton(e[ci], k[ci], pol[ci], rng)
            record(ci, 1, e[ci] - e_new)
            e[ci] = e_new; k[ci] = k_new; pol[ci] = pol_new
    unfinished = alive.copy()
    area_start = (2 * w) ** 2
    return {"ix": dep_ix, "iy": dep_iy, "e": dep_e, "code": dep_code, "escaped": escaped, "unfinished": unfinished,
            "n_start": n, "fid_scale": det.fiducial_area / area_start, "e0": float(e_kev), "n_deposits": nd}
