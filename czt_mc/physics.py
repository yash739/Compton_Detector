"""Physics of the CZT Monte Carlo: tabulated cross sections, exact polarised Klein-Nishina sampling, K fluorescence.

Everything is vectorised over photons (numpy arrays); randomness comes from a numpy Generator that the caller owns.
"""
import csv
import os

import numpy as np

from .constants import ELEMENTS, ME_KEV

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ----------------------------------------------------------------------------------------------------------------- cross sections
class CrossSections:
    """Linear attenuation coefficients (1/cm) of Cd(0.9)Zn(0.1)Te by process, from czt_mc/data/czt_xs.csv (log-log interpolation).

    photo : photoabsorption;  incoh : incoherent (Compton) scattering;  coh : coherent (Rayleigh) scattering.
    The table covers 5-800 keV; the Elam tables are not reliable above that.
    """

    def __init__(self, path=None):
        path = path or os.path.join(_DATA, "czt_xs.csv")
        with open(path) as f:
            rd = csv.DictReader(f)
            rows = [{k: float(v) for k, v in r.items()} for r in rd]
        self._e = np.array([r["E_keV"] for r in rows])
        self._le = np.log(self._e)
        self._cols = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "E_keV"}
        self._lcols = {k: np.log(np.maximum(v, 1e-300)) for k, v in self._cols.items() if k.startswith("mu")}
        tr = self._cols["mu_photo"] + self._cols["mu_incoh"]
        self._ltr = np.log(tr)                                      # combined transport table: one log and one interpolation per call in the hot loop
        self._pph = self._cols["mu_photo"] / tr

    def _interp(self, name, e):
        e = np.asarray(e, float)
        if np.any(e < self._e[0]) or np.any(e > self._e[-1]):
            raise ValueError("energy outside the 5-800 keV table")
        return np.exp(np.interp(np.log(e), self._le, self._lcols[name]))

    def photo(self, e): return self._interp("mu_photo", e)
    def incoh(self, e): return self._interp("mu_incoh", e)
    def coh(self, e): return self._interp("mu_coh", e)
    def total(self, e): return self.photo(e) + self.incoh(e) + self.coh(e)

    def transport_and_photo_fraction(self, e):
        """(transport attenuation, photoabsorption fraction of the interactions) at e (keV): the pair the transport loop needs, from a single log and two interpolations"""
        e = np.asarray(e, float)
        if np.any(e < self._e[0]) or np.any(e > self._e[-1]):
            raise ValueError("energy outside the 5-800 keV table")
        le = np.log(e)
        return np.exp(np.interp(le, self._le, self._ltr)), np.interp(le, self._le, self._pph)

    def transport(self, e):
        """attenuation that matters for energy deposition: photoabsorption + Compton. Coherent scattering deposits nothing and, in the small-angle approximation, leaves the photon undeflected."""
        return self.photo(e) + self.incoh(e)

    def photo_by_element(self, e):
        return np.stack([self._interp(f"mu_photo_{el}", e) for el in ELEMENTS], axis=-1)

    def k_fraction(self, e):
        """fraction of each element's photoabsorption that is K-shell (zero below the element's K edge), shape (..., 3)"""
        e = np.asarray(e, float)
        return np.stack([np.interp(e, self._e, self._cols[f"fK_{el}"]) for el in ELEMENTS], axis=-1)


class KFluorescence:
    """K fluorescence of Cd, Zn, Te: which element is photoabsorbed, whether the K shell is hit, and whether a K X-ray is emitted (czt_kfluor_lines.csv)."""

    def __init__(self, xs, path=None):
        path = path or os.path.join(_DATA, "czt_kfluor_lines.csv")
        self.xs = xs
        with open(path) as f:
            rows = [r for r in csv.DictReader(f)]
        self.omega = {el: float(next(r["fluor_yield_K"] for r in rows if r["element"] == el)) for el in ELEMENTS}
        self.lines = {}
        for el in ELEMENTS:
            lr = [r for r in rows if r["element"] == el and float(r["rel_intensity"]) > 1e-3]
            e = np.array([float(r["E_keV"]) for r in lr]); w = np.array([float(r["rel_intensity"]) for r in lr]); w /= w.sum()
            mu = np.array([float(r["mu_total_cm"]) for r in lr])
            self.lines[el] = (e, w, mu)
        self._pk = np.array([self.omega[el] for el in ELEMENTS])

    def probability_per_photoabsorption(self, e):
        """probability that a photoabsorption at energy e (keV) is followed by emission of a K X-ray (any element)"""
        e = np.asarray(e, float)
        w = self.xs.photo_by_element(e)
        w = w / w.sum(axis=-1, keepdims=True)
        return (w * self.xs.k_fraction(e) * self._pk).sum(axis=-1)

    def emit(self, e, rng):
        """For photoabsorptions at energies e (array): returns (emitted bool, line energy keV, line total attenuation 1/cm); the last two are 0 where nothing is emitted."""
        e = np.asarray(e, float)
        n = e.size
        w = self.xs.photo_by_element(e)
        cum = np.cumsum(w / w.sum(axis=-1, keepdims=True), axis=-1)
        u = rng.random(n)
        iel = np.minimum((u[:, None] > cum).sum(axis=1), len(ELEMENTS) - 1)
        fk = self.xs.k_fraction(e)[np.arange(n), iel]
        om = self._pk[iel]
        emitted = (rng.random(n) < fk) & (rng.random(n) < om)
        eline = np.zeros(n); mu = np.zeros(n)
        for j, el in enumerate(ELEMENTS):
            sel = np.where(emitted & (iel == j))[0]
            if sel.size:
                le, lw, lm = self.lines[el]
                k = rng.choice(len(le), size=sel.size, p=lw)
                eline[sel] = le[k]; mu[sel] = lm[k]
        return emitted, eline, mu


# ----------------------------------------------------------------------------------------------------------------- Klein-Nishina
def kn_epsilon(e_kev, cos_theta):
    """E'/E for Compton scattering through the angle theta"""
    return 1.0 / (1.0 + (np.asarray(e_kev, float) / ME_KEV) * (1.0 - cos_theta))


def kn_unpolarised_pdf_cos(e_kev, mu):
    """unnormalised dsigma/dcos(theta) of the unpolarised Klein-Nishina cross section"""
    eps = kn_epsilon(e_kev, mu)
    return eps ** 2 * (eps + 1.0 / eps - 1.0 + mu ** 2)


def kn_modulation(e_kev, cos_theta):
    """m(theta): the azimuthal pdf at fixed theta is proportional to 1 - m cos(2 phi), phi measured from the polarisation vector. m = sin^2 / (eps + 1/eps - sin^2)."""
    eps = kn_epsilon(e_kev, cos_theta)
    s2 = 1.0 - cos_theta ** 2
    return s2 / (eps + 1.0 / eps - s2)


def sample_compton(e_kev, k, pol, rng):
    """Sample the polarised Klein-Nishina cross section exactly (two nested rejection samplers, no discarded photons).

    e_kev (n,), k (n,3) unit direction, pol (n,3) unit polarisation vector perpendicular to k.
    Returns (cos_theta, phi, e_new, k_new, pol_new). phi is measured from pol, so the distribution is peaked at phi = +-90 deg (perpendicular to the E vector)."""
    n = e_kev.size
    mu = np.empty(n); todo = np.arange(n)
    while todo.size:                                   # cos(theta): f(mu) <= 2 for all energies (value 2 at mu = 1, eps = 1)
        m = rng.uniform(-1.0, 1.0, todo.size)
        ok = rng.uniform(0.0, 2.0, todo.size) < kn_unpolarised_pdf_cos(e_kev[todo], m)
        mu[todo[ok]] = m[ok]
        todo = todo[~ok]
    eps = kn_epsilon(e_kev, mu)
    s2 = 1.0 - mu ** 2
    msk = s2 / (eps + 1.0 / eps - s2)
    phi = np.empty(n); todo = np.arange(n)
    while todo.size:                                   # pdf(phi) ~ 1 - m cos(2 phi); bound 1 + m
        p = rng.uniform(0.0, 2.0 * np.pi, todo.size)
        ok = rng.uniform(0.0, 1.0 + msk[todo], todo.size) < 1.0 - msk[todo] * np.cos(2.0 * p)
        phi[todo[ok]] = p[ok]
        todo = todo[~ok]
    sin_t = np.sqrt(s2)
    kxe = np.cross(k, pol)
    k_new = mu[:, None] * k + sin_t[:, None] * (np.cos(phi)[:, None] * pol + np.sin(phi)[:, None] * kxe)
    k_new /= np.linalg.norm(k_new, axis=1)[:, None]
    # polarisation of the scattered photon: the incident E vector projected perpendicular to the new direction (first-order transport; matters only for multiple scattering)
    pe = pol - (pol * k_new).sum(axis=1)[:, None] * k_new
    nrm = np.linalg.norm(pe, axis=1)
    bad = nrm < 1e-9
    if np.any(bad):
        pe[bad] = kxe[bad] - (kxe[bad] * k_new[bad]).sum(axis=1)[:, None] * k_new[bad]
        nrm = np.linalg.norm(pe, axis=1)
    pe /= nrm[:, None]
    return mu, phi, e_kev * eps, k_new, pe


# ----------------------------------------------------------------------------------------------------------------- beam polarisation
def perpendicular_basis(k):
    """two unit vectors u1, u2 with (u1, u2, k) right-handed, for k of shape (3,) or (n,3)"""
    k = np.atleast_2d(k)
    ref = np.where(np.abs(k[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    u1 = np.cross(ref, k); u1 /= np.linalg.norm(u1, axis=1)[:, None]
    u2 = np.cross(k, u1)
    return u1, u2


def beam_polarisation(n, k, polarisation_fraction, pa_rad, rng):
    """Polarisation vectors for n photons of a beam along the unit vector k (shape (3,)).

    A fraction `polarisation_fraction` has its E vector along the projection of the detector-plane direction (cos pa, sin pa, 0) on the plane perpendicular to k;
    the rest have an E vector uniform in angle in that plane (a truly unpolarised beam: <e e^T> = identity/2 in the plane, for any k)."""
    k = np.asarray(k, float)
    d = np.array([np.cos(pa_rad), np.sin(pa_rad), 0.0])
    e0 = d - (d @ k) * k
    e0 /= np.linalg.norm(e0)
    u1, u2 = perpendicular_basis(k)
    ang = rng.uniform(0.0, 2.0 * np.pi, n)
    e = np.cos(ang)[:, None] * u1 + np.sin(ang)[:, None] * u2
    pol = rng.random(n) < polarisation_fraction
    e[pol] = e0
    return e
