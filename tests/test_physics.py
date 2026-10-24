"""Physics tests: cross sections against NIST, exact Klein-Nishina sampling, beam polarisation (regression for the review findings)."""
import csv
import os

import numpy as np
import pytest

from czt_mc import physics as ph
from czt_mc.constants import MASS_FRACTIONS, ME_KEV, RHO_G_CM3

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "czt_mc", "data")


@pytest.fixture(scope="module")
def xs():
    return ph.CrossSections()


def test_total_attenuation_matches_nist(xs):
    """the tabulated total (photo + incoherent + coherent) of the Cd(0.9)Zn(0.1)Te mixture agrees with the NIST element tables within 1% between 60 and 600 keV"""
    ref = {}
    with open(os.path.join(DATA, "nist_reference.csv")) as f:
        for r in csv.DictReader(f):
            ref[(r["element"], float(r["energy_keV"]))] = float(r["mu_over_rho_cm2_g_NIST"])
    for e in sorted({k[1] for k in ref}):
        nist = RHO_G_CM3 * sum(MASS_FRACTIONS[el] * ref[(el, e)] for el in ("Cd", "Zn", "Te"))
        assert abs(float(xs.total(e)) / nist - 1) < 0.01, e


def test_attenuation_is_energy_dependent_and_not_the_old_constant(xs):
    """the old notebooks used 2.64 /cm for every energy; the real value falls by a factor ~10 between 100 and 600 keV"""
    assert float(xs.total(100.0)) > 8 and float(xs.total(600.0)) < 0.6
    assert 150 < _energy_where_total_equals(xs, 2.64) < 200


def _energy_where_total_equals(xs, mu):
    e = np.geomspace(100, 600, 2000)
    return e[np.argmin(np.abs(xs.total(e) - mu))]


def test_process_fractions_sum_to_one(xs):
    for e in (100.0, 250.0, 600.0):
        s = float(xs.photo(e) + xs.incoh(e) + xs.coh(e))
        assert abs(s / float(xs.total(e)) - 1) < 1e-9


@pytest.mark.parametrize("energy", [100.0, 300.0, 600.0])
def test_kn_theta_marginal(energy):
    rng = np.random.default_rng(11)
    n = 300000
    e = np.full(n, energy); k = np.tile([0.0, 0.0, 1.0], (n, 1)); pol = np.tile([1.0, 0.0, 0.0], (n, 1))
    mu, phi, e_new, k_new, _ = ph.sample_compton(e, k, pol, rng)
    edges = np.linspace(-1, 1, 21)
    obs = np.histogram(mu, edges)[0]
    grid = np.linspace(-1, 1, 200001)
    pdf = ph.kn_unpolarised_pdf_cos(energy, grid)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    exp = n * np.diff(np.interp(edges, grid, cdf))
    chi2 = np.sum((obs - exp) ** 2 / exp)
    assert chi2 < 45.0                                            # 19 dof; p = 1e-3 is 43.8
    # scattered energy follows the Compton formula
    assert np.allclose(e_new, energy / (1 + (energy / ME_KEV) * (1 - mu)))
    assert np.allclose(np.linalg.norm(k_new, axis=1), 1.0)
    assert np.allclose((k_new * k).sum(axis=1), mu, atol=1e-9)    # the polar angle is measured from the incident direction


@pytest.mark.parametrize("energy", [100.0, 300.0])
def test_kn_azimuth_modulation_per_theta(energy):
    """at fixed theta the azimuth from the E vector has <cos 2 phi> = -m(theta)/2 with m = sin^2/(eps + 1/eps - sin^2)"""
    rng = np.random.default_rng(12)
    n = 600000
    e = np.full(n, energy); k = np.tile([0.0, 0.0, 1.0], (n, 1)); pol = np.tile([1.0, 0.0, 0.0], (n, 1))
    mu, phi, *_ = ph.sample_compton(e, k, pol, rng)
    for lo, hi in ((-0.2, 0.2), (0.4, 0.6), (-0.7, -0.5)):
        s = (mu > lo) & (mu < hi)
        meas = np.mean(np.cos(2 * phi[s])); err = np.std(np.cos(2 * phi[s])) / np.sqrt(s.sum())
        pred = -0.5 * np.mean(ph.kn_modulation(energy, mu[s]))
        assert abs(meas - pred) < 4 * err + 2e-3


def test_scattering_peaks_perpendicular_to_the_e_vector():
    rng = np.random.default_rng(13)
    n = 200000
    e = np.full(n, 200.0); k = np.tile([0.0, 0.0, 1.0], (n, 1)); pol = np.tile([1.0, 0.0, 0.0], (n, 1))
    _, _, _, k_new, _ = ph.sample_compton(e, k, pol, rng)
    az = np.degrees(np.arctan2(k_new[:, 1], k_new[:, 0])) % 180.0
    h = np.histogram(az, bins=18, range=(0, 180))[0]
    assert np.argmax(h) in (8, 9) and h[0] < h[9]


@pytest.mark.parametrize("theta,phi", [(0, 0), (30, 0), (30, 45), (45, 45), (60, 0), (60, 120)])
def test_unpolarised_beam_has_no_residual_polarisation(theta, phi):
    """regression: crossing the beam direction with a random in-plane vector (the old oblique generator) gave 7, 17, 33% polarisation at 30, 45, 60 deg"""
    rng = np.random.default_rng(14)
    t, p = np.radians(theta), np.radians(phi)
    k = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    e = ph.beam_polarisation(400000, k, 0.0, 0.0, rng)
    assert np.allclose((e @ k), 0.0, atol=1e-9) and np.allclose(np.linalg.norm(e, axis=1), 1.0)
    u1, u2 = ph.perpendicular_basis(k)
    t2 = np.array([[np.mean((e @ a[0]) * (e @ b[0])) for b in (u1, u2)] for a in (u1, u2)])
    assert abs(t2[0, 0] - 0.5) < 0.004 and abs(t2[0, 1]) < 0.004


def test_polarised_beam_is_perpendicular_to_k_and_fraction_respected():
    rng = np.random.default_rng(15)
    k = np.array([np.sin(0.5), 0.0, np.cos(0.5)])
    e = ph.beam_polarisation(200000, k, 0.5, np.radians(20.0), rng)
    d = np.array([np.cos(np.radians(20.0)), np.sin(np.radians(20.0)), 0.0]); d = d - (d @ k) * k; d /= np.linalg.norm(d)
    frac = np.mean(np.abs(e @ d) > 0.999999)
    assert abs(frac - 0.5) < 0.01


def test_k_fluorescence_probability(xs):
    kf = ph.KFluorescence(xs)
    rng = np.random.default_rng(16)
    for e in (150.0, 400.0):
        emitted, eline, mu = kf.emit(np.full(300000, e), rng)
        p = kf.probability_per_photoabsorption(e)
        assert abs(emitted.mean() - p) < 4 * np.sqrt(p * (1 - p) / 300000)
        assert np.all((eline[emitted] > 8) & (eline[emitted] < 32)) and np.all(eline[~emitted] == 0)
    assert 0.65 < p < 0.8                                         # about 0.72 of photoabsorptions give a K X-ray (Cd, Te dominate; omega_K 0.84, 0.88)
