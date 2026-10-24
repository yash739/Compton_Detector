"""8-bin modulation curves: geometry correction, cosine fit, polarisation fraction and angle recovery.

Bins: the double event's direction (lower- to higher-energy pixel) in steps of 45 deg, bin i centred on exactly i * 45 deg (0 = +x). Using the right angles matters:
the notebooks this package replaces binned with np.histogram(range=(-pi/4, 8 pi/4)), i.e. bins of 9 pi/32, and fitted at the wrong centres (amplitude low by 6-11%).
"""
import numpy as np

BIN_ANGLES = np.radians(45.0 * np.arange(8))


def histogram(bins):
    return np.bincount(np.asarray(bins, int), minlength=8)[:8].astype(float)


def corrected_curve(h_pol, h_unp):
    """geometry-corrected counts: pol_i / unp_i times the mean of unp (the correction of the report), with the Poisson error of the ratio"""
    h_pol = np.asarray(h_pol, float); h_unp = np.asarray(h_unp, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = h_pol / h_unp * h_unp.mean()
        s = c * np.sqrt(1.0 / np.maximum(h_pol, 1.0) + 1.0 / np.maximum(h_unp, 1.0))
    return c, np.where(s > 0, s, np.inf)


def fit_modulation(y, sigma):
    """Weighted linear least squares of y_i = B + a cos(2 phi_i) + b sin(2 phi_i) (exact, no local minima).

    Returns a dict: B, amp = hypot(a, b), peak (angle of the maximum, in [0, 180) deg), mu = amp / B, the covariance of (B, a, b), chi2 and the degrees of freedom.
    """
    x = np.stack([np.ones(8), np.cos(2 * BIN_ANGLES), np.sin(2 * BIN_ANGLES)], axis=1)
    w = 1.0 / np.asarray(sigma, float) ** 2
    cov = np.linalg.inv(x.T @ (x * w[:, None]))
    beta = cov @ (x.T @ (w * np.asarray(y, float)))
    B, a, b = beta
    res = np.asarray(y, float) - x @ beta
    return {"B": B, "amp": float(np.hypot(a, b)), "peak": float(np.degrees(0.5 * np.arctan2(b, a)) % 180.0), "mu": float(np.hypot(a, b) / B),
            "beta": beta, "cov": cov, "chi2": float(np.sum(w * res ** 2)), "dof": 5}


def _draw_mu_peak(fit, rng, n):
    d = rng.multivariate_normal(fit["beta"], fit["cov"], size=n)
    return np.hypot(d[:, 1], d[:, 2]) / d[:, 0], np.degrees(0.5 * np.arctan2(d[:, 2], d[:, 1])) % 180.0


def recover_pf_pa(h_meas, h_unp, h_100, rng=None, n_mc=4000):
    """Polarisation fraction and angle of a measured 8-bin histogram.

    mu_meas / mu_100, the modulation of the geometry-corrected curve of the measurement over that of a 100% polarised run with the same spectrum, incidence and PA.
    PA is the E-vector angle: the modulation peaks perpendicular to the polarisation, so PA = peak - 90 deg (mod 180).
    Errors come from the full fit covariances (the amplitude and the mean are correlated) by Monte Carlo; PF is a positive-definite estimator and is biased high for PF -> 0.
    """
    rng = rng or np.random.default_rng(1)
    fm = fit_modulation(*corrected_curve(h_meas, h_unp))
    f1 = fit_modulation(*corrected_curve(h_100, h_unp))
    mm, pm = _draw_mu_peak(fm, rng, n_mc)
    m1, _ = _draw_mu_peak(f1, rng, n_mc)
    pf = mm / m1
    dpa = ((pm - fm["peak"] + 90.0) % 180.0) - 90.0
    return {"pf": fm["mu"] / f1["mu"], "pf_err": float(np.std(pf)), "pf_median": float(np.median(pf)),
            "pa": float((fm["peak"] - 90.0) % 180.0), "pa_err": float(np.std(dpa)),
            "mu_meas": fm["mu"], "mu_100": f1["mu"], "chi2_meas": fm["chi2"], "chi2_100": f1["chi2"]}
