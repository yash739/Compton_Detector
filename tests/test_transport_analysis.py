"""Transport, event reconstruction and analysis tests."""
import numpy as np
import pytest

from czt_mc import analysis as an, detector as dt, physics as ph, transport as tr


@pytest.fixture(scope="module")
def env():
    xs = ph.CrossSections()
    return dt.Detector(), xs, ph.KFluorescence(xs)


@pytest.mark.parametrize("fluor", [True, False])
def test_energy_conservation(env, fluor):
    det, xs, kf = env
    r = tr.simulate(200000, 250.0, 20.0, 40.0, 0.5, 30.0, det, xs, kf, np.random.default_rng(21), fluorescence=fluor)
    contained = ~r["escaped"] & ~r["unfinished"] & (r["n_deposits"] > 0)
    total = r["e"].sum(axis=1)
    assert contained.sum() > 1000
    assert np.allclose(total[contained], 250.0, atol=1e-2)
    assert np.all(total <= 250.0 + 1e-2)


def test_first_interaction_probability_follows_the_real_attenuation(env):
    det, xs, kf = env
    n = 400000
    for e in (150.0, 300.0):
        r = tr.simulate(n, e, 0.0, 0.0, 0.0, 0.0, det, xs, kf, np.random.default_rng(22))
        # photons are started over the whole array footprint at normal incidence, so every photon sees the full thickness
        p = 1.0 - np.exp(-float(xs.transport(e)) * det.t)
        meas = np.mean(r["n_deposits"] > 0)
        assert abs(meas - p) < 4 * np.sqrt(p * (1 - p) / n)
    # the old notebooks: 2.64 /cm at every energy
    r = tr.simulate(n, 300.0, 0.0, 0.0, 0.0, 0.0, det, xs, kf, np.random.default_rng(23), mu_const=2.64)
    assert abs(np.mean(r["n_deposits"] > 0) - (1 - np.exp(-2.64 * det.t))) < 0.005
    assert np.all(r["code"][r["n_deposits"] > 0][:, 0] == 1)      # and every interaction was a Compton scatter


def test_process_mix_at_300_kev(env):
    det, xs, kf = env
    r = tr.simulate(400000, 300.0, 0.0, 0.0, 0.0, 0.0, det, xs, kf, np.random.default_rng(24))
    first = r["code"][r["n_deposits"] > 0][:, 0]
    frac_pe = np.mean(first == 2)
    expect = float(xs.photo(300.0) / xs.transport(300.0))
    assert abs(frac_pe - expect) < 0.01


def _deposits(positions, energies):
    n = 1
    m = len(positions)
    ix = np.full((n, m), -1, np.int16); iy = ix.copy(); e = np.zeros((n, m), np.float32); c = np.ones((n, m), np.int8)
    for j, ((a, b), en) in enumerate(zip(positions, energies)):
        ix[0, j], iy[0, j], e[0, j] = a, b, en
    return ix, iy, e, c


@pytest.mark.parametrize("low,high,expected_bin", [((3, 3), (4, 3), 0), ((3, 3), (4, 4), 1), ((3, 3), (3, 4), 2), ((3, 3), (2, 4), 3),
                                                   ((3, 3), (2, 3), 4), ((3, 3), (2, 2), 5), ((3, 3), (3, 2), 6), ((3, 3), (4, 2), 7)])
def test_direction_bins_run_from_the_lower_to_the_higher_energy_pixel(low, high, expected_bin):
    det = dt.Detector()
    ix, iy, e, c = _deposits([low, high], [50.0, 150.0])
    ev = dt.reconstruct(ix, iy, e, c, det)["double"]
    assert ev["bin"].tolist() == [expected_bin]
    ix, iy, e, c = _deposits([low, high], [150.0, 50.0])           # energies swapped: the direction reverses, i.e. +4 bins
    assert dt.reconstruct(ix, iy, e, c, det)["double"]["bin"].tolist() == [(expected_bin + 4) % 8]


def test_selection_threshold_adjacency_window_and_ratio():
    det = dt.Detector()
    sel = lambda pos, en, **kw: len(dt.reconstruct(*_deposits(pos, en), det, **kw)["double"]["bin"])
    assert sel([(3, 3), (4, 3)], [30.0, 120.0]) == 1
    assert sel([(3, 3), (4, 3)], [15.0, 120.0]) == 0              # lower pixel below the 20 keV threshold: a single, not a double
    assert sel([(3, 3), (5, 3)], [50.0, 100.0]) == 0              # not adjacent
    assert sel([(3, 3), (4, 3)], [50.0, 40.0]) == 0               # total energy 90 keV below the 100 keV window
    assert sel([(3, 3), (4, 3)], [25.0, 300.0]) == 1              # E_high / E_low = 12: accepted now that the upper ratio limit is gone
    assert sel([(3, 3), (4, 3)], [25.0, 300.0], ratio_max=6.0) == 0   # ... and rejected by the old cut
    assert sel([(0, 0), (1, 0)], [50.0, 150.0]) == 0              # lower pixel outside the fiducial block
    ix, iy, e, c = _deposits([(3, 3), (3, 3), (4, 3)], [30.0, 25.0, 100.0])
    assert dt.reconstruct(ix, iy, e, c, det)["double"]["e_low"].tolist() == [55.0]    # deposits in one pixel are summed


def test_bins_are_exactly_45_degrees():
    """regression: np.histogram(range=(-pi/4, 8 pi/4), bins=8) has bins of 9 pi/32"""
    assert np.allclose(np.diff(an.BIN_ANGLES), np.pi / 4)
    assert an.histogram([0, 0, 7, 3]).tolist() == [2, 0, 0, 1, 0, 0, 0, 1]


def test_cosine_fit_recovers_the_truth():
    rng = np.random.default_rng(31)
    for peak in (10.0, 45.0, 100.0, 170.0):
        mu = 0.3
        truth = 4000.0 * (1 + mu * np.cos(2 * (an.BIN_ANGLES - np.radians(peak))))
        fits = [an.fit_modulation(*an.corrected_curve(rng.poisson(truth), np.full(8, 4000.0) * 1000)) for _ in range(300)]
        m = np.array([f["mu"] for f in fits]); p = np.array([f["peak"] for f in fits])
        dp = ((p - peak + 90) % 180) - 90
        assert abs(m.mean() - mu) < 0.01 and abs(dp.mean()) < 1.0
        assert 0.6 < np.mean([f["chi2"] for f in fits]) / 5 < 1.5   # the errors are about right


def test_pf_and_pa_recovery_end_to_end(env):
    det, xs, kf = env
    rng = np.random.default_rng(32)
    def run(pf, pa):
        r = tr.simulate(2500000, 300.0, 0.0, 0.0, pf, pa, det, xs, kf, rng)
        return an.histogram(dt.reconstruct(r["ix"], r["iy"], r["e"], r["code"], det)["double"]["bin"])
    unp, full, half = run(0.0, 0.0), run(1.0, 45.0), run(0.5, 45.0)     # 45 deg: a symmetry axis of the square pixels, where the 8-bin cosine fit has no PA bias (see scripts/run_pa_scan.py)
    out = an.recover_pf_pa(half, unp, full)
    assert abs(out["pf"] - 0.5) < 3 * out["pf_err"] + 0.03
    assert abs(((out["pa"] - 45.0 + 90) % 180) - 90) < 3 * out["pa_err"] + 1.0


def test_double_classification_is_by_pixel_content():
    """a genuine Compton double whose K X-ray is reabsorbed in the same pixel must not be called fluorescence (the first version of the classifier did)"""
    C, P, F = dt.C_BIT, dt.P_BIT, dt.F_BIT
    cls = lambda lo, hi: dt.double_class(np.array([lo]), np.array([hi]))[0]
    assert cls(C, P) == "Compton + photoabsorption" and cls(P, C) == "Compton + photoabsorption"
    assert cls(C, P | F) == "Compton + photoabsorption"            # fluorescence absorbed in the pixel of the photoabsorption
    assert cls(P, F) == "K fluorescence photon in the neighbour" and cls(F, C | P) == "K fluorescence photon in the neighbour"
    assert cls(C, C) == "multiple Compton" and cls(C, C | P) == "multiple Compton"


def test_pixel_table_adapter_matches_the_native_selection(env):
    """the per-pixel table path (CZTPhysSim-style rows of event, pixel, summed energy) selects exactly the same doubles as the native deposit path"""
    from czt_mc import pixel_io
    det, xs, kf = env
    r = tr.simulate(600000, 200.0, 0.0, 0.0, 0.5, 20.0, det, xs, kf, np.random.default_rng(41))
    native = dt.reconstruct(r["ix"], r["iy"], r["e"], r["code"], det)["double"]
    valid = r["code"] > 0
    ev = np.repeat(np.arange(r["code"].shape[0])[:, None], r["code"].shape[1], axis=1)[valid]
    pid = (r["iy"].astype(np.int64) * det.n + r["ix"])[valid]
    key = ev.astype(np.int64) * (det.n * det.n) + pid
    uk, inv = np.unique(key, return_inverse=True)
    esum = np.zeros(uk.size); np.add.at(esum, inv, r["e"][valid].astype(float))
    table = pixel_io.pixel_table_doubles(uk // (det.n * det.n), uk % (det.n * det.n), esum, n_pix=det.n, fiducial=det.fid)
    assert len(table["bin"]) == len(native["bin"]) > 500
    assert np.array_equal(np.sort(table["event"]), np.sort(native["event"]))
    o1, o2 = np.argsort(table["event"]), np.argsort(native["event"])
    assert np.array_equal(table["bin"][o1], native["bin"][o2])
    assert np.allclose(table["e_low"][o1], native["e_low"][o2], atol=1e-3)


def _reference_reconstruct(ix, iy, e, code, det, e_thr=20.0, window=(100.0, 600.0), ratio_max=None):
    """a plain Python loop that states the selection rules literally"""
    out = {}
    for ev in range(ix.shape[0]):
        pix = {}
        for j in range(ix.shape[1]):
            if code[ev, j] > 0:
                pix[(int(ix[ev, j]), int(iy[ev, j]))] = pix.get((int(ix[ev, j]), int(iy[ev, j])), 0.0) + float(e[ev, j])
        trig = {p: v for p, v in pix.items() if v >= e_thr}
        if len(trig) != 2:
            continue
        (p0, e0), (p1, e1) = sorted(trig.items(), key=lambda kv: kv[1])
        if not (window[0] <= e0 + e1 <= window[1]) or max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) != 1:
            continue
        if ratio_max is not None and e1 > ratio_max * e0:
            continue
        if not (det.fid[0] <= p0[0] < det.fid[1] and det.fid[0] <= p0[1] < det.fid[1]):
            continue
        out[ev] = (int(round((np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])) % 360.0) / 45.0)) % 8, e0, e1)
    return out


@pytest.mark.parametrize("ratio_max", [None, 6.0])
def test_reconstruct_matches_a_literal_reference(env, ratio_max):
    det = env[0]
    rng = np.random.default_rng(51)
    n, m = 4000, 6
    nd = rng.integers(0, m + 1, n)
    code = (np.arange(m)[None, :] < nd[:, None]).astype(np.int8) * rng.integers(1, 4, (n, m)).astype(np.int8)
    base = rng.integers(2, 5, (n, 2))
    ix = np.clip(base[:, :1] + rng.integers(-1, 2, (n, m)), 0, det.n - 1).astype(np.int16)
    iy = np.clip(base[:, 1:] + rng.integers(-1, 2, (n, m)), 0, det.n - 1).astype(np.int16)
    e = (rng.exponential(60.0, (n, m)) * (code > 0)).astype(np.float32)
    ref = _reference_reconstruct(ix, iy, e, code, det, ratio_max=ratio_max)
    got = dt.reconstruct(ix, iy, e, code, det, ratio_max=ratio_max)["double"]
    assert len(ref) > 100
    assert sorted(got["event"].tolist()) == sorted(ref)
    order = np.argsort(got["event"])
    for i, ev in enumerate(got["event"][order]):
        b, lo, hi = ref[int(ev)]
        assert got["bin"][order][i] == b and abs(got["e_low"][order][i] - lo) < 1e-3 and abs(got["e_high"][order][i] - hi) < 1e-3


def test_results_do_not_depend_on_the_number_of_workers(env):
    from czt_mc import runner
    s = runner.Setup()
    a = runner.run(s, 3_000_000, 250.0, pf=0.5, pa_deg=20.0, seed=61, workers=1)
    b = runner.run(s, 3_000_000, 250.0, pf=0.5, pa_deg=20.0, seed=61, workers=3)
    assert np.array_equal(a["hist"], b["hist"]) and a["n_double"] == b["n_double"] and a["n_single"] == b["n_single"] and a["origin"] == b["origin"]
    assert a["n_double"] > 5000
