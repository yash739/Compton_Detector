"""Batched, parallel runs: simulate many photons in chunks and accumulate 8-bin histograms and double-event statistics.

Every chunk has its own random stream (numpy SeedSequence.spawn from the run seed), so a run gives bit-identical results for any number of workers.
The number of worker processes is the `workers` argument, else the environment variable CZT_WORKERS, else the number of CPUs (a worker needs about 0.7 GB for a 1M-photon chunk).
"""
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import analysis as an, constants as C, detector as dt, physics as ph, transport as tr

CHUNK = 1_000_000


class Setup:
    """bundles the geometry and the physics tables so that scripts build them once"""

    def __init__(self, det=None):
        self.det = det or dt.Detector()
        self.xs = ph.CrossSections()
        self.kf = ph.KFluorescence(self.xs)


_WORKER_SETUP = None


def _init_worker():
    global _WORKER_SETUP
    _WORKER_SETUP = Setup()


def _one_chunk(args):
    setup, n, e_kev, theta_inc, phi_inc, pf, pa_deg, seed_seq, mu_const, fluorescence, e_thr, e_window, ratio_max, keep_doubles = args
    setup = setup or _WORKER_SETUP
    rng = np.random.default_rng(seed_seq)
    r = tr.simulate(n, e_kev, theta_inc, phi_inc, pf, pa_deg, setup.det, setup.xs, setup.kf, rng, mu_const=mu_const, fluorescence=fluorescence)
    ev = dt.reconstruct(r["ix"], r["iy"], r["e"], r["code"], setup.det, e_thr=e_thr, e_window=e_window, ratio_max=ratio_max)
    d = ev["double"]
    c0 = r["code"][:, 0]
    out = {"hist": an.histogram(d["bin"]), "n_double": len(d["bin"]), "n_single": ev["n_single"], "mult": ev["n_triggered_pixels"], "n_fid": n * r["fid_scale"],
           "origin": Counter(dt.double_class(d["c_low"], d["c_high"]).tolist()),
           "first": {int(k): int(v) for k, v in zip(*np.unique(c0[c0 > 0], return_counts=True))}}
    if keep_doubles:
        out["keep"] = {k: d[k] for k in ("e_low", "e_high", "sig", "bin", "c_low", "c_high")}
    return out


def default_workers():
    return int(os.environ.get("CZT_WORKERS", os.cpu_count() or 1))


def run(setup, n_total, e_kev, theta_inc=0.0, phi_inc=0.0, pf=1.0, pa_deg=0.0, seed=1, chunk=CHUNK, mu_const=None, fluorescence=True,
        e_thr=C.E_THRESHOLD_KEV, e_window=C.E_WINDOW_KEV, ratio_max=C.RATIO_MAX, keep_doubles=False, workers=None):
    """Returns a dict: hist (8,), n_double, n_single, n_start, n_fid (photons incident on the fiducial block's entrance face), origin (Counter class name -> doubles, see detector.double_class),
    process_first (Counter of the first interaction code), pixel_multiplicity (9,), and, if keep_doubles, arrays e_low, e_high, sig, bin, c_low, c_high of every selected double."""
    workers = default_workers() if workers is None else int(workers)
    sizes = [chunk] * (int(n_total) // chunk) + ([int(n_total) % chunk] if int(n_total) % chunk else [])
    seeds = np.random.SeedSequence(seed).spawn(len(sizes))
    jobs = [(None if workers > 1 else setup, n, e_kev, theta_inc, phi_inc, pf, pa_deg, ss, mu_const, fluorescence, e_thr, e_window, ratio_max, keep_doubles) for n, ss in zip(sizes, seeds)]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(min(workers, len(jobs)), initializer=_init_worker) as ex:
            results = list(ex.map(_one_chunk, jobs))
    else:
        results = [_one_chunk((setup,) + j[1:]) for j in jobs]
    hist = np.zeros(8); mult = np.zeros(9, int); origin = Counter(); first = Counter(); n_double = n_single = 0; n_fid = 0.0
    keep = {k: [] for k in ("e_low", "e_high", "sig", "bin", "c_low", "c_high")}
    for r in results:
        hist += r["hist"]; mult += r["mult"]; origin.update(r["origin"]); first.update(r["first"]); n_double += r["n_double"]; n_single += r["n_single"]; n_fid += r["n_fid"]
        if keep_doubles:
            for k in keep:
                keep[k].append(r["keep"][k])
    out = {"hist": hist, "n_double": n_double, "n_single": n_single, "n_start": int(n_total), "n_fid": n_fid, "origin": origin, "process_first": first, "pixel_multiplicity": mult}
    if keep_doubles:
        out.update({k: np.concatenate(v) for k, v in keep.items()})
    return out
