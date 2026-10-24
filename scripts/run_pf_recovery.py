"""Polarisation fraction and angle recovery against injected values (replaces bulkrun_script.py).

    python scripts/run_pf_recovery.py [--energy 300] [--theta 0] [--phi 0] [--pfs 0,0.25,0.5,0.75,1] [--pas 0,22.5,45] [--photons 20000000]

For each PA a 100% polarised reference run gives mu_100; each injected PF is recovered as mu / mu_100 from the geometry-corrected cosine fit, with errors from the full
fit covariances. One unpolarised run per incidence is shared by all. Writes results/pf_recovery_E<E>_th<theta>.csv (one aggregated file) and a figure.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from czt_mc import analysis as an, runner  # noqa: E402

RES = os.path.join(HERE, "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", type=float, default=300.0)
    ap.add_argument("--theta", type=float, default=0.0)
    ap.add_argument("--phi", type=float, default=0.0)
    ap.add_argument("--pfs", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--pas", default="0,22.5,45")
    ap.add_argument("--photons", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    pfs = [float(v) for v in a.pfs.split(",")]; pas = [float(v) for v in a.pas.split(",")]
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    s = runner.Setup()
    kw = dict(e_kev=a.energy, theta_inc=a.theta, phi_inc=a.phi)
    unp = runner.run(s, 2 * a.photons, pf=0.0, seed=a.seed, **kw)
    rows = []
    print(f"E = {a.energy} keV, incidence ({a.theta}, {a.phi}) deg; unpolarised reference: {unp['n_double']} doubles")
    print(f"{'PA':>6s} {'PF in':>6s} {'doubles':>8s} {'PF out':>14s} {'PA out':>14s} {'chi2/5':>7s}")
    k = 0
    for pa in pas:
        ref = runner.run(s, a.photons, pf=1.0, pa_deg=pa, seed=a.seed + 1000 + k, **kw); k += 1
        for pf in pfs:
            m = runner.run(s, a.photons, pf=pf, pa_deg=pa, seed=a.seed + k, **kw); k += 1
            o = an.recover_pf_pa(m["hist"], unp["hist"], ref["hist"], rng=np.random.default_rng(k))
            rows.append([pa, pf, m["n_double"], o["pf"], o["pf_err"], o["pa"], o["pa_err"], o["mu_100"], o["chi2_meas"]])
            print(f"{pa:6.1f} {pf:6.2f} {m['n_double']:8d} {o['pf']:7.3f}+-{o['pf_err']:.3f} {o['pa']:7.1f}+-{o['pa_err']:4.1f} {o['chi2_meas']:7.1f}")
    path = os.path.join(RES, f"pf_recovery_E{int(a.energy)}_th{int(a.theta)}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pa_in_deg", "pf_in", "n_doubles", "pf_out", "pf_err", "pa_out_deg", "pa_err_deg", "mu_100", "chi2_5dof"])
        for r in rows:
            w.writerow([f"{v:.5g}" if isinstance(v, float) else v for v in r])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    r = np.array(rows, float)
    for pa in pas:
        m = r[:, 0] == pa
        ax[0].errorbar(r[m, 1], r[m, 3], r[m, 4], fmt="o-", label=f"PA {pa:g}")
        mm = m & (r[:, 1] >= 0.5)
        ax[1].errorbar(r[mm, 1], ((r[mm, 5] - pa + 90) % 180) - 90, r[mm, 6], fmt="o-", label=f"PA {pa:g}")
    ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set_xlabel("injected PF"); ax[0].set_ylabel("recovered PF"); ax[0].legend(fontsize=8)
    ax[1].axhline(0, color="0.6"); ax[1].set_xlabel("injected PF"); ax[1].set_ylabel("recovered - injected PA (deg)"); ax[1].legend(fontsize=8)
    fig.suptitle(f"{a.energy:g} keV, incidence ({a.theta:g}, {a.phi:g}) deg", fontsize=10); fig.tight_layout()
    fig.savefig(os.path.join(RES, "figures", f"pf_recovery_E{int(a.energy)}_th{int(a.theta)}.png"), dpi=130)
    print("wrote", path)


if __name__ == "__main__":
    main()
