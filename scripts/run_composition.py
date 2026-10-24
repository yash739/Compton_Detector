"""Compton : photoabsorption : fluorescence in CZT, at the level of cross sections and of selected double events.

    python scripts/run_composition.py [--photons 10000000] [--energies 100,150,200,300,400,500,600]

Writes results/composition_xs.csv         attenuation by process, first-interaction probabilities in 5 mm, K-fluorescence probabilities
       results/composition_events.csv     per energy: doubles per incident photon, class composition of the selected doubles, the share the old E_high/E_low <= 6 cut removed,
                                          the E_low of fluorescence doubles, and the dilution of the modulation by the unpolarised classes
       results/figures/composition.png
Normal incidence, unpolarised beam for the composition; a 100% polarised beam at PA 45 deg (a symmetry axis of the pixels) for the dilution.
"""
import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from czt_mc import analysis as an, constants as C, detector as dt, runner  # noqa: E402

RES = os.path.join(HERE, "..", "results")
CLASSES = dt.CLASS_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photons", type=int, default=10_000_000)
    ap.add_argument("--energies", default="100,150,200,300,400,500,600")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    energies = [float(e) for e in a.energies.split(",")]
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    s = runner.Setup()
    xs, kf, det = s.xs, s.kf, s.det

    # ---- cross sections
    rows = []
    print(f"{'E':>5s} {'photo':>8s} {'Compton':>8s} {'coherent':>8s} {'Compton:photo':>13s} {'P(int,5mm)':>10s} {'P(K x-ray|photo)':>16s} {'K x-rays per interacting photon':>32s}")
    for e in sorted(set(energies + [60.0, 80.0, 125.0, 250.0])):
        ph_, inc, coh = float(xs.photo(e)), float(xs.incoh(e)), float(xs.coh(e))
        tot = ph_ + inc + coh
        p_int = 1 - np.exp(-(ph_ + inc) * det.t)
        pk = float(kf.probability_per_photoabsorption(e))
        kx_per_int = pk * ph_ / (ph_ + inc)
        rows.append([e, ph_, inc, coh, tot, 100 * ph_ / tot, 100 * inc / tot, 100 * coh / tot, inc / ph_, p_int, pk, kx_per_int])
        print(f"{e:5.0f} {ph_:8.3f} {inc:8.3f} {coh:8.3f} {inc / ph_:13.2f} {p_int:10.3f} {pk:16.3f} {kx_per_int:32.3f}")
    with open(os.path.join(RES, "composition_xs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["E_keV", "mu_photo_cm", "mu_incoh_cm", "mu_coh_cm", "mu_total_cm", "pct_photo", "pct_incoh", "pct_coh", "compton_over_photo", "p_interact_5mm", "p_Kxray_per_photoabsorption", "Kxrays_per_interacting_photon"])
        for r in rows:
            w.writerow([f"{v:.6g}" for v in r])

    # ---- events
    erows = []
    print(f"\n{'E':>5s} {'doubles/1e6 photons':>20s} " + " ".join(f"{c[:18]:>20s}" for c in CLASSES) + f" {'ratio>6 removed':>16s} {'mu_all/mu_CP PA0':>17s} {'PA45':>6s}")
    for k, e in enumerate(energies):
        unp = runner.run(s, a.photons, e, pf=0.0, seed=a.seed + 2 * k, keep_doubles=True)
        pols = {pa: runner.run(s, a.photons, e, pf=1.0, pa_deg=pa, seed=a.seed + 10 * k + 1 + int(pa) // 45, keep_doubles=True) for pa in (0.0, 45.0)}
        n = unp["n_double"]
        per_class = Counter(); ratio_removed = Counter(); elow_f = []
        c_all = dt.double_class(unp["c_low"], unp["c_high"])
        ratio = unp["e_high"] / unp["e_low"]
        for c in CLASSES:
            m = c_all == c
            per_class[c] = int(m.sum()); ratio_removed[c] = int(np.sum(m & (ratio > 6.0)))
        elow_f = unp["e_low"][c_all == "K fluorescence photon in the neighbour"]
        # modulation of the whole curve against the curve of the genuine (Compton + photoabsorption) doubles only
        def hist_of(d, mask=None):
            b = d["bin"]
            return an.histogram(b if mask is None else b[mask])
        cp_u = c_all == CLASSES[0]
        fid = unp["n_fid"]
        row = {"E_keV": e, "doubles_per_1e6_photons": 1e6 * n / fid, "frac_removed_by_ratio6": float(np.mean(ratio > 6.0)), "fluor_elow_median_keV": float(np.median(elow_f)) if elow_f.size else 0.0}
        for pa, pol in pols.items():                     # the dilution depends on the PA: fluorescence pairs fill the edge bins, where a 0 deg signal sits, not the corner bins of a 45 deg signal
            cp_p = dt.double_class(pol["c_low"], pol["c_high"]) == CLASSES[0]
            mu_all = an.fit_modulation(*an.corrected_curve(pol["hist"], unp["hist"]))["mu"]
            mu_cp = an.fit_modulation(*an.corrected_curve(hist_of(pol, cp_p), hist_of(unp, cp_u)))["mu"]
            row.update({f"mu_all_pa{int(pa)}": mu_all, f"mu_genuine_pa{int(pa)}": mu_cp, f"dilution_pa{int(pa)}": mu_all / mu_cp})
        for c in CLASSES:
            row["pct " + c] = 100.0 * per_class[c] / max(n, 1)
            row["ratio>6 within " + c] = 100.0 * ratio_removed[c] / max(per_class[c], 1)
        erows.append(row)
        print(f"{e:5.0f} {row['doubles_per_1e6_photons']:20.0f} " + " ".join(f"{row['pct ' + c]:19.1f}%" for c in CLASSES) + f" {100 * row['frac_removed_by_ratio6']:15.1f}% {row['dilution_pa0']:17.3f} {row['dilution_pa45']:6.3f}")
    with open(os.path.join(RES, "composition_events.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(list(erows[0].keys()))
        for r in erows:
            w.writerow([f"{v:.6g}" for v in r.values()])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    e = np.array([r[0] for r in rows]); r = np.array(rows)
    ax[0].semilogy(e, r[:, 1], label="photoabsorption"); ax[0].semilogy(e, r[:, 2], label="Compton"); ax[0].semilogy(e, r[:, 3], label="coherent")
    ax[0].axhline(2.64, color="0.5", ls=":", label="old notebooks (2.64 /cm)")
    ax[0].set_xlabel("E (keV)"); ax[0].set_ylabel("attenuation (1/cm)"); ax[0].legend(fontsize=8); ax[0].set_title("Cd0.9Zn0.1Te")
    ax[1].plot(e, r[:, 10], label="P(K X-ray | photoabsorption)"); ax[1].plot(e, r[:, 9], label="P(interact in 5 mm)"); ax[1].plot(e, r[:, 5] / 100, label="photoabsorption share of interactions (incl. coh.)")
    ax[1].set_xlabel("E (keV)"); ax[1].legend(fontsize=8); ax[1].set_title("interaction and fluorescence probabilities")
    ee = np.array([q["E_keV"] for q in erows]); bottom = np.zeros(len(erows))
    for c in CLASSES:
        v = np.array([q["pct " + c] for q in erows]); ax[2].bar(ee, v, bottom=bottom, width=0.08 * ee, label=c[:42]); bottom += v
    ax[2].set_xscale("log"); ax[2].set_xlabel("E (keV)"); ax[2].set_ylabel("% of selected doubles"); ax[2].legend(fontsize=6); ax[2].set_title("what the selected doubles are")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "figures", "composition.png"), dpi=130)
    print("wrote results/composition_*.csv and figures/composition.png")


if __name__ == "__main__":
    main()
