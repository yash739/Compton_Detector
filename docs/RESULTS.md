# Results

All numbers are in `results/*.csv`; figures in `results/figures/`.

**Realistic interaction mix in 5 mm of Cd0.9Zn0.1Te** (`composition_xs.csv`). Shares are of the attenuation (photoabsorption + incoherent + coherent):

| E (keV) | photo | Compton | coherent | Compton : photo | P(interact in 5 mm) | K X-rays per interacting photon |
|---|---|---|---|---|---|---|
| 100 | 87.4% | 6.7% | 5.9% | 0.08 | 0.99 | 0.67 |
| 150 | 74.8% | 17.4% | 7.8% | 0.23 | 0.80 | 0.59 |
| 200 | 61.1% | 30.3% | 8.6% | 0.50 | 0.57 | 0.48 |
| 300 | 38.6% | 53.4% | 8.0% | 1.38 | 0.35 | 0.30 |
| 400 | 25.1% | 68.4% | 6.5% | 2.73 | 0.27 | 0.19 |
| 600 | 12.8% | 83.0% | 4.2% | 6.46 | 0.20 | 0.10 |

Compton and photoabsorption cross sections are equal near 250 keV. The old notebooks used 2.64 /cm at every energy and made every interaction a Compton scatter.
72% of photoabsorptions end in a K X-ray that leaves the interaction site.

**What the selected double events are made of** (`composition_events.csv`, 1e7 photons per energy):

| E (keV) | doubles per 1e6 photons | Compton + photoabsorption | multiple Compton | K fluorescence photon in the neighbour | removed by the old E_high/E_low <= 6 |
|---|---|---|---|---|---|
| 100 | 19339 | 7.1% | 2.8% | **90.1%** | 0% |
| 150 | 54530 | 47.1% | 6.8% | 46.1% | 2% |
| 200 | 58837 | 63.9% | 10.1% | 26.1% | 32% |
| 300 | 42332 | 66.5% | 19.2% | 14.3% | 20% |
| 400 | 31192 | 61.1% | 29.0% | 10.0% | 15% |
| 600 | 14297 | 42.7% | 51.8% | 5.5% | 16% |

The classes are defined by what the two triggered pixels contain (`detector.double_class`); "other" is 0.0% everywhere. Below about 150 keV most apparent doubles are a photoabsorption plus its fluorescence photon,
which carries no polarisation information (isotropic direction). Fluorescence doubles have E_low about 23 keV and E_high/E_low > 6 at 200 keV and above (93-100%), which is why the old ratio cut
acted as a fluorescence filter. With the cut removed, the fluorescence doubles must be modelled, and that is the main non-trivial effect in this package.

**Modulation dilution by fluorescence** (`mu_all / mu_genuine`, all doubles against the Compton-only subset): at PA = 0 it is 0.23 at 100 keV, 0.68 at 200 keV, 0.83 at 300 keV and 0.96 at 600 keV; at PA = 45 it is
0.32 at 100 keV, 0.92 at 150 keV and 0.98-1.02 above that. The dilution depends on the PA because fluorescence photons fill the edge (0, 90, 180, 270 degree) bins preferentially, so the effective modulation factor is not a single number per energy.

**Modulation and PA recovery** (`modulation_E*.csv`, `pa_scan_E300.csv`). For a 100% polarised beam at 300 keV the modulation A/B is 0.23 at PA = 0 and 90 and 0.48 at PA = 45 (the pixel geometry favours the diagonals), and the recovered PA is
179.4 / 45.2 / 88.4 deg for injected 0 / 45 / 90. Over a PA scan with 1.6e5 doubles per point the cosine fit has a systematic offset of up to 10 deg between the symmetry axes (+9.5 at 15 deg, -7.0 at 60 deg, +9.9 at 105 deg, ...)
and is zero at 0, 45, 90 and 135. This is a property of fitting cos 2phi to an 8-bin, square-pixel response, not a statistical error; a PA better than about 10 deg needs a template fit instead of the cosine.

**PF recovery at 300 keV, normal incidence** (`pf_recovery_E300_th0.csv`; PF = mu / mu_100): injected 0 / 0.25 / 0.5 / 0.75 / 1 gives at PA = 45 0.004 / 0.253 / 0.499 / 0.749 / 1.005 (errors 0.008-0.016) and at PA = 0
0.011 / 0.249 / 0.506 / 0.745 / 0.998 (errors 0.016-0.021). PF is unbiased and the errors match the scatter; PA is only meaningful for PF above about 0.25.

**Oblique incidence** (`oblique_E300.csv`, beam azimuth 45 deg): doubles per 1e6 photons rise from 42.6 k to 68-75 k as the polar angle goes 0 to 60 deg, A/B at PA = 45 drops from 0.48 to 0.29, and PF = 0.5 is recovered within errors (0.44-0.52, errors 0.03-0.04)
when the unpolarised and 100% references are taken at the same incidence angle. The chi2 of a pure cos 2phi fit is poor at 30-60 deg (about 50-110 for 5 dof): parallax adds higher harmonics, so oblique results need the template approach, not the cosine.

**What the old conventions did** (`legacy_comparison.csv`, 2e7 photons, 100% polarised; entries are doubles per 1e6 photons / A/B, or A/B only where the yield is unchanged):

| E (keV) | PA | this package | old bin centres in the fit | ratio <= 6 cut | old mu = 2.64 /cm, always Compton | fluorescence off |
|---|---|---|---|---|---|---|
| 200 | 0 | 58.6 k / 0.215 | 0.196 | 39.9 k / 0.311 | 76.1 k / 0.296 | 43.9 k / 0.318 |
| 200 | 45 | 58.6 k / 0.528 | 0.507 | 39.9 k / 0.569 | 76.0 k / 0.470 | 43.8 k / 0.518 |
| 300 | 0 | 42.7 k / 0.239 | 0.219 | 34.1 k / 0.303 | 138.6 k / 0.296 | 37.3 k / 0.286 |
| 300 | 45 | 42.7 k / 0.475 | 0.450 | 34.2 k / 0.492 | 138.1 k / 0.468 | 37.4 k / 0.476 |

The old attenuation overestimates the double yield by 30% at 200 keV and by a factor 3.2 at 300 keV; the old bin centres lower the fitted modulation by 0.02-0.025.

**Speed** (`benchmark_speed.py`): 1.27 M photons/s on one core (0.37 before the optimisations) and 3.16 M photons/s on 6 workers; results are identical for any number of workers.
