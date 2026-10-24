# Review of the original notebooks: findings, fixes and tests

| # | finding in the notebooks | fix | test |
|---|---|---|---|
| 1 | One attenuation coefficient (2.64 /cm) at all energies; every interaction a Compton scatter | Elam/NIST tables, photoabsorption : incoherent split, coherent = no interaction | `test_total_attenuation_matches_nist`, `test_attenuation_is_energy_dependent_and_not_the_old_constant`, `test_first_interaction_probability_follows_the_real_attenuation`, `test_process_mix_at_300_kev` |
| 2 | No photoabsorption, no fluorescence | K fluorescence of Cd, Zn and Te with real yields, lines and absorption lengths | `test_k_fluorescence_probability`, `test_energy_conservation` |
| 3 | Klein-Nishina rejection sampling discarded about 95% of photons (acceptance 1/(2 pi^2)) | exact two-stage sampling: cos theta with bound 2, azimuth from the E vector | `test_kn_theta_marginal`, `test_kn_azimuth_modulation_per_theta` |
| 4 | Polarisation carried as one azimuth angle relative to the x axis and rotated with the direction, not as a 3-vector perpendicular to k; the unpolarised beam drew that angle in the detector plane, which is wrong for oblique beams | `beam_polarisation`, `perpendicular_basis`: polarisation is a unit vector perpendicular to k (uniform angle in that plane for PF = 0, projection of the PA direction for PF > 0), and the scattered polarisation vector is updated exactly | `test_unpolarised_beam_has_no_residual_polarisation`, `test_polarised_beam_is_perpendicular_to_k_and_fraction_respected`, `test_scattering_peaks_perpendicular_to_the_e_vector` |
| 5 | `np.histogram(range=(-pi/4, 8 pi/4), bins=8)` has bins 9 pi/32 wide and the fit used their centres instead of 0, 45, ... deg | direction bins centred on i x 45 deg; exact linear fit | `test_bins_are_exactly_45_degrees`, `test_direction_bins_run_from_the_lower_to_the_higher_energy_pixel`; `legacy_comparison.csv` quantifies the bias |
| 6 | `curve_fit` with badly scaled data (non-convergence, local minima) | exact linear weighted least squares with a covariance | `test_cosine_fit_recovers_the_truth` |
| 7 | PA and the scattering direction confused (`pref_angle` is the scattering direction, 90 deg from the E vector) | PA = E-vector angle, modulation peaks at PA + 90 | `test_scattering_peaks_perpendicular_to_the_e_vector`, `test_pf_and_pa_recovery_end_to_end` |
| 8 | Only one interaction pair simulated; multiple Compton and fluorescence doubles not possible | multi-step transport (up to 10 deposits) and pixel-merged reconstruction | `test_reconstruct_matches_a_literal_reference`, `test_double_classification_is_by_pixel_content`, `test_selection_threshold_adjacency_window_and_ratio` |
| 9 | E_high/E_low <= 6 cut | removed (`RATIO_MAX = None`), still available for comparison | `legacy_comparison.csv` |
| 10 | Bulk runs in plain Python loops, single process | vectorised, chunked, seeded per chunk, parallel | `test_results_do_not_depend_on_the_number_of_workers`, `benchmark_speed.py` |
| 11 | 156 MB virtual environment and generated outputs committed | `.gitignore`; removed from the branch tip | - |

The 156 MB `new_env/` is still in the *history* of the original branch; purging it needs a history rewrite (`git filter-repo`) and a force-push, which has not been done.

Items in the BTP report text itself (not fixable here): PA defined as phi_0 on p. 20; Fig. 4.4 text and Fig. 4.7 caption; the missing table on p. 31; 200 vs 300 keV in the text; 280 vs 255 keV in the kinematics example; the ratio 1-6 criterion
is now outdated; the GRB 221009A numbers and "5 of 20 with Bayes factor > 3" are not backed by anything in the repository.
