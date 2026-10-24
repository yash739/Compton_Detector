"""Run the whole analysis: tests, validations, composition, modulation curves, PA scan, PF recovery, oblique incidence, legacy comparison.

    python scripts/run_all.py [--quick]       (--quick: 1/10 of the photons, a few minutes; the default takes about 30-40 minutes)
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def sh(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    n = lambda full: str(full // 10 if a.quick else full)
    py = sys.executable
    sh([py, "-m", "pytest", "-q", "tests"])
    sh([py, "scripts/validate_cross_sections.py"])
    sh([py, "scripts/validate_kn_sampling.py"])
    sh([py, "scripts/run_composition.py", "--photons", n(10_000_000)])
    sh([py, "scripts/run_modulation.py", "--energy", "300", "--photons", n(20_000_000)])
    sh([py, "scripts/run_modulation.py", "--energy", "200", "--photons", n(20_000_000)])
    sh([py, "scripts/run_pa_scan.py", "--energy", "300", "--photons", n(20_000_000), "--step", "15"])
    sh([py, "scripts/run_pf_recovery.py", "--energy", "300", "--photons", n(20_000_000)])
    sh([py, "scripts/run_oblique.py", "--energy", "300", "--photons", n(20_000_000)])
    sh([py, "scripts/run_legacy_comparison.py", "--photons", n(20_000_000)])


if __name__ == "__main__":
    main()
