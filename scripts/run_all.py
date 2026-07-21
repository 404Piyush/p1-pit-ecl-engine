"""Single-command entrypoint: `python scripts/run_all.py`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_loans", type=int, default=8000)
    parser.add_argument("--n_months", type=int, default=180)
    args = parser.parse_args()
    out = run(seed=args.seed, n_loans=args.n_loans, n_months=args.n_months)
    s = out["stress"]
    print()
    print("=" * 64)
    print("FINAL SUMMARY")
    print("=" * 64)
    print(f"  AUC (Platt-calibrated)        : {out['fit'].auc_calibrated:.4f}")
    print(f"  Brier (Platt-calibrated)      : {out['fit'].brier_calibrated:.4f}")
    print(f"  Vasicek rho                   : {out['vasicek'].rho:.4f}")
    print(f"  OU mean-reversion theta       : {out['ou'].theta:.4f}")
    print(f"  Baseline provisions mean      : INR {s.baseline_provisions_mean:,.0f}")
    print(f"  Adverse provisions mean       : INR {s.adverse_provisions_mean:,.0f}")
    print(f"  Realised adverse losses (mean): INR {s.realised_adverse_losses_mean:,.0f}")
    print(f"  Static-TTC provisions total   : INR {s.static_ttc_provisions_total:,.0f}")
    print(f"  PiT coverage ratio (adv)      : {s.adverse_coverage_ratio_pit:.3f}")
    print(f"  TTC coverage ratio (adv)      : {s.adverse_coverage_ratio_ttc:.3f}")
    print(f"  Forecast-error reduction      : {s.forecast_error_reduction_pct:.2f}%")
    print("=" * 64)
