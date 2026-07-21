"""Generate `notebooks/01_end_to_end.ipynb` from the canonical pipeline.

This script regenerates the notebook deterministically so that the figures,
tables, and quoted numbers stay in sync with `src/pipeline.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.pipeline import run


FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def md_image(rel_path: str, alt: str = "") -> str:
    return f"![{alt}]({rel_path})"


def build_notebook(out: dict) -> dict:
    nb = {}
    nb["cells"] = []
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 5

    stress = out["stress"]
    fit = out["fit"]
    vas = out["vasicek"]
    ou = out["ou"]
    ecl_b = out["ecl_baseline"]
    ecl_a = out["ecl_adverse"]
    pd_b = out["pd_curve_5y_baseline"]  # (n_paths, 5)
    pd_a = out["pd_curve_5y_adverse"]
    diag = out["term_structure_diag"]

    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        "# Project 1 — Point-in-Time ECL Engine\n",
        "End-to-end walkthrough of the Vasicek-calibrated, dynamically-staged\n",
        "Ind AS 109 expected credit loss engine. All numbers in this notebook\n",
        "are produced by `src/pipeline.py`; the entire pipeline runs with\n",
        "`python scripts/run_all.py`.\n",
    ]})

    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        "## 1. Synthetic retail loan book\n",
        "**Schema.** Each row is a single loan. `int_rate`, `loan_amnt`, `grade`,\n",
        "`annual_inc`, `dti`, `delinq_2yrs`, `inq_last_6mths`, `revol_util`,\n",
        "`home_ownership`, `purpose` mirror public LendingClub disclosure.\n",
        "`default_12m` is the 12-month default flag and `origination_z` is the\n",
        "Vasicek systematic factor at origination (leakage-free; we never feed\n",
        "this column into the classifier).\n",
    ]})

    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "synthetic_loans.csv")
    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
        "import pandas as pd\n",
        "df = pd.read_csv('data/synthetic_loans.csv')\n",
        "print('Shape:', df.shape)\n",
        "print('Default rate (%):', round(100 * df['default_12m'].mean(), 2))\n",
        "print('Grade distribution:')\n",
        "print(df['grade'].value_counts().sort_index())\n",
        f"df.head(3)\n",
    ]})

    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        "## 2. Macroeconomic series and default rates\n",
        "`synthetic_macro.csv` contains monthly observations of GDP growth,\n",
        "CPI, repo rate, unemployment, plus the latent systematic factor\n",
        "`z_systematic` and the implied `default_rate`.\n",
    ]})

    macro = pd.read_csv(ROOT / "data" / "synthetic_macro.csv", parse_dates=["month"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    axes[0, 0].plot(macro["month"], macro["default_rate"], color="C0")
    axes[0, 0].set_title("Observed default rate $DR_t$")
    axes[0, 0].set_ylabel("PD")
    axes[0, 1].plot(macro["month"], macro["gdp_growth"], color="C1")
    axes[0, 1].set_title("GDP growth")
    axes[1, 0].plot(macro["month"], macro["repo_rate"], color="C2")
    axes[1, 0].set_title("Repo rate")
    axes[1, 1].plot(macro["month"], macro["unemployment"], color="C3")
    axes[1, 1].set_title("Unemployment")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_macro_drivers.png", dpi=130)
    plt.close()

    fig2, ax = plt.subplots(figsize=(9, 4))
    ax.plot(macro["month"], macro["z_systematic"], color="black")
    ax.axhline(0.0, color="grey", lw=0.7)
    ax.set_title("Vasicek systematic factor $Z_t$ (extracted)")
    ax.set_ylabel("Z")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_z_systematic.png", dpi=130)
    plt.close()

    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
        "import pandas as pd\n",
        "macro = pd.read_csv('data/synthetic_macro.csv', parse_dates=['month'])\n",
        "macro[['month','default_rate','gdp_growth','repo_rate','unemployment']].head()\n",
    ]})
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        md_image("../reports/figures/fig_macro_drivers.png", "Macro drivers") + "\n\n" +
        md_image("../reports/figures/fig_z_systematic.png", "Z systematic") + "\n",
    ]})

    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        "## 3. Vasicek model and Ornstein-Uhlenbeck dynamics\n",
        "Closed-form inversion yields the systematic factor Z_t. Asset correlation\n",
        "ρ is calibrated by moment-matching default-rate volatility to the\n",
        "Vasicek identity. The mean-reverting Z_t dynamics are estimated by\n",
        "exact MLE on the discretised AR(1) form.\n",
        f"\n",
        f"- Asset correlation ρ = **{vas.rho:.4f}**\n",
        f"- Long-run PD_PiT = **{vas.pd_ttc:.4f}**\n",
        f"- OU θ = **{ou.theta:.4f}**   μ = **{ou.mu:+.4f}**   σ = **{ou.sigma:.4f}**\n",
    ]})

    fig3, ax = plt.subplots(figsize=(9, 4))
    ax.hist(vas.z_path, bins=40, color="C0", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_title("Distribution of $Z_t$ across history")
    ax.set_xlabel("Z")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_z_distribution.png", dpi=130)
    plt.close()
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        md_image("../reports/figures/fig_z_distribution.png", "Z distribution") + "\n",
    ]})

    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        "## 4. Forward OU simulation — baseline vs adverse\n",
        f"200 Monte-Carlo paths simulated for 60 monthly steps (5y) under\n",
        f"the calibrated OU process. The adverse scenario overrides both\n",
        f"the starting state and the long-run mean to model a lasting\n",
        f"regime shift.\n",
    ]})

    fig4, axes = plt.subplots(1, 2, figsize=(11, 4))
    paths_b = out["paths_baseline"]
    paths_a = out["paths_adverse"]
    paths_full_b = np.zeros((paths_b.shape[0], 6))
    paths_full_b[:, 0] = float(vas.z_path[-1])
    paths_full_b[:, 1:6] = paths_b
    paths_full_a = np.zeros((paths_a.shape[0], 6))
    paths_full_a[:, 0] = float(vas.z_path[-1]) - 1.5
    paths_full_a[:, 1:6] = paths_a
    t = np.arange(6)
    for p in paths_full_b[:80]:
        axes[0].plot(t, p, color="C0", alpha=0.15)
    axes[0].plot(t, paths_full_b.mean(axis=0), color="black", lw=2, label="mean path")
    axes[0].set_title("Baseline $Z_t$ paths (5y)")
    axes[0].set_xlabel("Year")
    axes[0].legend()
    for p in paths_full_a[:80]:
        axes[1].plot(t, p, color="C3", alpha=0.15)
    axes[1].plot(t, paths_full_a.mean(axis=0), color="black", lw=2, label="mean path")
    axes[1].set_title("Adverse $Z_t$ paths (5y)")
    axes[1].set_xlabel("Year")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_ou_paths.png", dpi=130)
    plt.close()
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        md_image("../reports/figures/fig_ou_paths.png", "OU paths") + "\n",
    ]})

    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        "### PD term structure (path-mean across simulations)\n",
        f"- Baseline: **{np.round(pd_b.mean(axis=0), 4).tolist()}**\n",
        f"- Adverse:  **{np.round(pd_a.mean(axis=0), 4).tolist()}**\n",
        "\n",
        "The Vasicek conditional PD translates each year's mean Z into a\n",
        "yearly marginal PD, giving the 5-year point-in-time term structure.\n",
    ]})

    fig5, ax = plt.subplots(figsize=(9, 4))
    yrs = np.arange(1, 6)
    ax.plot(yrs, pd_b.mean(axis=0), marker="o", color="C0", label="Baseline")
    ax.plot(yrs, pd_a.mean(axis=0), marker="o", color="C3", label="Adverse")
    ax.fill_between(yrs,
                    np.percentile(pd_b, 25, axis=0),
                    np.percentile(pd_b, 75, axis=0),
                    color="C0", alpha=0.15)
    ax.fill_between(yrs,
                    np.percentile(pd_a, 25, axis=0),
                    np.percentile(pd_a, 75, axis=0),
                    color="C3", alpha=0.15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Marginal PD")
    ax.set_title("Annual marginal PD term structure")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_pd_term_structure.png", dpi=130)
    plt.close()
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        md_image("../reports/figures/fig_pd_term_structure.png", "Term structure") + "\n",
    ]})

    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        "### Sample cohort diagnostic (from the Ind AS 109 brief)\n",
        "Reproducing the marginal-to-cumulative PD identity from the brief:\n",
        "$\\Pi_{t}(1 - {}_{t-1}q_t) = (1 - {}_0q_T)$.\n",
    ]})
    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
        "from src.pd_term_structure import term_structure_diagnostic\n",
        "diag = term_structure_diagnostic()\n",
        "for y, m, c in zip([1,2,3], diag['marginal'], diag['cumulative']):\n",
        "    print(f'  Year {y}: marginal={m:.4f}  cumulative={c:.4f}')\n",
    ]})

    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        "## 5. LightGBM classifier with Platt scaling\n",
        f"Classifier is trained on a leakage-clean feature set with time-series\n",
        f"split. Platt scaling maps raw scores to calibrated probabilities.\n",
        f"\n",
        f"- Test AUC (raw):  **{fit.auc:.4f}**\n",
        f"- Test AUC (Platt):**{fit.auc_calibrated:.4f}**\n",
        f"- Brier (raw):     **{fit.brier:.4f}**\n",
        f"- Brier (Platt):   **{fit.brier_calibrated:.4f}**\n",
    ]})

    fig6, axes = plt.subplots(1, 2, figsize=(11, 4))
    ecdf = pd.read_csv(ROOT / "data" / "ecl_output.csv")
    pos = ecdf[ecdf["default_12m"] == 1]["pd_test_baseline"]
    neg = ecdf[ecdf["default_12m"] == 0]["pd_test_baseline"]
    axes[0].hist([neg, pos], bins=30, color=["C0", "C3"], label=["non-default", "default"], alpha=0.8)
    axes[0].set_xlabel("Calibrated PD")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Score distribution by label")
    axes[0].legend()
    axes[1].scatter(ecdf["pd_test_baseline"], ecdf["pd_test_adverse"],
                     s=4, alpha=0.4, color="C0")
    axes[1].plot([0, 0.3], [0, 0.3], color="black", lw=0.5)
    axes[1].set_xlabel("Baseline PD")
    axes[1].set_ylabel("Adverse PD")
    axes[1].set_title("PiT shift under adverse scenario")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_classifier_outputs.png", dpi=130)
    plt.close()
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        md_image("../reports/figures/fig_classifier_outputs.png", "Classifier outputs") + "\n",
    ]})

    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        "## 6. SICR-driven staging\n",
        "Each loan is assigned to Stage 1, 2 or 3 using the rule:\n",
        "\n",
        "- Stage 3 if DPD > 90 days (defaulted at reporting date).\n",
        "- Stage 2 if 30 < DPD ≤ 90, **or** PiT-LifetimePD / OriginationPD > 1.5\n",
        "  (the SICR criterion that replaces static DPD triggers).\n",
        "- Stage 1 otherwise.\n",
    ]})

    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
        "from src.staging import stage_breakdown, STAGE1, STAGE2, STAGE3\n",
        "out_df = pd.read_csv('data/ecl_output.csv')\n",
        "print('Stage mix (baseline):', stage_breakdown(out_df['stage_baseline'].values))\n",
        "print('Stage mix (adverse) :', stage_breakdown(out_df['stage_adverse'].values))\n",
    ]})

    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        "## 7. ECL aggregation per stage and scenario\n",
        f"For Stage 1 we compute the 12-month ECL (sum of T=1 year PD · LGD · EAD · D).\n",
        f"For Stage 2/3 we run the lifetime (T=5 year) term structure with the\n",
        f"stressed-LGD uplift applied to Stage 3.\n",
    ]})

    labels = ["Stage 1", "Stage 2", "Stage 3"]
    b_vals = [ecl_b["stage1_ecl"], ecl_b["stage2_ecl"], ecl_b["stage3_ecl"]]
    a_vals = [ecl_a["stage1_ecl"], ecl_a["stage2_ecl"], ecl_a["stage3_ecl"]]
    fig7, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, b_vals, width=0.4, color="C0", label="Baseline")
    ax.bar(x + 0.2, a_vals, width=0.4, color="C3", label="Adverse")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("ECL (INR)")
    ax.set_title("Stage-wise ECL totals")
    ax.legend()
    for i, (b, a) in enumerate(zip(b_vals, a_vals)):
        ax.text(i - 0.2, b, f"{b:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, a, f"{a:,.0f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_ecl_per_stage.png", dpi=130)
    plt.close()
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        md_image("../reports/figures/fig_ecl_per_stage.png", "ECL per stage") + "\n",
    ]})

    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        "## 8. Stress test — forecast-error reduction\n",
        "We measure how well each candidate engine anticipates the realised\n",
        "first-year portfolio losses under the adverse scenario. Coverage\n",
        "ratio = mean(provisions) / mean(realised losses).\n",
        "\n",
        f"- PiT mean provisions (adverse) : INR **{stress.adverse_provisions_mean:,.0f}**\n",
        f"- Realised adverse losses (mean) : INR **{stress.realised_adverse_losses_mean:,.0f}**\n",
        f"- Static-TTC provisions          : INR **{stress.static_ttc_provisions_total:,.0f}**\n",
        f"- PiT coverage ratio (adverse)   : **{stress.adverse_coverage_ratio_pit:.3f}** (>1 = over-provisioned, regulatorily safe)\n",
        f"- TTC coverage ratio (adverse)   : **{stress.adverse_coverage_ratio_ttc:.3f}** (<1 = under-provisioned, regulatorily risky)\n",
        f"- **Forecast-error reduction**   : **{stress.forecast_error_reduction_pct:.2f}%**\n",
    ]})

    fig8, ax = plt.subplots(figsize=(8, 4))
    methods = ["Static TTC", "PiT (mean)", "Realised"]
    values = [
        stress.static_ttc_provisions_total,
        stress.adverse_provisions_mean,
        stress.realised_adverse_losses_mean,
    ]
    colors = ["grey", "C0", "black"]
    bars = ax.bar(methods, values, color=colors)
    ax.set_ylabel("Portfolio loss / provision (INR)")
    ax.set_title("Coverage under adverse scenario")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_stress_coverage.png", dpi=130)
    plt.close()
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        md_image("../reports/figures/fig_stress_coverage.png", "Stress coverage") + "\n",
    ]})

    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        "## 9. Audit trail\n",
        "Every run produces `reports/audit_trail.json` with hashed model\n",
        "manifests, calibrated parameters, ECL totals, and the Ind AS 109\n",
        "staging counts. This is what you would table during a regulatory\n",
        "review meeting.\n",
    ]})
    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
        "import json\n",
        "with open('reports/audit_trail.json') as f:\n",
        "    audit = json.load(f)\n",
        "print(json.dumps({k: v for k, v in audit.items() if k not in ('data_hashes',)}, indent=2, default=str)[:1800])\n",
    ]})

    return nb


def main():
    print("[gen_notebook] Running canonical pipeline...")
    out = run(seed=42, n_loans=8000, n_months=180, write_outputs=False, n_paths=200)
    print("[gen_notebook] Building notebook...")
    nb = build_notebook(out)
    nb_path = ROOT / "notebooks" / "01_end_to_end.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1))
    print(f"[gen_notebook] wrote {nb_path}")


if __name__ == "__main__":
    main()
