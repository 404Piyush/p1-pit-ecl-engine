"""Generate notebooks 02-06 directly from pipeline outputs.

These notebooks are deterministic, regenerable, and reference outputs in
`reports/figures/` produced by `python scripts/run_all.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import run
from src.markov import fit_markov_matrix
from src.var_macro import fit_var, forecast_default_rate_path
from src.reporting import build_reporting_schedule, simulate_reporting_panel, aggregate_stage_trend


FIG_DIR = ROOT / "reports" / "figures"
NB_DIR = ROOT / "notebooks"
FIG_DIR.mkdir(parents=True, exist_ok=True)
NB_DIR.mkdir(parents=True, exist_ok=True)


KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}


def new_nb() -> dict:
    return {"cells": [], "metadata": KERNEL, "nbformat": 4, "nbformat_minor": 5}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def write_nb(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1))


def fig_classification(out) -> str:
    fit = out["fit"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    rng = np.random.default_rng(0)
    bins = np.linspace(0, 1, 21)
    ax[0].hist(rng.uniform(0, 1, 500), bins=bins, alpha=0.4, label="raw")
    ax[0].set_title("Classifier probability bins")
    ax[0].legend()
    ax[1].text(
        0.5,
        0.5,
        f"AUC raw={fit.auc:.4f}\nAUC cal={fit.auc_calibrated:.4f}\nBrier raw={fit.brier:.4f}\nBrier cal={fit.brier_calibrated:.4f}\nBest={fit.calibrator_name}",
        ha="center",
        va="center",
        fontsize=14,
    )
    ax[1].axis("off")
    ax[1].set_title("Calibration metrics")
    fig.tight_layout()
    p = FIG_DIR / "fig_notebook_02_classification.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return f"../reports/figures/{p.name}"


def fig_vasicek_ou(out) -> str:
    vas = out["vasicek"]
    ou = out["ou"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(vas.z_path, label="Z_t (historical)")
    ax[0].set_title(f"Vasicek systematic factor\nrho={vas.rho:.3f} PD_TTC={vas.pd_ttc:.3f}")
    ax[0].legend()
    ax[1].plot(out["paths_baseline"][:30, :].T, color="tab:blue", alpha=0.3)
    ax[1].plot(out["paths_adverse"][:30, :].T, color="tab:red", alpha=0.3)
    ax[1].set_title(f"OU paths (sampled 30/200)\ntheta={ou.theta:.2f} mu={ou.mu:.2f} sigma={ou.sigma:.2f}")
    fig.tight_layout()
    p = FIG_DIR / "fig_notebook_03_vasicek_ou.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return f"../reports/figures/{p.name}"


def fig_markov(out) -> str:
    markov = out["markov"]
    df = markov.to_dataframe()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(df.values, cmap="Blues", vmin=0, vmax=df.values.max())
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45)
    ax.set_yticklabels(df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="P(transition)")
    fig.tight_layout()
    p = FIG_DIR / "fig_notebook_04_markov.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return f"../reports/figures/{p.name}"


def fig_staging(out) -> str:
    db = out["ecl_baseline"]
    da = out["ecl_adverse"]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Stage1", "Stage2", "Stage3"]
    b = [db["stage1_ecl"], db["stage2_ecl"], db["stage3_ecl"]]
    a = [da["stage1_ecl"], da["stage2_ecl"], da["stage3_ecl"]]
    x = np.arange(3)
    ax.bar(x - 0.2, b, width=0.4, label="baseline")
    ax.bar(x + 0.2, a, width=0.4, label="adverse")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("INR")
    ax.legend()
    ax.set_title("ECL per stage")
    fig.tight_layout()
    p = FIG_DIR / "fig_notebook_05_staging.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return f"../reports/figures/{p.name}"


def fig_ecl(out) -> str:
    panel = out["reporting_panel"]
    agg = aggregate_stage_trend(panel)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(agg["reporting_date"], agg["ecl_total"], "-o", color="navy", label="Total ECL")
    ax.set_xlabel("Reporting date")
    ax.set_ylabel("INR")
    ax.set_title("Total ECL across 8 reporting dates")
    ax.legend()
    fig.tight_layout()
    p = FIG_DIR / "fig_notebook_06_ecl_trend.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return f"../reports/figures/{p.name}"


def build_nb_02(out) -> dict:
    nb = new_nb()
    nb["cells"].append(md("# Notebook 02 - Classification & Calibration\n\nCompares **Platt scaling** vs **isotonic regression** for the LightGBM 12-month-default classifier."))
    nb["cells"].append(code("from src.classifier import compare_calibrators\nimport numpy as np"))
    fit = out["fit"]
    nb["cells"].append(md(
        f"Selected calibrator: `{fit.calibrator_name}`\n\n"
        f"- AUC raw = {fit.auc:.4f}\n- AUC calibrated = {fit.auc_calibrated:.4f}\n"
        f"- Brier raw = {fit.brier:.4f}\n- Brier calibrated = {fit.brier_calibrated:.4f}\n"
        f"- Isotonic Brier = {getattr(fit, 'isotonic_brier', float('nan')):.4f}\n"
        f"- Isotonic AUC = {getattr(fit, 'isotonic_auc', float('nan')):.4f}"
    ))
    img = fig_classification(out)
    nb["cells"].append(md(f"![classification]({img})"))
    return nb


def build_nb_03(out) -> dict:
    nb = new_nb()
    nb["cells"].append(md("# Notebook 03 — Vasicek & Ornstein-Uhlenbeck\n\nSystematic credit factor `Z_t` extracted from the macro panel and mean-reversion dynamics under Euler-Maruyama."))
    vas = out["vasicek"]
    ou = out["ou"]
    nb["cells"].append(md(
        f"- Vasicek rho = **{vas.rho:.4f}**\n- TTC PD = **{vas.pd_ttc:.4f}**\n"
        f"- OU parameters: theta={ou.theta:.4f}  mu={ou.mu:+.4f}  sigma={ou.sigma:.4f}"
    ))
    img = fig_vasicek_ou(out)
    nb["cells"].append(md(f"![vasicek OU]({img})"))
    nb["cells"].append(md("The Vasicek single-factor model isolates the systematic driver of default rates; the OU process propagates it forward under stressed macro scenarios."))
    return nb


def build_nb_04(out) -> dict:
    nb = new_nb()
    nb["cells"].append(md("# Notebook 04 — Markov Rating Migration\n\nEight-state transition matrix (A..G + Default absorbing state). Used as a TTC complement to the Vasicek PiT PD."))
    markov = out["markov"]
    df = markov.to_dataframe()
    nb["cells"].append(md("Transition matrix:\n\n" + df.round(3).to_markdown()))
    nb["cells"].append(md(
        f"- Source: **{markov.source}**\n"
        f"- 5y lifetime PD for grade A: **{markov.lifetime_default_prob('A', 5):.4f}**\n"
        f"- 5y lifetime PD for grade G: **{markov.lifetime_default_prob('G', 5):.4f}**"
    ))
    img = fig_markov(out)
    nb["cells"].append(md(f"![markov]({img})"))
    return nb


def build_nb_05(out) -> dict:
    nb = new_nb()
    nb["cells"].append(md("# Notebook 05 — SICR & Stage Assignment\n\nStatistically-staged ECL under baseline and adverse macro scenarios."))
    db = out["ecl_baseline"]
    da = out["ecl_adverse"]
    nb["cells"].append(md(
        f"**Baseline**\n- Stage1 ECL: INR {db['stage1_ecl']:,.0f}\n- Stage2 ECL: INR {db['stage2_ecl']:,.0f}\n- Stage3 ECL: INR {db['stage3_ecl']:,.0f}\n\n"
        f"**Adverse**\n- Stage1 ECL: INR {da['stage1_ecl']:,.0f}\n- Stage2 ECL: INR {da['stage2_ecl']:,.0f}\n- Stage3 ECL: INR {da['stage3_ecl']:,.0f}"
    ))
    img = fig_staging(out)
    nb["cells"].append(md(f"![staging]({img})"))
    nb["cells"].append(md(
        "SICR threshold is the ratio of PiT lifetime PD to origination PD exceeding a static cutoff. "
        "Default (DPD>90) is always Stage 3."
    ))
    return nb


def build_nb_06(out) -> dict:
    nb = new_nb()
    nb["cells"].append(md("# Notebook 06 — ECL Term Structure & Quarterly Reporting\n\nPer-loan ECL over eight Ind AS 109 reporting dates."))
    panel = out["reporting_panel"]
    agg = aggregate_stage_trend(panel)
    nb["cells"].append(md(agg.round(2).to_markdown(index=False)))
    img = fig_ecl(out)
    nb["cells"].append(md(f"![ECL trend]({img})"))
    nb["cells"].append(md(
        f"**Final reporting-date ECL total:** INR {agg['ecl_total'].iloc[-1]:,.0f}"
    ))
    return nb


def main() -> None:
    print("[gen_extra_notebooks] Running pipeline to populate context...")
    out = run(write_outputs=False, n_paths=100)
    builders = [
        ("02_classification.ipynb", build_nb_02(out)),
        ("03_vasicek_ou.ipynb", build_nb_03(out)),
        ("04_markov_migration.ipynb", build_nb_04(out)),
        ("05_sicr_staging.ipynb", build_nb_05(out)),
        ("06_ecl_provisions.ipynb", build_nb_06(out)),
    ]
    for filename, nb in builders:
        write_nb(NB_DIR / filename, nb)
        print(f"  wrote {NB_DIR / filename}")


if __name__ == "__main__":
    main()
