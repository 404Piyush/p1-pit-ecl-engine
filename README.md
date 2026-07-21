# Project 1 — Point-in-Time Expected Credit Loss (ECL) Engine

**Vasicek-calibrated · OU-simulated · Ind AS 109 compliant**

A production-grade loss-forecasting engine that replaces rule-based staging
triggers with a dynamically calibrated Point-in-Time term structure. The
pipeline ingests a retail loan book plus macroeconomic state variables,
produces a continuously updated Probability-of-Default term structure,
and computes per-loan regulatory provisions with a fully auditable
data lineage.

> Cross-references the Ind AS 109 framework, the Vasicek single-factor
> model, Ornstein-Uhlenbeck mean-reversion, Platt-vs-Isotonic-scaled
> LightGBM classification, a VAR(p) macro forecaster, an 8-state Markov
> rating-migration matrix, per-loan quarterly reporting, and a deterministic
> PII-redaction layer for audit-trail compliance.

---

## Build & security status

| Track | Status |
| --- | --- |
| Core MVP (LightGBM + Vasicek + OU + SICR + ECL) | ✅ passing |
| Phase 2 — Isotonic regression auto-selection | ✅ passing |
| Phase 2 — VAR(p) macro forecaster | ✅ passing (lag 1) |
| Phase 2 — 8-state Markov migration matrix | ✅ passing |
| Phase 2 — Per-loan 8-quarter reporting panel | ✅ passing (40,168 rows) |
| Phase 2 — Deterministic PII redaction (PAN/Aadhaar/mobile/email/account) | ✅ wired + tested |
| Phase 2 — Notebooks 02-06 | ✅ generated |
| Security audit (`bandit`, `pip_audit`, manual) | ✅ `SECURITY_AUDIT.md` (8 findings, 6 Phase-3 recommendations) |

---

## Results (measured on this build)

| Metric | Value |
| --- | --- |
| Test AUC (raw LightGBM) | 0.6059 |
| Test AUC (best-calibrated) | 0.6059 |
| Brier (raw) / (Platt) / (Isotonic) | 0.0920 / 0.0923 / 0.0930 |
| Selected calibrator | `platt` (by Brier) |
| Asset correlation ρ (Vasicek) | 0.0500 |
| VAR(p) lag order (AIC) | 1 |
| VAR AIC / BIC | -59.09 / -58.56 |
| Markov 5y PD — grade A / G | 0.4750 / 0.4238 (synthetic generator) |
| OU mean-reversion θ | 3.1546 / year |
| OU long-run mean μ | -0.172 |
| OU volatility σ | 1.620 |
| Mean PD term structure (adverse) | 9.79% → 10.35% over 5y |
| Baseline provisions (mean path) | INR 3,257,986 |
| Adverse provisions (mean path) | INR 6,248,282 |
| Realised first-year losses (adverse mean) | INR 4,934,505 |
| Static-TTC provisions | INR 2,262,861 |
| **PiT coverage ratio (adverse)** | **1.266** (over-provisioned) |
| **TTC coverage ratio (adverse)** | **0.459** (under-provisioned) |
| **Forecast-error reduction vs TTC** | **50.83%** |
| Reporting panel rows (8 dates × 5,021 loans) | 40,168 |
| Final reporting-date ECL total | INR 8,238,174 |

These figures are reproducible by running:

```bash
py -3.11 scripts/run_all.py
```

> The original brief quoted a generic "15% provisioning-variance
> reduction" figure. The well-defined analogue here is
> `|1 − coverage|`. Static TTC under-provisions by 54.1% under stress,
> whereas the PiT engine over-provisions by only 26.6% — a forecast-error
> reduction of 50.83%. The number that we report is the *measured* one.

---

## Resume blueprint

| Structural dimension | Detail |
| --- | --- |
| **Project title** | Point-in-Time Expected Credit Loss (ECL) Engine with Vasicek Macroeconomic Calibration and Dynamic Asset Staging |
| **Target dataset** | Modified LendingClub-schema retail loan book (8,000 loans) plus a synthetic 15-year monthly macroeconomic state series; engineered to be Ind AS 109 test-ready while remaining reproducible. |
| **Core models & math** | LightGBM classification, **Platt scaling vs Isotonic-regression calibration auto-selected** by Brier score, Vasicek single-factor systematic model, Ornstein-Uhlenbeck Euler-Maruyama simulation, **VAR(p) macro forecaster**, **eight-state Markov rating migration matrix**, survival-form PD term structure, Euler-Maruyama numerical integration, **per-loan quarterly Ind AS 109 reporting panel**. |
| **Unique selling proposition** | Replaces static DPD-based staging triggers with a forward-looking PD term structure derived from Vasicek + OU dynamics, enabling earlier SICR detection and a calibrated balance-sheet provision. |
| **Quantifiable impact** | Forecast-error reduction vs static TTC of 50.83% under an adverse macro scenario, with full audit trail and per-stage provision roll. |

### Ind AS 109 staging summary

| Stage | Criterion | ECL horizon | Operational metric |
| --- | --- | --- | --- |
| **Stage 1** | DPD ≤ 30, no SICR | 12-month ECL | stable payments, on-time servicing |
| **Stage 2** | 30 < DPD ≤ 90 **or** PiT-LifetimePD / OriginationPD > 1.5 | Lifetime ECL | rating downgrade, restructure flag |
| **Stage 3** | DPD > 90, bankruptcy, severe financial distress | Lifetime ECL + stressed LGD | default / write-off |

### Mathematical anchors implemented

- **ECL formula**

  $$\text{ECL}_i = \sum_{t=1}^{T} PD_{i,t} \cdot LGD_{i,t} \cdot EAD_{i,t} \cdot D_t$$

- **Vasicek systematic-factor inversion**

  $$Z_t = \frac{\Phi^{-1}(PD_{TTC}) - \sqrt{1-\rho}\,\Phi^{-1}(DR_t)}{\sqrt{\rho}}$$

- **Ornstein-Uhlenbeck Euler-Maruyama**

  $$Z_{t+\Delta t} = Z_t + \theta(\mu - Z_t)\Delta t + \sigma\sqrt{\Delta t}\,\varepsilon_t$$

- **Marginal-to-cumulative PD identity** (verified against the
  brief's sample cohort table)

  $$\prod_{t=1}^{T}(1 - {}_{t-1}q_t) = (1 - {}_0q_T)$$

---

## Repository layout

```
p1-pit-ecl-engine/
├── README.md
├── requirements.txt                # lightgbm, scikit-learn, statsmodels, ...
├── data/
│   ├── synthetic_loans.csv          # 8,000 loans, 18 cols
│   ├── synthetic_macro.csv          # 180 monthly obs, 7 cols
│   └── ecl_output.csv               # per-loan staging + ECL outputs
├── src/
│   ├── data_gen.py                  # synthetic lender + macro generator
│   ├── features.py                  # leakage-safe column config
│   ├── classifier.py                # LightGBM + Platt/Isotonic auto-selection
│   ├── vasicek.py                   # Z_t closed-form + ρ calibration
│   ├── ornstein_uhlenbeck.py        # OU fit (MLE) + Euler-Maruyama sim
│   ├── pd_term_structure.py         # marginal ↔ cumulative PD identity
│   ├── staging.py                   # SICR + Stage 1/2/3 engine
│   ├── ecl.py                       # LGD/EAD/ECL calculator
│   ├── stress.py                    # forecast-error reduction metric
│   ├── markov.py                    # 8-state rating migration matrix
│   ├── var_macro.py                 # VAR(p) macro forecaster
│   ├── reporting.py                 # 8-quarter per-loan Ind AS 109 panel
│   ├── redact.py                    # deterministic PII masker (PAN, Aadhaar, …)
│   └── pipeline.py                  # canonical orchestrator (12 steps)
├── scripts/
│   ├── run_all.py                   # one-command execution
│   ├── gen_notebook.py              # regenerate the E2E notebook
│   └── gen_extra_notebooks.py       # rebuild notebooks 02-06
├── notebooks/
│   ├── 01_end_to_end.ipynb
│   ├── 02_classification.ipynb
│   ├── 03_vasicek_ou.ipynb
│   ├── 04_markov_migration.ipynb
│   ├── 05_sicr_staging.ipynb
│   └── 06_ecl_provisions.ipynb
└── reports/
    ├── audit_trail.json             # Ind AS 109 lineage manifest
    ├── bandit_report.txt            # static-analysis report
    ├── pip_audit.txt                # dependency CVE scan
    ├── SECURITY_AUDIT.md            # full security audit (F-1…F-8, R-1…R-6)
    └── figures/                     # PNGs produced by the notebooks
```

---

## Running it

### 1. Install

```bash
py -3.11 -m pip install -r requirements.txt
```

### 2. Run the canonical pipeline

```bash
py -3.11 scripts/run_all.py
```

This writes:
- `data/synthetic_loans.csv`
- `data/synthetic_macro.csv`
- `data/ecl_output.csv` (per-loan staging + ECL)
- `reports/audit_trail.json` (model lineage, calibrated parameters,
  stress results)

### 3. (Optional) Rebuild the notebooks

```bash
py -3.11 scripts/gen_notebook.py        # rebuild 01_end_to_end.ipynb
py -3.11 scripts/gen_extra_notebooks.py # rebuild notebooks 02-06
```

These regenerate the notebooks deterministically from the canonical
pipeline, including all figures and the audit-trail snippets.

### 4. Open any notebook

```bash
py -3.11 -m jupyter notebook notebooks/
```

### 5. Security audit

```bash
py -3.11 -m bandit -r src scripts      # static analysis
py -3.11 -m pip_audit --requirement requirements.txt --strict   # CVE scan
```

Both reports are saved under `reports/` automatically. See
[`SECURITY_AUDIT.md`](SECURITY_AUDIT.md) for the full audit narrative,
findings (F-1…F-8), and the Phase 3 recommendations backlog (R-1…R-6).

---

## Pipeline architecture

```
synthetic loan book (8k rows)
        │
        ▼
[ leakage-safe feature table ]
        │
        ▼
[ LightGBM classifier (time-series CV, early stopping) ]
        │
        ▼
[ Platt scaling calibration ]
        │
        ▼
[ Vasicek Z_t  ←  observed DR_t and PD_TTC  +  ρ calibration ]
        │
        ▼
[ Ornstein-Uhlenbeck MLE on discretised AR(1) ]
        │
        ▼
[ N forward paths × 5y horizon × 2 scenarios ]
        │
        ▼
[ PiT-LifetimePD per loan ]
        │
        ▼
[ SICR + Stage 1/2/3 assignment ]
        │
        ▼
[ LGD / EAD / ECL = Σ_t PD_t · LGD_t · EAD_t · D_t ]
        │
        ▼
[ Per-stage provisions + path-wise stress + audit trail ]
```

---

## Key implementation choices

- **Calibrated probability scale.** LightGBM raw scores are
  monotonic but clustered near the boundaries. We apply Platt scaling
  on a time-aware validation split so the marginal PDs feed the ECL
  formula on a true probability scale.
- **Path-wise stress.** The OU simulator runs 200 paths under both a
  baseline and a long-run-shock adverse scenario. Each path's 5-year
  PD curve feeds a full re-evaluation of the ECL formula, producing
  an honest dispersion of provisions across macro futures rather than
  a marketing-style point estimate.
- **Forecast-error metric.** The metric of record is the absolute
  deviation from realised 12-month portfolio losses (`|1 − coverage|`).
  Ind AS 109 cares about under-provisioning more than over-provisioning
  because the latter is regulatory capital, but the former triggers
  shortfalls. We report both coverage ratios and the percentage
  forecast-error reduction.
- **Audit-trail JSON.** Every model version, calibrated parameter,
  data hash, and stress result is logged in a single JSON manifest so
  the run is fully reproducible.

---

## What is deferred to Phase 3

The MVP and Phase 2 are both complete. The remaining backlog is
operational/CI hygiene rather than modelling depth:

- Real LendingClub CSV ingest (currently we use a LendingClub-schema
  synthetic generator; plug-in path is documented).
- Dependency lockfile (`pip-compile` / `uv.lock`).
- CI integration of `bandit` and `pip_audit`.
- Schema-enforced PII redaction when promoted to production ingestion.

---

## Citation hook

A natural next step would be to extend the engine with **revenue
attribution** under a stable coin / tokenized credit asset framework,
but that is well outside the Ind AS 109 scope of this project.

---

## License & data provenance

The synthetic data generator embeds the *LendingClub schema* but
generates values internally; no proprietary data is used. The Vasicek
model and Ornstein-Uhlenbeck process follow the standard literature
(Vasicek 2002; OU process as the continuous-time limit of the
discretised AR(1)).

---

## What is this project useful for?

This engine is a **production-style reference implementation** of an
Ind AS 109 forward-looking credit-loss provisioning system. Three
distinct audiences can use it:

### 1. Retail lenders (NBFCs, HFCs, banks)
- **Regulatory provisioning.** Replaces rule-based DPD triggers with a
  statistically-grounded Significant Increase in Credit Risk (SICR)
  test, producing Stage 1 / 2 / 3 classifications and balance-sheet
  provisions compliant with Ind AS 109, IFRS 9, and CECL.
- **Macro stress testing.** The Vasicek + Ornstein-Uhlenbeck stack
  translates forward macro paths into PD term structures, so a CFO can
  ask *"what happens to provisions if unemployment rises 1.5 σ?"* and
  get an honest distribution, not a marketing point estimate.
- **Audit lineage.** `reports/audit_trail.json` is a versioned manifest
  of every model, parameter, hash, and metric — designed to drop
  straight into an internal model-validation memo.

### 2. Quantitative job candidates
- **CV blueprint.** The five-row resume block in §"Resume blueprint"
  maps directly to credit-risk job descriptions at firms like
  Godrej Capital, HDFC, ICICI, Bajaj Finance, PaySense, and LazyPay.
- **Interview talking points.** Vasicek inversion, Euler-Maruyama
  numerical stability, calibration drift, SICR threshold sensitivity,
  Stage-3 LGD overlap with collaterals — all are defensible under
  follow-up "why?" questions.
- **Portfolio piece.** Pair it with your other projects to form the
  "credit lifecycle" trio the brief recommended.

### 3. Researchers / students
- **Pure-math reference.** Every formula from the brief is implemented
  with explicit numerical guards (e.g. `eps = 1e-6` clipping on
  probabilities to keep `scipy.special.logit` finite).
- **Reproducible.** Single-seed determinism + line-by-line pipeline
  print makes the engine easy to dissect for a credit-modelling
  seminar.

---

## How to use this project

### A. Run the canonical pipeline (one command)

```bash
py -3.11 -m pip install -r requirements.txt
py -3.11 scripts/run_all.py
```

What you get on disk:
| File | Purpose |
| --- | --- |
| `data/synthetic_loans.csv` | 8,000-loan LendingClub-schema book (PII-redacted) |
| `data/synthetic_macro.csv` | 15-year monthly macro panel |
| `data/ecl_output.csv` | per-loan staging + ECL under baseline & adverse |
| `data/reporting_panel.csv` | per-loan × 8-quarter Ind AS 109 reporting grid |
| `reports/audit_trail.json` | machine-readable model lineage manifest |
| `reports/figures/*.png` | calibration, stage, OU-paths plots |

### B. Read the notebooks

```bash
py -3.11 -m jupyter notebook notebooks/
```

- `01_end_to_end.ipynb` — full walk-through (start here)
- `02_classification.ipynb` — Platt vs Isotonic calibration
- `03_vasicek_ou.ipynb` — systematic factor + mean-reversion sim
- `04_markov_migration.ipynb` — rating-transition matrix
- `05_sicr_staging.ipynb` — Stage 1/2/3 boundaries under stress
- `06_ecl_provisions.ipynb` — quarterly provisioning trend

### C. Use individual modules from Python

```python
from src.vasicek import fit_vasicek, conditional_pd
from src.ornstein_uhlenbeck import fit_ou_parameters, simulate_ou_paths
from src.markov import fit_markov_matrix
from src.redact import redact_dataframe_columns, RedactionConfig

# Vasicek systematic factor from an observed default-rate series
vfit = fit_vasicek(dr_series)
z = vfit.z_path                              # systematic factor
pd = conditional_pd(z, pd_ttc=0.05, rho=0.05) # conditional PDs

# Mean-reverting OU forward simulation
ou = fit_ou_parameters(z, dt=1/12)
paths = simulate_ou_paths(z0=z[-1], n_steps=60, n_paths=200, params=ou, dt=1/12)

# Redact PII before any external write
import pandas as pd
df = pd.DataFrame({"pan": ["ABCDE1234F"], "aadhaar": ["234123412346"]})
redact_dataframe_columns(df, cfg=RedactionConfig(salt="your-salt"))
```

### D. Plug in real LendingClub data (production path)

The pipeline reads `synthetic_loans.csv` by default. To swap in a real
LendingClub slice:

1. Drop the file at `data/raw/loans.csv` with the same column names
   (or open an issue for a schema-mapping helper).
2. Replace `generate_synthetic_loan_book(...)` in
   `src/pipeline.py:131` with
   `loans = pd.read_csv("data/raw/loans.csv")`.
3. Add a calibration step on the new PD_TTC before running.
4. Re-run `scripts/run_all.py` — the audit trail records the new
   lineage hash automatically.

### E. Run the security audit

```bash
py -3.11 -m bandit -r src scripts
py -3.11 -m pip_audit --requirement requirements.txt --strict
```

Full narrative: [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md).
