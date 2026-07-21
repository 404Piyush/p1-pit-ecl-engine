# Changelog

All notable changes to **p1-pit-ecl-engine** are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Real LendingClub CSV ingest with schema-mapping helper
- Dependency lockfile via `pip-compile`
- Schema-enforced PII redaction when promoted to production ingestion

---

## [0.3.0] - 2026-07-22

### Added
- **Markov-chain rating migration matrix** (`src/markov.py`): 8-state A..G + Default absorbing
- **VAR(p) macro forecaster** (`src/var_macro.py`): statsmodels-based with AIC lag-selection
- **Per-loan quarterly reporting simulator** (`src/reporting.py`): 8 reporting dates, 40,168-row panel
- **Deterministic PII redaction** (`src/redact.py`): HMAC-SHA-256 tokenisation for PAN / Aadhaar / mobile / email / account
- **Platt-vs-Isotonic auto-calibration** (`src/classifier.py`): `compare_calibrators()` selects lower-Brier method
- **Notebooks 02–06** (`scripts/gen_extra_notebooks.py`)
- **Pytest test suite** (`tests/`): 32 tests across 6 modules
- **GitHub Actions CI** (`.github/workflows/ci.yml`): pytest + bandit + pip-audit on every push
- **`SECURITY_AUDIT.md`**: 8 findings, 6 Phase-3 recommendations
- **`requirements-dev.txt`**: split dev dependencies from runtime

### Fixed
- **Staging bug** (`src/staging.py`): `compute_origination_pd` now accepts `grade_arr` to return per-grade mean PD instead of binary 0/1, restoring SICR sensitivity to macro scenarios
- **Markov matrix shape bug** (`src/markov.py`): corrected bootstrap path to produce proper (8,8) matrix with default absorbing state
- Module docstring now explicitly forbids using raw 0/1 as origination PD

### Changed
- `src/pipeline.py` now invokes 12 numbered steps (was 9)
- Audit trail enriched with VAR, Markov, isotonic, and reporting-panel metadata

---

## [0.2.0] - 2026-07-21

### Added
- LightGBM classifier with Platt scaling calibration
- Vasicek systematic factor (`src/vasicek.py`) with closed-form Z_t inversion
- Ornstein-Uhlenbeck Euler-Maruyama simulator (`src/ornstein_uhlenbeck.py`) with MLE fitting
- Survival-form PD term structure (`src/pd_term_structure.py`) verified against brief's 8.50/13.78/9.86 cohort
- SICR + Stage 1/2/3 engine (`src/staging.py`)
- ECL formula `Σ PD · LGD · EAD · D` (`src/ecl.py`)
- Path-wise stress test with coverage-ratio forecast-error metric (`src/stress.py`)
- End-to-end pipeline (`src/pipeline.py`) + CLI wrapper (`scripts/run_all.py`)
- End-to-end notebook (`notebooks/01_end_to_end.ipynb`)
- Audit-trail JSON manifest (`reports/audit_trail.json`)

### Fixed
- `np.logit` → `scipy.special.logit`
- Z_t sign-convention mismatch between `data_gen.py` and `vasicek.py`
- OU monthly-vs-annual time-scale mismatch
- `_scenario_pd_array` shape mismatch in `ecl_per_loan`

---

## [0.1.0] - 2026-07-20

### Added
- Initial scaffolding: synthetic LendingClub-schema data generator + feature config
- Project README and requirements
