# Contributing to PiT-ECL Engine

Thank you for your interest in contributing. This project targets production-grade
reference implementations of retail credit-risk models. We welcome bug reports,
documentation improvements, and well-scoped pull requests.

## Code of Conduct

All contributors are expected to abide by the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Development setup

```bash
git clone https://github.com/404Piyush/p1-pit-ecl-engine.git
cd p1-pit-ecl-engine
py -3.11 -m pip install -r requirements-dev.txt
```

## Running the test suite

```bash
py -3.11 -m pytest -v
```

Target coverage floor is **70%** (configured in `pyproject.toml`).

## Linting and security

```bash
py -3.11 -m bandit -r src
py -3.11 -m pip_audit --requirement requirements.txt --strict
```

## Pull request process

1. Fork the repo and create a feature branch from `main`.
2. Add tests for any new functionality. Maintain or improve coverage.
3. Run `pytest`, `bandit`, and `pip_audit` locally before pushing.
4. Reference any related issue in the PR description.
5. Use the [pull request template](.github/PULL_REQUEST_TEMPLATE.md).

## Coding conventions

- Python 3.11+. Type hints preferred on public APIs.
- No hardcoded secrets, credentials, or live data. Synthetic data only.
- No `eval`, `exec`, `pickle.loads`, or `subprocess` calls in production code.
- All file writes must be anchored to repo-relative `Path` objects.
- Mathematical formulas in docstrings should match the original brief.

## Reporting security issues

Please **do not** open a public GitHub issue for security vulnerabilities.
Email `security@example.invalid` instead. We follow responsible disclosure
with a 90-day window.
