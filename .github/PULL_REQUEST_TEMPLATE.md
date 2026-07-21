---
name: Pull Request
about: Submit changes for review
---

## What does this PR do?

A 1–3 sentence summary.

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that breaks existing behaviour)
- [ ] Documentation only

## Related issues

Closes #___ or references #___.

## How was this tested?

- [ ] `pytest -v` passes locally
- [ ] `bandit -r src` clean
- [ ] `pip_audit` clean
- [ ] New unit tests added (describe)

## Checklist

- [ ] Code follows the project's style (PEP 8 + `pyproject.toml`)
- [ ] Self-review completed
- [ ] Comments added for non-obvious logic
- [ ] Docs updated (`README.md`, `CHANGELOG.md`, docstrings)
- [ ] No new dependency without justification

## Screenshots / output

If the change affects numerical output, paste the relevant `audit_trail.json`
snippet or a before/after table.
