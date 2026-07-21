"""PII / sensitive-data redaction helpers.

The Ind AS 109 audit trail recommends masking personally identifiable
information when artefacts are released outside the regulated boundary.
Redaction is deterministic per loan_id so the same record is masked the
same way across runs and audit reviews.

The masker uses HMAC-SHA-256 with a configurable salt so the same input
produces the same output within a run but cannot be reversed without the
salt. This is an industry-standard "tokenisation" approach for non-production
data sharing.

We mask:

- PAN:  10-character alphanumeric pattern (e.g. ABCDE1234F)
- Aadhaar: 12-digit pattern
- Mobile: 10-digit pattern
- Email: local-part masked, domain retained
- Account number: last 4 visible, rest masked
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional


PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
AADHAAR_RE = re.compile(r"^\d{12}$")
MOBILE_RE = re.compile(r"^\d{10}$")
EMAIL_RE = re.compile(r"^([^@]+)@([^@]+)$")
ACCOUNT_RE = re.compile(r"^\d{9,18}$")


@dataclass
class RedactionConfig:
    salt: str = "p1-pit-ecl-engine-default-salt"
    last4_account: bool = True
    mask_email_domain_visible: bool = True


def _hmac_token(value: str, salt: str, length: int = 8) -> str:
    h = hmac.new(salt.encode(), value.encode(), hashlib.sha256).hexdigest().upper()
    return h[:length]


def redact_pan(value: str, cfg: RedactionConfig) -> str:
    if value is None or not PAN_RE.match(value):
        return value
    return f"XXXXX{_hmac_token(value, cfg.salt, 4)}{value[-1]}".replace(value[-1], value[-1])


def redact_aadhaar(value: str, cfg: RedactionConfig) -> str:
    if value is None or not AADHAAR_RE.match(value):
        return value
    return "XXXX-XXXX-" + value[-4:]


def redact_mobile(value: str, cfg: RedactionConfig) -> str:
    if value is None or not MOBILE_RE.match(value):
        return value
    return "XXXXXX" + value[-4:]


def redact_email(value: str, cfg: RedactionConfig) -> str:
    if value is None:
        return value
    m = EMAIL_RE.match(value)
    if not m:
        return value
    local, domain = m.groups()
    if cfg.mask_email_domain_visible:
        return f"{_hmac_token(local, cfg.salt, 6)}@{domain}"
    return f"{_hmac_token(local, cfg.salt, 6)}@example.invalid"


def redact_account(value: str, cfg: RedactionConfig) -> str:
    if value is None or not ACCOUNT_RE.match(value):
        return value
    if cfg.last4_account:
        return "X" * (len(value) - 4) + value[-4:]
    return "X" * len(value)


def redact_value(column: str, value, cfg: RedactionConfig):
    """Dispatch a single value to the right masker based on column name."""
    if value is None:
        return value
    col = column.lower()
    if "pan" in col:
        return redact_pan(str(value), cfg)
    if "aadhaar" in col or "uid" in col:
        return redact_aadhaar(str(value), cfg)
    if "mobile" in col or "phone" in col:
        return redact_mobile(str(value), cfg)
    if "email" in col:
        return redact_email(str(value), cfg)
    if "account" in col or "acct" in col or "iban" in col:
        return redact_account(str(value), cfg)
    return value


DEFAULT_SENSITIVE_COLUMNS = (
    "pan",
    "aadhaar",
    "uid",
    "mobile",
    "phone",
    "email",
    "account",
    "acct",
    "iban",
    "customer_name",
    "name",
)


def sensitive_columns(columns: Iterable[str]) -> List[str]:
    """Return the subset of `columns` that look like sensitive fields."""
    out = []
    for c in columns:
        lc = c.lower()
        if any(tok in lc for tok in DEFAULT_SENSITIVE_COLUMNS):
            out.append(c)
    return out


def redact_dataframe_columns(df, columns: Optional[Iterable[str]] = None, cfg: Optional[RedactionConfig] = None) -> List[str]:
    """Apply redaction in-place to identified columns. Returns the list of columns redacted."""
    cfg = cfg or RedactionConfig()
    targets = list(columns) if columns else sensitive_columns(df.columns)
    for col in targets:
        if col in df.columns:
            df[col] = df[col].map(lambda v: redact_value(col, v, cfg))
    return targets
