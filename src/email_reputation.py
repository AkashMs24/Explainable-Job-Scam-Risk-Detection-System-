# ==============================
# src/email_reputation.py
# Email authentication + reputation checks for job-posting contact emails.
#
# Checks performed:
#   1. SPF record present on sending domain (DNS TXT lookup)
#   2. DMARC policy present (DNS TXT lookup on _dmarc.<domain>)
#   3. Free/consumer email provider flag (gmail, yahoo, etc.)
#   4. Disposable/throwaway email domain flag (common temp-mail providers)
#
# Note: DKIM cannot be verified from a bare email address alone — DKIM is a
# per-message signature added by the sending mail server at send time, not a
# static DNS record you can look up from just "user@domain.com". What CAN be
# checked statically is whether the domain *publishes* a DMARC policy, which
# in practice requires the domain to also support DKIM/SPF alignment — so we
# surface that as "dmarc_present" rather than falsely claiming a DKIM check.
# ==============================

import re
from typing import Dict, List

# dnspython is an OPTIONAL dependency. It must be importable for SPF/DMARC
# checks to run, but if the deployment environment is missing it (wrong
# requirements.txt picked up, stale build cache, etc.) we do NOT want that
# to take down the entire FastAPI app at import time — every endpoint,
# including /predict, would 500. So the import is wrapped, and SPF/DMARC
# checks degrade to "unavailable" instead of crashing the module.
try:
    import dns.resolver
    _DNS_AVAILABLE = True
except ImportError:
    dns = None
    _DNS_AVAILABLE = False

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "mail.com",
    "ymail.com", "aol.com", "icloud.com", "protonmail.com", "tutanota.com",
    "rediffmail.com", "live.com", "msn.com",
}

# Commonly used disposable/temporary email providers seen in scam job postings
DISPOSABLE_DOMAINS = {
    "mailinator.com", "10minutemail.com", "guerrillamail.com", "tempmail.com",
    "throwawaymail.com", "yopmail.com", "trashmail.com", "getnada.com",
    "fakeinbox.com", "sharklasers.com", "temp-mail.org",
}

EMAIL_RE = re.compile(r"^[^@\s]+@([^@\s]+\.[^@\s]+)$")

_resolver = None
if _DNS_AVAILABLE:
    _resolver = dns.resolver.Resolver()
    _resolver.timeout = 3
    _resolver.lifetime = 3


def _extract_domain(email: str) -> str:
    m = EMAIL_RE.match((email or "").strip().lower())
    return m.group(1) if m else ""


def _has_spf(domain: str) -> bool:
    if not _DNS_AVAILABLE:
        return False
    try:
        answers = _resolver.resolve(domain, "TXT")
        for rdata in answers:
            txt = b"".join(rdata.strings).decode("utf-8", errors="ignore") if hasattr(rdata, "strings") else str(rdata)
            if txt.lower().startswith("v=spf1"):
                return True
    except Exception:
        pass
    return False


def _has_dmarc(domain: str) -> bool:
    if not _DNS_AVAILABLE:
        return False
    try:
        answers = _resolver.resolve(f"_dmarc.{domain}", "TXT")
        for rdata in answers:
            txt = b"".join(rdata.strings).decode("utf-8", errors="ignore") if hasattr(rdata, "strings") else str(rdata)
            if txt.lower().startswith("v=dmarc1"):
                return True
    except Exception:
        pass
    return False


def check_email_reputation(email: str) -> Dict:
    """
    Returns a reputation report for a single contact email address.
    DNS lookups fail gracefully (timeouts/no records) — a failed lookup is
    reported as False, not raised as an error, since scam postings often
    use domains with minimal DNS configuration in the first place.
    """
    domain = _extract_domain(email)
    if not domain:
        return {
            "email": email,
            "valid_format": False,
            "domain": None,
            "is_free_provider": False,
            "is_disposable": False,
            "spf_present": False,
            "dmarc_present": False,
            "risk_flags": ["invalid_email_format"],
        }

    is_free = domain in FREE_EMAIL_DOMAINS
    is_disposable = domain in DISPOSABLE_DOMAINS
    spf = False if is_disposable else _has_spf(domain)
    dmarc = False if is_disposable else _has_dmarc(domain)

    flags: List[str] = []
    if is_disposable:
        flags.append("disposable_email_domain")
    if is_free:
        flags.append("free_consumer_email_provider")
    if not _DNS_AVAILABLE:
        flags.append("dns_check_unavailable_on_server")
    else:
        if not is_disposable and not spf:
            flags.append("no_spf_record")
        if not is_disposable and not dmarc:
            flags.append("no_dmarc_policy")

    return {
        "email": email,
        "valid_format": True,
        "domain": domain,
        "is_free_provider": is_free,
        "is_disposable": is_disposable,
        "spf_present": spf,
        "dmarc_present": dmarc,
        "dns_check_available": _DNS_AVAILABLE,
        "risk_flags": flags,
        "email_risk_score": min(100, len(flags) * 25),  # simple 0-100 scale
    }
