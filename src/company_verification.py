# ==============================
# src/company_verification.py
# GSTIN / CIN structural + checksum validation for Indian company verification.
#
# IMPORTANT SCOPE NOTE (be upfront about this in interviews):
# This performs STRUCTURAL validation — format + official checksum algorithm —
# NOT a live lookup against the government GST/MCA database. A live lookup
# requires a registered GSP (GST Suvidha Provider) API key or MCA21 API access,
# which needs paid registration and isn't something to fake with placeholder
# credentials. What this DOES catch: obviously fabricated numbers, wrong
# state codes, malformed CIN/GSTIN strings — a real and common scam-detection
# signal, since many fake postings quote GSTIN/CIN numbers that are simply
# invalid strings, not real registered numbers.
# ==============================

import re
from typing import Dict, Optional

# Official GST state codes (first 2 digits of a GSTIN)
GST_STATE_CODES = {
    "01": "Jammu & Kashmir", "02": "Himachal Pradesh", "03": "Punjab",
    "04": "Chandigarh", "05": "Uttarakhand", "06": "Haryana", "07": "Delhi",
    "08": "Rajasthan", "09": "Uttar Pradesh", "10": "Bihar", "11": "Sikkim",
    "12": "Arunachal Pradesh", "13": "Nagaland", "14": "Manipur",
    "15": "Mizoram", "16": "Tripura", "17": "Meghalaya", "18": "Assam",
    "19": "West Bengal", "20": "Jharkhand", "21": "Odisha", "22": "Chhattisgarh",
    "23": "Madhya Pradesh", "24": "Gujarat", "25": "Daman & Diu",
    "26": "Dadra & Nagar Haveli", "27": "Maharashtra", "28": "Andhra Pradesh (Old)",
    "29": "Karnataka", "30": "Goa", "31": "Lakshadweep", "32": "Kerala",
    "33": "Tamil Nadu", "34": "Puducherry", "35": "Andaman & Nicobar",
    "36": "Telangana", "37": "Andhra Pradesh", "38": "Ladakh",
}

GSTIN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]$")
CIN_RE = re.compile(r"^[UL]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}$")

_GSTIN_CHECKSUM_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _gstin_checksum_valid(gstin: str) -> bool:
    """Validates the official GSTIN checksum (last character, mod-36 algorithm)."""
    try:
        factor = 1
        total = 0
        code_point_chars = _GSTIN_CHECKSUM_CHARS
        for ch in gstin[:-1]:
            digit = code_point_chars.index(ch)
            addend = factor * digit
            digit = (addend // 36) + (addend % 36)
            total += digit
            factor = 2 if factor == 1 else 1
        remainder = total % 36
        check_digit = (36 - remainder) % 36
        return code_point_chars[check_digit] == gstin[-1]
    except Exception:
        return False


def validate_gstin(gstin: str) -> Dict:
    gstin = (gstin or "").strip().upper()
    result = {
        "input": gstin,
        "type": "GSTIN",
        "format_valid": False,
        "checksum_valid": False,
        "state": None,
        "pan_embedded": None,
        "risk_flags": [],
    }
    if not gstin:
        result["risk_flags"].append("empty_input")
        return result

    if not GSTIN_RE.match(gstin):
        result["risk_flags"].append("malformed_gstin_format")
        return result

    result["format_valid"] = True
    state_code = gstin[:2]
    result["state"] = GST_STATE_CODES.get(state_code)
    if not result["state"]:
        result["risk_flags"].append("unknown_state_code")

    result["pan_embedded"] = gstin[2:12]
    result["checksum_valid"] = _gstin_checksum_valid(gstin)
    if not result["checksum_valid"]:
        result["risk_flags"].append("invalid_checksum_likely_fabricated")

    return result


def validate_cin(cin: str) -> Dict:
    cin = (cin or "").strip().upper()
    result = {
        "input": cin,
        "type": "CIN",
        "format_valid": False,
        "listing_status": None,
        "incorporation_year": None,
        "risk_flags": [],
    }
    if not cin:
        result["risk_flags"].append("empty_input")
        return result

    if not CIN_RE.match(cin):
        result["risk_flags"].append("malformed_cin_format")
        return result

    result["format_valid"] = True
    result["listing_status"] = "Listed" if cin[0] == "L" else "Unlisted"
    year_str = cin[8:12]
    if year_str.isdigit():
        result["incorporation_year"] = int(year_str)
        if result["incorporation_year"] > 2026 or result["incorporation_year"] < 1900:
            result["risk_flags"].append("implausible_incorporation_year")

    return result


def verify_company_identifier(identifier: str) -> Dict:
    """Auto-detects GSTIN (15 chars) vs CIN (21 chars) and validates accordingly."""
    identifier = (identifier or "").strip().upper()
    if len(identifier) == 15:
        return validate_gstin(identifier)
    elif len(identifier) == 21:
        return validate_cin(identifier)
    else:
        return {
            "input": identifier,
            "type": "unknown",
            "format_valid": False,
            "risk_flags": ["unrecognized_identifier_length"],
        }
