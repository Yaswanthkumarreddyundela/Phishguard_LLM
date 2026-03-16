# tests/test_phase2.py  — v3.1 (fixed double-instantiation + logging)
# ─────────────────────────────────────────────────────────────────────────────
# CHANGES FROM PREVIOUS VERSION:
#
#   1. Added logging.basicConfig() so logger.info() messages are visible
#      (previously only print() showed output; now both work)
#
#   2. DomainIntelligenceManager instantiated ONCE at module level,
#      then shared with FeatureExtractor.
#      Previously it was instantiated twice → "Loading..." printed twice.
#
#   3. EmailParser instantiated at module level alongside the others.
#      No functional change, just consistent placement.
#
# No test cases changed. All assertions are identical.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import logging

# ── Configure logging so logger.info() is visible in the console ──────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)

# ── Make sure project root is on sys.path ────────────────────────────────────
# tests/ is one level below project root; this ensures `modules` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.email_parser import EmailParser
from modules.domain_intelligence import DomainIntelligenceManager
from modules.feature_extractor import FeatureExtractor

# ── Instantiate ONCE — share everywhere ──────────────────────────────────────
# DomainIntelligenceManager loads Tranco (1M rows) and threat feeds into memory.
# Creating it multiple times wastes seconds and causes duplicate log output.
print("Loading domain intelligence (first load may take a moment)...")
intel     = DomainIntelligenceManager()
extractor = FeatureExtractor(domain_intel=intel)
parser    = EmailParser()
print("Ready.\n")


# ── Test helpers ─────────────────────────────────────────────────────────────

def print_features(features: dict, title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for k, v in features.items():
        flag = " ⚠️" if (isinstance(v, (int, float)) and v > 0) else ""
        print(f"  {k:<42}: {v}{flag}")


def assert_features(features: dict, assertions: dict, test_name: str):
    passed = failed = 0
    for feat, expected in assertions.items():
        actual = features.get(feat)
        if callable(expected):
            ok      = expected(actual)
            exp_str = f"lambda({actual})"
        else:
            ok      = (actual == expected)
            exp_str = str(expected)

        if ok:
            passed += 1
        else:
            print(f"  ❌ FAIL [{test_name}] {feat}: expected={exp_str}, got={actual}")
            failed += 1

    status = "✅ ALL PASSED" if failed == 0 else f"❌ {failed} FAILED"
    print(f"\n  {status} ({passed}/{passed + failed})")
    return failed == 0


def run_test(name: str, raw_email: str, assertions: dict, extra_attachments=None):
    parsed = parser.parse_from_string(raw_email)
    if extra_attachments:
        parsed["attachments"] = extra_attachments
    features = extractor.extract(parsed)
    print_features(features, name)
    return assert_features(features, assertions, name)


# ════════════════════════════════════════════════════════════════════════════
# TEST CASES  (unchanged from previous version)
# ════════════════════════════════════════════════════════════════════════════

# ── TEST 1: HDFC Phishing ────────────────────────────────────────────────────
run_test(
    "TEST 1: HDFC Bank Phishing (.co TLD, IP URL, urgency)",
    """From: HDFC Bank <alerts@hdfcbank-secure.co>
To: user@gmail.com
Reply-To: verify@secure-update.cn
Subject: Immediate Action Required: Account Locked

Dear Customer,
We detected unusual activity in your bank account.
Your account will be permanently locked in 12 hours.
Verify your identity immediately:
https://hdfcbank-secure.co/verify-login
http://45.113.22.19/authenticate""",
    {
        "has_ip_based_url":      1,
        "compound_brand_domain": 1,
        "reply_to_differs":      1,
        "subject_has_urgency":   1,
        "urgency_phrase_count":  lambda x: x >= 1,
        "sender_in_tranco":      0,
        "impersonation_score":   lambda x: x >= 1,
    },
)

# ── TEST 2: Amazon Refund Phishing ───────────────────────────────────────────
run_test(
    "TEST 2: Amazon Refund Phishing (compound brand domain)",
    """From: Amazon Support <support@amazon-refund.net>
To: victim@gmail.com
Subject: Refund Pending - Confirm Now!

Hello,
We attempted to process your refund of $329.99 but it failed.
Click below to confirm your billing details:
http://amazon-refund.net/confirm?id=8872
If not confirmed within 6 HOURS, the refund will be cancelled.
Amazon Billing Team""",
    {
        "compound_brand_domain": 1,
        "has_http_only":         1,
        "impersonation_score":   lambda x: x >= 1,
        "urgency_phrase_count":  lambda x: x >= 1,
        "subject_has_urgency":   1,
        "sender_in_tranco":      0,
    },
)

# ── TEST 3: Microsoft Homograph ──────────────────────────────────────────────
run_test(
    "TEST 3: Microsoft Homograph Attack (micr0soft leet-speak)",
    """From: Microsoft Office <no-reply@micr0soft-login.com>
To: employee@company.com
Subject: Password Expiring Today

<html><body>
Your password expires TODAY.
<a href="https://micr0soft-login.com/reset">Reset Password</a>
<a href="http://bit.ly/o365-reset">Alternate Link</a>
</body></html>""",
    {
        "homograph_attack":        1,
        "has_url_shortener":       1,
        "sender_display_mismatch": 1,
        "subject_has_urgency":     1,
        "impersonation_score":     lambda x: x >= 1,
    },
)

# ── TEST 4: DHL Attachment Phishing ──────────────────────────────────────────
run_test(
    "TEST 4: DHL Attachment Phishing (.zip, .info TLD, lure filename)",
    """From: DHL Express <shipment@dhl-track.info>
To: user@gmail.com
Subject: Package Delivery Failed

Your package delivery failed.
Please see the attached file.

Regards, DHL Support""",
    {
        "has_dangerous_attachment":    1,
        "suspicious_attachment_name":  1,
        "compound_brand_domain":       1,
        "subject_has_urgency":         1,
        "sender_in_tranco":            0,
    },
    extra_attachments=[{
        "filename":     "Invoice_Receipt.zip",
        "extension":    ".zip",
        "is_dangerous": True,
        "content_type": "application/zip",
    }],
)

# ── TEST 5: Legitimate HDFC Bank (false-positive test) ───────────────────────
run_test(
    "TEST 5: ✅ LEGITIMATE HDFC Bank — must NOT be flagged",
    """From: HDFC Bank <alerts@hdfcbank.com>
To: user@gmail.com
Subject: Monthly Account Statement

Dear Customer,
Your monthly account statement is now available.
Please login securely via: https://netbanking.hdfcbank.com
Thank you, HDFC Bank""",
    {
        "urgency_phrase_count":  0,
        "subject_has_urgency":   0,
        "impersonation_score":   0,
        "has_ip_based_url":      0,
        "compound_brand_domain": 0,
        "homograph_attack":      0,
        "reply_to_differs":      0,
        "sender_in_tranco":      1,
    },
)

# ── TEST 6: Legitimate GitHub (false-positive test) ──────────────────────────
run_test(
    "TEST 6: ✅ LEGITIMATE GitHub — must NOT be flagged",
    """From: GitHub <noreply@github.com>
To: developer@gmail.com
Subject: Your weekly GitHub digest

Hi Developer,
You had 5 commits and 3 pull requests this week.
Visit https://github.com/dashboard

Thanks, The GitHub Team""",
    {
        "urgency_phrase_count":  0,
        "subject_has_urgency":   0,
        "impersonation_score":   0,
        "has_ip_based_url":      0,
        "compound_brand_domain": 0,
        "homograph_attack":      0,
        "sender_in_tranco":      1,
        "sender_is_threat":      0,
    },
)

# ── TEST 7: PayPal Phishing (regression) ─────────────────────────────────────
run_test(
    "TEST 7: PayPal Phishing (regression — must still pass)",
    """From: PayPal Security <security@paypa1.ru>
To: victim@gmail.com
Reply-To: harvest@evilserver.cn
Subject: URGENT: Your PayPal account has been SUSPENDED!

<html><body>
<p>Your account has been suspended. Verify immediately or it expires in 24 HOURS!</p>
<a href="http://paypa1-secure.ru/verify?token=xyz">Click here to verify your account</a>
<a href="http://192.168.100.1/login">backup link</a>
<a href="http://bit.ly/2xPhish">Alternative link</a>
</body></html>""",
    {
        "has_ip_based_url":      1,
        "homograph_attack":      1,
        "reply_to_differs":      1,
        "has_url_shortener":     1,
        "subject_has_urgency":   1,
        "urgency_phrase_count":  lambda x: x >= 1,
        "impersonation_score":   lambda x: x >= 1,
        "sender_in_tranco":      0,
    },
)

print("\n" + "=" * 60)
print("  ALL TESTS COMPLETE")
print("=" * 60)