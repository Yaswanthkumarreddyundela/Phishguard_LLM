# tests/test_phase1.py
"""
How to run: python -m pytest tests/test_phase1.py -v

This tests the parser WITHOUT needing a live IMAP connection.
We construct a synthetic phishing email as a raw string.
"""

import json
import sys
sys.path.append("..")

from modules.email_parser import EmailParser

# --- Synthetic phishing email for testing ---
SAMPLE_PHISHING_EMAIL = """From: security@paypa1.com
To: victim@gmail.com
Reply-To: harvest@evilsite.ru
Subject: URGENT: Your account has been suspended!
Date: Mon, 23 Feb 2026 10:00:00 +0000
Message-ID: <fake123@paypa1.com>
Content-Type: text/plain; charset=utf-8

Dear Customer,

Your PayPal account has been SUSPENDED due to suspicious activity!
You must verify your account within 24 HOURS or it will be permanently deleted.

Click here to verify: http://paypa1-secure.ru/verify?token=abc123
Or visit: http://192.168.1.1/phishing-page

Failure to act will result in permanent account closure.

PayPal Security Team
"""

SAMPLE_LEGIT_EMAIL = """From: newsletter@github.com
To: developer@gmail.com
Subject: Your weekly GitHub digest
Date: Mon, 23 Feb 2026 09:00:00 +0000
Message-ID: <legit456@github.com>
Content-Type: text/plain; charset=utf-8

Hi Developer,

Here is your weekly summary of activity on GitHub.
You had 5 commits, 3 pull requests, and 2 issues this week.

Visit your dashboard: https://github.com/dashboard

Thanks,
The GitHub Team
"""


def test_parser_output_schema():
    """Test that parser returns all required keys."""
    parser = EmailParser()
    result = parser.parse_from_string(SAMPLE_PHISHING_EMAIL)
    
    required_keys = ["email_hash", "headers", "body", "urls", "attachments", "metadata"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    print("[PASS] Schema validation")


def test_url_extraction():
    """Test URL extraction from phishing email."""
    parser = EmailParser()
    result = parser.parse_from_string(SAMPLE_PHISHING_EMAIL)
    
    assert len(result["urls"]) == 2
    assert "http://paypa1-secure.ru/verify?token=abc123" in result["urls"]
    assert "http://192.168.1.1/phishing-page" in result["urls"]
    print(f"[PASS] URL extraction: found {result['urls']}")


def test_header_parsing():
    """Test header extraction."""
    parser = EmailParser()
    result = parser.parse_from_string(SAMPLE_PHISHING_EMAIL)
    
    assert result["headers"]["from"] == "security@paypa1.com"
    assert result["headers"]["reply_to"] == "harvest@evilsite.ru"
    assert result["headers"]["subject"] == "URGENT: Your account has been suspended!"
    print("[PASS] Header parsing")


def test_email_hash_uniqueness():
    """Test that different emails get different hashes."""
    parser = EmailParser()
    hash1 = parser.parse_from_string(SAMPLE_PHISHING_EMAIL)["email_hash"]
    hash2 = parser.parse_from_string(SAMPLE_LEGIT_EMAIL)["email_hash"]
    
    assert hash1 != hash2
    print(f"[PASS] Hash uniqueness: {hash1[:16]}... vs {hash2[:16]}...")


def test_metadata():
    """Test metadata computation."""
    parser = EmailParser()
    result = parser.parse_from_string(SAMPLE_PHISHING_EMAIL)
    
    assert result["metadata"]["url_count"] == 2
    assert result["metadata"]["has_html"] == False
    print(f"[PASS] Metadata: {result['metadata']}")


def run_all_tests():
    print("\n=== PHASE 1 TESTS ===\n")
    test_parser_output_schema()
    test_url_extraction()
    test_header_parsing()
    test_email_hash_uniqueness()
    test_metadata()
    
    print("\n--- Sample Output ---")
    parser = EmailParser()
    result = parser.parse_from_string(SAMPLE_PHISHING_EMAIL)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run_all_tests()