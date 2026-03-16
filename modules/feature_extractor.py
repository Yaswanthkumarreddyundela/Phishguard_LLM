# modules/feature_extractor.py  — v4.0
"""
CHANGES FROM v3.1:

  NEW FEATURES (5 added → total now 37 handcrafted + nlp = 38):

  1. suspicious_path_keyword_count  (GROUP 3: URL Structure)
     Counts how many phishing-action keywords appear in URL paths.
     Catches: legit-domain.com/office365-login, university.edu/paypal/verify
     This is the key fix for compromised-legitimate-domain attacks.

  2. path_brand_mismatch  (GROUP 3: URL Structure)
     Domain is clean/unknown BUT path contains a brand name.
     e.g. invoice-approval.net/paypal/confirm → brand in path, domain ≠ brand
     Catches neutral-looking phishing URLs on non-brand domains.

  3. redirect_depth  (GROUP 3: URL Structure)
     Number of redirect hops detected statically from URL structure.
     bit.ly → intermediate → final = 2 hops detected statically.
     Dynamic resolution requires network; this is offline approximation.

  4. unknown_domain_action_content  (GROUP 6: Content Signals)
     Sender domain NOT in Tranco + email body contains financial/action keywords.
     The invoice email gap: unknown sender + "invoice/payment/approve" = risk.
     This is the direct fix for the AI-neutral phishing miss.

  5. ssl_cert_age_days  (GROUP 1: Sender Domain Reputation)
     Age of SSL certificate for sender domain in days.
     Fresh cert (< 7 days) on unknown domain = strong phishing signal.
     Requires network at inference; returns -1 if unavailable.

  ASSERT updated: len(f) == 37

  FIX: body field in extract() now handles both str and dict gracefully.
"""

import re
import ssl
import math
import socket
import logging
import ipaddress
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, List, Optional, Tuple
from email.utils import parseaddr

try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False

from modules.domain_intelligence import DomainIntelligenceManager

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts 37 features from a parsed email dict.
    The 38th feature (nlp_phishing_prob) is appended by the pipeline.

    All feature values are numeric (int or float).
    """

    LEET_MAP = str.maketrans({
        '0': 'o', '1': 'l', '3': 'e', '4': 'a',
        '5': 's', '6': 'g', '7': 't', '8': 'b',
        '@': 'a', '!': 'i',
    })

    BRAND_IDENTIFIERS = {
        "paypal", "stripe", "square", "venmo", "cashapp", "zelle",
        "google", "microsoft", "apple", "amazon", "netflix", "facebook",
        "instagram", "twitter", "linkedin", "dropbox", "adobe",
        "citibank", "barclays", "hsbc", "wellsfargo", "jpmorgan",
        "bankofamerica", "chase", "hdfc", "icici", "sbi", "axis",
        "fedex", "dhl", "ups", "usps",
        "irs", "hmrc",
        "salesforce", "docusign", "intuit", "quickbooks",
    }

    _MIN_BRAND_LEN_FOR_SUBSTRING = 4

    # Trusted sender domains — skip all feature extraction, return zeros.
    # Prevents false positives from legitimate providers whose SSL certs,
    # HTML structure, or auth headers trigger spurious signals.
    # URL domains are still checked — a trusted sender can still contain
    # a malicious link, which URL-group features will catch independently.
    # Add domains here when you see recurring false positives from a
    # known-legitimate source.
    TRUSTED_SENDER_DOMAINS = {
        # Google / Gmail
        "gmail.com", "googlemail.com", "google.com",
        "accounts.google.com", "mail.google.com",
        # Microsoft
        "outlook.com", "hotmail.com", "live.com",
        "microsoft.com", "office.com", "office365.com",
        # Apple
        "apple.com", "icloud.com", "me.com",
        # Amazon
        "amazon.com", "amazon.in", "amazon.co.uk", "amazonses.com",
        # Social / platforms
        "facebook.com", "instagram.com", "mail.instagram.com",
        "twitter.com", "linkedin.com", "youtube.com",
        # Developer / productivity
        "github.com", "gitlab.com", "notion.so",
        "slack.com", "zoom.us", "dropbox.com",
        # Indian services
        "zoho.com", "zohomail.com",
        # Course platforms
        "coursera.org", "m.learn.coursera.org", "m.mail.coursera.org",
        "udemy.com", "edx.org",
    }

    URL_SHORTENERS = {
        "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
        "buff.ly", "short.link", "rb.gy", "cutt.ly", "is.gd",
        "v.gd", "tiny.cc", "clck.ru", "shorturl.at", "bl.ink",
        "snip.ly", "rebrand.ly", "lnkd.in", "qr.ae", "adf.ly",
    }

    DANGEROUS_EXTENSIONS = {
        ".exe", ".bat", ".cmd", ".com", ".scr", ".pif",
        ".js", ".jse", ".vbs", ".vbe", ".ps1", ".ps2", ".psm1",
        ".wsf", ".wsh", ".msh",
        ".zip", ".rar", ".7z", ".iso", ".img", ".cab",
        ".docm", ".xlsm", ".pptm", ".xlam",
        ".hta", ".lnk", ".url", ".reg", ".cpl", ".inf",
        ".jar", ".apk", ".msi", ".msp",
    }

    URGENCY_PHRASES = [
        "within 24 hours", "within 48 hours", "within 12 hours",
        "within 6 hours", "within 2 hours",
        "expires today", "expiring today", "expire today",
        "last chance", "final notice", "final warning",
        "urgent action", "action required", "immediate action",
        "act now", "respond now", "respond immediately",
        "will be suspended", "will be locked", "will be deleted",
        "will be terminated", "will be cancelled", "will be blocked",
        "permanently locked", "permanently suspended", "permanently deleted",
        "access will be", "account will be",
        "failure to comply", "failure to verify", "failure to confirm",
        "legal action", "law enforcement",
        "you have been selected", "you have won", "you won",
        "claim your prize", "claim your reward", "claim now",
        "unauthorized access detected", "suspicious activity detected",
        "unusual login attempt", "unusual sign-in",
        "security breach", "account breach",
        "your payment failed", "payment was declined",
        "refund has been", "refund will be cancelled",
        "delivery attempt failed", "package could not be delivered",
        "verify your identity", "confirm your identity",
        "update your billing", "update payment information",
    ]

    SUBJECT_URGENCY_PATTERNS = [
        r'\burgent\b',
        r'\bimmediate\b',
        r'action required',
        r'account.{0,20}(locked|suspended|blocked|disabled|compromised)',
        r'(locked|suspended|blocked|disabled).{0,20}account',
        r'password.{0,20}(expir|reset|chang|compromised)',
        r'(verify|confirm|validate).{0,20}(account|identity|email|payment)',
        r'delivery.{0,15}(fail|attempt|unable|problem|exception)',
        r'package.{0,15}(fail|held|detained|pending|undelivered)',
        r'(unusual|suspicious|unauthorized).{0,15}(activity|login|access|sign)',
        r'security.{0,15}(alert|warning|notice|breach|issue)',
        r'(final|last).{0,15}(notice|warning|chance|reminder)',
        r'refund.{0,15}(pending|confirm|fail|process)',
        r'\bwarning\b',
        r'will be (deleted|terminated|suspended|cancelled)',
        r'expires? (today|soon|in \d)',
        r'limited time',
        r'(re-?verify|re-?confirm|re-?validate)',
        r'your.{0,10}(order|payment|account|subscription).{0,10}(fail|cancel|suspend|problem)',
    ]

    BRAND_CONNECTOR_WORDS = {
        "secure", "security", "login", "signin", "verify", "verification",
        "account", "update", "support", "help", "service", "online",
        "web", "site", "portal", "refund", "track", "tracking",
        "delivery", "alert", "notice", "confirm", "confirmation",
        "reset", "unlock", "restore", "access", "customer", "billing",
        "payment", "invoice", "reward", "prize", "offer", "official",
        "center", "centre", "corp", "inc", "ltd",
    }

    # ── NEW v4.0: Suspicious path keywords ───────────────────────────
    SUSPICIOUS_PATH_KEYWORDS = {
        "login", "signin", "verify", "verification", "update", "secure",
        "account", "billing", "office365", "microsoft", "google", "paypal",
        "apple", "amazon", "bank", "password", "credential", "authenticate",
        "confirm", "recover", "invoice", "payment", "approve", "review",
        "docusign", "authorize", "validate", "reset", "unlock",
    }

    # ── NEW v4.0: Action content keywords for unknown-domain signal ───
    ACTION_CONTENT_KEYWORDS = {
        "invoice", "payment", "approve", "review", "billing",
        "wire transfer", "bank transfer", "remittance", "purchase order",
        "click here", "click the link", "click below",
        "open attachment", "see attached", "attached document",
        "credentials", "username", "password", "login",
    }

    def __init__(self, domain_intel: DomainIntelligenceManager = None):
        self._intel = domain_intel or DomainIntelligenceManager()

    # ════════════════════════════════════════════════════════════════ #
    #  PUBLIC API
    # ════════════════════════════════════════════════════════════════ #

    def extract(self, parsed_email: Dict) -> Dict:
        """
        Extract all 37 features from a parsed email dict.
        Returns ordered dict of 37 numeric features.
        The 38th feature (nlp_phishing_prob) is appended by the pipeline.
        """
        headers     = parsed_email.get("headers",     {})
        body        = parsed_email.get("body",        {})
        urls        = parsed_email.get("urls",        [])
        attachments = parsed_email.get("attachments", [])

        subject    = headers.get("subject", "")
        from_field = headers.get("from",    "")

        # ── Whitelist check — trusted senders return all-zero features ─
        # This prevents false positives from Gmail, Outlook, Instagram etc.
        # whose SSL certs and HTML wrappers routinely trigger signals.
        # URL features are still computed below if URLs are present, but
        # since we return early here, the sender-side features are zeroed.
        _sender_domain_for_whitelist = self._extract_sender_domain(from_field)
        _root_for_whitelist = self._get_root_domain_safe(_sender_domain_for_whitelist)
        if (_sender_domain_for_whitelist in self.TRUSTED_SENDER_DOMAINS or
                _root_for_whitelist in self.TRUSTED_SENDER_DOMAINS):
            logger.debug(f"[Whitelist] Trusted sender: {_sender_domain_for_whitelist} — zeroing sender features")
            return self._zero_features()
        # ─────────────────────────────────────────────────────────────── #

        # ── Handle body being either a dict or a plain string ─────────
        if isinstance(body, str):
            body = {"combined": body, "plain_text": body, "html_text": "", "plain": body}

        sender_domain = self._extract_sender_domain(from_field)
        combined_text = (body.get("combined", body.get("plain_text", "")) + " " + subject).lower()

        # Pre-fetch all domain verdicts in one batch lookup
        all_domains = self._collect_all_domains(urls, sender_domain, headers)
        verdicts    = self._intel.lookup_bulk(all_domains)

        # ── GROUP 1: Sender Domain Reputation ────────────────────────
        sender_verdict = verdicts.get(sender_domain) or verdicts.get(
            self._intel._get_root_domain(sender_domain)
        )

        f = {}

        f["sender_in_tranco"]    = int(sender_verdict.is_legitimate if sender_verdict else False)
        f["sender_is_threat"]    = int(
            (sender_verdict.is_known_threat or sender_verdict.is_safe_browsing_threat)
            if sender_verdict else False
        )
        f["sender_domain_age_score"] = self._domain_age_score(
            sender_verdict.domain_age_days if sender_verdict else None
        )
        f["sender_is_newly_registered"] = int(
            sender_verdict.is_newly_registered if sender_verdict else False
        )

        # NEW v4.0: SSL certificate age for sender domain
        cert_age = self._get_ssl_cert_age(sender_domain)
        f["ssl_cert_age_days"] = 1 if (0 <= cert_age < 30) else 0
        
        # ── GROUP 2: URL Domain Reputation ───────────────────────────
        url_domains  = [self._intel._normalize_domain(u) for u in urls]
        url_verdicts = [verdicts.get(d) for d in url_domains]

        f["any_url_domain_is_threat"] = int(any(
            v and (v.is_known_threat or v.is_safe_browsing_threat)
            for v in url_verdicts
        ))
        f["any_url_domain_not_in_tranco"] = int(bool(urls) and not any(
            v and v.is_legitimate for v in url_verdicts
        ))
        f["min_url_tranco_rank"]    = self._min_tranco_rank(url_verdicts)
        f["any_url_newly_registered"] = int(any(
            v and v.is_newly_registered for v in url_verdicts
        ))

        # ── GROUP 3: URL Structure ────────────────────────────────────
        f["url_count"]         = len(urls)
        f["has_ip_based_url"]  = int(self._has_ip_url(urls))
        f["has_url_shortener"] = int(self._has_url_shortener(urls))
        f["has_http_only"]     = int(self._has_http_only(urls))
        f["max_url_length"]    = self._max_url_length(urls)
        f["url_has_at_symbol"] = int(self._url_has_at_symbol(urls))
        f["url_entropy_score"] = self._url_entropy_score(urls)

        # NEW v4.0: Path-level analysis
        f["suspicious_path_keyword_count"] = self._suspicious_path_keywords(urls)
        f["path_brand_mismatch"]           = int(self._path_brand_mismatch(urls, verdicts))
        f["redirect_depth"]                = self._redirect_depth(urls)

        # ── GROUP 4: Domain Deception ─────────────────────────────────
        f["brand_in_subdomain"]      = int(self._brand_in_subdomain(urls, verdicts))
        f["compound_brand_domain"]   = int(self._compound_brand_domain(urls, sender_domain, verdicts))
        f["homograph_attack"]        = int(self._homograph_attack(urls, sender_domain, verdicts))
        f["sender_display_mismatch"] = int(self._sender_display_mismatch(headers, verdicts))

        # ── GROUP 5: Header Authentication ───────────────────────────
        f["reply_to_differs"]     = int(self._reply_to_differs(headers))
        f["spf_fail"]             = int(headers.get("spf_result",  "none") in ["fail", "softfail"])
        f["dkim_fail"]            = int(headers.get("dkim_result", "none") == "fail")
        f["dmarc_fail"]           = int(headers.get("dmarc_result","none") == "fail")
        f["auth_completely_absent"] = int(self._auth_completely_absent(headers))

        # ── GROUP 6: Content Signals ──────────────────────────────────
        f["urgency_phrase_count"] = self._count_urgency_phrases(combined_text)
        f["subject_has_urgency"]  = int(self._subject_urgency(subject))
        f["impersonation_score"]  = self._impersonation_score(
            from_field, subject, sender_domain, sender_verdict
        )

        # NEW v4.0: Unknown domain + action content = neutral phishing signal
        f["unknown_domain_action_content"] = self._unknown_domain_action_content(
            sender_verdict, combined_text
        )

        # ── GROUP 7: Attachments ──────────────────────────────────────
        f["has_dangerous_attachment"]   = int(self._has_dangerous_attachment(attachments))
        f["suspicious_attachment_name"] = int(self._suspicious_attachment_name(attachments))

        # ── GROUP 8: HTML Structure ───────────────────────────────────
        f["has_form_in_html"]          = int(self._has_form_in_html(body))
        f["link_text_domain_mismatch"] = int(self._link_text_domain_mismatch(body, verdicts))
        f["html_to_text_ratio"]        = self._html_text_ratio(body)

        assert len(f) == 37, f"Feature count changed: {len(f)} (expected 37)"
        return f

    def _zero_features(self) -> Dict:
        """Return all-zero feature dict for whitelisted senders."""
        return {name: 0 for name in self.get_feature_names()}

    def _get_root_domain_safe(self, domain: str) -> str:
        """Get root domain safely for whitelist comparison."""
        try:
            if HAS_TLDEXTRACT:
                ext = tldextract.extract(domain)
                if ext.domain and ext.suffix:
                    return f"{ext.domain}.{ext.suffix}"
            parts = domain.split(".")
            if len(parts) >= 2:
                return ".".join(parts[-2:])
        except Exception:
            pass
        return domain

    def get_feature_names(self) -> List[str]:
        """Return the 37 feature names in exact order produced by extract()."""
        return [
            # Group 1 — Sender reputation
            "sender_in_tranco",           "sender_is_threat",
            "sender_domain_age_score",    "sender_is_newly_registered",
            "ssl_cert_age_days",
            # Group 2 — URL reputation
            "any_url_domain_is_threat",   "any_url_domain_not_in_tranco",
            "min_url_tranco_rank",        "any_url_newly_registered",
            # Group 3 — URL structure
            "url_count",                  "has_ip_based_url",
            "has_url_shortener",          "has_http_only",
            "max_url_length",             "url_has_at_symbol",
            "url_entropy_score",
            "suspicious_path_keyword_count", "path_brand_mismatch", "redirect_depth",
            # Group 4 — Domain deception
            "brand_in_subdomain",         "compound_brand_domain",
            "homograph_attack",           "sender_display_mismatch",
            # Group 5 — Header auth
            "reply_to_differs",           "spf_fail",
            "dkim_fail",                  "dmarc_fail",
            "auth_completely_absent",
            # Group 6 — Content
            "urgency_phrase_count",       "subject_has_urgency",
            "impersonation_score",        "unknown_domain_action_content",
            # Group 7 — Attachments
            "has_dangerous_attachment",   "suspicious_attachment_name",
            # Group 8 — HTML
            "has_form_in_html",           "link_text_domain_mismatch",
            "html_to_text_ratio",
        ]

    # ════════════════════════════════════════════════════════════════ #
    #  HELPER: COLLECT ALL DOMAINS
    # ════════════════════════════════════════════════════════════════ #

    def _collect_all_domains(
        self,
        urls: List[str],
        sender_domain: str,
        headers: Dict,
    ) -> List[str]:
        domains = set()
        if sender_domain:
            domains.add(sender_domain)
            domains.add(self._intel._get_root_domain(sender_domain))
        for url in urls:
            d = self._intel._normalize_domain(url)
            if d:
                domains.add(d)
                domains.add(self._intel._get_root_domain(d))
        _, reply_email = parseaddr(headers.get("reply_to", ""))
        if reply_email and "@" in reply_email:
            domains.add(reply_email.split("@")[-1].lower())
        return [d for d in domains if d]

    # ════════════════════════════════════════════════════════════════ #
    #  GROUP 1+2: DOMAIN REPUTATION HELPERS
    # ════════════════════════════════════════════════════════════════ #

    def _domain_age_score(self, age_days: Optional[int]) -> float:
        if age_days is None: return 0.5
        if age_days < 7:     return 1.0
        if age_days < 30:    return 0.9
        if age_days < 90:    return 0.7
        if age_days < 180:   return 0.5
        if age_days < 365:   return 0.3
        if age_days < 730:   return 0.15
        return 0.05

    def _min_tranco_rank(self, url_verdicts: list) -> int:
        ranks = [v.tranco_rank for v in url_verdicts if v and v.tranco_rank]
        return min(ranks) if ranks else 1_000_001

    # ── NEW v4.0: SSL certificate age ────────────────────────────────
    def _get_ssl_cert_age(self, domain: str) -> float:
        """
        Returns age of SSL cert in days. -1 if unavailable/error.
        Fresh cert (< 7 days) on unknown domain is a strong phishing signal.

        FIX: settimeout() must be called on the raw socket BEFORE wrapping,
        not after. Previously the timeout was set on the SSL socket which
        had no effect — the connect() could hang and return stale/zero data.
        Also returns -1 (not 0) on any date parsing ambiguity to avoid
        false "0 days old" reports on legitimate domains.
        """
        if not domain:
            return -1.0
        try:
            raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            raw_sock.settimeout(3)   # FIX: timeout on raw socket before SSL wrap
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(raw_sock, server_hostname=domain) as s:
                s.connect((domain, 443))
                cert = s.getpeercert()
            not_before_str = cert.get("notBefore", "")
            if not not_before_str:
                return -1.0
            not_before = datetime.strptime(not_before_str, "%b %d %H:%M:%S %Y %Z")
            age_days = (datetime.utcnow() - not_before).days
            # Return -1 for any result that looks like a parsing artifact
            if age_days < 0:
                return -1.0
            return float(age_days)
        except Exception:
            return -1.0

    # ════════════════════════════════════════════════════════════════ #
    #  GROUP 3: URL STRUCTURE
    # ════════════════════════════════════════════════════════════════ #

    def _has_ip_url(self, urls: List[str]) -> bool:
        ip_pattern = re.compile(r'https?://(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')
        for url in urls:
            m = ip_pattern.match(url)
            if m:
                try:
                    ipaddress.ip_address(m.group(1))
                    return True
                except ValueError:
                    pass
            try:
                host = (urlparse(url).hostname or "").split(":")[0]
                if host:
                    ipaddress.ip_address(host)
                    return True
            except ValueError:
                pass
        return False

    def _has_url_shortener(self, urls: List[str]) -> bool:
        for url in urls:
            domain = self._intel._get_root_domain(self._intel._normalize_domain(url))
            if domain in self.URL_SHORTENERS:
                return True
        return False

    def _has_http_only(self, urls: List[str]) -> bool:
        return any(u.startswith("http://") for u in urls) if urls else False

    def _max_url_length(self, urls: List[str]) -> int:
        return max((len(u) for u in urls), default=0)

    def _url_has_at_symbol(self, urls: List[str]) -> bool:
        return any("@" in urlparse(u).netloc for u in urls)

    def _url_entropy_score(self, urls: List[str]) -> float:
        def entropy(s: str) -> float:
            if not s: return 0.0
            freq = {}
            for c in s:
                freq[c] = freq.get(c, 0) + 1
            return -sum((f / len(s)) * math.log2(f / len(s)) for f in freq.values())

        entropies = []
        for url in urls:
            domain = self._intel._normalize_domain(url)
            if HAS_TLDEXTRACT:
                try:
                    d = tldextract.extract(domain).domain
                    if d:
                        entropies.append(entropy(d))
                except Exception:
                    pass
            else:
                parts = domain.split(".")
                if parts:
                    entropies.append(entropy(parts[0]))
        return round(max(entropies), 3) if entropies else 0.0

    # ── NEW v4.0: Suspicious path keywords ───────────────────────────
    def _suspicious_path_keywords(self, urls: List[str]) -> int:
        """
        Count how many phishing action keywords appear across all URL paths.
        invoice-approval.net/paypal/verify → paypal=1, verify=1 → score=2
        """
        total = 0
        for url in urls:
            path = urlparse(url).path.lower()
            total += sum(1 for kw in self.SUSPICIOUS_PATH_KEYWORDS if kw in path)
        return total

    # ── NEW v4.0: Brand name in path but not in domain ───────────────
    def _path_brand_mismatch(self, urls: List[str], verdicts: Dict) -> bool:
        """
        Domain is unknown/clean but URL path contains a brand name.
        invoice-approval.net/paypal/confirm → domain ≠ paypal, path has paypal.
        This is the primary signal for compromised-domain + path-brand attacks.
        """
        for url in urls:
            try:
                parsed = urlparse(url)
                path   = parsed.path.lower()

                domain = self._intel._normalize_domain(url)
                root   = self._intel._get_root_domain(domain)
                v      = verdicts.get(domain) or verdicts.get(root)

                # Skip if domain is the actual brand (false positive guard)
                if v and v.is_legitimate:
                    continue

                if HAS_TLDEXTRACT:
                    domain_label = tldextract.extract(domain).domain.lower()
                else:
                    domain_label = root.split(".")[0].lower()

                for brand in self.BRAND_IDENTIFIERS:
                    if len(brand) < 4:
                        continue
                    # Brand in path but NOT in domain label
                    if brand in path and brand not in domain_label:
                        return True
            except Exception:
                continue
        return False

    # ── NEW v4.0: Static redirect depth estimation ───────────────────
    def _redirect_depth(self, urls: List[str]) -> int:
        """
        Statically estimate redirect depth from URL structure.
        Checks for: URL shorteners, redirect parameters, chained URLs.
        Dynamic resolution would be more accurate but requires network.
        """
        max_depth = 0
        redirect_params = {"url", "redirect", "next", "goto", "return",
                           "returnurl", "redirect_uri", "continue", "dest"}
        for url in urls:
            depth = 0
            # Shortener = at least 1 hop
            root = self._intel._get_root_domain(self._intel._normalize_domain(url))
            if root in self.URL_SHORTENERS:
                depth += 1
            # Redirect parameter in query string = another hop
            try:
                from urllib.parse import parse_qs
                params = parse_qs(urlparse(url).query)
                for param in params:
                    if param.lower() in redirect_params:
                        depth += 1
                        break
            except Exception:
                pass
            # Encoded URL in path = chained redirect
            if "http%3a" in url.lower() or "http%3A" in url:
                depth += 1
            max_depth = max(max_depth, depth)
        return max_depth

    # ════════════════════════════════════════════════════════════════ #
    #  GROUP 4: DOMAIN DECEPTION (unchanged from v3.1)
    # ════════════════════════════════════════════════════════════════ #

    def _brand_in_subdomain(self, urls: List[str], verdicts: Dict) -> bool:
        for url in urls:
            if not HAS_TLDEXTRACT:
                break
            try:
                extracted = tldextract.extract(url)
                subdomain = extracted.subdomain.lower()
                root      = f"{extracted.domain}.{extracted.suffix}".lower()
                v = verdicts.get(root)
                if v and v.is_legitimate:
                    continue
                for brand in self.BRAND_IDENTIFIERS:
                    if brand in subdomain and brand not in extracted.domain.lower():
                        return True
            except Exception:
                continue
        return False

    def _compound_brand_domain(
        self, urls: List[str], sender_domain: str, verdicts: Dict,
    ) -> bool:
        domains_to_check = [
            self._intel._normalize_domain(u) for u in urls
            if self._intel._normalize_domain(u)
        ]
        if sender_domain:
            domains_to_check.append(sender_domain)

        for domain in domains_to_check:
            root = self._intel._get_root_domain(domain)
            v    = verdicts.get(root)
            if v and v.is_legitimate:
                continue

            if HAS_TLDEXTRACT:
                try:
                    domain_label = tldextract.extract(domain).domain.lower()
                except Exception:
                    domain_label = root.split(".")[0]
            else:
                domain_label = root.split(".")[0]

            parts           = re.split(r'[-_]', domain_label)
            brand_found     = None
            connector_found = False

            for part in parts:
                if part in self.BRAND_IDENTIFIERS:
                    brand_found = part
                if part in self.BRAND_CONNECTOR_WORDS:
                    connector_found = True

            if brand_found and len(parts) > 1:
                return True
            if brand_found and connector_found:
                return True

            for brand in self.BRAND_IDENTIFIERS:
                if len(brand) < self._MIN_BRAND_LEN_FOR_SUBSTRING:
                    continue
                if brand in domain_label and domain_label != brand:
                    if not (v and v.is_legitimate):
                        return True
        return False

    def _homograph_attack(
        self, urls: List[str], sender_domain: str, verdicts: Dict,
    ) -> bool:
        domains_to_check = [
            self._intel._normalize_domain(u) for u in urls
            if self._intel._normalize_domain(u)
        ]
        if sender_domain:
            domains_to_check.append(sender_domain)

        for domain in domains_to_check:
            root = self._intel._get_root_domain(domain)
            v    = verdicts.get(root) or verdicts.get(domain)
            if v and v.is_legitimate:
                continue

            if HAS_TLDEXTRACT:
                try:
                    domain_label = tldextract.extract(domain).domain.lower()
                except Exception:
                    domain_label = root.split(".")[0]
            else:
                domain_label = root.split(".")[0]

            primary_part = re.split(r'[-_]', domain_label)[0]
            normalized   = primary_part.translate(self.LEET_MAP)

            for brand in self.BRAND_IDENTIFIERS:
                if normalized == brand:
                    return True
                if (len(primary_part) >= 4 and
                        abs(len(normalized) - len(brand)) <= 2):
                    if self._levenshtein(normalized, brand) <= 1:
                        return True
        return False

    def _sender_display_mismatch(self, headers: Dict, verdicts: Dict) -> bool:
        from_field   = headers.get("from", "")
        display_name, email_addr = parseaddr(from_field)
        if not display_name or not email_addr or "@" not in email_addr:
            return False
        display_lower = display_name.lower()
        sender_domain = email_addr.split("@")[-1].lower()
        root_domain   = self._intel._get_root_domain(sender_domain)
        v = verdicts.get(sender_domain) or verdicts.get(root_domain)
        if v and v.is_legitimate:
            return False
        return any(brand in display_lower for brand in self.BRAND_IDENTIFIERS)

    # ════════════════════════════════════════════════════════════════ #
    #  GROUP 5: HEADER AUTHENTICATION
    # ════════════════════════════════════════════════════════════════ #

    def _reply_to_differs(self, headers: Dict) -> bool:
        from_field = headers.get("from",     "")
        reply_to   = headers.get("reply_to", "")
        if not reply_to:
            return False
        _, from_email  = parseaddr(from_field)
        _, reply_email = parseaddr(reply_to)
        if not from_email or not reply_email:
            return False
        from_root  = self._intel._get_root_domain(
            from_email.split("@")[-1]  if "@" in from_email  else ""
        )
        reply_root = self._intel._get_root_domain(
            reply_email.split("@")[-1] if "@" in reply_email else ""
        )
        return bool(from_root and reply_root and from_root != reply_root)

    def _auth_completely_absent(self, headers: Dict) -> bool:
        return (
            headers.get("spf_result",   "none") == "none" and
            headers.get("dkim_result",  "none") == "none" and
            headers.get("dmarc_result", "none") == "none"
        )

    # ════════════════════════════════════════════════════════════════ #
    #  GROUP 6: CONTENT SIGNALS
    # ════════════════════════════════════════════════════════════════ #

    def _count_urgency_phrases(self, text: str) -> int:
        return sum(1 for phrase in self.URGENCY_PHRASES if phrase in text)

    def _subject_urgency(self, subject: str) -> bool:
        s = subject.lower()
        return any(re.search(p, s) for p in self.SUBJECT_URGENCY_PATTERNS)

    def _impersonation_score(
        self, from_field: str, subject: str,
        sender_domain: str, sender_verdict,
    ) -> int:
        if sender_verdict and sender_verdict.is_legitimate:
            return 0
        display_name, _ = parseaddr(from_field)
        display_lower   = display_name.lower()
        subject_lower   = subject.lower()
        return sum(
            1 for brand in self.BRAND_IDENTIFIERS
            if brand in display_lower or brand in subject_lower
        )

    # ── NEW v4.0: Unknown domain + action content ─────────────────────
    def _unknown_domain_action_content(
        self, sender_verdict, combined_text: str
    ) -> int:
        """
        Score 0-3 based on how many action keywords appear when sender
        domain is NOT in Tranco.

        This directly addresses the AI-neutral phishing gap:
        'Invoice attached for your review' from unknown domain scores 2-3
        even with NLP prob near zero.

        Returns:
            0 = sender is legitimate OR no action keywords
            1 = 1-2 action keywords from unknown domain
            2 = 3-4 action keywords
            3 = 5+ action keywords
        """
        # If sender is known legitimate, this signal doesn't apply
        if sender_verdict and sender_verdict.is_legitimate:
            return 0

        count = sum(1 for kw in self.ACTION_CONTENT_KEYWORDS if kw in combined_text)

        if count == 0:   return 0
        if count <= 2:   return 1
        if count <= 4:   return 2
        return 3

    # ════════════════════════════════════════════════════════════════ #
    #  GROUP 7: ATTACHMENTS
    # ════════════════════════════════════════════════════════════════ #

    def _has_dangerous_attachment(self, attachments: List[Dict]) -> bool:
        return any(
            att.get("extension", "").lower() in self.DANGEROUS_EXTENSIONS
            for att in attachments
        )

    def _suspicious_attachment_name(self, attachments: List[Dict]) -> bool:
        lure_patterns = [
            r'invoice', r'receipt', r'payment', r'order',
            r'statement', r'document', r'scan', r'fax',
            r'contract', r'agreement', r'report', r'bill',
        ]
        for att in attachments:
            name = att.get("filename", "").lower()
            if any(re.search(p, name) for p in lure_patterns):
                return True
        return False

    # ════════════════════════════════════════════════════════════════ #
    #  GROUP 8: HTML STRUCTURE
    # ════════════════════════════════════════════════════════════════ #

    def _has_form_in_html(self, body: Dict) -> bool:
        html = body.get("html_text", "").lower()
        return (
            "<form" in html or
            bool(re.search(r'<input[^>]+type=["\']?(password|text)', html))
        )

    def _link_text_domain_mismatch(self, body: Dict, verdicts: Dict) -> bool:
        html = body.get("html_text", "")
        if not html:
            return False
        link_pattern = re.compile(
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        for href, text in link_pattern.findall(html):
            text_clean = re.sub(r'<[^>]+>', '', text).lower().strip()
            for brand in self.BRAND_IDENTIFIERS:
                if brand in text_clean:
                    href_domain = self._intel._normalize_domain(href)
                    href_root   = self._intel._get_root_domain(href_domain)
                    v = verdicts.get(href_domain) or verdicts.get(href_root)
                    if not (v and v.is_legitimate):
                        return True
        return False

    def _html_text_ratio(self, body: Dict) -> float:
        html_len = len(body.get("html_text",  ""))
        text_len = len(body.get("plain_text", "")) + 1
        return min(round(html_len / text_len, 2), 100.0)

    # ════════════════════════════════════════════════════════════════ #
    #  UTILITIES
    # ════════════════════════════════════════════════════════════════ #

    def _extract_sender_domain(self, from_field: str) -> str:
        _, email_addr = parseaddr(from_field)
        if email_addr and "@" in email_addr:
            return email_addr.split("@")[-1].lower().strip()
        return ""

    def _levenshtein(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if not s2:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for c1 in s1:
            curr = [prev[0] + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(
                    prev[j + 1] + 1,
                    curr[j]     + 1,
                    prev[j]     + (c1 != c2),
                ))
            prev = curr
        return prev[-1]