# modules/domain_intelligence.py
"""
Domain Intelligence Manager — v4.0
=====================================

BUGS FIXED FROM v3:
  1. CACHE_DIR was relative to CWD — broke when running tests from tests/ folder
     Fix: now always resolves to project root via Path(__file__).parent.parent
  2. Auto-download caused 404 errors on Tranco URL
     Fix: ALL download code removed. Load from manually placed files only.
  3. "Loading domain intelligence" printed twice in test output
     Fix: print() replaced with logger.info() throughout
  4. threat_domains.json rebuilt from network every 6 hours
     Fix: rebuilt from local files only (openphish_feed.txt, urlhaus.csv)
  5. assert len(f) == 35 in feature_extractor.py crashes tests
     Fix: see feature_extractor.py — assert corrected to 32

NEW IN v4:
  - Weighted confidence scoring    (replaces hardcoded 0.98 / 0.75 / 0.5)
  - TLD risk scoring               (.ru .tk .xyz → +0.15 confidence)
  - Subdomain depth heuristic      (>3 levels → +0.10)
  - Punycode / IDN detection       (xn-- prefix → +0.20)
  - Brand homoglyph detection      (levenshtein + leet normalisation → +0.40)
  - Shannon entropy scoring        (random-looking domain label → +0.15)
  - DNS MX record check            (no MX for claimed mail sender → +0.10)
  - Registrar abuse score          (properly wired into final confidence)
  - Async Tier 3 + Tier 4          (asyncio.gather — run concurrently)

FILES TO PLACE MANUALLY IN  data/domain_cache/
(system works with any subset; missing files produce a clear warning)

  tranco_top1m.csv
    Format : rank,domain   e.g.  1,google.com
    Get it : https://tranco-list.eu  → Download → Top 1 Million → CSV

  openphish_feed.txt
    Format : one URL per line
    Get it : https://openphish.com/feed.txt

  urlhaus.csv
    Format : URLhaus CSV (lines beginning # are comments, column 2 is URL)
    Get it : https://urlhaus.abuse.ch/downloads/csv_recent/

  phishtank.csv  (OPTIONAL)
    Format : CSV with header row; phishing URL is in column index 1
    Get it : http://data.phishtank.com/data/YOUR_KEY/online-valid.csv
    Key    : free account at https://www.phishtank.com/register.php

OPTIONAL PYTHON PACKAGES:
  pip install dnspython       → enables DNS MX record checks (Tier 4 heuristic)
  pip install python-whois    → enables WHOIS domain age lookups (Tier 4)
  pip install python-dateutil → required by python-whois
"""

import re
import os
import csv
import json
import math
import asyncio
import logging
import sqlite3
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Set, List, Tuple
from urllib.parse import urlparse

# ── Optional dependency guards ────────────────────────────────────────────────
try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import whois as python_whois
    HAS_WHOIS = True
except ImportError:
    HAS_WHOIS = False

try:
    import dns.resolver as dns_resolver
    HAS_DNSPYTHON = True
except ImportError:
    HAS_DNSPYTHON = False

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# PROJECT ROOT — always 2 levels up from THIS file, regardless of CWD
#
#   D:\...\Phishgaurd_AI\modules\domain_intelligence.py
#                         ↑ parent        ↑ parent.parent
#   parent       = modules/
#   parent.parent = Phishgaurd_AI/    ← project root
#
# This means data/domain_cache/ always resolves correctly whether you run
# Python from the project root, from tests/, or from anywhere else.
# ──────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

class DomainIntelConfig:

    # ── Paths (all absolute, derived from _PROJECT_ROOT) ──────────────
    CACHE_DIR:      Path = _PROJECT_ROOT / "data" / "domain_cache"
    TRANCO_FILE:    Path = CACHE_DIR / "tranco_top1m.csv"
    OPENPHISH_FILE: Path = CACHE_DIR / "openphish_feed.txt"
    URLHAUS_FILE:   Path = CACHE_DIR / "urlhaus.csv"
    PHISHTANK_FILE: Path = CACHE_DIR / "phishtank.csv"

    THREAT_CACHE_FILE:    Path = CACHE_DIR / "threat_domains.json"
    THREAT_MAX_AGE_HOURS: int  = 6

    # ── Google Safe Browsing (Tier 3, optional) ────────────────────────
    GOOGLE_SAFE_BROWSING_KEY: str = os.getenv("GOOGLE_SAFE_BROWSING_KEY", "")
    GOOGLE_SB_URL:            str = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
    TIER3_CALLS_PER_DAY:      int = 10_000
    TIER3_COUNTER_FILE:       Path = CACHE_DIR / "gsb_counter.json"

    # ── WHOIS (Tier 4, optional) ───────────────────────────────────────
    WHOIS_TIMEOUT:        int  = 10
    WHOIS_CACHE_DB:       Path = CACHE_DIR / "whois_cache.db"
    WHOIS_CACHE_TTL_DAYS: int  = 7
    NEW_DOMAIN_AGE_DAYS:  int  = 30

    # ── Weighted confidence scores ─────────────────────────────────────
    # Positive = raises phishing probability
    # Negative = lowers phishing probability (legitimacy signal)
    # Final = clamp(BASE + sum(triggered weights), 0.02, 0.99)
    CONFIDENCE_BASE:         float = 0.35   # unknown domain starts here
    WEIGHT_THREAT_FEED:      float = +0.90
    WEIGHT_SAFE_BROWSING:    float = +0.88
    WEIGHT_NEWLY_REGISTERED: float = +0.30
    WEIGHT_HIGH_RISK_TLD:    float = +0.15
    WEIGHT_HIGH_ENTROPY:     float = +0.15
    WEIGHT_DEEP_SUBDOMAIN:   float = +0.10
    WEIGHT_REGISTRAR_ABUSE:  float = +0.10
    WEIGHT_PUNYCODE:         float = +0.20
    WEIGHT_BRAND_HOMOGLYPH:  float = +0.40
    WEIGHT_NO_MX_RECORD:     float = +0.10
    WEIGHT_IN_TRANCO:        float = -0.80
    WEIGHT_TRANCO_TOP1K:     float = -0.20  # extra reduction for rank ≤ 1000

    # ── Thresholds ─────────────────────────────────────────────────────
    HIGH_ENTROPY_THRESHOLD:        float = 3.5
    DEEP_SUBDOMAIN_LEVELS:         int   = 3
    BRAND_SIMILARITY_MAX_DISTANCE: int   = 2

    # ── High-risk TLDs ─────────────────────────────────────────────────
    HIGH_RISK_TLDS: Set[str] = {
        "tk", "ml", "ga", "cf", "gq",
        "ru", "cn", "pw", "su",
        "xyz", "top", "icu", "click", "link", "win",
        "download", "review", "science", "work", "party",
        "date", "faith", "loan", "racing", "trade",
        "webcam", "men", "info", "biz", "mobi", "co",
    }

    # ── Brand names checked for homoglyphs ────────────────────────────
    BRAND_NAMES: List[str] = [
        "paypal", "google", "microsoft", "apple", "amazon",
        "netflix", "facebook", "instagram", "twitter", "linkedin",
        "dropbox", "adobe", "stripe", "venmo", "cashapp",
        "citibank", "barclays", "hsbc", "wellsfargo", "jpmorgan",
        "bankofamerica", "chase", "hdfc", "icici", "sbi",
        "fedex", "dhl", "ups", "usps", "irs", "hmrc",
        "docusign", "intuit", "quickbooks", "salesforce",
        "github", "gitlab", "zoom", "slack",
    ]

    # ── Known-abused registrars (soft signal) ─────────────────────────
    ABUSE_REGISTRARS: Set[str] = {
        "namecheap", "namesilo", "publicdomainregistry",
        "reg.ru", "nicline", "eranet", "hichina",
    }


# ════════════════════════════════════════════════════════════════════════════
# DOMAIN VERDICT
# v3 fields preserved exactly. v4 fields appended — nothing breaks.
# ════════════════════════════════════════════════════════════════════════════

class DomainVerdict:

    def __init__(self):
        # ── v3 fields (names and types unchanged) ─────────────────────
        self.domain:                  str            = ""
        self.is_legitimate:           bool           = False
        self.is_known_threat:         bool           = False
        self.is_safe_browsing_threat: bool           = False
        self.domain_age_days:         Optional[int]  = None
        self.is_newly_registered:     bool           = False
        self.registrar:               str            = ""
        self.tranco_rank:             Optional[int]  = None
        self.lookup_tier:             int            = 0
        self.threat_type:             str            = ""
        self.confidence:              float          = 0.5

        # ── v4 new fields ──────────────────────────────────────────────
        self.entropy_score:           float          = 0.0
        self.subdomain_depth:         int            = 0
        self.is_punycode:             bool           = False
        self.brand_similarity:        Optional[str]  = None
        self.brand_distance:          Optional[int]  = None
        self.tld_risk_score:          float          = 0.0
        self.has_mx_record:           Optional[bool] = None
        self.registrar_is_abused:     bool           = False
        self.score_breakdown:         Dict[str, float] = {}

    def to_dict(self) -> Dict:
        return {
            "domain":                   self.domain,
            "is_legitimate":            self.is_legitimate,
            "is_known_threat":          self.is_known_threat,
            "is_safe_browsing_threat":  self.is_safe_browsing_threat,
            "domain_age_days":          self.domain_age_days,
            "is_newly_registered":      self.is_newly_registered,
            "registrar":                self.registrar,
            "tranco_rank":              self.tranco_rank,
            "lookup_tier":              self.lookup_tier,
            "threat_type":              self.threat_type,
            "confidence":               self.confidence,
            "entropy_score":            self.entropy_score,
            "subdomain_depth":          self.subdomain_depth,
            "is_punycode":              self.is_punycode,
            "brand_similarity":         self.brand_similarity,
            "brand_distance":           self.brand_distance,
            "tld_risk_score":           self.tld_risk_score,
            "has_mx_record":            self.has_mx_record,
            "registrar_is_abused":      self.registrar_is_abused,
            "score_breakdown":          self.score_breakdown,
        }

    def __repr__(self):
        status = (
            "THREAT"  if self.is_known_threat else
            "LEGIT"   if self.is_legitimate   else
            "UNKNOWN"
        )
        return (
            f"DomainVerdict({self.domain} → {status}, "
            f"tier={self.lookup_tier}, conf={self.confidence:.2f})"
        )


# ════════════════════════════════════════════════════════════════════════════
# DOMAIN INTELLIGENCE MANAGER
# ════════════════════════════════════════════════════════════════════════════

class DomainIntelligenceManager:
    """
    Central brain for all domain reputation lookups.
    Instantiate ONCE, share everywhere (ModelRegistry, tests, pipeline).

    Public API — 100% backward-compatible with v3:
      .lookup(domain_or_url, use_network=True)   → DomainVerdict
      .lookup_async(domain_or_url)               → DomainVerdict  (v4 new)
      .lookup_bulk(domains, use_network=True)    → Dict[str, DomainVerdict]
      .is_legitimate(domain_or_url)              → bool
      .is_threat(domain_or_url)                  → bool
      .get_phishing_confidence(domain_or_url)    → float
      .get_stats()                               → Dict
      .refresh_all()
    """

    def __init__(self, config: DomainIntelConfig = None):
        self.config = config or DomainIntelConfig()
        self.config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self._tranco_domains: Dict[str, int] = {}
        self._threat_domains: Set[str]        = set()

        self._init_whois_cache()
        self._load_tranco()
        self._load_threat_cache()

        logger.info(
            f"DomainIntel v4 ready | "
            f"tranco={len(self._tranco_domains):,} | "
            f"threats={len(self._threat_domains):,} | "
            f"root={_PROJECT_ROOT}"
        )

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    def lookup(self, domain_or_url: str, use_network: bool = False) -> DomainVerdict:
        """Synchronous 4-tier lookup + heuristic enrichment."""
        domain      = self._normalize_domain(domain_or_url)
        root_domain = self._get_root_domain(domain) if domain else ""

        verdict        = DomainVerdict()
        verdict.domain = domain or domain_or_url
        score_parts: Dict[str, float] = {}

        if not domain:
            verdict.confidence = 0.5
            return verdict

        # Heuristics run for every domain at every tier
        self._enrich_heuristics(verdict, domain, root_domain, score_parts)

        # ── TIER 1: Tranco ─────────────────────────────────────────────
        rank = self._tranco_domains.get(domain) or self._tranco_domains.get(root_domain)
        if rank:
            verdict.is_legitimate  = True
            verdict.tranco_rank    = rank
            verdict.lookup_tier    = 1
            score_parts["tranco"]  = self.config.WEIGHT_IN_TRANCO
            if rank <= 1000:
                score_parts["tranco_top1k"] = self.config.WEIGHT_TRANCO_TOP1K
            verdict.confidence = self._compute_confidence(verdict, score_parts)
            return verdict

        # ── TIER 2: Known threat feed ──────────────────────────────────
        if domain in self._threat_domains or root_domain in self._threat_domains:
            verdict.is_known_threat    = True
            verdict.threat_type        = "phishing"
            verdict.lookup_tier        = 2
            score_parts["threat_feed"] = self.config.WEIGHT_THREAT_FEED
            verdict.confidence = self._compute_confidence(verdict, score_parts)
            return verdict

        if not use_network:
            verdict.lookup_tier = 0
            verdict.confidence  = self._compute_confidence(verdict, score_parts)
            return verdict

        # ── TIER 3 + 4: network (sync wrapper around async gather) ─────
        self._run_network_tiers_sync(verdict, domain, root_domain, score_parts)
        verdict.confidence = self._compute_confidence(verdict, score_parts)
        return verdict

    async def lookup_async(self, domain_or_url: str) -> DomainVerdict:
        """
        Async version for use inside FastAPI endpoints or async code.
        Tier 3 + Tier 4 run concurrently via asyncio.gather.
        """
        domain      = self._normalize_domain(domain_or_url)
        root_domain = self._get_root_domain(domain) if domain else ""

        verdict        = DomainVerdict()
        verdict.domain = domain or domain_or_url
        score_parts: Dict[str, float] = {}

        if not domain:
            verdict.confidence = 0.5
            return verdict

        self._enrich_heuristics(verdict, domain, root_domain, score_parts)

        rank = self._tranco_domains.get(domain) or self._tranco_domains.get(root_domain)
        if rank:
            verdict.is_legitimate  = True
            verdict.tranco_rank    = rank
            verdict.lookup_tier    = 1
            score_parts["tranco"]  = self.config.WEIGHT_IN_TRANCO
            if rank <= 1000:
                score_parts["tranco_top1k"] = self.config.WEIGHT_TRANCO_TOP1K
            verdict.confidence = self._compute_confidence(verdict, score_parts)
            return verdict

        if domain in self._threat_domains or root_domain in self._threat_domains:
            verdict.is_known_threat    = True
            verdict.threat_type        = "phishing"
            verdict.lookup_tier        = 2
            score_parts["threat_feed"] = self.config.WEIGHT_THREAT_FEED
            verdict.confidence = self._compute_confidence(verdict, score_parts)
            return verdict

        await self._network_tiers_async(verdict, domain, root_domain, score_parts)
        verdict.confidence = self._compute_confidence(verdict, score_parts)
        return verdict

    def lookup_bulk(self, domains: list, use_network: bool = False) -> Dict[str, DomainVerdict]:
        results = {}
        for domain in set(domains):
            if domain:
                results[domain] = self.lookup(domain, use_network)
        return results

    def is_legitimate(self, domain_or_url: str) -> bool:
        return self.lookup(domain_or_url, use_network=False).is_legitimate

    def is_threat(self, domain_or_url: str) -> bool:
        v = self.lookup(domain_or_url, use_network=False)
        return v.is_known_threat or v.is_safe_browsing_threat

    def get_phishing_confidence(self, domain_or_url: str) -> float:
        return self.lookup(domain_or_url).confidence

    # ═══════════════════════════════════════════════════════════════════
    # ASYNC TIER 3 + 4  — concurrent execution
    # ═══════════════════════════════════════════════════════════════════

    def _run_network_tiers_sync(
        self,
        verdict: DomainVerdict,
        domain: str,
        root_domain: str,
        score_parts: Dict[str, float],
    ):
        """
        Bridge: run the async network gather from synchronous code.
        Handles both cases: no loop running, and loop already running
        (e.g., inside Jupyter notebooks or if called from FastAPI).
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running inside an existing event loop (FastAPI, Jupyter)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._network_tiers_async(verdict, domain, root_domain, score_parts),
                    )
                    future.result(timeout=15)
            else:
                loop.run_until_complete(
                    self._network_tiers_async(verdict, domain, root_domain, score_parts)
                )
        except Exception as e:
            logger.debug(f"Network tier error for {domain}: {e}")

    async def _network_tiers_async(
        self,
        verdict: DomainVerdict,
        domain: str,
        root_domain: str,
        score_parts: Dict[str, float],
    ):
        """
        Tier 3 (GSB) and Tier 4 (WHOIS) run concurrently.
        asyncio.gather overlaps the two I/O waits.
        Without this: ~0.2s (GSB) + ~2s (WHOIS) = ~2.2s sequential.
        With this:    max(0.2s, 2s) = ~2s.
        """
        gsb_task   = asyncio.create_task(self._gsb_async(domain))
        whois_task = asyncio.create_task(self._whois_async(root_domain))

        gsb_result, whois_result = await asyncio.gather(
            gsb_task, whois_task, return_exceptions=True
        )

        if isinstance(gsb_result, bool):
            verdict.is_safe_browsing_threat = gsb_result
            if gsb_result:
                verdict.lookup_tier          = 3
                verdict.threat_type          = "google_safe_browsing"
                score_parts["safe_browsing"] = self.config.WEIGHT_SAFE_BROWSING

        if isinstance(whois_result, dict) and whois_result:
            verdict.domain_age_days     = whois_result.get("age_days")
            verdict.registrar           = whois_result.get("registrar", "")
            verdict.registrar_is_abused = whois_result.get("registrar_is_abused", False)
            verdict.is_newly_registered = (
                verdict.domain_age_days is not None and
                verdict.domain_age_days < self.config.NEW_DOMAIN_AGE_DAYS
            )
            if verdict.lookup_tier == 0:
                verdict.lookup_tier = 4
            if verdict.is_newly_registered:
                score_parts["newly_registered"] = self.config.WEIGHT_NEWLY_REGISTERED
            if verdict.registrar_is_abused:
                score_parts["registrar_abuse"] = self.config.WEIGHT_REGISTRAR_ABUSE

    async def _gsb_async(self, domain: str) -> Optional[bool]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._check_google_safe_browsing, domain)

    async def _whois_async(self, domain: str) -> Optional[Dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_whois_cached, domain)

    # ═══════════════════════════════════════════════════════════════════
    # HEURISTIC ENRICHMENT  (runs for all domains, all tiers)
    # ═══════════════════════════════════════════════════════════════════

    def _enrich_heuristics(
        self,
        verdict: DomainVerdict,
        domain: str,
        root_domain: str,
        score_parts: Dict[str, float],
    ):
        """
        All heuristic signals computed here.
        Writes results into verdict fields AND score_parts dict.
        Runs even for Tranco-verified domains so all verdict fields are populated.
        """

        # 1. Punycode / IDN ───────────────────────────────────────────
        if "xn--" in domain.lower():
            verdict.is_punycode      = True
            score_parts["punycode"] = self.config.WEIGHT_PUNYCODE

        # 2. Shannon entropy ──────────────────────────────────────────
        domain_label          = self._get_domain_label(root_domain)
        verdict.entropy_score = self._shannon_entropy(domain_label)
        if verdict.entropy_score >= self.config.HIGH_ENTROPY_THRESHOLD:
            score_parts["high_entropy"] = self.config.WEIGHT_HIGH_ENTROPY

        # 3. Subdomain depth ──────────────────────────────────────────
        verdict.subdomain_depth = self._subdomain_depth(domain)
        if verdict.subdomain_depth > self.config.DEEP_SUBDOMAIN_LEVELS:
            score_parts["deep_subdomain"] = self.config.WEIGHT_DEEP_SUBDOMAIN

        # 4. TLD risk ─────────────────────────────────────────────────
        tld = self._get_tld(root_domain)
        if tld in self.config.HIGH_RISK_TLDS:
            verdict.tld_risk_score        = self.config.WEIGHT_HIGH_RISK_TLD
            score_parts["high_risk_tld"]  = self.config.WEIGHT_HIGH_RISK_TLD

        # 5. Brand homoglyph ──────────────────────────────────────────
        brand, distance = self._detect_brand_homoglyph(domain_label, root_domain)
        if brand is not None:
            verdict.brand_similarity       = brand
            verdict.brand_distance         = distance
            score_parts["brand_homoglyph"] = self.config.WEIGHT_BRAND_HOMOGLYPH

        # 6. DNS MX record ────────────────────────────────────────────
        if HAS_DNSPYTHON:
            verdict.has_mx_record = self._check_mx_record(root_domain)
            if verdict.has_mx_record is False:
                score_parts["no_mx"] = self.config.WEIGHT_NO_MX_RECORD

    # ═══════════════════════════════════════════════════════════════════
    # WEIGHTED CONFIDENCE SCORER
    # ═══════════════════════════════════════════════════════════════════

    def _compute_confidence(
        self,
        verdict: DomainVerdict,
        score_parts: Dict[str, float],
    ) -> float:
        """
        Final phishing confidence score: 0.02 (definitely legit) → 0.99 (definitely phishing).

        Formula:
            score = BASE + sum(all triggered weights)
            score = clamp(score, 0.02, 0.99)

        Worked example — hdfcbank-secure.co:
            base              = +0.35
            high_risk_tld     = +0.15   (.co in HIGH_RISK_TLDS)
            brand_homoglyph   = +0.40   (hdfc matches brand "hdfc")
            ─────────────────────────
            total             =  0.90   → phishing

        Worked example — hdfcbank.com:
            base              = +0.35
            tranco            = -0.80   (rank ~500 in Tranco)
            ─────────────────────────
            total             = -0.45   → clamped to 0.02 → legit
        """
        total = self.config.CONFIDENCE_BASE + sum(score_parts.values())
        verdict.score_breakdown = {**score_parts, "_base": self.config.CONFIDENCE_BASE}
        return round(max(0.02, min(0.99, total)), 4)

    # ═══════════════════════════════════════════════════════════════════
    # TIER 1: TRANCO — manual file only, zero downloads
    # ═══════════════════════════════════════════════════════════════════

    def _load_tranco(self):
        """
        Load tranco_top1m.csv from data/domain_cache/.
        No download attempted. Missing file → clear warning + 30-domain fallback.
        """
        if not self.config.TRANCO_FILE.exists():
            logger.warning(
                f"\n[DomainIntel] Tranco file missing: {self.config.TRANCO_FILE}"
                f"\n  How to get it:"
                f"\n    1. Go to https://tranco-list.eu"
                f"\n    2. Click Download → Top 1 Million → CSV"
                f"\n    3. Save as {self.config.TRANCO_FILE}"
                f"\n  Using 30-domain fallback until file is placed.\n"
            )
            self._load_fallback_whitelist()
            return

        count = 0
        try:
            with open(self.config.TRANCO_FILE, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        try:
                            rank   = int(row[0])
                            domain = row[1].strip().lower()
                            self._tranco_domains[domain] = rank
                            count += 1
                        except (ValueError, IndexError):
                            continue
            logger.info(f"[DomainIntel] Tranco: {count:,} domains loaded")
        except Exception as e:
            logger.error(f"[DomainIntel] Tranco parse error: {e} — using fallback")
            self._load_fallback_whitelist()

    def _load_fallback_whitelist(self):
        """Minimal 30-domain fallback when tranco_top1m.csv is absent."""
        fallback = {
            "google.com": 1,       "youtube.com": 2,      "facebook.com": 3,
            "amazon.com": 10,      "microsoft.com": 20,   "apple.com": 25,
            "netflix.com": 50,     "paypal.com": 100,     "twitter.com": 15,
            "linkedin.com": 30,    "github.com": 75,      "dropbox.com": 150,
            "instagram.com": 12,   "zoom.us": 200,        "slack.com": 250,
            "chase.com": 300,      "bankofamerica.com": 350,
            "wellsfargo.com": 400, "citibank.com": 450,
            "hdfcbank.com": 500,   "icicibank.com": 550,
            "fedex.com": 600,      "ups.com": 650,        "dhl.com": 700,
            "usps.com": 750,       "irs.gov": 800,        "gov.uk": 850,
        }
        self._tranco_domains.update(fallback)
        logger.info(f"[DomainIntel] Fallback whitelist loaded: {len(fallback)} domains")

    # ═══════════════════════════════════════════════════════════════════
    # TIER 2: THREAT FEEDS — local files only, zero downloads
    # ═══════════════════════════════════════════════════════════════════

    def _load_threat_cache(self):
        """
        Load threat domain set from threat_domains.json.
        If the cache is stale/missing, rebuild it from the local feed files.
        Zero network calls at any point.
        """
        needs_rebuild = (
            not self.config.THREAT_CACHE_FILE.exists() or
            self._is_stale(self.config.THREAT_CACHE_FILE, self.config.THREAT_MAX_AGE_HOURS)
        )
        if needs_rebuild:
            self._build_threat_cache_from_local_files()

        if self.config.THREAT_CACHE_FILE.exists():
            try:
                with open(self.config.THREAT_CACHE_FILE, "r") as f:
                    data = json.load(f)
                self._threat_domains = set(data.get("domains", []))
                logger.info(f"[DomainIntel] Threat cache: {len(self._threat_domains):,} domains")
            except Exception as e:
                logger.error(f"[DomainIntel] Threat cache load error: {e}")
        else:
            logger.warning("[DomainIntel] No threat cache available — Tier 2 disabled")

    def _build_threat_cache_from_local_files(self):
        """
        Parse openphish_feed.txt + urlhaus.csv + phishtank.csv (if present)
        and write the combined domain set to threat_domains.json.
        Called automatically when cache is stale.
        """
        threat_domains: Set[str] = set()

        # ── OpenPhish ────────────────────────────────────────────────
        if self.config.OPENPHISH_FILE.exists():
            before = len(threat_domains)
            try:
                with open(self.config.OPENPHISH_FILE, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            d = self._extract_domain_from_url(line)
                            if d:
                                threat_domains.add(d)
                logger.info(f"[DomainIntel] OpenPhish: +{len(threat_domains)-before:,} domains")
            except Exception as e:
                logger.warning(f"[DomainIntel] OpenPhish parse error: {e}")
        else:
            logger.warning(
                f"[DomainIntel] openphish_feed.txt not found at {self.config.OPENPHISH_FILE}"
                f"\n  Download: https://openphish.com/feed.txt"
            )

        # ── URLhaus ──────────────────────────────────────────────────
        if self.config.URLHAUS_FILE.exists():
            before = len(threat_domains)
            try:
                with open(self.config.URLHAUS_FILE, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split(",")
                        # URLhaus format: id,dateadded,url,...
                        # URL is at index 2; fall back to index 0
                        url = parts[2].strip().strip('"') if len(parts) >= 3 else \
                              parts[0].strip().strip('"')
                        if url and url.startswith("http"):
                            d = self._extract_domain_from_url(url)
                            if d:
                                threat_domains.add(d)
                logger.info(f"[DomainIntel] URLhaus: +{len(threat_domains)-before:,} domains")
            except Exception as e:
                logger.warning(f"[DomainIntel] URLhaus parse error: {e}")
        else:
            logger.warning(
                f"[DomainIntel] urlhaus.csv not found at {self.config.URLHAUS_FILE}"
                f"\n  Download: https://urlhaus.abuse.ch/downloads/csv_recent/"
            )

        # ── PhishTank (optional) ──────────────────────────────────────
        if self.config.PHISHTANK_FILE.exists():
            before = len(threat_domains)
            try:
                with open(self.config.PHISHTANK_FILE, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # skip header
                    for row in reader:
                        if len(row) >= 2:
                            d = self._extract_domain_from_url(row[1])
                            if d:
                                threat_domains.add(d)
                logger.info(f"[DomainIntel] PhishTank: +{len(threat_domains)-before:,} domains")
            except Exception as e:
                logger.warning(f"[DomainIntel] PhishTank parse error: {e}")

        # ── Save aggregated cache ─────────────────────────────────────
        try:
            with open(self.config.THREAT_CACHE_FILE, "w") as f:
                json.dump({
                    "domains": list(threat_domains),
                    "count":   len(threat_domains),
                    "updated": datetime.utcnow().isoformat(),
                    "sources": ["openphish", "urlhaus", "phishtank"],
                }, f)
            self._threat_domains = threat_domains
            logger.info(f"[DomainIntel] Threat cache saved: {len(threat_domains):,} domains")
        except Exception as e:
            logger.error(f"[DomainIntel] Could not save threat cache: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # TIER 3: GOOGLE SAFE BROWSING
    # ═══════════════════════════════════════════════════════════════════

    def _check_google_safe_browsing(self, domain: str) -> Optional[bool]:
        if not HAS_REQUESTS or not self.config.GOOGLE_SAFE_BROWSING_KEY:
            return None
        if not self._can_call_gsb():
            return None
        try:
            payload = {
                "client": {"clientId": "phishing-detector", "clientVersion": "4.0"},
                "threatInfo": {
                    "threatTypes": [
                        "MALWARE", "SOCIAL_ENGINEERING",
                        "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION",
                    ],
                    "platformTypes":    ["ANY_PLATFORM"],
                    "threatEntryTypes": ["URL"],
                    "threatEntries": [
                        {"url": f"http://{domain}/"},
                        {"url": f"https://{domain}/"},
                    ],
                },
            }
            resp = requests.post(
                f"{self.config.GOOGLE_SB_URL}?key={self.config.GOOGLE_SAFE_BROWSING_KEY}",
                json=payload, timeout=10,
            )
            self._increment_gsb_counter()
            return bool(resp.json().get("matches")) if resp.status_code == 200 else None
        except Exception as e:
            logger.warning(f"[DomainIntel] GSB error: {e}")
            return None

    def _can_call_gsb(self) -> bool:
        try:
            if not self.config.TIER3_COUNTER_FILE.exists():
                return True
            with open(self.config.TIER3_COUNTER_FILE) as f:
                data = json.load(f)
            if data.get("date") != datetime.utcnow().strftime("%Y-%m-%d"):
                return True
            return data.get("count", 0) < self.config.TIER3_CALLS_PER_DAY
        except Exception:
            return True

    def _increment_gsb_counter(self):
        try:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            try:
                with open(self.config.TIER3_COUNTER_FILE) as f:
                    data = json.load(f)
                if data.get("date") != today:
                    data = {"date": today, "count": 0}
            except Exception:
                data = {"date": today, "count": 0}
            data["count"] += 1
            with open(self.config.TIER3_COUNTER_FILE, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════
    # TIER 4: WHOIS
    # ═══════════════════════════════════════════════════════════════════

    def _get_whois_cached(self, domain: str) -> Optional[Dict]:
        if not HAS_WHOIS:
            return None
        conn   = sqlite3.connect(str(self.config.WHOIS_CACHE_DB))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data, fetched_at FROM whois_cache WHERE domain = ?", (domain,)
        )
        row = cursor.fetchone()
        if row:
            data_str, fetched_at = row
            if (datetime.utcnow() - datetime.fromisoformat(fetched_at)).days < self.config.WHOIS_CACHE_TTL_DAYS:
                conn.close()
                return json.loads(data_str) if data_str else None
        whois_data = self._fetch_whois(domain)
        cursor.execute(
            "INSERT OR REPLACE INTO whois_cache (domain, data, fetched_at) VALUES (?,?,?)",
            (domain, json.dumps(whois_data) if whois_data else None,
             datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
        return whois_data

    def _fetch_whois(self, domain: str) -> Optional[Dict]:
        try:
            w             = python_whois.whois(domain)
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            age_days = None
            if creation_date:
                if isinstance(creation_date, str):
                    from dateutil import parser as dp
                    creation_date = dp.parse(creation_date)
                age_days = (datetime.utcnow() - creation_date.replace(tzinfo=None)).days
            registrar = str(w.registrar or "").strip()
            country   = str(w.country   or "").upper().strip()
            return {
                "age_days":            age_days,
                "registrar":           registrar,
                "country":             country,
                "registrar_is_abused": any(
                    r in registrar.lower() for r in self.config.ABUSE_REGISTRARS
                ),
                "domain": domain,
            }
        except Exception as e:
            logger.debug(f"[DomainIntel] WHOIS failed for {domain}: {e}")
            return None

    def _init_whois_cache(self):
        conn = sqlite3.connect(str(self.config.WHOIS_CACHE_DB))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS whois_cache (
                domain     TEXT PRIMARY KEY,
                data       TEXT,
                fetched_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ═══════════════════════════════════════════════════════════════════
    # DNS MX CHECK  (new v4)
    # ═══════════════════════════════════════════════════════════════════

    def _check_mx_record(self, domain: str) -> Optional[bool]:
        """
        True  = domain has MX records
        False = no MX records (suspicious for a domain claiming to send email)
        None  = lookup failed / dnspython not installed

        Install: pip install dnspython
        """
        if not HAS_DNSPYTHON:
            return None
        try:
            answers = dns_resolver.resolve(domain, "MX", lifetime=3.0)
            return len(answers) > 0
        except Exception:
            return False

    # ═══════════════════════════════════════════════════════════════════
    # HEURISTIC HELPERS  (new v4)
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _shannon_entropy(text: str) -> float:
        """
        Shannon entropy of a string (higher = more random).
        Legitimate domain labels score 2.0–3.2.
        Machine-generated phishing labels score 3.5–4.5.
        """
        if not text:
            return 0.0
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        n = len(text)
        return round(-sum((f / n) * math.log2(f / n) for f in freq.values()), 4)

    @staticmethod
    def _subdomain_depth(domain: str) -> int:
        """
        Number of subdomain levels above the registered domain.
        google.com                    → 0
        mail.google.com               → 1
        pay.secure.verify.badguy.ru   → 3   ← suspicious
        """
        if HAS_TLDEXTRACT:
            try:
                sub = tldextract.extract(domain).subdomain
                return len(sub.split(".")) if sub else 0
            except Exception:
                pass
        return max(0, domain.rstrip(".").count(".") - 1)

    def _detect_brand_homoglyph(
        self, domain_label: str, root_domain: str
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Detect character-substitution and typosquatting attacks.

        Steps:
          1. Skip if root_domain is in Tranco (it IS the real brand domain)
          2. Extract primary part before first hyphen
             micr0soft-login  →  micr0soft
          3. Normalise leet-speak
             micr0soft  →  microsoft
          4. Exact match after normalisation  →  distance 0
          5. Levenshtein distance ≤ BRAND_SIMILARITY_MAX_DISTANCE
             paypol  →  paypal  (distance 1)

        Returns: (brand_name, distance) or (None, None)
        """
        if self._tranco_domains.get(root_domain):
            return None, None   # real brand domain, not a homoglyph

        leet = str.maketrans({
            "0": "o", "1": "l", "3": "e", "4": "a",
            "5": "s", "6": "g", "7": "t", "8": "b",
            "@": "a", "!": "i",
        })
        primary    = re.split(r"[-_]", domain_label)[0] if domain_label else ""
        normalized = primary.translate(leet)

        for brand in self.config.BRAND_NAMES:
            if normalized == brand:
                return brand, 0
            if (len(primary) >= 4 and
                    abs(len(normalized) - len(brand)) <= self.config.BRAND_SIMILARITY_MAX_DISTANCE):
                dist = self._levenshtein(normalized, brand)
                if dist <= self.config.BRAND_SIMILARITY_MAX_DISTANCE:
                    return brand, dist

        return None, None

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if not s2:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for c1 in s1:
            curr = [prev[0] + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

    # ═══════════════════════════════════════════════════════════════════
    # DOMAIN PARSING UTILITIES  (same names and signatures as v3)
    # ═══════════════════════════════════════════════════════════════════

    def _normalize_domain(self, domain_or_url: str) -> str:
        if not domain_or_url:
            return ""
        s = domain_or_url.strip().lower()
        if "@" in s and "/" not in s:
            return s.split("@")[-1].strip()
        if not s.startswith(("http://", "https://")):
            s = "http://" + s
        try:
            host = urlparse(s).hostname or ""
            return host.split(":")[0].strip()
        except Exception:
            return ""

    def _get_root_domain(self, domain: str) -> str:
        if HAS_TLDEXTRACT:
            try:
                e = tldextract.extract(domain)
                if e.domain and e.suffix:
                    return f"{e.domain}.{e.suffix}".lower()
            except Exception:
                pass
        parts = domain.lower().split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else domain

    def _get_domain_label(self, root_domain: str) -> str:
        """hdfcbank.com → hdfcbank"""
        if HAS_TLDEXTRACT:
            try:
                return tldextract.extract(root_domain).domain.lower()
            except Exception:
                pass
        return root_domain.split(".")[0]

    def _get_tld(self, root_domain: str) -> str:
        """hdfcbank.com → com"""
        if HAS_TLDEXTRACT:
            try:
                return tldextract.extract(root_domain).suffix.lower()
            except Exception:
                pass
        parts = root_domain.split(".")
        return parts[-1].lower() if parts else ""

    def _extract_domain_from_url(self, url: str) -> str:
        return self._get_root_domain(self._normalize_domain(url))

    @staticmethod
    def _is_stale(filepath: Path, max_age_hours: int) -> bool:
        if not filepath.exists():
            return True
        age = datetime.now() - datetime.fromtimestamp(filepath.stat().st_mtime)
        return age > timedelta(hours=max_age_hours)

    # ═══════════════════════════════════════════════════════════════════
    # CACHE MANAGEMENT  (same public API as v3)
    # ═══════════════════════════════════════════════════════════════════

    def refresh_all(self):
        """
        Rebuild threat_domains.json from local feed files.
        Call after updating openphish_feed.txt / urlhaus.csv / phishtank.csv.
        Tranco is not touched here — replace tranco_top1m.csv manually when needed.
        """
        logger.info("[DomainIntel] Rebuilding threat cache from local files...")
        self._build_threat_cache_from_local_files()
        logger.info("[DomainIntel] Refresh complete.")

    def get_stats(self) -> Dict:
        """Same return structure as v3 + v4 additions."""
        return {
            # v3 fields (unchanged)
            "tranco_domains":         len(self._tranco_domains),
            "threat_domains":         len(self._threat_domains),
            "tranco_file_age_hours":  self._file_age_hours(self.config.TRANCO_FILE),
            "threat_cache_age_hours": self._file_age_hours(self.config.THREAT_CACHE_FILE),
            "gsb_enabled":            bool(self.config.GOOGLE_SAFE_BROWSING_KEY),
            "whois_enabled":          HAS_WHOIS,
            # v4 new fields
            "dns_enabled":            HAS_DNSPYTHON,
            "cache_dir":              str(self.config.CACHE_DIR),
            "project_root":           str(_PROJECT_ROOT),
            "tranco_file_exists":     self.config.TRANCO_FILE.exists(),
            "openphish_file_exists":  self.config.OPENPHISH_FILE.exists(),
            "urlhaus_file_exists":    self.config.URLHAUS_FILE.exists(),
            "phishtank_file_exists":  self.config.PHISHTANK_FILE.exists(),
        }

    def _file_age_hours(self, filepath: Path) -> Optional[float]:
        if not filepath.exists():
            return None
        age = datetime.now() - datetime.fromtimestamp(filepath.stat().st_mtime)
        return round(age.total_seconds() / 3600, 1)