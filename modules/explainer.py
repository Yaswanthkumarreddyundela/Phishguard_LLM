# modules/explainer.py — v4.0
"""
CHANGES FROM v3.0:
  REPLACED: Phi-2 local model → Claude API (claude-sonnet-4-20250514)
  NEW: Rich, context-aware explanations using actual email content
  NEW: Confidence-calibrated language (certain vs suspicious vs borderline)
  NEW: Specific signal callouts with human-readable descriptions
  NEW: Actionable advice per risk level
  KEEP: Rule-based fallback if API call fails
  KEEP: nlp_weight tiered logic (NLP-dominant / structural / mixed)
"""

import os
import json
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = "claude-sonnet-4-6"

RISK_LEVELS = {
    "critical": (0.85, 1.00),
    "high":     (0.65, 0.85),
    "medium":   (0.45, 0.65),
    "low":      (0.25, 0.45),
    "safe":     (0.00, 0.25),
}


def _get_risk_level(probability: float) -> str:
    for level, (lo, hi) in RISK_LEVELS.items():
        if lo <= probability <= hi:
            return level
    return "medium"


def _build_signal_summary(features: dict, nlp_prob: float, nlp_weight: float) -> str:
    """Convert raw feature dict into human-readable signal list."""
    signals = []

    # NLP signal
    if nlp_weight >= 0.5:
        signals.append(f"Email language matches known phishing patterns ({nlp_prob*100:.0f}% NLP confidence)")

    # Auth signals
    if features.get("spf_fail"):
        signals.append("SPF authentication failed — sender domain cannot be verified")
    if features.get("dkim_fail"):
        signals.append("DKIM signature invalid — email may have been tampered with")
    if features.get("dmarc_fail"):
        signals.append("DMARC policy violation — sender is likely spoofing a legitimate domain")
    if features.get("auth_completely_absent"):
        signals.append("No email authentication headers present (no SPF/DKIM/DMARC)")

    # Sender signals
    if features.get("sender_display_mismatch"):
        signals.append("Display name doesn't match actual sender address")
    if features.get("reply_to_differs"):
        signals.append("Reply-To address differs from sender — replies go to a different address")
    if features.get("sender_is_newly_registered"):
        signals.append("Sender domain was registered very recently")
    if features.get("sender_is_threat"):
        signals.append("Sender domain is on a known threat feed")

    # URL signals
    if features.get("any_url_domain_is_threat"):
        signals.append("Links point to domains on known threat/phishing lists")
    if features.get("any_url_domain_not_in_tranco"):
        signals.append("Links point to obscure/unknown domains not in top 1M websites")
    if features.get("has_ip_based_url"):
        signals.append("Links use raw IP addresses instead of domain names")
    if features.get("url_has_at_symbol"):
        signals.append("URL contains @ symbol — a known phishing obfuscation trick")
    if features.get("brand_in_subdomain"):
        signals.append("Legitimate brand name used as subdomain of suspicious domain")
    if features.get("link_text_domain_mismatch"):
        signals.append("Link text shows a different domain than the actual destination")
    if features.get("has_url_shortener"):
        signals.append("Email uses URL shorteners to hide the real destination")

    # HTML signals
    if features.get("has_form_in_html"):
        signals.append("Email contains an HTML form — possibly harvesting credentials")
    if features.get("html_to_text_ratio", 0) > 0.5:
        signals.append("Very high HTML-to-text ratio — heavy use of HTML obfuscation")

    # Content signals
    if features.get("urgency_phrase_count", 0) >= 2:
        signals.append(f"High urgency language detected ({features['urgency_phrase_count']} urgency phrases)")
    if features.get("impersonation_score", 0) > 0.5:
        signals.append("Email appears to impersonate a trusted brand or organization")
    if features.get("homograph_attack"):
        signals.append("Domain uses lookalike Unicode characters to impersonate a real site")

    # Attachment signals
    if features.get("has_dangerous_attachment"):
        signals.append("Dangerous attachment type detected (.exe, .js, .vbs, etc.)")

    # Domain reputation
    if features.get("ssl_cert_suspicious"):
        signals.append("SSL certificate appears suspicious")

    return signals


def _rule_based_explanation(
    parsed_email: dict,
    features: dict,
    probability: float,
    risk_score: float,
    nlp_prob: float,
    nlp_weight: float,
) -> str:
    """Fallback explanation when Claude API is unavailable."""
    risk  = _get_risk_level(probability)
    pct   = f"{probability*100:.1f}%"
    signals = _build_signal_summary(features, nlp_prob, nlp_weight)

    if risk == "safe":
        return (
            f"This email appears legitimate (confidence: {pct}). "
            "No significant phishing indicators were detected."
        )

    risk_phrases = {
        "critical": f"This email is almost certainly a phishing attempt ({pct} confidence).",
        "high":     f"This email shows strong phishing indicators ({pct} confidence).",
        "medium":   f"This email shows some suspicious characteristics ({pct} confidence).",
        "low":      f"This email has minor suspicious elements ({pct} confidence).",
    }
    base = risk_phrases.get(risk, f"Phishing probability: {pct}.")

    if not signals:
        if nlp_weight >= 0.6:
            return f"{base} The email language closely resembles known phishing messages."
        return f"{base} Multiple weak signals combined to trigger detection."

    top_signals = signals[:4]
    signal_text = "; ".join(top_signals)

    if nlp_weight >= 0.7:
        return (
            f"{base} Flagged primarily based on language patterns. "
            f"Key indicators: {signal_text}."
        )
    elif nlp_weight <= 0.3:
        return (
            f"{base} Flagged based on structural red flags. "
            f"Key indicators: {signal_text}. "
            "Email language appeared neutral but technical signals suggest phishing."
        )
    else:
        return (
            f"{base} Both language analysis ({nlp_prob*100:.0f}% NLP) and "
            f"structural signals indicate phishing. Key indicators: {signal_text}."
        )


def _claude_explanation(
    parsed_email: dict,
    features: dict,
    probability: float,
    risk_score: float,
    nlp_prob: float,
    nlp_weight: float,
) -> Optional[str]:
    """Generate explanation using Claude API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    risk    = _get_risk_level(probability)
    signals = _build_signal_summary(features, nlp_prob, nlp_weight)

    # Build context for Claude
    email_context = {
        "from":    parsed_email.get("from", "Unknown"),
        "subject": parsed_email.get("subject", "No subject"),
        "risk_level": risk,
        "phishing_probability": f"{probability*100:.1f}%",
        "nlp_confidence": f"{nlp_prob*100:.1f}%",
        "nlp_driven": nlp_weight >= 0.6,
        "structural_driven": nlp_weight <= 0.3,
        "detected_signals": signals,
        "url_count": features.get("url_count", 0),
        "has_attachments": bool(features.get("has_dangerous_attachment")),
    }

    prompt = f"""You are PhishGuard, an email security assistant. Analyze this phishing detection result and write a clear, helpful explanation for the user.

Email details:
{json.dumps(email_context, indent=2)}

Write a 2-4 sentence explanation that:
1. States the risk level clearly ({risk.upper()})
2. Explains the TOP 2-3 most important reasons why this was flagged
3. Gives one specific action the user should take
4. Uses plain language — no technical jargon unless necessary

Do NOT start with "I" or "As PhishGuard". Start directly with the assessment.
Keep it under 100 words. Be direct and specific."""

    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        text = data["content"][0]["text"].strip()
        logger.debug(f"Claude explanation generated ({len(text)} chars)")
        return text
    except Exception as e:
        logger.warning(f"Claude API call failed: {e} — using rule-based fallback")
        return None


class PhishingExplainer:
    """
    Generates human-readable phishing explanations.
    Uses Claude API when available, falls back to rule-based logic.
    """

    def __init__(self):
        self.api_available = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
        if self.api_available:
            logger.info("Explainer: Claude API mode")
        else:
            logger.info("Explainer: Rule-based mode (set ANTHROPIC_API_KEY to enable Claude)")

    def _extract_key_signals(self, features: dict, parsed_email: dict) -> list:
        """
        Return a list of the most important human-readable threat signals
        for this email. Used by monitor.py and notifier.py to populate the
        key_signals field in detection results.

        Args:
            features     : dict from FeatureExtractor
            parsed_email : dict from EmailParser

        Returns:
            List of up to 5 signal strings, highest priority first.
        """
        # Derive nlp_prob/nlp_weight from features if stored there,
        # otherwise default to 0.5 so structural signals still enumerate.
        nlp_prob   = float(features.get("nlp_prob",   0.5))
        nlp_weight = float(features.get("nlp_weight", 0.5))

        all_signals = _build_signal_summary(features, nlp_prob, nlp_weight)

        # Prioritise: auth failures → known-bad domains → URL obfuscation → content
        priority_keywords = [
            "SPF", "DKIM", "DMARC",
            "threat", "phishing list",
            "IP address", "shortener",
            "spoofing", "mismatch",
            "urgency", "impersonat",
            "form",
        ]
        ordered = []
        remainder = list(all_signals)
        for keyword in priority_keywords:
            for sig in list(remainder):
                if keyword.lower() in sig.lower() and sig not in ordered:
                    ordered.append(sig)
                    remainder.remove(sig)
        ordered.extend(remainder)

        return ordered[:5]

    def generate_explanation(
        self,
        parsed_email: dict,
        features: dict,
        prediction: int,
        probability: float,
        risk_score: float,
        nlp_prob: float  = 0.5,
        nlp_weight: float = 0.5,
    ) -> str:
        """
        Generate explanation for a phishing detection result.

        Args:
            parsed_email : dict from EmailParser
            features     : dict from FeatureExtractor
            prediction   : 0 or 1
            probability  : float 0-1 (final ensemble probability)
            risk_score   : float 0-1 (risk score)
            nlp_prob     : float 0-1 (NLP model probability)
            nlp_weight   : float 0-1 (how much NLP drove the decision)
        """
        if prediction == 0 or probability < 0.25:
            return (
                f"This email appears legitimate "
                f"(phishing probability: {probability*100:.1f}%). "
                "No significant threats detected."
            )

        # Try Claude API first
        if self.api_available:
            explanation = _claude_explanation(
                parsed_email, features, probability, risk_score, nlp_prob, nlp_weight
            )
            if explanation:
                return explanation

        # Fallback to rule-based
        return _rule_based_explanation(
            parsed_email, features, probability, risk_score, nlp_prob, nlp_weight
        )


# Legacy compatibility — allow direct instantiation as Explainer()
Explainer = PhishingExplainer