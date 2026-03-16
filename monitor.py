# monitor.py  — v4.0
"""
CHANGES FROM v3.0:
  FIX: Pass nlp_prob and nlp_weight into generate_explanation() so the
       explainer can produce tiered explanations based on what actually
       drove the detection (NLP vs structural vs mixed).
       Previously nlp_prob defaulted to 0.5 and nlp_weight to 0.5 —
       so every explanation was "mixed" even when NLP was 99.9%.
"""

import time, logging
import torch

from modules.email_fetcher       import EmailFetcher
from modules.email_parser        import EmailParser
from modules.feature_extractor   import FeatureExtractor
from modules.nlp_model           import NLPModel
from modules.lgbm_classifier     import LGBMClassifier
from modules.explainer           import PhishingExplainer
from modules.database            import DatabaseManager
from modules.domain_intelligence import DomainIntelligenceManager
from modules.notifier            import PhishingNotifier, _decode_subject
from config.settings             import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

POLL_INTERVAL = 60
FETCH_COUNT   = 10


def load_models():
    logger.info("Loading models...")
    domain_intel = DomainIntelligenceManager()
    nlp          = NLPModel();       nlp.load()
    extractor    = FeatureExtractor(domain_intel=domain_intel)
    lgbm         = LGBMClassifier(); lgbm.load()
    explainer    = PhishingExplainer()
    db           = DatabaseManager(); db.init_db()
    notifier     = PhishingNotifier()
    logger.info(f"Ready | threshold={lgbm.threshold:.2f}")
    stats = domain_intel.get_stats()
    logger.info(f"Domain intel: {stats['tranco_domains']:,} legit + {stats['threat_domains']:,} threat")
    return dict(nlp=nlp, extractor=extractor, lgbm=lgbm,
                explainer=explainer, db=db, notifier=notifier)


def process_email(parsed, models):
    features = models["extractor"].extract(parsed)
    body     = parsed.get("body", {})
    text     = body.get("combined") or body.get("plain_text") or ""
    subject  = parsed.get("headers", {}).get("subject", "")
    sender   = parsed.get("headers", {}).get("from", "")
    nlp_prob, _ = models["nlp"].predict(f"Subject: {subject}\nFrom: {sender}\n\n{text}")
    nlp_prob    = float(nlp_prob)

    probability, label = models["lgbm"].predict(features, nlp_prob)
    probability = float(probability)
    prediction  = 1 if label == "PHISHING" else 0
    risk_score  = round(probability * 100, 1)

    # Estimate nlp_weight: how much did NLP drive vs structural?
    # Use the ratio of nlp_prob to final probability as a proxy.
    # High nlp_prob + high final prob = NLP dominated.
    # Low nlp_prob + high final prob = structural dominated.
    if probability > 0:
        nlp_weight = min(nlp_prob / probability, 1.0)
    else:
        nlp_weight = 0.5

    explanation = models["explainer"].generate_explanation(
        parsed, features, prediction, probability, risk_score,
        nlp_prob=nlp_prob,       # NEW
        nlp_weight=nlp_weight,   # NEW
    )
    key_signals = models["explainer"]._extract_key_signals(features, parsed)

    return {
        "email_hash":      parsed["email_hash"],
        "is_phishing":     bool(prediction),
        "label":           label.lower(),
        "confidence":      round(probability, 4),
        "risk_score":      risk_score,
        "nlp_probability": round(nlp_prob, 4),
        "key_signals":     key_signals,
        "explanation":     explanation,
    }


def run_monitor():
    models  = load_models()
    parser  = EmailParser()
    fetcher = EmailFetcher()

    seen = {r["email_hash"] for r in models["db"].get_history(limit=500)}
    logger.info(f"Skipping {len(seen)} previously processed emails")
    logger.info(f"Watching    : {config.EMAIL_ADDRESS}")
    logger.info(f"Alert sender: {config.ALERT_SENDER_EMAIL}")
    logger.info(f"Alert to    : {config.ALERT_RECIPIENT_EMAIL}")
    logger.info(f"Polling every {POLL_INTERVAL}s — Ctrl+C to stop\n")

    while True:
        try:
            if not fetcher.connect():
                logger.error("IMAP connection failed — retrying in 60s")
                time.sleep(60)
                continue

            raw_emails = fetcher.fetch_emails(folder="INBOX", count=FETCH_COUNT)
            fetcher.disconnect()

            new_count = phishing_count = skipped = 0

            for raw in raw_emails:
                parsed     = parser.parse(raw)
                email_hash = parsed["email_hash"]

                if email_hash in seen:
                    continue

                sender  = parsed.get("headers", {}).get("from", "")
                subject = _decode_subject(parsed.get("headers", {}).get("subject", ""))

                # Skip PhishGuard alert emails — prevents infinite loop
                if config.ALERT_SENDER_EMAIL.lower() in sender.lower():
                    seen.add(email_hash)
                    skipped += 1
                    logger.debug(f"[Skip] PhishGuard alert: {subject[:60]}")
                    continue

                seen.add(email_hash)
                new_count += 1

                try:
                    result = process_email(parsed, models)
                    models["db"].store_result(
                        email_hash=email_hash,
                        is_phishing=result["is_phishing"],
                        confidence=result["confidence"],
                        risk_score=result["risk_score"],
                        label=result["label"],
                        explanation=result["explanation"],
                    )
                    status = "PHISHING !" if result["is_phishing"] else "legitimate"
                    logger.info(
                        f"[{status}] {sender[:40]} | "
                        f"{subject[:50]} | "
                        f"conf={result['confidence']:.1%} "
                        f"nlp={result['nlp_probability']:.1%}"
                    )
                    if result["is_phishing"]:
                        phishing_count += 1
                        models["notifier"].notify(result, parsed)

                except Exception as e:
                    logger.error(f"Pipeline error — {subject[:40]}: {e}")

            summary = f"Cycle: {new_count} new, {phishing_count} phishing"
            if skipped:
                summary += f", {skipped} PhishGuard alerts skipped"
            logger.info(summary)

        except KeyboardInterrupt:
            logger.info("Monitor stopped")
            break
        except Exception as e:
            logger.error(f"Monitor error: {e} — retrying in 60s")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_monitor()