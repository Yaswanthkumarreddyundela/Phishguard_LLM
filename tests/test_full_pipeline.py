# tests/test_full_pipeline.py — v5.0
"""
FULL PIPELINE TEST — runs real DistilBERT + ensemble LightGBM.

Both models are loaded through their proper interfaces:
  - NLPModel              → modules/nlp_model.py
  - LGBMClassifier        → modules/lgbm_classifier.py  (ensemble-aware)
  - FeatureExtractor      → modules/feature_extractor.py

This is the production inference path. Do NOT call lgbm_model.pkl directly.
"""
import torch  # MUST be first — Windows DLL order fix

import sys, os, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from pathlib import Path

from modules.domain_intelligence import DomainIntelligenceManager
from modules.feature_extractor   import FeatureExtractor
from modules.lgbm_classifier     import LGBMClassifier
from modules.nlp_model           import NLPModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LGBM_PATH    = PROJECT_ROOT / "data" / "models" / "lgbm_model.pkl"

print("Loading models...")
clf = LGBMClassifier(model_path=str(LGBM_PATH))
clf.load()

nlp = NLPModel()
nlp.load()
print(f"DistilBERT loaded on {'GPU' if nlp.device.type == 'cuda' else 'CPU'} \u2713")

intel     = DomainIntelligenceManager()
extractor = FeatureExtractor(domain_intel=intel)
print("All models loaded \u2713\n")

TEST_EMAILS = [
    # \u2500\u2500 OBVIOUS PHISHING \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    (
        "PHISHING",
        "URGENT: Your PayPal account has been suspended",
        """Dear Customer, We detected unusual activity. Your account is suspended.
        Verify immediately: http://paypa1-secure-login.xyz/verify/account/update
        Failure to verify within 24 hours will result in permanent suspension.
        PayPal Security Team""",
        "security@paypa1-secure-login.xyz"
    ),
    # \u2500\u2500 IP-BASED URL \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    (
        "PHISHING",
        "Your Apple ID requires verification",
        """Your Apple ID was used from a new device.
        Verify here: http://192.168.1.45/apple/login/verify
        Apple Support""",
        "noreply@apple-support-center.tk"
    ),
    # \u2500\u2500 SUBDOMAIN TRICK \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    (
        "PHISHING",
        "Microsoft 365 - Action Required",
        """Your Microsoft 365 subscription is expiring.
        Update billing: https://microsoft.com.billing-update.ru/office365/renew
        Microsoft Billing Team""",
        "billing@microsoft.com.billing-update.ru"
    ),
    # \u2500\u2500 AI-STYLE NEUTRAL PHISHING \u2014 THE KEY TEST CASE \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # NLP \u2248 0 (clean text) \u2192 weight = 0 \u2192 100% structural model
    # structural model sees: unknown domain + no auth + action content
    (
        "PHISHING",
        "Invoice attached for your review",
        """Hi, Please find the attached invoice for services rendered this month.
        You can review and approve the payment here:
        https://invoice-approval.net/review/payment/doc/48273
        Let me know if you have any questions. Best regards, James""",
        "james.wilson@invoice-approval.net"
    ),
    # \u2500\u2500 COMPROMISED DOMAIN + BRAND IN PATH \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    (
        "PHISHING",
        "Verify your PayPal account",
        """Please verify your account credentials at:
        https://small-business-site.com/paypal/login/verify/credentials
        PayPal Team""",
        "support@small-business-site.com"
    ),
    # \u2500\u2500 LEGITIMATE EMAILS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    (
        "LEGITIMATE",
        "Your GitHub pull request was merged",
        """Hi there, Your pull request #1234 has been merged into main.
        View it at: https://github.com/your-repo/pull/1234
        The GitHub Team""",
        "noreply@github.com"
    ),
    (
        "LEGITIMATE",
        "Your Amazon order has shipped",
        """Hello, Your order #123-456-789 has shipped.
        Track: https://amazon.com/orders/123-456-789
        Amazon Logistics""",
        "shipment-tracking@amazon.com"
    ),
    (
        "LEGITIMATE",
        "Weekly team standup reminder",
        """Hi team, Standup tomorrow at 10am.
        Zoom: https://zoom.us/j/123456789
        See you all then!""",
        "manager@yourcompany.com"
    ),
    (
        "LEGITIMATE",
        "Re: Project proposal feedback",
        """Thanks for the proposal. Looks solid overall.
        Few minor comments in the doc, let's discuss Thursday. Cheers""",
        "colleague@gmail.com"
    ),
]


def predict_email(subject: str, body: str, sender: str) -> dict:
    """
    Run a single email through the full production pipeline.
    Uses real DistilBERT NLP inference + ensemble LightGBM.
    """
    text = f"Subject: {subject}\nFrom: {sender}\n\n{body}"

    # Step 1: Real NLP score from DistilBERT
    nlp_prob, _ = nlp.predict(text)

    # Step 2: Extract 37 structural features
    parsed = {
        "subject": subject,
        "body": {
            "combined":   body,
            "plain_text": body,
            "html_text":  "",
        },
        "sender":     sender,
        "urls":       re.findall(r"https?://[^\s<>"{}|\^\ `\[\]]+", body),
        "headers": {
            "from":         sender,
            "subject":      subject,
            "reply_to":     "",
            "spf_result":   "none",
            "dkim_result":  "none",
            "dmarc_result": "none",
        },
        "attachments": [],
        "email_hash": hashlib.sha256(
            (subject + body + sender).encode()
        ).hexdigest(),
    }
    features = extractor.extract(parsed)

    # Step 3: Ensemble prediction (NLP-adaptive dual model)
    prob, label = clf.predict(features, float(nlp_prob))

    # Step 4: Detailed breakdown for debugging
    detail = clf.explain(features, float(nlp_prob))

    return {
        "probability":    round(float(prob), 4),
        "label":          label,
        "nlp_prob_raw":   round(float(nlp_prob), 4),
        "nlp_weight":     detail.get("nlp_weight", "N/A"),
        "prob_full":      detail.get("prob_full", "N/A"),
        "prob_struct":    detail.get("prob_structural", "N/A"),
        "path_keywords":  features.get("suspicious_path_keyword_count", 0),
        "path_brand":     features.get("path_brand_mismatch", 0),
        "action_content": features.get("unknown_domain_action_content", 0),
        "active_signals": {k: v for k, v in features.items() if v != 0},
    }


print("=" * 65)
print("  END-TO-END PIPELINE TEST \u2014 v5.0 (real DistilBERT + ensemble)")
print("=" * 65)
print(f"  Decision threshold: {clf.threshold:.2f} (auto-tuned during training)")

correct = 0
total   = len(TEST_EMAILS)
results = []

for expected_label, subject, body, sender in TEST_EMAILS:
    print(f"\n[{expected_label}] {subject[:55]}")
    print(f"  Sender: {sender}")

    try:
        result    = predict_email(subject, body, sender)
        predicted = result["label"]
        prob      = result["probability"]

        flag = "\u2713" if predicted == expected_label else "\u2717 WRONG"
        if predicted == expected_label:
            correct += 1

        print(f"  Prediction    : {predicted} ({prob:.1%} phishing) {flag}")
        print(f"  NLP prob      : {result['nlp_prob_raw']}")
        if isinstance(result["nlp_weight"], float):
            print(f"  NLP weight    : {result['nlp_weight']:.3f}  "
                  f"(full={result['prob_full']:.3f}, struct={result['prob_struct']:.3f})")
        print(f"  Path keywords : {result['path_keywords']}")
        print(f"  Path brand    : {result['path_brand']}")
        print(f"  Action content: {result['action_content']}")
        results.append((expected_label, predicted, prob, result["nlp_prob_raw"]))

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

print("\n" + "=" * 65)
print(f"  RESULTS: {correct}/{total} correct ({correct/total:.1%})")
print("=" * 65)

print("\n[Threshold Analysis]")
print("  Probability scores:")
for expected, predicted, prob, nlp_raw in results:
    bar    = "\u2588" * int(prob * 30)
    marker = "\u2190 WRONG" if expected != predicted else ""
    print(f"  {expected:10} | {bar:<30} | {prob:.4f}  nlp={nlp_raw:.4f} {marker}")

# If invoice still fails, show what structural model sees
print("\n[Invoice Email — Expected Ensemble Behavior]")
print("  NLP \u2248 0.0001 \u2192 nlp_weight = 0.000 \u2192 100% structural model")
print("  Structural signals: unknown_domain_action_content=2, auth_absent=1,")
print("                      any_url_domain_not_in_tranco=1")
print("  Target: structural model score > 0.50")