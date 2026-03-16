# tests/test_adversarial.py — adversarial edge cases
import torch  # MUST be first

import sys, os, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
intel     = DomainIntelligenceManager()
extractor = FeatureExtractor(domain_intel=intel)
print("All models loaded\n")

TEST_EMAILS = [
    (
        "PHISHING",
        "1 — Executive Impersonation (BEC)",
        "Quick Review Needed – Vendor Settlement",
        """Hi Priya,
I'm currently in transit and have limited connectivity.
Could you process the attached invoice from our Singapore vendor today?
The amount matches the revised agreement we discussed last quarter.
Kindly confirm once completed.
Let's avoid delays as this is part of the Q4 closure.
Regards, Rajesh
Chief Operations Officer""",
        "rajesh.mehta@larsentoubro.co",
        "No link, domain typo (larsentoubro.co), social engineering"
    ),
    (
        "PHISHING",
        "2 — OAuth Token Harvesting",
        "Action Required: Conditional Access Policy Update",
        """Hello,
Your organization has implemented a new Conditional Access policy
requiring revalidation of active sessions.
To prevent temporary sign-out from SharePoint and Outlook,
please confirm your session below:
https://login.microsoft365-support.com/verify-session
This process takes less than 30 seconds.
Microsoft Identity Protection""",
        "security-noreply@microsoft365-support.com",
        "Fake subdomain, enterprise terminology, procedural tone"
    ),
    (
        "PHISHING",
        "3 — Reply-Chain Hijacking",
        "Re: Updated Banking Details",
        """Hi Arun,
As discussed in the previous thread, please find our updated remittance details attached.
Kindly use the new account number for the upcoming payment cycle.
Let me know if anything else is needed.
Best, Nisha
Vendor Relations""",
        "accounts@vendor-services.com",
        "No links at all, pretends to be existing thread"
    ),
    (
        "PHISHING",
        "4 — HR Portal Credential Harvest",
        "Benefits Enrollment Reminder",
        """Hello,
Open enrollment closes this Friday.
If you have not reviewed your 2026 benefits package,
please access the employee portal below:
https://portal.workday-support.net/login
Failure to confirm selections will result in default allocation.
HR Operations Team""",
        "hr.operations@workday-support.net",
        "SaaS domain mimic, soft threat, real HR context"
    ),
    (
        "PHISHING",
        "5 — Crypto Spear Phish",
        "Custodial Asset Confirmation Required",
        """Dear Client,
During a routine reconciliation process,
we identified an unverified device associated with your custodial dashboard.
As a precaution, access has been limited.
Please confirm your authentication credentials below to restore access:
https://coinbase-custody.io/device-verify
Thank you,
Coinbase Institutional Security""",
        "support@coinbase-custody.io",
        "Realistic security workflow, subtle credential harvest"
    ),
    (
        "PHISHING",
        "6 — Internal IT Phish",
        "Mailbox Storage Optimization",
        """Team,
We are optimizing mailbox storage ahead of the April server migration.
Accounts exceeding 85% capacity will be archived automatically.
To retain your mailbox configuration, confirm your credentials here:
https://mail.corp-internal.com/retain
IT Infrastructure""",
        "it.support@corp-internal.com",
        "Corporate tone, operational framing, no urgency"
    ),
    (
        "PHISHING",
        "7 — AI-Generated Contextual Phishing",
        "Paper Submission Status – AI Security Track",
        """Dear Yaswanth,
Your submission titled Adaptive Threat Modeling in Cloud Environments
has been shortlisted for secondary review.
Kindly upload the final revision via the reviewer portal:
https://review.ieee-symposium.org/author-login
Deadline for revision is 48 hours.
Regards, Program Committee""",
        "events@ieee-symposium.org",
        "Academic context, tailored content, IEEE domain mimic"
    ),
    (
        "PHISHING",
        "8 — Zero-Link Gift Card BEC",
        "Quick Task Before 4PM",
        """Can you purchase 6 Apple gift cards worth 10,000 each?
I need them for client appreciation distribution.
Send me the codes once purchased.
Thanks.""",
        "cfo@company-finance.com",
        "Zero links, pure behavioral, no technical signals"
    ),
]


def predict_email(subject, body, sender):
    import hashlib
    text = f"Subject: {subject}\nFrom: {sender}\n\n{body}"
    nlp_prob, _ = nlp.predict(text)
    parsed = {
        "subject": subject,
        "body": {"combined": body, "plain_text": body, "html_text": ""},
        "sender": sender,
        "urls": [],
        "headers": {
            "from": sender, "subject": subject, "reply_to": "",
            "spf_result": "none", "dkim_result": "none", "dmarc_result": "none",
        },
        "attachments": [],
        "email_hash": hashlib.sha256((subject+body+sender).encode()).hexdigest(),
    }
    features = extractor.extract(parsed)
    prob, label = clf.predict(features, float(nlp_prob))
    detail = clf.explain(features, float(nlp_prob))
    return {
        "probability":    round(float(prob), 4),
        "label":          label,
        "nlp_prob":       round(float(nlp_prob), 4),
        "nlp_weight":     detail.get("nlp_weight", 0),
        "prob_full":      detail.get("prob_full", 0),
        "prob_struct":    detail.get("prob_structural", 0),
        "active_signals": {k: v for k, v in features.items() if v != 0},
    }


print("=" * 70)
print("  ADVERSARIAL TEST — 8 hard phishing cases")
print(f"  Threshold: {clf.threshold:.2f}")
print("=" * 70)

caught = 0
missed = 0
results = []

for expected, name, subject, body, sender, why_hard in TEST_EMAILS:
    print(f"\n[{name}]")
    print(f"  From   : {sender}")
    print(f"  Hard because: {why_hard}")

    r = predict_email(subject, body, sender)
    flag = "CAUGHT ✓" if r["label"] == "PHISHING" else "MISSED ✗"
    if r["label"] == "PHISHING":
        caught += 1
    else:
        missed += 1

    print(f"  Result : {r['label']} ({r['probability']:.1%})  [{flag}]")
    print(f"  NLP    : {r['nlp_prob']}  weight={r['nlp_weight']:.3f}  "
          f"(full={r['prob_full']:.3f}, struct={r['prob_struct']:.3f})")

    active = r["active_signals"]
    key_signals = {k: active[k] for k in [
        "sender_in_tranco", "any_url_domain_not_in_tranco",
        "auth_completely_absent", "unknown_domain_action_content",
        "suspicious_path_keyword_count", "path_brand_mismatch",
        "brand_in_subdomain", "compound_brand_domain",
        "sender_display_mismatch", "urgency_phrase_count",
    ] if k in active}
    if key_signals:
        print(f"  Signals: {key_signals}")
    else:
        print(f"  Signals: (none triggered)")

    results.append((name, r["label"], r["probability"], r["nlp_prob"]))

print("\n" + "=" * 70)
print(f"  CAUGHT: {caught}/8   MISSED: {missed}/8")
print("=" * 70)

print("\n[Score Breakdown]")
for name, label, prob, nlp in results:
    bar    = "█" * int(prob * 35)
    result = "✓" if label == "PHISHING" else "✗"
    print(f"  {result} {name[:35]:<35} | {bar:<35} | {prob:.3f}  nlp={nlp:.3f}")

if missed > 0:
    print(f"\n[Missed Cases Analysis]")
    print(f"  These emails have no structural signals PhishGuard can detect.")
    print(f"  They rely purely on social engineering (BEC, gift card, reply-chain).")
    print(f"  No ML model catches zero-link BEC without sender behavior history.")
    print(f"  Fix: SPF/DKIM enforcement + sender allowlisting at mail gateway level.")