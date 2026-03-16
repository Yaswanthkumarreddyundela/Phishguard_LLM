# training/test_distilbert.py  — v2.1
"""
CHANGES FROM v1.0:

  BUG FIX:
    test_custom() parameter was named 'label' but called with 'expected_label'
    Fixed: renamed to 'expected_label' throughout.

  ADDED — Group 5: Known failure modes (documented)
    These are cases the model currently gets wrong, with explanations.
    They stay in the test suite as regression tests — if a future
    retrain fixes them, the score improves automatically.

  ADDED — Group 6: BEC / spear phishing
    Business Email Compromise phishing uses casual conversational tone.
    No urgency keywords, no ALL CAPS, no threats.
    These are the most financially damaging real-world attacks.
    Current model misses these — training data needs BEC examples.
    Added 4 BEC test cases to measure progress across retrains.

  DOCUMENTED — Which failures are fixable vs by-design:
    Fixable via more training data:
      - Failure 1 (password changed notification) → add legit security alerts
      - Failure 3 (conversational phishing) → add BEC examples
    Expected / by-design:
      - Failure 2 (bank fraud alert) → hard even for humans, pipeline corrects
      - Failure 4 (IT impersonation) → URL domain check, handled by LightGBM

USAGE:
  python training/test_distilbert.py

  On Colab:
  !python test_distilbert.py
"""

import torch
from pathlib import Path
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

# ── Model path ────────────────────────────────────────────────────────────────
_IS_COLAB  = Path("/content").exists()
_MODEL_DIR = (
    Path("/content/distilbert_phishing")
    if _IS_COLAB
    else Path(__file__).resolve().parent.parent / "data" / "models" / "distilbert_phishing"
)


# ════════════════════════════════════════════════════════════════════════════
# TEST CASES
# ════════════════════════════════════════════════════════════════════════════

TEST_CASES = [

    # ── Group 1: Obvious phishing ─────────────────────────────────────────
    # All should score > 0.90. If any fail the model has a serious problem.

    ("PHISHING", "1", "Classic account suspension",
     """Subject: URGENT: Your account has been suspended

     Dear Customer,
     We have detected unusual activity on your account. Your account has been
     temporarily suspended. You must verify your identity within 24 hours or
     your account will be permanently deleted.
     Click here immediately: http://paypal-secure-login.xyz/verify
     Failure to comply will result in legal action."""),

    ("PHISHING", "1", "Prize scam",
     """Subject: Congratulations! You've been selected

     You have been selected as our lucky winner! Claim your $5,000 prize NOW.
     This offer expires TODAY. Act immediately to receive your reward.
     Send your bank details to claim@prize-winners.ru to receive your winnings."""),

    ("PHISHING", "1", "Bank credential harvest",
     """Subject: Your Wells Fargo account requires immediate attention

     Dear Wells Fargo Customer,
     We have noticed suspicious login attempts on your account.
     To secure your account, please verify your credentials immediately:
     Username: ___________ Password: ___________ SSN: ___________
     Reply to this email or your account will be closed within 12 hours."""),

    ("PHISHING", "1", "Delivery failure lure",
     """Subject: Package delivery failed - Action required

     We attempted to deliver your package but were unable to complete delivery.
     Your package will be returned unless you confirm your address and pay a
     redelivery fee of $2.99.
     Confirm here: http://dhl-redelivery-secure.tk/confirm?id=8821983"""),

    ("PHISHING", "1", "IRS tax refund",
     """Subject: IRS Notice: You have an unclaimed tax refund of $3,241.00

     Our records show you are entitled to a tax refund.
     To receive your refund you must verify your identity.
     Provide your Social Security Number and bank routing number at:
     http://irs-refund-portal.info/claim
     This offer expires in 48 hours."""),

    # ── Group 2: Obvious legitimate ───────────────────────────────────────
    # All should score < 0.20. Failures here mean too many false positives.

    ("LEGIT", "2", "GitHub PR notification",
     """Subject: [GitHub] Pull request merged: fix typo in README

     Hi there,
     Your pull request #142 'fix typo in README' was merged into main by @jsmith.
     View the pull request: https://github.com/org/repo/pull/142
     You're receiving this because you authored the thread.
     Manage your notifications: https://github.com/settings/notifications"""),

    ("LEGIT", "2", "Standard meeting invite",
     """Subject: Q3 planning meeting - Thursday 2pm

     Hi team,
     Confirming our Q3 planning session for Thursday at 2pm in Conference Room B.
     Agenda: roadmap review, resource allocation, OKR check-in.
     Please review the attached slides before the meeting.
     Let me know if you have any conflicts.
     Thanks, Sarah"""),

    ("LEGIT", "2", "E-commerce order confirmation",
     """Subject: Your Amazon order has shipped - Order #112-3456789

     Hello,
     Your order has shipped and is on its way.
     Order #112-3456789: Sony WH-1000XM5 Headphones
     Estimated delivery: Tuesday, March 4
     Track your package: https://www.amazon.com/gp/your-account/order-details
     Thank you for shopping with Amazon."""),

    ("LEGIT", "2", "Newsletter unsubscribe",
     """Subject: Your weekly digest from Medium

     Here are this week's top stories based on your reading history.
     - The Future of Renewable Energy
     - Understanding Transformer Models
     - 10 Python Tricks You Should Know
     To unsubscribe from these emails, click here.
     Medium, 799 Market St, San Francisco, CA 94103"""),

    # ── Group 3: Edge cases ───────────────────────────────────────────────
    # Tests that look ambiguous but have a clear answer.
    # Two phishing emails that lack obvious urgency markers — model should
    # still catch them via the fake/typosquat domains in the URLs.
    # One real Google alert — model should NOT flag this.

    ("PHISHING", "3", "Soft urgency — no ALL CAPS, links to fake domain",
     # This IS phishing. netflix-billing-update.co is not Netflix.
     # Was mislabelled LEGIT in v1.0 — model was correct all along.
     """Subject: A quick note about your account

     Hi there, we noticed your billing information may be out of date.
     To avoid any interruption to your service, please take a moment
     to update your payment details at your earliest convenience.
     This only takes a minute: https://netflix-billing-update.co/secure
     Thank you for being a valued customer."""),

    ("PHISHING", "3", "Typosquat domain impersonating Microsoft",
     # This IS phishing. microsofft-account-verify.com (double-f) is not Microsoft.
     # Was mislabelled LEGIT in v1.0 — model was correct all along.
     """Subject: Important security update required

     Dear Microsoft Customer,
     As part of our ongoing security improvements, all users must
     re-verify their accounts by March 10th.
     Please sign in at: https://microsofft-account-verify.com/signin
     Failure to verify will result in account suspension.
     Microsoft Security Team"""),

    ("LEGIT", "3", "Real Google sign-in alert",
     """Subject: New sign-in to your Google Account

     Your Google Account was signed in to from a new device.
     Windows, Chrome, New York NY USA — Tuesday, March 4 at 9:32 AM
     If this was you, you can ignore this email.
     If you don't recognise this activity, check your account immediately.
     Google LLC, 1600 Amphitheatre Parkway, Mountain View, CA 94043"""),

    # ── Group 4: Adversarial phishing ─────────────────────────────────────
    # Only the invoice lure currently passes. Others need BEC training data.

    ("PHISHING", "4", "Adversarial: invoice attachment lure",
     """Subject: Invoice #INV-2024-0892 attached

     Please find attached invoice #INV-2024-0892 for services rendered
     in February 2024, totalling $4,250.00.
     Payment is due within 30 days. Please process at your earliest convenience.
     If you have any questions regarding this invoice, please don't hesitate
     to contact our billing department.
     Regards, Accounts Receivable"""),

    # ── Group 5: Known failure modes (regression tests) ───────────────────
    # These currently fail. They stay in the suite as targets.
    # When a future retrain fixes them, the score improves automatically.
    # See comments for why each fails and what the fix is.

    ("LEGIT", "5", "[KNOWN FAIL] Password changed confirmation — false positive",
     # WHY IT FAILS: "password...changed" + "immediately" maps to phishing
     # in all 9 training sources. No legit security notification emails exist
     # in our dataset. Google/Microsoft transactional emails are not present.
     # FIX: Add ~500 legitimate security notification emails to training data.
     # (francescogreco97 LLM dataset includes some — re-check that source)
     """Subject: Your Google account password was changed

     Hi,
     Your Google Account password was recently changed.
     If you made this change, you can disregard this email.
     If you didn't change your password, visit our Help Center immediately.
     The Google Accounts Team"""),

    ("LEGIT", "5", "[KNOWN FAIL] Bank fraud alert — false positive (hard case)",
     # WHY IT FAILS: Real fraud alerts are structurally nearly identical to
     # phishing fraud alerts. Even humans misclassify these. Score 0.8363
     # (not 1.0) shows the model is uncertain — correct behaviour.
     # FIX: Partially by-design. Full pipeline corrects via sender domain +
     # SPF/DKIM + link analysis in LightGBM. DistilBERT alone cannot.
     """Subject: Fraud alert: unusual activity on your Chase card

     We noticed an unusual charge on your Chase card ending in 4821.
     $847.00 at ELECTRONICS STORE, Miami FL on March 4 at 2:14 PM.
     If you recognise this charge, no action needed.
     If you don't recognise it, call the number on the back of your card.
     Chase Fraud Department"""),

    ("PHISHING", "5", "[KNOWN FAIL] BEC: conversational tone, no urgency",
     # WHY IT FAILS: No urgency keywords, no ALL CAPS, no threats.
     # Conversational phishing (BEC) is absent from all 9 training sources.
     # The model learned phishing = formal + urgent + threatening.
     # FIX: Add BEC email examples to training data. See Group 6 for more.
     """Subject: Following up

     Hey,
     Just wanted to check in about that document I sent over last week.
     I noticed you hadn't opened it yet — here's the link again in case
     it got buried: http://docs-share.xyz/view?file=contract_final.pdf
     Let me know if you have any trouble accessing it.
     Best, Tom"""),

    ("PHISHING", "5", "[KNOWN FAIL] IT impersonation — URL domain is only signal",
     # WHY IT FAILS: "scheduled maintenance" + "IT Operations" sounds like
     # real Enron corporate ham. The only phishing signal is the domain
     # corp-sso-portal.net — but DistilBERT sees URLs as raw text tokens,
     # not parsed domains. It cannot assess domain legitimacy.
     # FIX: By-design limitation. Feature extractor catches this:
     #   - corp-sso-portal.net not in Tranco top 1M
     #   - compound brand token ("corp", "sso") triggers brand impersonation
     #   - domain likely newly registered
     # LightGBM scores this correctly via those features.
     """Subject: Scheduled maintenance - action needed

     Hi,
     As part of this weekend's scheduled infrastructure maintenance,
     all employees need to re-authenticate their corporate credentials
     by end of day Friday.
     Please log in here to complete the process: http://corp-sso-portal.net/auth
     IT Operations"""),

    # ── Group 6: BEC / spear phishing ─────────────────────────────────────
    # Business Email Compromise — the most dangerous real-world attack type.
    # Casual tone, no red flags, targets individuals not bulk recipients.
    # All currently fail. Goal: fix with BEC training data in next retrain.

    ("PHISHING", "6", "BEC: CEO wire transfer request",
     """Subject: Quick favour needed

     Hi,
     I'm in a meeting and can't talk. I need you to process an urgent wire
     transfer for a vendor payment before end of day.
     Amount: $24,500 to account details I'll send shortly.
     Please keep this between us for now — I'll explain later.
     Thanks"""),

    ("PHISHING", "6", "BEC: gift card request",
     """Subject: Are you available?

     Hi,
     I need a quick favour. Can you purchase some Google Play gift cards
     for me? I need 5 x $100 cards — it's for a client gift and I'm
     tied up in meetings all day. Scratch off the codes and email them to me.
     I'll reimburse you today.
     Thanks"""),

    ("PHISHING", "6", "BEC: HR payroll redirect",
     """Subject: Payroll bank account update

     Hi HR,
     I recently changed my bank account and would like to update my direct
     deposit details for this month's payroll.
     New account: BSB 062-000, Account 10294847
     Could you please update this before the next pay run?
     Thanks for your help"""),

    ("PHISHING", "6", "BEC: vendor payment redirect",
     """Subject: Updated payment details

     Dear Accounts,
     Please be advised that our banking details have changed effective
     immediately. All future payments should be directed to our new account.
     Bank: First National, BSB: 083-170, Account: 229938471
     Please update your records and confirm receipt of this email.
     Kind regards, Supplier Accounts Team"""),
]


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ════════════════════════════════════════════════════════════════════════════

class PhishingTester:
    def __init__(self, model_dir: Path):
        if not model_dir.exists():
            raise FileNotFoundError(
                f"\nModel not found: {model_dir}"
                f"\nTrain first: python training/train_distilbert.py"
            )
        print(f"Loading model from {model_dir}...")
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
        self.model     = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded. Device: {self.device}\n")

    def predict(self, text: str) -> tuple:
        """Returns (phishing_probability, predicted_label)"""
        inputs = self.tokenizer(
            text[:2048],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs  = torch.nn.functional.softmax(logits, dim=-1)

        phishing_prob = probs[0][1].item()
        label = "PHISHING" if phishing_prob > 0.65 else "LEGIT"
        return phishing_prob, label

    @staticmethod
    def _bar(prob: float, width: int = 20) -> str:
        filled = int(prob * width)
        return f"[{'█' * filled}{'░' * (width - filled)}]"

    def run_tests(self, test_cases: list):
        """Run all test cases grouped by group number."""

        group_meta = {
            "1": ("Group 1 — Obvious phishing",           "All expect score > 0.90"),
            "2": ("Group 2 — Obvious legitimate",          "All expect score < 0.20"),
            "3": ("Group 3 — Edge cases",                  "Subtle phishing + real security alerts"),
            "4": ("Group 4 — Adversarial phishing",        "Expect score > 0.60"),
            "5": ("Group 5 — Known failure modes",         "Regression tests — currently fail"),
            "6": ("Group 6 — BEC / spear phishing",        "Hardest — model needs BEC data"),
        }

        # Group the cases
        groups = {}
        for case in test_cases:
            g = case[1]
            groups.setdefault(g, []).append(case)

        print("=" * 70)
        print("  DISTILBERT PHISHING CLASSIFIER — TEST RESULTS v2.0")
        print("=" * 70)

        total_all = correct_all = 0
        group_scores = {}
        all_fails = []

        for g_id, cases in sorted(groups.items()):
            name, hint = group_meta.get(g_id, (f"Group {g_id}", ""))
            print(f"\n{'─' * 70}")
            print(f"  {name}")
            print(f"  {hint}")
            print(f"{'─' * 70}")

            g_total = g_correct = 0
            for true_label, _, description, email_text in cases:
                prob, pred = self.predict(email_text)
                match = "✓" if pred == true_label else "✗"
                g_total   += 1
                total_all += 1
                if pred == true_label:
                    g_correct   += 1
                    correct_all += 1
                else:
                    all_fails.append((g_id, description, true_label, pred, prob))

                print(f"  {match} [{true_label:<8}] {description}")
                print(f"    Score: {prob:.4f}  {self._bar(prob)}  → {pred}")
                print()

            group_scores[g_id] = (g_correct, g_total)

        # ── Overall summary ───────────────────────────────────────────
        print("=" * 70)
        print("  SUMMARY BY GROUP")
        print("=" * 70)
        for g_id, (correct, total) in sorted(group_scores.items()):
            name, _ = group_meta.get(g_id, (f"Group {g_id}", ""))
            bar = "█" * correct + "░" * (total - correct)
            print(f"  Group {g_id}: {correct}/{total}  [{bar}]  {name.split('—')[1].strip()}")

        print(f"\n  TOTAL (excl. known failures):  ", end="")
        exc5 = [(c, t) for g, (c, t) in group_scores.items() if g != "5"]
        ex_correct = sum(c for c, t in exc5)
        ex_total   = sum(t for c, t in exc5)
        print(f"{ex_correct}/{ex_total}  ({ex_correct/ex_total*100:.0f}%)")
        print(f"  TOTAL (all groups):             "
              f"{correct_all}/{total_all}  ({correct_all/total_all*100:.0f}%)")

        if all_fails:
            print(f"\n  ALL MISCLASSIFIED:")
            for g_id, desc, true, pred, prob in all_fails:
                marker = " [known]" if g_id == "5" else ""
                print(f"    ✗ G{g_id}{marker} {desc}")
                print(f"       True: {true:<8} | Pred: {pred:<8} | Score: {prob:.4f}")

        # ── Generalisation verdict ─────────────────────────────────────
        print(f"\n{'=' * 70}")
        print(f"  GENERALISATION VERDICT")
        print(f"{'─' * 70}")

        g4_correct, g4_total = group_scores.get("4", (0, 0))
        g6_correct, g6_total = group_scores.get("6", (0, 0))
        g2_correct, g2_total = group_scores.get("2", (0, 0))

        print(f"  Adversarial phishing (G4) : {g4_correct}/{g4_total}")
        print(f"  BEC / spear phishing (G6) : {g6_correct}/{g6_total}")
        print(f"  False positive rate  (G2) : {g2_total - g2_correct}/{g2_total} flagged legit as phishing")

        if g4_correct == g4_total and g2_correct == g2_total:
            print(f"\n  ✓ Excellent generalisation on standard adversarial cases")
        if g6_correct == 0:
            print(f"\n  ✗ BEC phishing completely missed — highest priority fix")
            print(f"    Action: Add BEC email examples to training data and retrain")
        elif g6_correct < g6_total:
            print(f"\n  ~ BEC phishing partially caught — retrain with more BEC data")

        print(f"\n  WHAT TO DO NEXT:")
        print(f"  1. Add BEC/conversational phishing to dataset_loader.py")
        print(f"     Suggested source: francescogreco97 LLM-generated phishing")
        print(f"     (includes conversation-style emails the others lack)")
        print(f"  2. Add ~500 legit security notification emails to training data")
        print(f"     (Google/Microsoft password alerts, sign-in notifications)")
        print(f"  3. Retrain and re-run this script — group 5 and 6 scores show progress")
        print(f"{'=' * 70}\n")

    def test_custom(self, email_text: str, expected_label: str = "?"):
        """
        Test a single custom email.
        Paste any email into the MY_EMAIL variable at the bottom of this file.
        Set expected_label to 'PHISHING' or 'LEGIT', or leave as '?' if unknown.
        """
        prob, pred = self.predict(email_text)
        print(f"\n{'─' * 55}")
        print(f"  CUSTOM EMAIL TEST")
        print(f"{'─' * 55}")
        print(f"  Score     : {prob:.4f}  {self._bar(prob)}")
        print(f"  Predicted : {pred}")
        if expected_label != "?":
            match = "✓ CORRECT" if pred == expected_label else "✗ WRONG"
            print(f"  Expected  : {expected_label}  →  {match}")
        preview = email_text.strip()[:120].replace("\n", " ")
        print(f"  Preview   : {preview}...")
        print(f"{'─' * 55}\n")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    tester = PhishingTester(_MODEL_DIR)

    # ── Full test suite ───────────────────────────────────────────────
    tester.run_tests(TEST_CASES)

    # ── Test your own email ───────────────────────────────────────────
    # Replace the text below with any email you want to test.
    # Change expected_label to "PHISHING" or "LEGIT", or leave as "?".

    MY_EMAIL = """
    Subject: Your subscription renewal

    Hi,
    This is a reminder that your annual subscription renews on April 1st.
    No action is needed — your card on file will be charged automatically.
    To manage your subscription, visit your account settings.
    Thanks for being a subscriber.
    """
    tester.test_custom(MY_EMAIL, expected_label="LEGIT")