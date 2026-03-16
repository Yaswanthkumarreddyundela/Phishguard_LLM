"""
training/add_invoice_phishing_samples.py

Appends realistic invoice-style phishing samples to combined_dataset.csv,
then optionally deletes the feature cache so the next retrain processes them.

These are AI-neutral phishing emails: professional tone (NLP near 0),
but unknown sender domain, suspicious URL path, and financial action in body.

Usage:
    python training/add_invoice_phishing_samples.py
    python training/add_invoice_phishing_samples.py --delete-cache
    del data\\datasets\\feature_cache_v4.pkl
    python training/train_lgbm.py
"""

import sys, csv
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH   = _PROJECT_ROOT / "data" / "datasets" / "combined_dataset.csv"
CACHE_PATH = _PROJECT_ROOT / "data" / "datasets" / "feature_cache_v4.pkl"

SAMPLES = [
    (
        "Subject: Invoice #INV-2847 ready for approval\n"
        "From: accounts@billing-remittance.net\n\n"
        "Hi, Please find invoice #INV-2847 for services rendered in February. "
        "Total: $3,450.00. Review and approve payment: "
        "https://billing-remittance.net/invoice/INV-2847/approve "
        "Please process within 5 business days. Accounts Receivable"
    ),
    (
        "Subject: Action required: outstanding invoice from TechSupply Co\n"
        "From: invoices@techsupply-billing.com\n\n"
        "Dear Finance Team, Invoice #8821 for $12,500 remains outstanding. "
        "Please review and authorize payment: "
        "https://techsupply-billing.com/portal/invoice/8821/authorize "
        "Contact us if you have questions. TechSupply Billing"
    ),
    (
        "Subject: New invoice from CloudServices Ltd\n"
        "From: noreply@cloudservices-invoicing.biz\n\n"
        "Hello, Invoice CS-00392 for $875.00 is now available. "
        "Review and pay: https://cloudservices-invoicing.biz/account/invoices/CS-00392/pay "
        "Due within 30 days."
    ),
    (
        "Subject: Purchase order #PO-5521 requires your sign-off\n"
        "From: procurement@po-approval-system.net\n\n"
        "Hi, PO #PO-5521 for office supplies ($2,100) needs your approval. "
        "Review and sign off: https://po-approval-system.net/orders/PO-5521/approve "
        "Vendor requires confirmation by Thursday."
    ),
    (
        "Subject: Approval needed: vendor payment authorization\n"
        "From: finance@vendor-payment-hub.com\n\n"
        "Please authorize vendor payment: Global Office Solutions, $6,750.00. "
        "Ref: VPA-2024-0334. Authorize: "
        "https://vendor-payment-hub.com/payments/VPA-2024-0334/authorize "
        "Payment processes within 24 hours of approval."
    ),
    (
        "Subject: Wire transfer instructions for contract payment\n"
        "From: payments@contractpay-secure.net\n\n"
        "Hello, Contract payment of $28,400 is ready to process. "
        "Review payment details and confirm transfer: "
        "https://contractpay-secure.net/transfer/confirm/CPY-3847 "
        "Please verify banking details before confirming."
    ),
    (
        "Subject: Remittance advice — Invoice settled\n"
        "From: remittance@payment-portal-secure.com\n\n"
        "Hi, Payment for invoice #REF-2290 has been initiated. "
        "Download your remittance advice: "
        "https://payment-portal-secure.com/remittance/REF-2290/download "
        "If you did not request this payment, contact us immediately."
    ),
    (
        "Subject: Your subscription renewal invoice is ready\n"
        "From: billing@subscription-renewal-center.net\n\n"
        "Hi, Annual subscription renewal invoice #SUB-8833 for $299.00 is available. "
        "Review and process payment: "
        "https://subscription-renewal-center.net/billing/SUB-8833/pay "
        "Subscription stays active once payment is received."
    ),
    (
        "Subject: License renewal — action required\n"
        "From: accounts@license-renewal-portal.com\n\n"
        "Your software license (Ref: LIC-4421) expires in 14 days. "
        "Renew to avoid service interruption: "
        "https://license-renewal-portal.com/renew/LIC-4421/checkout "
        "Annual renewal: $1,200.00."
    ),
    (
        "Subject: Expense report #EXP-0092 submitted for reimbursement\n"
        "From: hr@expense-reimbursement-portal.net\n\n"
        "Hi, Expense report #EXP-0092 has been approved. Reimbursement: $347.50. "
        "Submit your bank details: "
        "https://expense-reimbursement-portal.net/claims/EXP-0092/submit-details "
        "Processing takes 3-5 business days. HR Finance Team"
    ),
    (
        "Subject: Payment ready for freelance project — Invoice #FL-2209\n"
        "From: payments@freelance-payment-gateway.com\n\n"
        "Hello, Payment for Invoice #FL-2209, $4,500.00, is ready to be released. "
        "Verify your bank account details: "
        "https://freelance-payment-gateway.com/payout/FL-2209/verify-account "
        "Funds transfer within 2 business days."
    ),
    (
        "Subject: Outstanding payment — project delivery confirmation\n"
        "From: finance@project-delivery-payments.net\n\n"
        "Hi, Please confirm receipt of project deliverables to release payment of $7,200. "
        "Confirm delivery: https://project-delivery-payments.net/confirm/delivery/PRJ-3381 "
        "Payment issued within 48 hours of confirmation."
    ),
    (
        "Subject: Refund initiated — please provide account details\n"
        "From: refunds@customer-refund-processing.com\n\n"
        "Hi, A refund of $189.99 has been processed for your recent order. "
        "Verify your payment details to complete the refund: "
        "https://customer-refund-processing.com/refunds/confirm/RF-8821 "
        "Required to ensure funds reach the correct account."
    ),
    (
        "Subject: Credit note CN-4430 has been issued to your account\n"
        "From: accounts@credit-adjustment-center.biz\n\n"
        "Dear Customer, A credit of $560.00 has been applied to your account. "
        "Redeem or request bank transfer: "
        "https://credit-adjustment-center.biz/credits/CN-4430/redeem "
        "Please action within 30 days or credit will expire."
    ),
    (
        "Subject: Payroll update — please verify your bank details\n"
        "From: hr@payroll-verification-system.net\n\n"
        "Hi, As part of our annual payroll audit, please verify your bank account details. "
        "Update here: https://payroll-verification-system.net/employees/verify-bank-details "
        "Ensures salary is paid to the correct account. HR Team"
    ),
    (
        "Subject: Your updated pay stub is available\n"
        "From: noreply@paystub-employee-portal.com\n\n"
        "Hi, Your pay stub for the period ending March 31 is now available. "
        "Log in to review earnings and deductions: "
        "https://paystub-employee-portal.com/employees/login/review-payslip "
        "Contact payroll within 5 days if you notice discrepancies."
    ),
    (
        "Subject: Contract payment schedule — signature required\n"
        "From: legal@contract-payment-docs.net\n\n"
        "Please review and sign the payment schedule for contract #CTR-2201. "
        "Payment terms: $15,000 over 3 installments. "
        "Sign electronically: https://contract-payment-docs.net/sign/CTR-2201/payment-schedule "
        "Return by Friday to avoid delays. Legal and Finance"
    ),
    (
        "Subject: Your tax document is ready for download\n"
        "From: documents@tax-doc-retrieval.com\n\n"
        "Hi, Your annual tax summary is available. "
        "Download your tax forms: "
        "https://tax-doc-retrieval.com/account/tax-documents/2024/download "
        "You will need these to complete your tax filing."
    ),
    (
        "Subject: VAT invoice #VAT-3390 — payment due\n"
        "From: vat@vat-invoice-processing.net\n\n"
        "Dear Accounts Team, VAT invoice #VAT-3390 for Q1. "
        "Total VAT due: 2,340.00. "
        "Review and pay: https://vat-invoice-processing.net/invoices/VAT-3390/pay "
        "Deadline: 30 days from invoice date."
    ),
    (
        "Subject: Shipment payment confirmation required\n"
        "From: logistics@shipment-payment-confirm.com\n\n"
        "Hi, Your shipment SHP-88219 has arrived at customs. "
        "A clearance fee of $340 is required before release. "
        "Pay fee: https://shipment-payment-confirm.com/customs/SHP-88219/pay-fee "
        "Delays may result in storage charges."
    ),
    (
        "Subject: Outstanding freight invoice — FRT-0041\n"
        "From: billing@freight-invoice-center.net\n\n"
        "Dear Sir/Madam, Invoice FRT-0041 for freight services ($1,890) is due. "
        "Review and remit payment: "
        "https://freight-invoice-center.net/invoices/FRT-0041/remit "
        "We appreciate your prompt payment."
    ),
    (
        "Subject: Billing alert — update payment method\n"
        "From: billing@account-billing-update.com\n\n"
        "Hi, Your payment method on file is expiring soon. "
        "Update your billing details to avoid service interruption: "
        "https://account-billing-update.com/billing/update-payment-method "
        "This takes less than 2 minutes."
    ),
    (
        "Subject: Rent payment due — March 2025\n"
        "From: accounts@rental-payment-hub.net\n\n"
        "Hi, Your monthly rent of $1,850 is due on the 1st. "
        "Submit payment: https://rental-payment-hub.net/tenants/pay/march-2025 "
        "Early payment is appreciated. Property Management"
    ),
    (
        "Subject: Medical billing statement — balance due\n"
        "From: billing@medical-billing-statements.com\n\n"
        "Dear Patient, After insurance processing, a balance of $215.00 is owed. "
        "Review statement and pay: "
        "https://medical-billing-statements.com/patients/pay/STM-4492 "
        "Payment plans available. Medical Billing Department"
    ),
    (
        "Subject: Insurance premium payment reminder\n"
        "From: noreply@insurance-premium-pay.net\n\n"
        "Hi, Your insurance premium of $340/month is due on the 15th. "
        "Make your payment here to avoid a lapse in coverage: "
        "https://insurance-premium-pay.net/policy/PAY-8821/premium "
        "You can also set up automatic payments."
    ),
    (
        "Subject: Grant disbursement — bank details required\n"
        "From: grants@grant-disbursement-office.net\n\n"
        "Congratulations! Your grant application has been approved. Award: $5,000. "
        "Submit your banking details to receive disbursement: "
        "https://grant-disbursement-office.net/grants/submit-banking-details "
        "Funds transfer within 10 business days."
    ),
    (
        "Subject: Consulting invoice for Q1 engagement\n"
        "From: finance@consulting-invoice-portal.biz\n\n"
        "Dear Client, Q1 consulting invoice: 48 hours at $200/hr = $9,600. "
        "Review and approve: "
        "https://consulting-invoice-portal.biz/invoices/Q1-2025/approve "
        "Thank you for your continued business."
    ),
    (
        "Subject: Invoice approval and payment portal\n"
        "From: ap@accounts-payable-billing.net\n\n"
        "Hi, Three invoices require your approval this week. "
        "Review and approve payments: "
        "https://accounts-payable-billing.net/portal/invoices/approve/batch "
        "Total outstanding: $24,300. All approvals due by Friday."
    ),
    (
        "Subject: Overdue account — payment verification required\n"
        "From: collections@overdue-account-verify.com\n\n"
        "Hi, Your account has an overdue balance of $780. "
        "Verify your identity and set up a payment arrangement: "
        "https://overdue-account-verify.com/account/verify/payment-plan "
        "Ignoring this notice may affect your account standing."
    ),
    (
        "Subject: Payment receipt for your recent purchase\n"
        "From: receipts@purchase-receipt-verify.net\n\n"
        "Hi, We have processed a charge of $499 to your account. "
        "If you did not make this purchase, review and dispute here: "
        "https://purchase-receipt-verify.net/transactions/dispute/TXN-8821 "
        "Act within 48 hours to reverse this charge."
    ),
]

def main():
    if not CSV_PATH.exists():
        print(f"ERROR: Dataset not found: {CSV_PATH}")
        return

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        headers = next(csv.reader(f))

    print(f"CSV columns: {headers}")
    if "text" not in headers or "label" not in headers:
        print(f"ERROR: Expected 'text' and 'label' columns. Got: {headers}")
        return

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        existing = sum(1 for _ in f) - 1
    print(f"Existing samples: {existing:,}")

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        for text in SAMPLES:
            row = {col: "" for col in headers}
            row["text"]  = text.strip()
            row["label"] = 1
            writer.writerow(row)

    print(f"Added {len(SAMPLES)} invoice-style phishing samples")
    print(f"New total: {existing + len(SAMPLES):,}")

    if "--delete-cache" in sys.argv:
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
            print(f"Deleted cache: {CACHE_PATH}")
        else:
            print("Cache already absent.")
    elif CACHE_PATH.exists():
        print(f"\nIMPORTANT: Delete the cache before retraining:")
        print(f"  del {CACHE_PATH}")
        print(f"  or re-run with: --delete-cache")

    print(f"\nNext:")
    print(f"  python training/train_lgbm.py  (~35-50 min, cache will be rebuilt)")

if __name__ == "__main__":
    main()