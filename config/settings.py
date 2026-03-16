import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # IMAP Settings — account being monitored
    IMAP_SERVER    = os.getenv("IMAP_SERVER",   "imap.gmail.com")
    IMAP_PORT      = int(os.getenv("IMAP_PORT", 993))
    EMAIL_ADDRESS  = os.getenv("EMAIL_ADDRESS",  "your@gmail.com")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "your_app_password")

    # Alert sender — dedicated PhishGuard account
    ALERT_SENDER_EMAIL    = os.getenv("ALERT_SENDER_EMAIL",    "phishgaurdai@gmail.com")
    ALERT_SENDER_PASSWORD = os.getenv("ALERT_SENDER_PASSWORD", "")
    ALERT_RECIPIENT_EMAIL = os.getenv("ALERT_RECIPIENT_EMAIL", EMAIL_ADDRESS)

    # Model paths
    DISTILBERT_MODEL_PATH = "data/models/distilbert_phishing"
    LGBM_MODEL_PATH       = "data/models/lgbm_model.pkl"

    # Detection threshold
    # Do NOT apply manually — LGBMClassifier loads this from the pkl.
    # clf.predict() applies clf.threshold internally.
    # Current trained value: 0.35 (auto-tuned, FNR=2.01% FPR=3.88%)
    PHISHING_THRESHOLD = 0.35

    # Database
    DATABASE_URL = "sqlite:///phishing_detector.db"

    # Phi-2 settings
    PHI2_MODEL             = "microsoft/phi-2"
    MAX_EXPLANATION_TOKENS = 200

    # Domain Intelligence
    DOMAIN_CACHE_DIR         = "data/domain_cache"
    GOOGLE_SAFE_BROWSING_KEY = os.getenv("GOOGLE_SAFE_BROWSING_KEY", "")
    PHISHTANK_API_KEY        = os.getenv("PHISHTANK_API_KEY",        "")
    DOMAIN_INTEL_USE_NETWORK = True
    NEW_DOMAIN_AGE_DAYS      = 30

config = Config()