# training/dataset_loader.py  — v2.0
"""
Merges combined_dataset.csv + eml_dataset.csv at train time for LightGBM.

combined_dataset.csv — 149K text-only rows (no headers/HTML/attachments)
  Features always 0: spf_fail, dkim_fail, dmarc_fail, auth_completely_absent,
  reply_to_differs, sender_display_mismatch, has_form_in_html,
  html_to_text_ratio, link_text_domain_mismatch, has_dangerous_attachment,
  url_has_at_symbol, brand_in_subdomain, homograph_attack, sender_in_tranco

eml_dataset.csv — real .eml files with full headers/HTML/attachments
  ALL 37 features populated from actual email structure.
  Built by: python training/build_eml_dataset.py

Merge strategy:
  combined_dataset.csv rows -> NLP extraction -> feature extraction -> zeros for header features
  eml_dataset.csv rows      -> features already extracted -> NLP extraction only
  Combined -> train on full 38-feature matrix

USAGE:
  from training.dataset_loader import load_all_training_data
  feature_dicts, nlp_probs, labels = load_all_training_data(cache_path, max_rows)
"""

import sys
import re
import time
import pickle
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from modules.nlp_model           import NLPModel
from modules.feature_extractor   import FeatureExtractor
from modules.domain_intelligence import DomainIntelligenceManager

logger = logging.getLogger(__name__)

COMBINED_DATASET_PATH = _PROJECT_ROOT / "data" / "datasets" / "combined_dataset.csv"
EML_DATASET_PATH      = _PROJECT_ROOT / "data" / "datasets" / "eml_dataset.csv"
FEATURE_CACHE_PATH    = _PROJECT_ROOT / "data" / "datasets" / "feature_cache_v4.pkl"

FEATURE_NAMES_37 = [
    "sender_in_tranco", "sender_is_threat",
    "sender_domain_age_score", "sender_is_newly_registered",
    "ssl_cert_age_days",
    "any_url_domain_is_threat", "any_url_domain_not_in_tranco",
    "min_url_tranco_rank", "any_url_newly_registered",
    "url_count", "has_ip_based_url", "has_url_shortener", "has_http_only",
    "max_url_length", "url_has_at_symbol", "url_entropy_score",
    "suspicious_path_keyword_count", "path_brand_mismatch", "redirect_depth",
    "brand_in_subdomain", "compound_brand_domain", "homograph_attack",
    "sender_display_mismatch", "reply_to_differs",
    "spf_fail", "dkim_fail", "dmarc_fail", "auth_completely_absent",
    "urgency_phrase_count", "subject_has_urgency",
    "impersonation_score", "unknown_domain_action_content",
    "has_dangerous_attachment", "suspicious_attachment_name",
    "has_form_in_html", "link_text_domain_mismatch",
    "html_to_text_ratio",
]


def _text_to_parsed_email(text: str) -> Dict:
    """Convert raw text string to parsed_email dict for feature extraction."""
    subject_match = re.search(r"^Subject:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    from_match    = re.search(r"^From:\s*(.+)$",    text, re.MULTILINE | re.IGNORECASE)
    subject    = subject_match.group(1).strip() if subject_match else ""
    from_field = from_match.group(1).strip()    if from_match    else ""
    url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)
    urls        = list(set(u.rstrip(".,;)") for u in url_pattern.findall(text)))
    return {
        "email_hash": hashlib.sha256(text[:200].encode()).hexdigest(),
        "headers": {
            "from": from_field, "to": "", "reply_to": "",
            "subject": subject, "date": "", "message_id": "",
            "x_mailer": "", "received": [],
            "spf_result":  "none",
            "dkim_result": "none",
            "dmarc_result": "none",
            "content_type": "text/plain",
        },
        "body": {
            "plain_text": text[:5000],
            "html_text":  "",
            "combined":   text[:5000],
        },
        "urls": urls,
        "attachments": [],
        "metadata": {
            "has_html": False, "has_attachments": False,
            "url_count": len(urls), "raw_size_bytes": len(text),
        },
    }


def load_csv_dataset(
    csv_path: Path,
    nlp: NLPModel,
    extractor: FeatureExtractor,
    max_rows: int = 80_000,
) -> Tuple[List[Dict], List[float], List[int]]:
    """
    Load combined_dataset.csv — text-only rows.
    Runs NLP inference + feature extraction on each row.
    Header features will be 0 (no real headers in CSV).
    """
    logger.info(f"Loading {csv_path.name}...")
    df = pd.read_csv(csv_path)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled to {max_rows:,} rows")

    total = len(df)
    logger.info(f"{total:,} emails to process from CSV")

    feature_dicts: List[Dict]  = []
    nlp_probs:     List[float] = []
    labels:        List[int]   = []
    errors = 0
    t_start = time.time()

    for i, row in df.iterrows():
        try:
            text  = str(row.get("text", ""))
            label = int(row.get("label", 0))

            nlp_prob, _ = nlp.predict(text)
            parsed      = _text_to_parsed_email(text)
            features    = extractor.extract(parsed)

            feature_dicts.append(features)
            nlp_probs.append(float(nlp_prob))
            labels.append(label)

            n_done = len(feature_dicts)
            if n_done % 1000 == 0:
                elapsed  = time.time() - t_start
                rate     = n_done / elapsed
                eta_min  = (total - n_done) / rate / 60
                logger.info(f"  CSV: {n_done:>6,}/{total:,} | {rate:.0f}/sec | ETA {eta_min:.1f}m")

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"Row {i} skipped: {e}")

    elapsed = (time.time() - t_start) / 60
    logger.info(f"CSV done in {elapsed:.1f} min. {len(feature_dicts):,} rows, {errors} errors.")
    return feature_dicts, nlp_probs, labels


def load_eml_dataset(
    eml_path: Path,
    nlp: NLPModel,
) -> Tuple[List[Dict], List[float], List[int]]:
    """
    Load eml_dataset.csv — features already extracted by build_eml_dataset.py.
    Only runs NLP inference (no re-extraction needed).

    This is what activates the dead features:
      spf_fail, dkim_fail, dmarc_fail, auth_completely_absent,
      reply_to_differs, sender_display_mismatch, has_form_in_html,
      html_to_text_ratio, link_text_domain_mismatch, has_dangerous_attachment,
      url_has_at_symbol, brand_in_subdomain, homograph_attack
    """
    if not eml_path.exists():
        logger.info(f"eml_dataset.csv not found at {eml_path} — skipping EML merge")
        logger.info("To activate dead features, run: python training/build_eml_dataset.py")
        return [], [], []

    logger.info(f"Loading {eml_path.name}...")
    df = pd.read_csv(eml_path)
    total = len(df)
    logger.info(f"{total:,} EML emails to process (features pre-extracted)")

    # Report which dead features are now alive
    dead_features = [
        "spf_fail", "dkim_fail", "dmarc_fail", "auth_completely_absent",
        "has_form_in_html", "html_to_text_ratio", "link_text_domain_mismatch",
        "has_dangerous_attachment", "url_has_at_symbol", "reply_to_differs",
    ]
    alive = []
    for feat in dead_features:
        if feat in df.columns:
            rate = (df[feat] > 0).sum() / len(df) * 100
            if rate > 0.1:
                alive.append(f"{feat} ({rate:.1f}%)")
    if alive:
        logger.info(f"Activated features: {', '.join(alive)}")

    feature_dicts: List[Dict]  = []
    nlp_probs:     List[float] = []
    labels:        List[int]   = []
    errors = 0
    t_start = time.time()

    for i, row in df.iterrows():
        try:
            label = int(row.get("label", 0))

            # Extract pre-computed feature values
            features = {}
            for feat in FEATURE_NAMES_37:
                features[feat] = float(row.get(feat, 0.0))

            # Run NLP on subject + source (body not stored in eml_dataset)
            subject = str(row.get("subject", ""))
            sender  = str(row.get("source",  ""))
            text_for_nlp = f"Subject: {subject}\nFrom: {sender}"
            nlp_prob, _ = nlp.predict(text_for_nlp)

            feature_dicts.append(features)
            nlp_probs.append(float(nlp_prob))
            labels.append(label)

            n_done = len(feature_dicts)
            if n_done % 500 == 0:
                elapsed = time.time() - t_start
                rate    = n_done / max(elapsed, 0.001)
                eta_min = (total - n_done) / rate / 60
                logger.info(f"  EML: {n_done:>5,}/{total:,} | {rate:.0f}/sec | ETA {eta_min:.1f}m")

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"EML row {i} skipped: {e}")

    elapsed = (time.time() - t_start) / 60
    logger.info(f"EML done in {elapsed:.1f} min. {len(feature_dicts):,} rows, {errors} errors.")
    return feature_dicts, nlp_probs, labels


def load_all_training_data(
    cache_path: Path = FEATURE_CACHE_PATH,
    max_csv_rows: int = 80_000,
    delete_cache: bool = False,
) -> Tuple[List[Dict], List[float], List[int]]:
    """
    Master loader. Merges combined_dataset.csv + eml_dataset.csv.
    Uses cache for combined_dataset (slow) but always re-runs EML (fast,
    features pre-extracted).

    Args:
        cache_path:    Path to feature cache pkl
        max_csv_rows:  Max rows from combined_dataset.csv
        delete_cache:  Force cache rebuild

    Returns:
        feature_dicts, nlp_probs, labels
    """
    if delete_cache and cache_path.exists():
        cache_path.unlink()
        logger.info(f"Cache deleted: {cache_path}")

    # Load NLP model once — used by both loaders
    nlp = NLPModel()
    nlp.load()
    logger.info(f"NLP model loaded on {'GPU' if nlp.device.type == 'cuda' else 'CPU'}")

    intel     = DomainIntelligenceManager()
    extractor = FeatureExtractor(domain_intel=intel)

    # ── Step 1: Load combined_dataset (cached) ────────────────────────
    if cache_path.exists():
        logger.info(f"Loading CSV features from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        csv_features = cached["feature_dicts"]
        csv_nlp      = cached["nlp_probs"]
        csv_labels   = cached["labels"]
        logger.info(f"Cache loaded: {len(csv_features):,} rows (skipping DistilBERT re-inference)")
    else:
        logger.info("No cache — running full CSV extraction...")
        csv_features, csv_nlp, csv_labels = load_csv_dataset(
            COMBINED_DATASET_PATH, nlp, extractor, max_csv_rows
        )
        logger.info(f"Saving cache to {cache_path}...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({
                "feature_dicts": csv_features,
                "nlp_probs":     csv_nlp,
                "labels":        csv_labels,
            }, f)
        logger.info("Cache saved.")

    # ── Step 2: Load eml_dataset (always fresh — fast since pre-extracted)
    eml_features, eml_nlp, eml_labels = load_eml_dataset(EML_DATASET_PATH, nlp)

    # ── Step 3: Merge ─────────────────────────────────────────────────
    all_features = csv_features + eml_features
    all_nlp      = csv_nlp      + eml_nlp
    all_labels   = csv_labels   + eml_labels

    csv_phish = sum(csv_labels)
    eml_phish = sum(eml_labels) if eml_labels else 0

    logger.info(f"\nMerged dataset:")
    logger.info(f"  CSV rows : {len(csv_features):,}  ({csv_phish:,} phishing)")
    logger.info(f"  EML rows : {len(eml_features):,}  ({eml_phish:,} phishing)")
    logger.info(f"  Total    : {len(all_features):,}")
    logger.info(f"  Phishing : {sum(all_labels):,} ({sum(all_labels)/len(all_labels)*100:.1f}%)")

    return all_features, all_nlp, all_labels


if __name__ == "__main__":
    # Quick test — print dataset stats without training
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")
    features, nlp_probs, labels = load_all_training_data()
    logger.info(f"\nReady to train on {len(features):,} samples.")