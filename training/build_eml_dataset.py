# training/build_eml_dataset.py  — v4.0
"""
CHANGES FROM v3.0:
  NEW: Auto 50/50 balance — caps legitimate emails to match phishing count
  NEW: Windows trailing-dot fix (\\\\?\\ prefix for Enron files)
  NEW: os.walk instead of rglob (Windows compatibility)
  NEW: Speed mode — disables SSL cert lookup during EML building
       (SSL is a live-inference feature, not useful at train time)
  FIX: Label corruption bug from --append
  FIX: mbox detection + raw email detection both improved

TARGET DATASET:
  ~15,000 emails, 50/50 split
  Phishing  (~7,500): Nazario 2022-2025 + SpamAssassin spam
  Legit     (~7,500): SpamAssassin ham + Enron (5-10 folders)
"""

import sys
import os
import mailbox
import hashlib
import argparse
import logging
import platform
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from modules.email_parser        import EmailParser
from modules.feature_extractor   import FeatureExtractor
from modules.domain_intelligence import DomainIntelligenceManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SKIP_EXTENSIONS = {
    ".py", ".csv", ".pkl", ".json", ".md", ".gz", ".bz2",
    ".zip", ".tar", ".html", ".htm", ".pdf", ".png", ".jpg",
    ".log", ".xml", ".yml", ".yaml", ".sh", ".bat", ".db",
}

EML_SCHEMA_COLUMNS = [
    "label", "source", "email_hash",
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


def _open_windows_safe(path: Path):
    """
    Open a file safely on Windows — handles trailing-dot filenames
    like Enron's '1.', '2.' by using the \\\\?\\ long path prefix.
    """
    if platform.system() == "Windows":
        abs_path = str(path.resolve())
        if not abs_path.startswith("\\\\?\\"):
            abs_path = "\\\\?\\" + abs_path
        return open(abs_path, "rb")
    return open(path, "rb")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data[:500]).hexdigest()


def _is_mbox(path: Path) -> bool:
    try:
        with _open_windows_safe(path) as f:
            return f.read(5) == b"From "
    except Exception:
        return False


def _is_raw_email(path: Path) -> bool:
    RAW_STARTS = (
        b"Message-ID:", b"message-id:",
        b"Return-Path:", b"Received:",
        b"Date:", b"From:", b"MIME-Version:",
        b"X-", b"Subject:",
    )
    try:
        with _open_windows_safe(path) as f:
            header = f.read(50)
        return any(header.startswith(s) for s in RAW_STARTS)
    except Exception:
        return False


def _parse_message(msg, parser: EmailParser) -> Optional[dict]:
    try:
        raw = msg.as_bytes()
        if hasattr(parser, "parse_from_bytes"):
            return parser.parse_from_bytes(raw)
        if hasattr(parser, "parse_from_string"):
            return parser.parse_from_string(raw.decode("utf-8", errors="replace"))
        if hasattr(parser, "parse"):
            try:
                return parser.parse(raw)
            except Exception:
                return parser.parse(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None


def process_mbox(path: Path, label: int, extractor, parser, source, seen, max_rows=None):
    rows = []
    try:
        mbox = mailbox.mbox(str(path))
        total = len(mbox)
        logger.info(f"  {path.name}: {total:,} messages")
    except Exception as e:
        logger.error(f"Cannot open mbox {path.name}: {e}")
        return []

    for msg in tqdm(mbox, desc=path.name, unit="msg"):
        if max_rows and len(rows) >= max_rows:
            break
        try:
            raw       = msg.as_bytes()
            email_hash = _hash_bytes(raw)
            if email_hash in seen:
                continue
            seen.add(email_hash)

            parsed = _parse_message(msg, parser)
            if not parsed:
                continue

            features = extractor.extract(parsed)
            row = {"label": label, "source": source, "email_hash": email_hash}
            row.update(features)
            rows.append(row)
        except Exception:
            continue

    logger.info(f"  {path.name}: {len(rows):,} extracted")
    return rows


def process_raw_files(files, label, extractor, parser, source, seen, max_rows=None):
    rows   = []
    errors = 0

    for path in tqdm(files, desc=source, unit="email"):
        if max_rows and len(rows) >= max_rows:
            break
        try:
            with _open_windows_safe(path) as f:
                raw = f.read()

            email_hash = _hash_bytes(raw)
            if email_hash in seen:
                continue
            seen.add(email_hash)

            # Parse using available method
            parsed = None
            if hasattr(parser, "parse_from_bytes"):
                parsed = parser.parse_from_bytes(raw)
            elif hasattr(parser, "parse_from_string"):
                parsed = parser.parse_from_string(raw.decode("utf-8", errors="replace"))
            elif hasattr(parser, "parse"):
                try:
                    parsed = parser.parse(raw)
                except Exception:
                    parsed = parser.parse(raw.decode("utf-8", errors="replace"))

            if not parsed:
                errors += 1
                continue

            features = extractor.extract(parsed)
            row = {"label": label, "source": source, "email_hash": email_hash}
            row.update(features)
            rows.append(row)

        except Exception as e:
            errors += 1

    logger.info(f"  {source}: {len(rows):,} extracted, {errors} errors")
    return rows


def collect_files(directory: Path):
    """Collect all files using os.walk (Windows-safe, handles trailing dots)."""
    all_files  = []
    mbox_files = []

    for root, dirs, files in os.walk(str(directory)):
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix.lower() in SKIP_EXTENSIONS:
                continue
            all_files.append(fpath)

    # Separate mbox from raw
    raw_files = []
    for f in all_files:
        if f.suffix.lower() == ".mbox" or _is_mbox(f):
            mbox_files.append(f)
        else:
            raw_files.append(f)

    return mbox_files, raw_files


def process_directory(directory: Path, label: int, extractor, parser,
                      seen: set, max_rows=None) -> list:
    source = directory.name
    logger.info(f"Scanning {directory}...")
    mbox_files, raw_files = collect_files(directory)

    # TREC inmail files are detected as mbox (start with "From ")
    # but each file is a single message — treat as raw for speed
    # Move single-message mbox files to raw_files list
    true_mbox  = []  # multi-message mbox files (Nazario style)
    single_msg = []  # single-message files falsely detected as mbox (TREC style)

    for mf in mbox_files:
        try:
            size = mf.stat().st_size
            # Nazario mbox files are multi-MB; TREC inmail files are <100KB
            if size > 500_000:
                true_mbox.append(mf)
            else:
                single_msg.append(mf)
        except Exception:
            single_msg.append(mf)

    raw_files = single_msg + raw_files
    logger.info(
        f"  {source}: {len(true_mbox)} multi-msg mbox, "
        f"{len(raw_files):,} raw/single-msg files"
    )

    rows = []
    remaining = max_rows

    for mbox_path in true_mbox:
        r = process_mbox(mbox_path, label, extractor, parser, source, seen, remaining)
        rows.extend(r)
        if max_rows:
            remaining = max_rows - len(rows)
            if remaining <= 0:
                break

    if raw_files and (not max_rows or len(rows) < max_rows):
        r = process_raw_files(raw_files, label, extractor, parser, source, seen, remaining)
        rows.extend(r)

    logger.info(f"  {source} total: {len(rows):,}")
    return rows


def build_eml_dataset(
    phishing_dirs: list,
    legit_dirs: list,
    output_path: Path,
    append: bool = False,
    balance: bool = True,
    max_per_class: int = 10_000,
):
    logger.info("="*60)
    logger.info("PhishGuard EML Dataset Builder v4.0")
    logger.info(f"Target: {'50/50 balanced' if balance else 'unbalanced'}, max {max_per_class:,}/class")
    logger.info("="*60)

    # Disable SSL lookups during dataset building — it's a live feature
    # SSL age is always -1 in historical emails anyway (cert has changed)
    os.environ["PHISHGUARD_DISABLE_SSL"] = "1"

    logger.info("Loading domain intelligence...")
    domain_intel = DomainIntelligenceManager()
    extractor    = FeatureExtractor(domain_intel=domain_intel)
    parser       = EmailParser()

    seen_hashes = set()
    existing_df = None

    if append and output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            # Validate it's a real dataset (not corrupted)
            if "label" not in existing_df.columns or "email_hash" not in existing_df.columns:
                logger.warning("Existing file appears corrupted — starting fresh")
                existing_df = None
            else:
                seen_hashes = set(existing_df["email_hash"].dropna().tolist())
                logger.info(f"Appending to {len(existing_df):,} existing rows")
        except Exception as e:
            logger.warning(f"Could not read existing file: {e} — starting fresh")
            existing_df = None

    # ── Process phishing dirs ──────────────────────────────────────── #
    phishing_rows = []
    for d in phishing_dirs:
        d = Path(d)
        if not d.exists():
            logger.warning(f"Not found: {d}")
            continue
        remaining = max_per_class - len(phishing_rows)
        if remaining <= 0:
            break
        rows = process_directory(d, label=1, extractor=extractor,
                                 parser=parser, seen=seen_hashes,
                                 max_rows=remaining)
        phishing_rows.extend(rows)

    # ── Process legit dirs ─────────────────────────────────────────── #
    legit_rows = []
    # If balancing, cap legit to match phishing count
    legit_cap = len(phishing_rows) if balance else max_per_class
    logger.info(f"\nPhishing collected: {len(phishing_rows):,}")
    logger.info(f"Legitimate target : {legit_cap:,} ({'balanced' if balance else 'uncapped'})")

    for d in legit_dirs:
        d = Path(d)
        if not d.exists():
            logger.warning(f"Not found: {d}")
            continue
        remaining = legit_cap - len(legit_rows)
        if remaining <= 0:
            break
        rows = process_directory(d, label=0, extractor=extractor,
                                 parser=parser, seen=seen_hashes,
                                 max_rows=remaining)
        legit_rows.extend(rows)

    all_rows = phishing_rows + legit_rows

    if not all_rows:
        logger.error("No emails processed.")
        return

    df = pd.DataFrame(all_rows)
    for col in EML_SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df[EML_SCHEMA_COLUMNS]

    # Append to existing if valid
    if existing_df is not None:
        df = pd.concat([existing_df, df], ignore_index=True)

    # Stats
    phish_n = (df["label"] == 1).sum()
    legit_n = (df["label"] == 0).sum()
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL DATASET")
    logger.info(f"{'='*60}")
    logger.info(f"  Total      : {len(df):,}")
    logger.info(f"  Phishing   : {phish_n:,}  ({phish_n/len(df)*100:.1f}%)")
    logger.info(f"  Legitimate : {legit_n:,}  ({legit_n/len(df)*100:.1f}%)")
    logger.info(f"  Balance    : {'GOOD (40-60% range)' if 0.4 <= phish_n/len(df) <= 0.6 else 'WARNING: imbalanced'}")

    # Feature activation
    check = [
        "spf_fail", "dkim_fail", "dmarc_fail", "auth_completely_absent",
        "reply_to_differs", "sender_display_mismatch",
        "has_form_in_html", "html_to_text_ratio", "link_text_domain_mismatch",
        "has_dangerous_attachment", "url_has_at_symbol",
        "brand_in_subdomain", "homograph_attack",
    ]
    logger.info(f"\nPreviously-dead feature activation:")
    for feat in check:
        if feat in df.columns:
            rate = (df[feat] > 0).sum() / len(df) * 100
            status = "OK" if rate > 0.5 else "still zero"
            logger.info(f"  {feat:<40} {rate:>5.1f}%  {status}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved: {output_path}")
    logger.info("Next: python training/train_lgbm.py --delete-cache")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phishing_dirs", nargs="+", default=[])
    p.add_argument("--legit_dirs",    nargs="+", default=[])
    p.add_argument("--phishing_dir",  default=None)
    p.add_argument("--legit_dir",     default=None)
    p.add_argument("--output",        default="data/datasets/eml_dataset.csv")
    p.add_argument("--append",        action="store_true")
    p.add_argument("--no-balance",    action="store_true", help="Disable 50/50 balancing")
    p.add_argument("--max-per-class", type=int, default=10_000)
    args = p.parse_args()

    phishing_dirs = list(args.phishing_dirs)
    legit_dirs    = list(args.legit_dirs)
    if args.phishing_dir: phishing_dirs.append(args.phishing_dir)
    if args.legit_dir:    legit_dirs.append(args.legit_dir)

    if not phishing_dirs and not legit_dirs:
        p.print_help()
        sys.exit(1)

    build_eml_dataset(
        phishing_dirs  = phishing_dirs,
        legit_dirs     = legit_dirs,
        output_path    = Path(args.output),
        append         = args.append,
        balance        = not args.no_balance,
        max_per_class  = args.max_per_class,
    )