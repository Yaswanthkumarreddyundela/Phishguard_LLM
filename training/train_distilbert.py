# training/train_distilbert.py  — v4.0
"""
REWRITE FROM v3.0:
  Removed all HuggingFace API loaders.
  Now reads ONLY from combined_dataset.csv produced by dataset_loader.py.

  dataset_loader.py already handles everything:
    ✓ Downloads 3 HuggingFace datasets automatically
    ✓ Loads all your Kaggle CSV files
    ✓ Merges, deduplicates, and balances everything
    ✓ Saves the final result to data/datasets/combined_dataset.csv

  So train_distilbert.py has one job: read that CSV and train.
  No duplicate data logic. No API calls. No redundancy.

WORKFLOW:
  Step 1 — Run dataset_loader.py first:
    python training/dataset_loader.py
    → Produces: data/datasets/combined_dataset.csv  (~100k emails)

  Step 2 — Then train:
    python training/train_distilbert.py
    → Reads combined_dataset.csv, tokenizes, trains, saves model

GOOGLE COLAB:
  1. Runtime → Change runtime type → T4 GPU
  2. !pip install transformers datasets torch scikit-learn accelerate
  3. Upload to Colab (Files panel → Upload button):
       training/train_distilbert.py
       combined_dataset.csv            ← produced by dataset_loader.py
  4. Run:
       !python train_distilbert.py
  5. After training, download /content/distilbert_phishing/ and place at:
       Phishgaurd_AI/data/models/distilbert_phishing/

LOCAL:
  python training/dataset_loader.py
  python training/train_distilbert.py
  → Model saves to: data/models/distilbert_phishing/
"""

import os
import json
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Project root — works from any working directory ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Colab detection — /content/ always exists in any Colab runtime ───────────
_IS_COLAB = Path("/content").exists()

# ── CSV path: Colab looks in /content/, local looks in data/datasets/ ────────
_CSV_PATH = (
    Path("/content/combined_dataset.csv")
    if _IS_COLAB
    else _PROJECT_ROOT / "data" / "datasets" / "combined_dataset.csv"
)

# ── Where the trained model is saved ─────────────────────────────────────────
_MODEL_DIR = (
    Path("/content/distilbert_phishing")
    if _IS_COLAB
    else _PROJECT_ROOT / "data" / "models" / "distilbert_phishing"
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD combined_dataset.csv
# ════════════════════════════════════════════════════════════════════════════

def load_combined_csv(filepath: Path = _CSV_PATH) -> pd.DataFrame:
    """
    Load combined_dataset.csv produced by dataset_loader.py.

    This single file already contains all your data:
      - HuggingFace datasets (auto-downloaded by dataset_loader.py)
      - All Kaggle CSV files you placed in data/datasets/
      - Deduplicated and balanced (max 50k per class)

    Required columns: text, label (0 = legitimate, 1 = phishing)
    """
    logger.info(f"  Looking for CSV at: {filepath}")

    if not filepath.exists():
        raise FileNotFoundError(
            f"\n{'='*55}"
            f"\n  CSV not found: {filepath}"
            f"\n"
            f"\n  Run dataset_loader.py first to generate it:"
            f"\n    python training/dataset_loader.py"
            f"\n"
            f"\n  On Colab: upload combined_dataset.csv to /content/"
            f"\n    (Files panel → drag and drop → it lands at /content/combined_dataset.csv)"
            f"\n{'='*55}"
        )

    df = pd.read_csv(filepath)

    # Validate required columns
    missing = [c for c in ["text", "label"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {filepath.name}: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Re-run dataset_loader.py to regenerate the file."
        )

    # Clean
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip().str.len() > 10]
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)

    # Summary
    total    = len(df)
    phishing = df["label"].sum()
    legit    = total - phishing

    logger.info(f"\n{'='*55}")
    logger.info(f"  DATASET LOADED")
    logger.info(f"{'='*55}")
    logger.info(f"  File           : {filepath.name}")
    logger.info(f"  Total samples  : {total:,}")
    logger.info(f"  Phishing   (1) : {phishing:,}  ({phishing/total*100:.1f}%)")
    logger.info(f"  Legitimate (0) : {legit:,}  ({legit/total*100:.1f}%)")

    # Show source breakdown if column exists
    if "source" in df.columns:
        logger.info(f"  Source breakdown:")
        for src, count in df["source"].value_counts().items():
            logger.info(f"    {src:<38}: {count:,}")

    logger.info(f"{'='*55}\n")

    # Warn if class balance is very skewed
    phish_pct = phishing / total
    if phish_pct > 0.8 or phish_pct < 0.2:
        logger.warning(
            f"  ⚠️  Class imbalance: {phish_pct:.0%} phishing. "
            f"Re-run dataset_loader.py — it balances classes automatically."
        )

    return df[["text", "label"]]


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — TOKENIZATION
# ════════════════════════════════════════════════════════════════════════════

def tokenize_dataset(df: pd.DataFrame, tokenizer, max_length: int = 256):
    """
    Tokenize email texts for DistilBERT.

    max_length=256 tokens:
      DistilBERT supports 512 but phishing signals appear early.
      256 = 2x faster training with same accuracy.
      NLPModel inference uses max_length=256 — must match here.

    Stratified 80/20 split:
      Preserves phishing/legit ratio in both train and val sets.
    """
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["label"].tolist(),
    )

    logger.info(f"  Train : {len(train_texts):,}  |  Val : {len(val_texts):,}")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds   = Dataset.from_dict({"text": val_texts,   "label": val_labels})

    train_tok = train_ds.map(tokenize_fn, batched=True)
    val_tok   = val_ds.map(tokenize_fn,   batched=True)

    # HuggingFace Trainer requires "labels" (plural), not "label"
    train_tok = train_tok.rename_column("label", "labels")
    val_tok   = val_tok.rename_column("label", "labels")

    train_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_tok.set_format("torch",   columns=["input_ids", "attention_mask", "labels"])

    return train_tok, val_tok


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — EVALUATION METRICS
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(eval_pred):
    """
    RECALL is the priority metric.
    Missing a phishing email (false negative) costs more than
    a false alarm on a legitimate email (false positive).

    Production targets:
      Accuracy  > 97%
      Recall    > 98%   ← most important
      Precision > 95%
      F1        > 96%
      AUC-ROC   > 0.99
    """
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)
    probs          = torch.nn.functional.softmax(
        torch.tensor(logits), dim=-1
    ).numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    cm             = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy":        accuracy_score(labels, predictions),
        "precision":       precision,
        "recall":          recall,
        "f1":              f1,
        "auc_roc":         roc_auc_score(labels, probs[:, 1]),
        "true_positives":  int(tp),
        "true_negatives":  int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train_model(train_dataset, val_dataset, output_dir: str = None) -> tuple:
    """
    Fine-tune DistilBERT for phishing detection.

    Expected on T4 GPU with ~80k training samples: ~15–25 minutes.
    Expected metrics: Recall > 98%, F1 > 0.97, AUC-ROC > 0.99

    Hyperparameters:
      lr=2e-5        Standard for transformer fine-tuning.
      batch_size=32  Fills a T4 GPU (16GB) at max_length=256.
      epochs=5       Early stopping triggers at epoch 3–4 typically.
      warmup=100     Prevents loss divergence in the first steps.
      weight_decay   L2 regularisation, reduces overfitting.
    """
    if output_dir is None:
        output_dir = str(_MODEL_DIR)
    os.makedirs(output_dir, exist_ok=True)

    use_fp16 = torch.cuda.is_available()
    logger.info(f"  Device : {'GPU ✓  (fp16 on)' if use_fp16 else 'CPU  (fp16 off — slower)'}")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label={0: "legitimate", 1: "phishing"},
        label2id={"legitimate": 0, "phishing": 1},
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        report_to="none",
        fp16=use_fp16,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("\n=== FINAL EVALUATION ===")
    eval_results = trainer.evaluate()
    for k, v in eval_results.items():
        logger.info(f"  {k:<25}: {v:.4f}" if isinstance(v, float) else f"  {k:<25}: {v}")

    trainer.save_model(output_dir)
    DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased"
    ).save_pretrained(output_dir)

    with open(f"{output_dir}/eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    logger.info(f"\n✅ Model saved → {output_dir}")
    if _IS_COLAB:
        logger.info("   In the Files panel (left sidebar):")
        logger.info("   Right-click /content/distilbert_phishing/ → Download")
        logger.info("   Place at: Phishgaurd_AI/data/models/distilbert_phishing/")

    return trainer, eval_results


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  DistilBERT Phishing Classifier — Training v4.0")
    logger.info("=" * 60)
    logger.info(f"  Environment : {'Google Colab' if _IS_COLAB else 'Local machine'}")
    logger.info(f"  CSV path    : {_CSV_PATH}")
    logger.info(f"  Model saves : {_MODEL_DIR}")
    logger.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────
    logger.info("\nLoading dataset...")
    df = load_combined_csv()

    # ── Tokenize ──────────────────────────────────────────────────────
    logger.info("Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    logger.info("Tokenizing...")
    train_dataset, val_dataset = tokenize_dataset(df, tokenizer)

    # ── Train ─────────────────────────────────────────────────────────
    trainer, results = train_model(train_dataset, val_dataset)

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Recall  : {results.get('eval_recall',  0):.4f}")
    logger.info(f"  F1      : {results.get('eval_f1',      0):.4f}")
    logger.info(f"  AUC-ROC : {results.get('eval_auc_roc', 0):.4f}")
    logger.info(f"  Saved   : {_MODEL_DIR}")
    logger.info("=" * 60)