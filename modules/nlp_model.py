# modules/nlp_model.py
"""
CHANGES FROM v1 (what was wrong and why):

  FIX 1: `if not self.model` → `if self.model is None`
    Was: `if not self.model:` in predict()
    Problem: self.model is a PyTorch nn.Module. Calling `not` on a loaded
    Module raises:
      RuntimeError: Boolean value of Tensor with more than one element is ambiguous
    This crashes on the FIRST inference call after a successful load().
    Fix: `if self.model is None`

  FIX 2: text[:512] is wrong — 512 chars ≈ 100–130 tokens
    Was: prob, _ = self.predict(text[:512])  # DistilBERT input limit
    Problem: 512 is the TOKEN limit, not the CHARACTER limit.
    Average English text has ~4-5 chars per token.
    512 chars → only ~100–130 tokens → model sees only the first ~2 sentences.
    Phishing signals are often in the body, not just the first sentence.
    Fix: truncate to MAX_CHAR_INPUT = 2048 chars (≈ 400–500 tokens).
    The tokenizer's max_length=256 handles the final token truncation correctly.

  FIX 3: print() → logger.info()
    Consistent with the rest of the codebase.

  FIX 4: Comment said "DistilBERT input limit" for char truncation
    Incorrect — the token limit and char limit are very different things.
    Comment updated to explain what the truncation actually does.

  FIX 5: No file/directory existence check in load()
    Was: DistilBertTokenizerFast.from_pretrained() raises a confusing
    OSError with a long HuggingFace error if model_path doesn't exist.
    Fix: check path exists first with a clear, actionable error message.
"""

import logging
import torch
from pathlib import Path
from typing import Dict, Tuple

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from config.settings import config

logger = logging.getLogger(__name__)

# Each token is roughly 4–5 English characters.
# We train with max_length=256 tokens → feed ~2048 chars to be safe.
# The tokenizer's truncation handles the final cut to exactly 256 tokens.
_MAX_CHAR_INPUT = 2048


class NLPModel:
    """
    DistilBERT inference wrapper.

    Outputs a single probability score (0.0–1.0) representing how likely
    an email is phishing. This float is passed to LightGBM as one feature.

    The model is loaded once and reused for all predictions.
    Inference speed:
      CPU : ~50ms per email
      GPU : ~5ms per email

    Usage:
        nlp = NLPModel()
        nlp.load()
        prob = nlp.predict_from_parsed_email(parsed_email)
        # or
        prob, label = nlp.predict("Subject: URGENT...\n\nYour account...")
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.DISTILBERT_MODEL_PATH
        self.model      = None
        self.tokenizer  = None
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # FIX 3: logger instead of print
        logger.info(f"[NLP] Device: {self.device}")

    def load(self):
        """
        Load DistilBERT model and tokenizer from disk.

        Model must be trained and saved first via:
          python training/train_distilbert.py

        Then placed at the path in config.DISTILBERT_MODEL_PATH:
          data/models/distilbert_phishing/
        """
        # FIX 5: check path exists with clear error before HuggingFace's OSError
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"\n[NLP] Model directory not found: {path}"
                f"\n  Train the model first:"
                f"\n    python training/train_distilbert.py"
                f"\n  Then place the output folder at: {path}"
            )

        logger.info(f"[NLP] Loading model from {path}...")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(path))
        self.model     = DistilBertForSequenceClassification.from_pretrained(str(path))
        self.model.to(self.device)
        self.model.eval()

        logger.info("[NLP] Model loaded successfully")

    def predict(self, text: str) -> Tuple[float, str]:
        """
        Run inference on email text.

        Args:
            text: Combined email text — subject + body.
                  Long texts are automatically truncated to 2048 chars
                  before tokenisation (≈ 400–500 tokens). The tokenizer
                  then truncates to exactly 256 tokens as trained.

        Returns:
            (phishing_probability, predicted_label)
            - phishing_probability: 0.0–1.0
            - predicted_label:      "phishing" or "legitimate"
        """
        # FIX 1: `is None` not `not self.model` — PyTorch Module raises
        # "Boolean value of Tensor with more than one element is ambiguous"
        if self.model is None:
            raise RuntimeError(
                "[NLP] Model not loaded. Call load() before predict()."
            )

        # FIX 2 + FIX 4: 2048 chars ≈ 400–500 tokens (correct comment)
        # The tokenizer's max_length=256 handles the final token truncation.
        inputs = self.tokenizer(
            text[:_MAX_CHAR_INPUT],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            outputs       = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        phishing_prob = probabilities[0][1].item()
        label = (
            "phishing"
            if phishing_prob > config.PHISHING_THRESHOLD
            else "legitimate"
        )

        return phishing_prob, label

    def predict_from_parsed_email(self, parsed_email: Dict) -> float:
        """
        Extract text from a parsed email dict and return phishing probability.
        This is the method called by the pipeline to generate the nlp_phishing_prob
        feature that gets passed to LightGBM.

        Args:
            parsed_email: Output from EmailParser.parse()

        Returns:
            Float 0.0–1.0 phishing probability.
        """
        subject = parsed_email.get("headers", {}).get("subject", "")
        body    = parsed_email.get("body",    {}).get("combined", "")
        text    = f"Subject: {subject}\n\n{body}"

        # FIX 2: was text[:512] which fed only ~100 tokens; now 2048 chars
        prob, _ = self.predict(text[:_MAX_CHAR_INPUT])
        return prob