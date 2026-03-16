# modules/lgbm_classifier.py  — v5.0
"""
CHANGES FROM v4.0:

  ENSEMBLE INFERENCE
  ==================
  v5.0 uses two LightGBM models trained separately:
    model_full       (38 features, structural + NLP)
    model_structural (37 features, NO nlp_phishing_prob)

  The final phishing score is a weighted blend:
    nlp_weight    = normalize(squished_nlp_prob) from 0.0 to 1.0
    final_score   = nlp_weight * prob_full + (1 - nlp_weight) * prob_structural

  This means:
    - When NLP says "clean text" (weight≈0) → structural model decides alone
    - When NLP says "phishing text" (weight≈1) → full model decides alone
    - In between → proportional blend

  The invoice email case:
    nlp_prob = 0.0001 → squished ≈ 0.275 → weight = 0.0
    structural model sees: unknown domain + no auth + suspicious path + action content
    structural model scores: ~0.75-0.90 phishing → CAUGHT ✓

  BACKWARD COMPATIBILITY
  ======================
  If pkl contains only "model" (v4.0 or older), falls back to single-model
  inference automatically. No breaking changes for existing deployments.
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# NLP squish range — must match train_lgbm.py v5.0
_NLP_SQUISH_MIN = 0.275
_NLP_SQUISH_MAX = 0.725

# Default model path
_PROJECT_ROOT   = Path(__file__).resolve().parent.parent
_DEFAULT_PATH   = _PROJECT_ROOT / "data" / "models" / "lgbm_model.pkl"

# Feature name lists — kept here for reference and validation
FEATURE_NAMES_V5 = [
    "sender_in_tranco", "sender_is_threat",
    "sender_domain_age_score", "sender_is_newly_registered",
    "ssl_cert_age_days",
    "any_url_domain_is_threat", "any_url_domain_not_in_tranco",
    "min_url_tranco_rank", "any_url_newly_registered",
    "url_count", "has_ip_based_url", "has_url_shortener", "has_http_only",
    "max_url_length", "url_has_at_symbol", "url_entropy_score",
    "suspicious_path_keyword_count", "path_brand_mismatch", "redirect_depth",
    "brand_in_subdomain", "compound_brand_domain",
    "homograph_attack", "sender_display_mismatch",
    "reply_to_differs", "spf_fail", "dkim_fail", "dmarc_fail",
    "auth_completely_absent",
    "urgency_phrase_count", "subject_has_urgency",
    "impersonation_score", "unknown_domain_action_content",
    "has_dangerous_attachment", "suspicious_attachment_name",
    "has_form_in_html", "link_text_domain_mismatch",
    "html_to_text_ratio",
    "nlp_phishing_prob",  # last — full model only
]

FEATURE_NAMES_STRUCTURAL = FEATURE_NAMES_V5[:-1]  # 37, no NLP


class LGBMClassifier:
    """
    Phishing classifier using LightGBM.

    v5.0: Dual-model NLP-adaptive ensemble.
    v4.0: Single model with adversarial training (still supported via fallback).

    Usage:
        clf = LGBMClassifier()
        clf.load()
        prob = clf.predict(features, nlp_prob)
        label = "PHISHING" if prob >= 0.5 else "LEGITIMATE"
    """

    def __init__(self, model_path: str = None):
        self.model_path       = Path(model_path) if model_path else _DEFAULT_PATH
        self.model_full       = None
        self.model_structural = None
        self.feature_names    = None
        self.feature_names_struct = None
        self.version          = None
        self.nlp_squish_range = (_NLP_SQUISH_MIN, _NLP_SQUISH_MAX)
        self.threshold        = 0.5   # updated from pkl at load() time
        self._is_ensemble     = False
        self._is_loaded       = False

    # ── Load ─────────────────────────────────────────────────────────────── #

    def load(self, path: str = None) -> None:
        load_path = Path(path) if path else self.model_path

        if not load_path.exists():
            raise FileNotFoundError(
                f"LightGBM model not found: {load_path}\n"
                f"Run: python training/train_lgbm.py"
            )

        with open(load_path, "rb") as f:
            artifact = pickle.load(f)

        self.version = artifact.get("version", "unknown")

        if self.version == "5.0" and "model_structural" in artifact:
            # ── v5.0 dual ensemble ───────────────────────────────────────
            self.model_full           = artifact["model_full"]
            self.model_structural     = artifact["model_structural"]
            self.feature_names        = artifact.get("feature_names_full",   FEATURE_NAMES_V5)
            self.feature_names_struct = artifact.get("feature_names_struct", FEATURE_NAMES_STRUCTURAL)
            self.nlp_squish_range     = artifact.get("nlp_squish_range",     (_NLP_SQUISH_MIN, _NLP_SQUISH_MAX))
            self.threshold            = artifact.get("optimal_threshold",    0.5)
            self._is_ensemble         = True

            logger.info(
                f"[LGBMClassifier] Loaded v5.0 dual ensemble | "
                f"threshold={self.threshold:.2f} | "
                f"full({len(self.feature_names)}) + structural({len(self.feature_names_struct)})"
            )

        elif "model" in artifact:
            # ── v4.0 or older single model (backward compat) ────────────
            self.model_full           = artifact["model"]
            self.feature_names        = artifact.get("feature_names", FEATURE_NAMES_V5)
            self.feature_names_struct = None
            self._is_ensemble         = False

            logger.warning(
                f"[LGBMClassifier] Loaded v{self.version} single model. "
                f"Upgrade: run python training/train_lgbm.py for v5.0 ensemble."
            )

        else:
            raise ValueError(
                f"Unrecognized model format in {load_path}. "
                f"Expected keys: 'model_full'/'model_structural' (v5.0) or 'model' (v4.0)"
            )

        self._is_loaded = True
        print(f"[LGBMClassifier] v{self.version} loaded | "
              f"{'dual ensemble' if self._is_ensemble else 'single model'} | "
              f"{len(self.feature_names)} features | "
              f"threshold={self.threshold:.2f}")

    # ── Predict ──────────────────────────────────────────────────────────── #

    def predict(
        self,
        features: Dict[str, float],
        nlp_prob: float,
    ) -> Tuple[float, str]:
        """
        Predict phishing probability for one email.

        Args:
            features:  dict of 37 structural features from FeatureExtractor.extract()
            nlp_prob:  raw NLP probability from NLPModel (0.0 to 1.0)

        Returns:
            (probability, label) where label is "PHISHING" or "LEGITIMATE"
        """
        if not self._is_loaded:
            raise RuntimeError("Call load() before predict()")

        prob = self._predict_prob(features, nlp_prob)
        label = "PHISHING" if prob >= self.threshold else "LEGITIMATE"
        return float(prob), label

    def predict_proba(
        self,
        features: Dict[str, float],
        nlp_prob: float,
    ) -> float:
        """Return phishing probability only (convenience wrapper)."""
        if not self._is_loaded:
            raise RuntimeError("Call load() before predict_proba()")
        return self._predict_prob(features, nlp_prob)

    def predict_batch(
        self,
        feature_list: List[Dict[str, float]],
        nlp_prob_list: List[float],
    ) -> List[Tuple[float, str]]:
        """
        Predict for a batch of emails.

        Args:
            feature_list:  list of feature dicts
            nlp_prob_list: list of NLP probabilities

        Returns:
            list of (probability, label) tuples
        """
        if not self._is_loaded:
            raise RuntimeError("Call load() before predict_batch()")

        results = []
        for features, nlp_prob in zip(feature_list, nlp_prob_list):
            prob  = self._predict_prob(features, nlp_prob)
            label = "PHISHING" if prob >= self.threshold else "LEGITIMATE"
            results.append((float(prob), label))
        return results

    # ── Internal ─────────────────────────────────────────────────────────── #

    def _predict_prob(
        self,
        features: Dict[str, float],
        nlp_prob: float,
    ) -> float:
        """Core prediction — handles both ensemble (v5.0) and single model (v4.0)."""

        if self._is_ensemble:
            return self._ensemble_predict(features, nlp_prob)
        else:
            return self._single_model_predict(features, nlp_prob)

    def _ensemble_predict(
        self,
        features: Dict[str, float],
        nlp_prob: float,
    ) -> float:
        """
        v5.0 ensemble inference:
          1. Squish NLP probability to [0.275, 0.725]
          2. Compute nlp_weight from squished NLP position in range
          3. Get prob_full from model_full (38 features)
          4. Get prob_structural from model_structural (37 features, no NLP)
          5. final = nlp_weight * prob_full + (1 - nlp_weight) * prob_structural

        When nlp_prob ≈ 0 (clean-text phishing):
          squished ≈ 0.275 → weight = 0.0 → 100% structural model
          The structural model was trained on synthetic samples specifically
          designed to catch this pattern.

        When nlp_prob ≈ 1 (obviously phishing text):
          squished ≈ 0.725 → weight = 1.0 → 100% full model
        """

        # Step 1: Squish NLP
        squished_nlp = self._squish_nlp(nlp_prob)

        # Step 2: NLP weight in [0.0, 1.0]
        nlp_min, nlp_max = self.nlp_squish_range
        nlp_weight = (squished_nlp - nlp_min) / (nlp_max - nlp_min)
        nlp_weight = float(np.clip(nlp_weight, 0.0, 1.0))

        # Step 3: Build structural feature vector (37 features)
        x_structural = np.array(
            [features.get(name, 0.0) for name in self.feature_names_struct],
            dtype=np.float32,
        ).reshape(1, -1)

        # Step 4: Build full feature vector (38 features — structural + squished NLP)
        x_full = np.array(
            [features.get(name, 0.0) for name in self.feature_names[:-1]] + [squished_nlp],
            dtype=np.float32,
        ).reshape(1, -1)

        # Step 5: Get probabilities from each model
        prob_full       = float(self.model_full.predict_proba(x_full)[:, 1][0])
        prob_structural = float(self.model_structural.predict_proba(x_structural)[:, 1][0])

        # Step 6: Weighted blend
        final_prob = nlp_weight * prob_full + (1.0 - nlp_weight) * prob_structural

        logger.debug(
            f"Ensemble: nlp_raw={nlp_prob:.4f} squished={squished_nlp:.4f} "
            f"weight={nlp_weight:.3f} | "
            f"prob_full={prob_full:.4f} prob_struct={prob_structural:.4f} "
            f"→ final={final_prob:.4f}"
        )

        return float(final_prob)

    def _single_model_predict(
        self,
        features: Dict[str, float],
        nlp_prob: float,
    ) -> float:
        """v4.0 backward-compatible single-model prediction."""
        capped_nlp = float(np.clip(nlp_prob, 0.05, 0.95))
        x = np.array(
            [features.get(name, 0.0) for name in self.feature_names[:-1]] + [capped_nlp],
            dtype=np.float32,
        ).reshape(1, -1)
        return float(self.model_full.predict_proba(x)[:, 1][0])

    @staticmethod
    def _squish_nlp(nlp_prob: float) -> float:
        """Map [0.05, 0.95] → [0.275, 0.725] via linear compression toward 0.5."""
        clipped = float(np.clip(nlp_prob, 0.05, 0.95))
        return 0.5 + (clipped - 0.5) * 0.5

    # ── Diagnostics ──────────────────────────────────────────────────────── #

    def explain(
        self,
        features: Dict[str, float],
        nlp_prob: float,
    ) -> Dict:
        """
        Return a detailed breakdown of the prediction for debugging.
        Shows both model scores and the ensemble weight.

        Returns dict with:
          final_prob, label, nlp_weight,
          prob_full, prob_structural (if ensemble),
          top_structural_signals (features with non-zero values)
        """
        if not self._is_loaded:
            raise RuntimeError("Call load() before explain()")

        squished_nlp = self._squish_nlp(nlp_prob)
        nlp_min, nlp_max = self.nlp_squish_range
        nlp_weight = float(np.clip(
            (squished_nlp - nlp_min) / (nlp_max - nlp_min), 0.0, 1.0
        ))

        result = {
            "nlp_prob_raw":      round(nlp_prob, 4),
            "nlp_prob_squished": round(squished_nlp, 4),
            "nlp_weight":        round(nlp_weight, 3),
            "model_mode":        "ensemble" if self._is_ensemble else "single",
        }

        if self._is_ensemble:
            x_structural = np.array(
                [features.get(name, 0.0) for name in self.feature_names_struct],
                dtype=np.float32,
            ).reshape(1, -1)
            x_full = np.array(
                [features.get(name, 0.0) for name in self.feature_names[:-1]] + [squished_nlp],
                dtype=np.float32,
            ).reshape(1, -1)

            prob_full   = float(self.model_full.predict_proba(x_full)[:, 1][0])
            prob_struct = float(self.model_structural.predict_proba(x_structural)[:, 1][0])
            final_prob  = nlp_weight * prob_full + (1.0 - nlp_weight) * prob_struct

            result.update({
                "prob_full":       round(prob_full, 4),
                "prob_structural": round(prob_struct, 4),
                "final_prob":      round(final_prob, 4),
                "threshold":       self.threshold,
                "label":           "PHISHING" if final_prob >= self.threshold else "LEGITIMATE",
            })
        else:
            final_prob = self._single_model_predict(features, nlp_prob)
            result.update({
                "final_prob": round(final_prob, 4),
                "threshold":  self.threshold,
                "label":      "PHISHING" if final_prob >= self.threshold else "LEGITIMATE",
            })

        # Top non-zero structural signals
        result["active_signals"] = {
            k: v for k, v in features.items() if v != 0
        }

        return result

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def is_ensemble(self) -> bool:
        return self._is_ensemble