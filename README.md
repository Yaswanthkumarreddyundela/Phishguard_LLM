# PhishGuard AI 🛡️

**A real-time, explainable phishing detection system that runs entirely on your local machine — no cloud, no GPU required.**

PhishGuard AI monitors your inbox continuously, classifies every incoming email before you open it, and explains exactly why a message was flagged — in plain language.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Authors](#authors)
- [License](#license)

---

## Overview

Email phishing causes billions in annual losses and drives over 80% of data breaches worldwide. Existing defenses all have the same blind spot: **the window between message delivery and the moment you open it.**

PhishGuard AI closes that window. It runs as a background process on your own machine, scanning each arriving message through a multi-stage ML pipeline before your inbox renders. When a threat is detected, an on-device language model writes a short explanation naming the specific warning signs — no binary alerts, no mystery scores.

**Three things no prior system did simultaneously:**
- Fully local execution — nothing leaves your device
- Detection coverage including link-free phishing attacks
- Natural-language explanations accessible to non-specialist users

---

## Key Results

Evaluated on a held-out test set of **10,000 emails**:

| Metric | Value | 95% CI |
|---|---|---|
| Overall Accuracy | **96.8%** | [96.2%, 97.4%] |
| F1-Score (Phishing) | **94.9%** | [94.2%, 95.7%] |
| False Positive Rate | **4.8%** | [4.2%, 5.4%] |
| False Negative Rate | **5.3%** | [4.7%, 5.9%] |
| AUC-ROC | **0.980** | [0.977, 0.983] |
| Matthews Corr. Coeff. | **0.930** | [0.925, 0.935] |
| Suspicious Detection | **89.3%** | [87.9%, 90.7%] |
| Sync Classification Latency | **~230 ms** | — |
| Memory Footprint | **~310 MB** | — |

**Baseline comparison:**

| System | Accuracy | FNR | Local? | Explainable? |
|---|---|---|---|---|
| **PhishGuard AI (ours)** | **96.8%** | **5.3%** | ✅ | ✅ |
| Chiew et al. [24] hybrid | 94.8% | 7.4% | ❌ | ❌ |
| DistilBERT standalone | 94.3% | 6.9% | ✅ | ❌ |
| LightGBM (structural) | 93.1% | 8.7% | ✅ | Partial |
| Random Forest | 92.4% | 10.2% | ✅ | Partial |
| SVM (RBF) | 89.7% | 14.1% | ✅ | ❌ |
| Naive Bayes | 87.2% | 18.3% | ✅ | ❌ |

All improvements statistically significant via McNemar paired tests (p < 0.001 vs all baselines).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PhishGuard AI Pipeline                  │
│                  (fully client-side, zero egress)           │
└─────────────────────────────────────────────────────────────┘

  IMAP / Gmail API
        │
        ▼
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│    Email     │───▶│    Feature       │───▶│   DistilBERT    │
│   Fetcher    │    │    Extractor     │    │   NLP Scorer    │
└──────────────┘    │  (37 structural  │    │  (feature #38)  │
                    │    features)     │    └────────┬────────┘
                    └──────────────────┘             │
                                                     ▼
                                            ┌─────────────────┐
                                            │    LightGBM     │
                                            │   Classifier    │
                                            │ Safe/Susp/Phish │
                                            └────────┬────────┘
                                                     │
                              ┌──────────────────────┼────────────────────┐
                              │                      │                    │
                              ▼                      ▼                    ▼
                    ┌──────────────────┐   ┌──────────────┐   ┌──────────────────┐
                    │  Quarantine +    │   │  Phi-2 LLM   │   │  SQLite Audit    │
                    │  Notification    │   │  Explainer   │   │  Log + Feedback  │
                    │  (sync, ~230ms)  │   │  (async,~1s) │   │                  │
                    └──────────────────┘   └──────────────┘   └──────────────────┘
```

**Six modules, all local:**

1. **Email Fetcher** — IMAP / Gmail API (read-only, OAuth 2.0)
2. **Feature Extractor** — 38-element vector (headers, URLs, content, NLP)
3. **DistilBERT NLP Scorer** — fine-tuned phishing likelihood score
4. **LightGBM Classifier** — 3-class output: Safe / Suspicious / Phishing
5. **Phi-2 Explainer** — on-device natural-language explanation (via Ollama)
6. **Response + Logger** — quarantine, notifications, SQLite audit log

---

## Features

- 🔒 **Fully local** — no email content ever leaves your device
- ⚡ **Fast** — synchronous classification in ~230 ms
- 🧠 **Hybrid ML** — structural features + NLP + LLM explanation
- 📊 **Three-class output** — Safe / Suspicious / Phishing (not just binary)
- 💬 **Plain-language alerts** — Phi-2 explains *why* a message was flagged
- 🔄 **Adaptive** — user corrections feed back into periodic retraining
- 🖥️ **Desktop notifications** — Plyer / ToastNotifier integration
- 📁 **Auto-quarantine** — flagged messages moved to a dedicated folder
- 🔍 **Link-free detection** — catches phishing with no URL using NLP alone
- 📝 **Audit log** — full SQLite history of every classification

---

## Tech Stack

| Component | Technology |
|---|---|
| Core Language | Python 3.10+ |
| ML Classifier | LightGBM + Scikit-learn |
| NLP Model | DistilBERT (fine-tuned, Hugging Face) |
| LLM Explainer | Phi-2 via Ollama (2.7B parameters) |
| Email Access | IMAPClient, Gmail API |
| HTML Parsing | BeautifulSoup4 |
| URL Analysis | tldextract, python-whois |
| Scheduling | APScheduler |
| Database | SQLite |
| Notifications | Plyer, ToastNotifier |
| Packaging | PyInstaller |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.ai) installed and running locally
- Gmail API credentials (optional — IMAP also supported)

### 1. Clone the repository

```bash
git clone https://github.com/Yaswanthkumarreddyundela/Phishguard_LLM.git
cd Phishguard_LLM
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the Phi-2 model via Ollama

```bash
ollama pull phi
```

### 5. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials (see [Configuration](#configuration)).

### 6. Run the monitor

```bash
python monitor.py
```

---

## Configuration

Copy `.env.example` to `.env` and fill in:

```env
# Email account to monitor
EMAIL_ADDRESS=your@email.com
EMAIL_PASSWORD=your_app_password

# IMAP settings (leave blank to use Gmail API instead)
IMAP_SERVER=imap.gmail.com
IMAP_PORT=993

# Alert notifications — send alerts FROM this address
ALERT_SENDER_EMAIL=phishguardai@gmail.com
ALERT_SENDER_PASSWORD=your_alert_app_password

# Alert notifications — send alerts TO this address
ALERT_RECIPIENT_EMAIL=your@email.com

# Anthropic API key (optional — for Claude-powered explanations)
ANTHROPIC_API_KEY=your_key_here

# Polling interval in seconds (default: 60)
POLL_INTERVAL=60
```

**Gmail users:** You need an [App Password](https://support.google.com/accounts/answer/185833), not your regular password.

---

## Usage

### Start monitoring

```bash
python monitor.py
```

The monitor will:
1. Connect to your inbox
2. Classify every new email
3. Quarantine phishing detections automatically
4. Send a desktop notification with the AI-generated explanation
5. Log all results to `data/phishguard.db`

### View detection history

```bash
python dashboard.py
```

### Retrain the model with your corrections

```bash
python training/train_lgbm.py
```

### Run the test suite

```bash
pytest tests/
```

---

## Dataset

We trained on approximately **130,000 labeled emails** from seven publicly available sources:

| Source | Class | Volume | Period |
|---|---|---|---|
| Nazario Phishing Corpus | Phishing | ~3,661 | 2020–2025 |
| APWG Public Repository | Phishing | ~15,000 | 2018–2023 |
| Academic Datasets (Berkeley / MIT) | Phishing | ~20,000 | Mixed |
| TREC 2007 Spam Track | Phishing | ~40,842 | 2007 |
| SpamAssassin Public Corpus | Phishing | ~497 | 2002–2006 |
| Enron Email Dataset (anonymized) | Legitimate | ~60,000 | 1998–2002 |
| Synthetic Business Email | Legitimate | ~10,000 | 2023 |

- Two independent reviewers labeled all samples (κ = 0.91)
- 60 / 40 legitimate-to-phishing split
- Temporal ordering preserved across splits to prevent data leakage

---

## Model Training

### Feature extraction (38 features)

| Group | Count | Examples |
|---|---|---|
| Header auth | 8 | SPF/DKIM/DMARC outcomes, Reply-To mismatch, routing anomalies |
| Body content | 7 | TF-IDF urgency score, HTML complexity, image count |
| URL analysis | 10 | Domain age, shortener detection, TLD risk, threat feeds |
| NLP (DistilBERT) | 1 | Fine-tuned phishing likelihood score |

### LightGBM hyperparameters

| Parameter | Value |
|---|---|
| n_estimators | 200 |
| learning_rate | 0.05 |
| max_depth | 7 |
| num_leaves | 31 |
| feature_fraction | 0.8 |
| bagging_fraction | 0.8 |
| class_weight | balanced |
| objective | multiclass |
| early_stopping | 50 rounds |

### Retrain from scratch

```bash
# Build the dataset
python training/build_dataset.py

# Train LightGBM
python training/train_lgbm.py

# Fine-tune DistilBERT (optional — pre-trained weights included)
python training/train_distilbert.py
```

---

## Project Structure

```
phishguard-ai/
├── monitor.py                  # Main monitoring loop
├── config/
│   └── settings.py             # Configuration loader
├── modules/
│   ├── email_fetcher.py        # IMAP / Gmail API connector
│   ├── email_parser.py         # Header + body + URL extraction
│   ├── feature_extractor.py    # 38-feature pipeline
│   ├── nlp_model.py            # DistilBERT inference
│   ├── lgbm_classifier.py      # LightGBM classifier
│   ├── explainer.py            # Phi-2 / Claude explanation generator
│   ├── notifier.py             # Desktop + email alerts
│   ├── database.py             # SQLite audit log
│   └── domain_intelligence.py  # Threat feed + WHOIS lookups
├── training/
│   ├── train_lgbm.py           # LightGBM training script
│   ├── train_distilbert.py     # DistilBERT fine-tuning script
│   └── build_dataset.py        # Dataset assembly pipeline
├── data/
│   ├── models/                 # Saved model weights
│   └── phishguard.db           # SQLite detection log
├── tests/
│   ├── test_full_pipeline.py
│   ├── test_adversarial.py
│   └── test_phase2.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## Known Limitations

| Limitation | Impact | Planned Fix |
|---|---|---|
| No attachment analysis | PDF/Office macro payloads undetected | Sandbox integration (v2) |
| Image-only phishing | Near-empty text degrades NLP | OCR integration (v2) |
| English only | Non-English campaigns unreliable | Multilingual DistilBERT (v2) |
| Manual retraining | Concept drift requires user corrections | Active learning (v2) |
| WHOIS latency | Up to 3s per lookup, occasional failures | Cached threat intel (ongoing) |

---

## Roadmap

- [ ] PDF and Office macro sandbox analysis
- [ ] OCR for image-only phishing
- [ ] Multilingual DistilBERT variants
- [ ] Entropy-based active learning for automatic drift correction
- [ ] Enterprise mode — SIEM integration, STIX/TAXII audit export
- [ ] Gmail push API + IMAP IDLE (replace polling, eliminate detection delay)

---

## Authors

**Lovely Professional University — Computer Science and Engineering**

| Name | Email |
|---|---|
| Undela Yaswanth Kumar Reddy | undelayaswanthreddy143@gmail.com |
| Maddineni Raviteja | ravi9347783160@gmail.com |
| Minnekanti Durga Varun | varundurga80@gmail.com |
| G. Tharun Venkata Sai Kumar | tsai3822@gmail.com |
| Bokku Venkat Nanda Kishore | bvnkishore2004@gmail.com |

**Supervisor:** Hariom, Assistant Professor — hariomsoniiimtgn@gmail.com

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

> **Note:** PhishGuard AI is a research prototype. Use it as a supplementary layer alongside your existing email security, not as a replacement.
