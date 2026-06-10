# 🌐 English → French Machine Translation — Transformer (PyTorch)

A **Seq2Seq Transformer** built entirely from scratch in PyTorch for English to French translation.

## 📁 Folder Structure

```
transformer_translation_project/
├── data/
│   └── en_fr_pairs.csv          ← 130 curated EN-FR sentence pairs
├── models/
│   ├── transformer_model.pth    ← trained model state dict
│   ├── src_vocab.pkl            ← English vocabulary
│   ├── tgt_vocab.pkl            ← French vocabulary
│   └── model_meta.pkl           ← BLEU, loss, params
├── notebooks/
│   └── eda.ipynb                ← EDA + model training + evaluation
├── app.py                       ← Streamlit frontend
├── requirements.txt
├── runtime.txt                  ← python-3.11
├── .gitignore
└── README.md
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# 1. Run notebook to train & save model
# Open notebooks/eda.ipynb and run all cells

# 2. Launch app
streamlit run app.py
```

> **Deployment:** `app.py` auto-trains on first launch if `models/` is absent.

## 🧠 Model Architecture

```
Input (English tokens)
    ↓
Token Embedding + Sinusoidal Positional Encoding
    ↓
Encoder: 3× Transformer Layers (d_model=128, heads=4, FFN=256)
    ↓
Decoder: 3× Transformer Layers + Causal Mask
    ↓
Linear(128 → fr_vocab_size) → softmax
    ↓
French tokens (greedy decoding)
```

## 📊 Key Features

- Scratch-built Transformer (no HuggingFace)
- Sinusoidal positional encoding
- Multi-head self-attention + cross-attention
- Label smoothing (0.1) for better generalisation
- Greedy decoding with EOS stopping
- BLEU-2 evaluation
- 130 curated EN→FR sentence pairs
