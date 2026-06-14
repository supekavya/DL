import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Sentiment LSTM",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif;}
.badge{display:inline-block;padding:.3rem 1rem;border-radius:20px;font-size:.72rem;font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:.8rem;
  background:#1a1f40;color:#06b6d4;border:1px solid #0891b2;}
.theory-box{background:linear-gradient(135deg,#1a1f35,#1e2540);border-left:4px solid #0891b2;
  border-radius:0 10px 10px 0;padding:1rem 1.4rem;margin:.8rem 0;color:#cbd5e1;
  line-height:1.7;font-size:.9rem;}
.mrow{display:flex;gap:.8rem;margin:.8rem 0;}
.mchip{background:#1e2130;border:1px solid #3a3f5c;border-radius:10px;padding:.7rem 1rem;text-align:center;flex:1;}
.mval{font-family:'DM Mono',monospace;font-size:1.4rem;color:#06b6d4;}
.mlbl{font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.1em;}
.pred-box{background:linear-gradient(135deg,#1a1040,#1e1560);border:2px solid #0891b2;
  border-radius:16px;padding:1.5rem;text-align:center;}
.pred-val{font-family:'DM Mono',monospace;font-size:2.8rem;font-weight:700;color:#06b6d4;}
</style>
""",
    unsafe_allow_html=True,
)

DARK_FIG, DARK_AX, GRID, TEXT = "#0d1117", "#161b27", "#2a2f45", "#cbd5e1"


def dfig(w=10, h=5, nc=1, nr=1):
    fig, ax = plt.subplots(nr, nc, figsize=(w, h))
    fig.patch.set_facecolor(DARK_FIG)
    for a in np.array([ax]).flatten():
        a.set_facecolor(DARK_AX)
        for sp in a.spines.values():
            sp.set_edgecolor(GRID)
        a.tick_params(colors=TEXT, labelsize=9)
        a.xaxis.label.set_color(TEXT)
        a.yaxis.label.set_color(TEXT)
        a.title.set_color("#e2e8f0")
        a.grid(True, alpha=0.15, color=GRID)
    return fig, ax


def mchips(acc, prec, rec, f1):
    st.markdown(
        f"""<div class="mrow">
      <div class="mchip"><div class="mval">{acc:.3f}</div><div class="mlbl">Accuracy</div></div>
      <div class="mchip"><div class="mval">{prec:.3f}</div><div class="mlbl">Precision</div></div>
      <div class="mchip"><div class="mval">{rec:.3f}</div><div class="mlbl">Recall</div></div>
      <div class="mchip"><div class="mval">{f1:.3f}</div><div class="mlbl">F1-Score</div></div>
    </div>""",
        unsafe_allow_html=True,
    )


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/sentiment.csv")
    return df


df = load_data()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💬 Sentiment LSTM")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 EDA", "🏗️ Build & Train", "🔮 Live Prediction", "📐 NLP Concepts"],
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption(f"Synthetic Product Reviews")
    st.caption(f"Samples: **{df.shape[0]}**")
    st.caption(f"Classes: Positive / Negative")

# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 EDA":
    st.markdown(
        '<span class="badge">Exploratory Data Analysis</span>', unsafe_allow_html=True
    )
    st.title("Dataset Exploration")

    tab1, tab2 = st.tabs(["Overview", "Word Frequency"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Dataset Preview**")
            st.dataframe(df.head(10), use_container_width=True)
        with c2:
            st.markdown("**Class Distribution**")
            counts = df["label"].value_counts()
            fig, ax = dfig(8, 4)
            colors = ["#06b6d4", "#f87171"]
            ax.bar(
                counts.index,
                counts.values,
                color=colors,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.set_title("Sentiment Distribution", color="#e2e8f0")
            ax.set_ylabel("Count")
            for i, v in enumerate(counts.values):
                ax.text(
                    i, v + 10, str(v), ha="center", color="white", fontweight="bold"
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab2:
        st.markdown("**Top Words by Sentiment**")
        pos_words = Counter()
        neg_words = Counter()

        for _, row in df.iterrows():
            words = clean_text(row["text"]).split()
            if row["sentiment"] == 1:
                pos_words.update(words)
            else:
                neg_words.update(words)

        # Filter out common stop words
        stop_words = {
            "the",
            "is",
            "and",
            "to",
            "a",
            "of",
            "it",
            "this",
            "i",
            "my",
            "with",
            "for",
            "on",
            "in",
        }
        pos_top = {k: v for k, v in pos_words.most_common(15) if k not in stop_words}
        neg_top = {k: v for k, v in neg_words.most_common(15) if k not in stop_words}

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = dfig(10, 5)
            ax.barh(
                list(pos_top.keys())[::-1],
                list(pos_top.values())[::-1],
                color="#06b6d4",
                edgecolor="white",
            )
            ax.set_title("Top Positive Words", color="#e2e8f0")
            ax.set_facecolor(DARK_AX)
            ax.tick_params(colors=TEXT)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with c2:
            fig, ax = dfig(10, 5)
            ax.barh(
                list(neg_top.keys())[::-1],
                list(neg_top.values())[::-1],
                color="#f87171",
                edgecolor="white",
            )
            ax.set_title("Top Negative Words", color="#e2e8f0")
            ax.set_facecolor(DARK_AX)
            ax.tick_params(colors=TEXT)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# BUILD & TRAIN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Build & Train":
    st.markdown(
        '<span class="badge">Build & Train NLP Model</span>', unsafe_allow_html=True
    )
    st.title("Build & Train Sentiment LSTM")
    st.markdown(
        '<div class="theory-box">Configure the NLP pipeline. Text is tokenized, padded to a fixed length, and passed through an Embedding layer before the LSTM processes the sequential word relationships.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**NLP Pipeline**")
        vocab_size = st.select_slider(
            "Vocabulary Size", options=[1000, 2000, 5000, 10000], value=2000
        )
        max_len = st.select_slider(
            "Max Sequence Length", options=[10, 15, 20, 30, 50], value=20
        )
        st.markdown("**Architecture**")
        embed_dim = st.select_slider(
            "Embedding Dimension", options=[16, 32, 64, 128], value=32
        )
        lstm_units = st.select_slider("LSTM Units", options=[16, 32, 64, 128], value=32)
        dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
        st.markdown("**Training**")
        lr = st.select_slider(
            "Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005], value=0.001
        )
        batch_sz = st.select_slider("Batch Size", options=[16, 32, 64], value=32)
        epochs = st.slider("Max Epochs", 10, 100, 30)
        run_btn = st.button("🚀 Train Model", use_container_width=True)

    with c2:
        if run_btn:
            try:
                import tensorflow as tf
                from tensorflow import keras
                from tensorflow.keras import layers
                from tensorflow.keras.callbacks import EarlyStopping
                from tensorflow.keras.preprocessing.text import Tokenizer
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import (
                    precision_score,
                    recall_score,
                    f1_score,
                    confusion_matrix,
                )

                texts = df["text"].values
                labels = df["sentiment"].values

                X_train, X_test, y_train, y_test = train_test_split(
                    texts, labels, test_size=0.2, random_state=42, stratify=labels
                )

                tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
                tokenizer.fit_on_texts(X_train)

                X_train_seq = pad_sequences(
                    tokenizer.texts_to_sequences(X_train),
                    maxlen=max_len,
                    padding="post",
                    truncating="post",
                )
                X_test_seq = pad_sequences(
                    tokenizer.texts_to_sequences(X_test),
                    maxlen=max_len,
                    padding="post",
                    truncating="post",
                )

                model = keras.Sequential(
                    [
                        layers.Embedding(vocab_size, embed_dim, input_length=max_len),
                        layers.LSTM(lstm_units, return_sequences=False),
                        layers.Dropout(dropout),
                        layers.Dense(16, activation="relu"),
                        layers.Dropout(dropout),
                        layers.Dense(1, activation="sigmoid"),
                    ]
                )

                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr),
                    loss="binary_crossentropy",
                    metrics=["accuracy"],
                )

                cbs = [
                    EarlyStopping(
                        monitor="val_accuracy",
                        patience=5,
                        restore_best_weights=True,
                        mode="max",
                    )
                ]

                with st.spinner("Training…"):
                    hist = model.fit(
                        X_train_seq,
                        y_train,
                        validation_split=0.15,
                        epochs=epochs,
                        batch_size=batch_sz,
                        callbacks=cbs,
                        verbose=0,
                    )

                y_pred_prob = model.predict(X_test_seq, verbose=0).flatten()
                y_pred = (y_pred_prob >= 0.5).astype(int)

                acc = np.mean(y_pred == y_test)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                mchips(acc, prec, rec, f1)

                hdf = pd.DataFrame(hist.history)
                fig, axes = dfig(14, 5, 2)
                axes = np.array(axes).flatten()

                axes[0].plot(hdf["loss"], color="#f87171", lw=2, label="Train")
                axes[0].plot(hdf["val_loss"], color="#06b6d4", lw=2, label="Val")
                axes[0].set_title("Training Loss (Binary Crossentropy)")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Loss")
                axes[0].legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)

                axes[1].plot(hdf["accuracy"], color="#34d399", lw=2, label="Train")
                axes[1].plot(hdf["val_accuracy"], color="#fbbf24", lw=2, label="Val")
                axes[1].set_title("Training Accuracy")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Accuracy")
                axes[1].legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Confusion Matrix
                fig, ax = dfig(8, 6)
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=ax,
                    xticklabels=["Negative (0)", "Positive (1)"],
                    yticklabels=["Negative (0)", "Positive (1)"],
                    linewidths=1,
                    linecolor="white",
                    annot_kws={"color": "white", "size": 14},
                )
                ax.set_title("Confusion Matrix", color="#e2e8f0")
                ax.set_facecolor(DARK_AX)
                ax.tick_params(colors=TEXT)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Save to session state for prediction tab
                st.session_state["nlp_model"] = model
                st.session_state["nlp_tokenizer"] = tokenizer
                st.session_state["nlp_max_len"] = max_len
                st.success("Model trained and saved to session!")

            except ImportError:
                st.error("TensorFlow not installed. Run: `pip install tensorflow`")
        else:
            st.info(
                "👈 Configure the NLP pipeline on the left and click **Train Model**."
            )
            st.markdown("**Planned Architecture:**")
            arch_lines = ["```", f"Input  (Max Length: {max_len})"]
            arch_lines += [
                f"  Embedding(vocab_size={vocab_size}, dimension={embed_dim})"
            ]
            arch_lines += [f"  LSTM({lstm_units})"]
            arch_lines += [f"  Dropout({dropout})"]
            arch_lines += [
                "  Dense(16, relu)",
                f"  Dropout({dropout})",
                "  Dense(1, sigmoid)",
                "Output (Probability of Positive Sentiment)",
                "```",
            ]
            st.markdown("\n".join(arch_lines))

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Prediction":
    st.markdown(
        '<span class="badge">Live Sentiment Analysis</span>', unsafe_allow_html=True
    )
    st.title("Analyze Custom Text")
    st.markdown(
        '<div class="theory-box">Type or paste a review below. The trained LSTM will tokenize the text, pad it to the correct length, and predict the sentiment probability based on learned word embeddings and sequential context.</div>',
        unsafe_allow_html=True,
    )

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        # Train a default model if none exists in session
        if "nlp_model" not in st.session_state:
            with st.spinner("Training default model for prediction..."):
                texts = df["text"].values
                labels = df["sentiment"].values
                tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
                tokenizer.fit_on_texts(texts)
                seqs = pad_sequences(
                    tokenizer.texts_to_sequences(texts),
                    maxlen=20,
                    padding="post",
                    truncating="post",
                )

                model = keras.Sequential(
                    [
                        layers.Embedding(2000, 32, input_length=20),
                        layers.LSTM(32, return_sequences=False),
                        layers.Dropout(0.2),
                        layers.Dense(1, activation="sigmoid"),
                    ]
                )
                model.compile(
                    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
                )
                model.fit(
                    seqs,
                    labels,
                    epochs=20,
                    batch_size=32,
                    verbose=0,
                    validation_split=0.15,
                )

                st.session_state["nlp_model"] = model
                st.session_state["nlp_tokenizer"] = tokenizer
                st.session_state["nlp_max_len"] = 20

        model = st.session_state["nlp_model"]
        tokenizer = st.session_state["nlp_tokenizer"]
        max_len = st.session_state["nlp_max_len"]

        user_text = st.text_area(
            "Enter a review or sentence:",
            value="I absolutely love this product, it is amazing and works perfectly!",
            height=100,
        )

        if st.button("🔮 Analyze Sentiment", use_container_width=True):
            seq = pad_sequences(
                tokenizer.texts_to_sequences([user_text]),
                maxlen=max_len,
                padding="post",
                truncating="post",
            )
            prob = float(model.predict(seq, verbose=0).flatten()[0])
            pred_class = "Positive 😊" if prob >= 0.5 else "Negative 😞"
            color = "#06b6d4" if prob >= 0.5 else "#f87171"

            st.markdown(
                f"""
            <div class="pred-box">
              <div style="color:#94a3b8;font-size:.85rem;text-transform:uppercase;letter-spacing:.15em">Prediction</div>
              <div class="pred-val" style="color:{color}">{pred_class}</div>
              <div style="color:#94a3b8;margin-top:.5rem">Positive Probability: <b style="color:#06b6d4">{prob:.4f}</b></div>
            </div>""",
                unsafe_allow_html=True,
            )

            # Visualize word contributions (simplified: highlight known vs OOV)
            words = user_text.lower().split()
            known_words = [w for w in words if w in tokenizer.word_index]
            oov_words = [w for w in words if w not in tokenizer.word_index]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Known Vocabulary Words:** {len(known_words)}")
                st.caption(", ".join(known_words) if known_words else "None")
            with c2:
                st.markdown(f"**Out-of-Vocabulary (OOV):** {len(oov_words)}")
                st.caption(", ".join(oov_words) if oov_words else "None")

    except ImportError:
        st.error("TensorFlow required. Install with: `pip install tensorflow`")

# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE STUDY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📐 NLP Concepts":
    st.markdown(
        '<span class="badge">NLP Architecture Study</span>', unsafe_allow_html=True
    )
    st.title("Text Processing & LSTM Concepts")

    tab1, tab2 = st.tabs(["🔬 The NLP Pipeline", "📊 Embedding Space"])

    with tab1:
        st.markdown("**From Raw Text to Neural Network Input**")
        st.markdown(
            '<div class="theory-box">Neural networks cannot process raw text. We must convert words into numerical representations through a standardized pipeline.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("""
        | Step | Description | Example |
        |---|---|---|
        | **1. Tokenization** | Split text into words/tokens and assign a unique integer ID to each based on frequency. | "I love this" → `[1, 15, 4]` |
        | **2. Padding/Truncating** | Ensure all sequences have the exact same length (`max_len`) by adding zeros or cutting off excess. | `[1, 15, 4]` → `[1, 15, 4, 0, 0]` |
        | **3. Embedding Layer** | A trainable lookup table that maps each integer ID to a dense vector of floats (e.g., 32 dimensions). Captures semantic meaning. | `4` → `[0.2, -0.5, 0.8, ...]` |
        | **4. LSTM Processing** | Processes the sequence of embedding vectors, maintaining a hidden state that captures the contextual meaning of the sentence. | Outputs a single context vector. |
        | **5. Dense Output** | A final sigmoid neuron outputs a probability between 0 and 1 for binary classification. | `0.87` (87% Positive) |
        """)

    with tab2:
        st.markdown("**Why Embeddings Matter**")
        st.markdown(
            '<div class="theory-box">Unlike one-hot encoding (which treats every word as completely independent), an <b>Embedding layer</b> learns dense, continuous vector representations. Words with similar meanings (e.g., "great" and "excellent") will have similar vectors and cluster together in this high-dimensional space.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Visualizing a Simplified 2D Embedding Projection**")
        fig, ax = dfig(10, 6)

        # Mock embedding coordinates for demonstration
        pos_words = {
            "amazing": (0.8, 0.7),
            "excellent": (0.7, 0.8),
            "great": (0.6, 0.6),
            "fantastic": (0.9, 0.5),
        }
        neg_words = {
            "terrible": (-0.8, -0.7),
            "awful": (-0.7, -0.8),
            "bad": (-0.6, -0.6),
            "worst": (-0.9, -0.5),
        }
        neu_words = {"the": (0.1, 0.1), "is": (0.0, 0.2), "it": (-0.1, 0.0)}

        for word, (x, y) in pos_words.items():
            ax.scatter(x, y, color="#06b6d4", s=150, zorder=5)
            ax.text(
                x + 0.05,
                y + 0.05,
                word,
                color="#06b6d4",
                fontsize=10,
                fontweight="bold",
            )

        for word, (x, y) in neg_words.items():
            ax.scatter(x, y, color="#f87171", s=150, zorder=5)
            ax.text(
                x + 0.05,
                y + 0.05,
                word,
                color="#f87171",
                fontsize=10,
                fontweight="bold",
            )

        for word, (x, y) in neu_words.items():
            ax.scatter(x, y, color="#94a3b8", s=100, zorder=5)
            ax.text(x + 0.05, y + 0.05, word, color="#94a3b8", fontsize=10)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(0, color=GRID, lw=1)
        ax.axvline(0, color=GRID, lw=1)
        ax.set_title(
            "Conceptual 2D Word Embedding Space",
            color="#e2e8f0",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors=TEXT)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.markdown("---")
st.markdown(
    "<center style='color:#4a5568;font-size:.78rem'>Sentiment Analysis with LSTM &nbsp;|&nbsp; TensorFlow · Keras · Scikit-learn</center>",
    unsafe_allow_html=True,
)
