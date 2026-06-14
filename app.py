import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="BERT Explorer",
    page_icon="🤖",
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
  background:#1a1f40;color:#f97316;border:1px solid #ea580c;}
.theory-box{background:linear-gradient(135deg,#1a1f35,#1e2540);border-left:4px solid #ea580c;
  border-radius:0 10px 10px 0;padding:1rem 1.4rem;margin:.8rem 0;color:#cbd5e1;
  line-height:1.7;font-size:.9rem;}
.mrow{display:flex;gap:.8rem;margin:.8rem 0;}
.mchip{background:#1e2130;border:1px solid #3a3f5c;border-radius:10px;padding:.7rem 1rem;text-align:center;flex:1;}
.mval{font-family:'DM Mono',monospace;font-size:1.4rem;color:#f97316;}
.mlbl{font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.1em;}
.pred-box{background:linear-gradient(135deg,#1a1040,#1e1560);border:2px solid #ea580c;
  border-radius:16px;padding:1.5rem;text-align:center;}
.pred-val{font-family:'DM Mono',monospace;font-size:2.8rem;font-weight:700;color:#f97316;}
.token-box{display:inline-block;background:#1e293b;border:1px solid #f97316;border-radius:6px;
  padding:0.3rem 0.6rem;margin:0.2rem;font-family:'DM Mono',monospace;font-size:0.85rem;color:#fbbf24;}
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


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/reviews.csv")
    return df


df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 BERT Explorer")
    st.markdown("---")
    page = st.radio(
        "Navigate", ["📊 EDA", "🔮 Live Analysis", "📐 Transformer Concepts"]
    )
    st.markdown("---")
    st.markdown("**Model**")
    st.caption(f"RoBERTa-base (Latest)")
    st.caption(f"Fine-tuned on diverse real-world data")
    st.caption(f"Task: 3-Class Sentiment (Pos/Neg/Neu)")

# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 EDA":
    st.markdown(
        '<span class="badge">Exploratory Data Analysis</span>', unsafe_allow_html=True
    )
    st.title("Dataset Exploration")

    tab1, tab2 = st.tabs(["Overview", "Text Length Distribution"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Dataset Preview**")
            st.dataframe(df.head(10), use_container_width=True)
        with c2:
            st.markdown("**Class Distribution**")
            counts = df["label"].value_counts()
            fig, ax = dfig(8, 4)
            colors = ["#f97316", "#3b82f6"]
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
        st.markdown("**Sequence Length Distribution**")
        df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
        fig, ax = dfig(10, 5)
        sns.histplot(df["word_count"], bins=20, color="#f97316", kde=True, ax=ax)
        ax.set_title("Word Count per Review", color="#e2e8f0")
        ax.set_xlabel("Number of Words")
        ax.set_ylabel("Frequency")
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors=TEXT)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(
            '<div class="theory-box">BERT has a maximum input length of 512 tokens. As seen above, typical reviews easily fit within this limit. BERT uses <b>WordPiece</b> tokenization, so the number of tokens will be slightly higher than the word count.</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# LIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Analysis":
    st.markdown(
        '<span class="badge">Live BERT Inference</span>', unsafe_allow_html=True
    )
    st.title("Analyze Text with BERT")
    st.markdown(
        '<div class="theory-box">BERT understands context bidirectionally. Unlike LSTMs that read left-to-right (or right-to-left), BERT\'s <b>Self-Attention</b> mechanism looks at all words simultaneously to understand nuanced phrases like "not bad" or "not good".</div>',
        unsafe_allow_html=True,
    )

    try:
        from transformers import pipeline, AutoTokenizer

        @st.cache_resource
        def load_bert():
            # Using a robust, state-of-the-art RoBERTa model for highly accurate predictions
            # Explicitly use PyTorch to avoid Keras 3 / TensorFlow conflicts
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                framework="pt",
                return_all_scores=False,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return sentiment_pipeline, tokenizer

        with st.spinner("Loading BERT model (this may take a moment on first run)..."):
            classifier, tokenizer = load_bert()

        user_text = st.text_area(
            "Enter a review or sentence:",
            value="The product is not bad at all, I actually love it!",
            height=100,
        )

        if st.button("🤖 Analyze with BERT", use_container_width=True):
            # 1. Tokenization Visualization
            st.markdown("### 🔤 Step 1: WordPiece Tokenization")
            st.markdown(
                "BERT breaks text into subword tokens. Notice how unknown words or punctuation are handled:"
            )

            tokens = tokenizer.tokenize(user_text)
            # Add [CLS] and [SEP]
            full_tokens = ["[CLS]"] + tokens + ["[SEP]"]

            token_html = "".join(
                [f'<span class="token-box">{t}</span>' for t in full_tokens]
            )
            st.markdown(token_html, unsafe_allow_html=True)

            input_ids = tokenizer.encode(user_text, add_special_tokens=True)
            st.caption(f"Total Tokens: {len(input_ids)} (Max limit: 512)")

            # 2. Prediction
            st.markdown("### 🎯 Step 2: Model Prediction")
            result = classifier(user_text)[0]
            label = result["label"].lower()  # 'positive', 'negative', or 'neutral'
            score = result["score"]

            # Map to friendly display
            if label == "positive":
                pred_class = "Positive 😊"
                color = "#f97316"
            elif label == "negative":
                pred_class = "Negative 😞"
                color = "#3b82f6"
            else:
                pred_class = "Neutral 😐"
                color = "#94a3b8"

            st.markdown(
                f"""
            <div class="pred-box">
              <div style="color:#94a3b8;font-size:.85rem;text-transform:uppercase;letter-spacing:.15em">BERT Prediction</div>
              <div class="pred-val" style="color:{color}">{pred_class}</div>
              <div style="color:#94a3b8;margin-top:.5rem">Confidence Score: <b style="color:#fbbf24">{score:.4f}</b></div>
            </div>""",
                unsafe_allow_html=True,
            )

            # Confidence Bar
            fig, ax = dfig(10, 2)
            if label == "positive":
                ax.barh(["Confidence"], [score], color="#f97316", height=0.5)
                ax.barh(
                    ["Confidence"],
                    [1 - score],
                    left=[score],
                    color="#3b82f6",
                    height=0.5,
                )
            elif label == "negative":
                ax.barh(["Confidence"], [score], color="#3b82f6", height=0.5)
                ax.barh(
                    ["Confidence"],
                    [1 - score],
                    left=[score],
                    color="#f97316",
                    height=0.5,
                )
            else:
                ax.barh(["Confidence"], [score], color="#94a3b8", height=0.5)
                ax.barh(
                    ["Confidence"],
                    [1 - score],
                    left=[score],
                    color="#475569",
                    height=0.5,
                )

            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["0%", "50%", "100%"])
            ax.set_facecolor(DARK_AX)
            ax.tick_params(colors=TEXT)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # 3. Nuance Test
            st.markdown("### 🧠 Why this Model is Powerful: Contextual Nuance")
            st.markdown(
                "Try these examples to see how this robust model handles negation, sarcasm, and word order, which often confuse simpler models:"
            )

            examples = [
                "I actually love it!",
                "The movie was not good.",
                "The movie was good, not great.",
                "I love it, not bad at all.",
                "Not bad at all, I love it.",
            ]

            for ex in examples:
                res = classifier(ex)[0]
                ex_label = res["label"].upper()
                if ex_label == "POSITIVE":
                    ex_color = "#f97316"
                elif ex_label == "NEGATIVE":
                    ex_color = "#3b82f6"
                else:
                    ex_color = "#94a3b8"
                st.markdown(
                    f"- **'{ex}'** → <span style='color:{ex_color};font-weight:bold'>{ex_label} ({res['score']:.2f})</span>",
                    unsafe_allow_html=True,
                )

    except ImportError:
        st.error(
            "Transformers library not installed. Run: `pip install transformers torch`"
        )

# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE STUDY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Transformer Concepts":
    st.markdown(
        '<span class="badge">Transformer Architecture</span>', unsafe_allow_html=True
    )
    st.title("Inside the BERT Model")

    tab1, tab2 = st.tabs(["🔬 Self-Attention Mechanism", "📊 BERT vs LSTM"])

    with tab1:
        st.markdown("**The Core of Transformers: Multi-Head Self-Attention**")
        st.markdown(
            '<div class="theory-box">Instead of processing words sequentially, Self-Attention calculates a "relevance score" between <b>every pair of words</b> in the sentence simultaneously. This allows BERT to understand that "it" refers to "product" regardless of distance.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(r"""
        | Component | Purpose |
        |---|---|
        | **Query (Q)** | What I am looking for in other words. |
        | **Key (K)** | What I contain that other words might look for. |
        | **Value (V)** | The actual content/information of the word. |
        | **Attention Score** | $Softmax(\frac{Q \cdot K^T}{\sqrt{d_k}}) \cdot V$ |
        """)

        st.markdown("**Conceptual Attention Flow**")
        fig, ax = dfig(12, 6)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        words = ["The", "product", "is", "not", "bad", "at", "all"]
        x_pos = np.linspace(1, 9, len(words))

        # Draw words
        for i, word in enumerate(words):
            ax.text(
                x_pos[i],
                8,
                word,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="#e2e8f0",
                bbox=dict(
                    facecolor="#1e293b", edgecolor="#f97316", boxstyle="round,pad=0.3"
                ),
            )

        # Draw attention lines (simplified)
        # "not" attends strongly to "bad"
        ax.plot([x_pos[3], x_pos[4]], [7, 7], color="#fbbf24", lw=3, alpha=0.8)
        ax.annotate(
            "",
            xy=(x_pos[4], 6.8),
            xytext=(x_pos[3], 6.8),
            arrowprops=dict(arrowstyle="<->", color="#fbbf24", lw=2),
        )
        ax.text(
            (x_pos[3] + x_pos[4]) / 2,
            7.3,
            "Strong Attention",
            ha="center",
            color="#fbbf24",
            fontsize=9,
            fontweight="bold",
        )

        # "it" (if present) would attend to "product"
        ax.plot(
            [x_pos[1], x_pos[1]],
            [6, 5],
            color="#3b82f6",
            lw=2,
            alpha=0.6,
            linestyle="--",
        )

        ax.set_title(
            "Self-Attention: Words evaluating each other simultaneously",
            color="#e2e8f0",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown("**Why BERT Replaced LSTMs for Most NLP Tasks**")
        st.markdown(
            '<div class="theory-box">While LSTMs process data step-by-step (making them slow to train and prone to forgetting very long-range dependencies), Transformers process the entire sequence in parallel and explicitly model relationships between all words.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("""
        | Feature | LSTM / BiLSTM | BERT (Transformer) |
        |---|---|---|
        | **Processing** | Sequential (step-by-step) | **Parallel** (all tokens at once) |
        | **Context** | Forward, Backward, or Concatenated | **Truly Bidirectional** (Self-Attention) |
        | **Long-range Dependencies** | Degrades over long distances | **Excellent** (direct connections between any two words) |
        | **Training Speed** | Slow (cannot parallelize well) | **Fast** (highly parallelizable on GPUs) |
        | **Pre-training** | Limited (e.g., word2vec) | **Massive** (Masked Language Modeling on billions of words) |
        | **Best For** | Simple sequences, real-time streaming | **State-of-the-art NLP** (Classification, QA, NER) |
        """)

        st.markdown("### 📝 Special Tokens in BERT")
        st.markdown("""
        - **`[CLS]`**: Added at the start of every sequence. The final hidden state of this token is used as the aggregate sequence representation for classification tasks.
        - **`[SEP]`**: Separates sentences in a pair (e.g., for Next Sentence Prediction) or marks the end of a single sentence.
        - **`[MASK]`**: Used during pre-training to hide random words, forcing the model to predict them based on surrounding context (Masked Language Modeling).
        """)

st.markdown("---")
st.markdown(
    "<center style='color:#4a5568;font-size:.78rem'>BERT (Bidirectional Encoder Representations from Transformers) Explorer &nbsp;|&nbsp; Hugging Face · PyTorch</center>",
    unsafe_allow_html=True,
)
