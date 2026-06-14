import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="GPT Explorer",
    page_icon="✨",
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
  background:#1a1f40;color:#10b981;border:1px solid #059669;}
.theory-box{background:linear-gradient(135deg,#1a1f35,#1e2540);border-left:4px solid #059669;
  border-radius:0 10px 10px 0;padding:1rem 1.4rem;margin:.8rem 0;color:#cbd5e1;
  line-height:1.7;font-size:.9rem;}
.mrow{display:flex;gap:.8rem;margin:.8rem 0;}
.mchip{background:#1e2130;border:1px solid #3a3f5c;border-radius:10px;padding:.7rem 1rem;text-align:center;flex:1;}
.mval{font-family:'DM Mono',monospace;font-size:1.4rem;color:#10b981;}
.mlbl{font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.1em;}
.gen-box{background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #10b981;
  border-radius:12px;padding:1.5rem;margin-top:1rem;}
.gen-text{font-family:'DM Mono',monospace;font-size:1.1rem;color:#e2e8f0;line-height:1.8;}
.prompt-text{color:#94a3b8;font-style:italic;}
.continuation-text{color:#10b981;font-weight:600;}
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
    df = pd.read_csv("data/stories.csv")
    return df


df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✨ GPT Explorer")
    st.markdown("---")
    page = st.radio(
        "Navigate", ["📊 Dataset", "✨ Text Generation", "📐 Transformer Concepts"]
    )
    st.markdown("---")
    st.markdown("**Model**")
    st.caption(f"DistilGPT-2")
    st.caption(f"Decoder-only Transformer")
    st.caption(f"Task: Causal Language Modeling")

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dataset":
    st.markdown('<span class="badge">Training Data</span>', unsafe_allow_html=True)
    st.title("Story Continuation Dataset")

    tab1, tab2 = st.tabs(["Overview", "Length Distribution"])

    with tab1:
        st.markdown("**Sample Prompts and Continuations**")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown(
            '<div class="theory-box">This dataset consists of creative writing prompts and their corresponding continuations. A GPT model is trained to take the <b>prompt</b> and autoregressively generate the <b>continuation</b> token by token.</div>',
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown("**Token/Word Length Distribution**")
        df["prompt_len"] = df["prompt"].apply(lambda x: len(str(x).split()))
        df["cont_len"] = df["continuation"].apply(lambda x: len(str(x).split()))

        fig, ax = dfig(12, 5)
        ax.hist(
            df["prompt_len"],
            bins=15,
            alpha=0.7,
            color="#94a3b8",
            label="Prompt Length",
            edgecolor="white",
        )
        ax.hist(
            df["cont_len"],
            bins=15,
            alpha=0.7,
            color="#10b981",
            label="Continuation Length",
            edgecolor="white",
        )
        ax.set_title("Word Count Distribution", color="#e2e8f0")
        ax.set_xlabel("Number of Words")
        ax.set_ylabel("Frequency")
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors=TEXT)
        ax.legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TEXT GENERATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✨ Text Generation":
    st.markdown(
        '<span class="badge">Autoregressive Generation</span>', unsafe_allow_html=True
    )
    st.title("Generate Text with GPT")
    st.markdown(
        '<div class="theory-box">GPT generates text <b>autoregressively</b>: it predicts the next token based on all previous tokens, then adds that token to the sequence and repeats. Parameters like <b>Temperature</b> and <b>Top-K</b> control the creativity and randomness of the output.</div>',
        unsafe_allow_html=True,
    )

    try:
        from transformers import pipeline, AutoTokenizer

        @st.cache_resource
        def load_gpt():
            # Using DistilGPT-2 for fast, lightweight text generation
            # Explicitly use PyTorch to avoid Keras 3 / TensorFlow conflicts
            generator = pipeline("text-generation", model="distilgpt2", framework="pt")
            tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            # Ensure pad token exists for batched generation
            if generator.tokenizer.pad_token is None:
                generator.tokenizer.pad_token = generator.tokenizer.eos_token
            return generator, tokenizer

        with st.spinner("Loading GPT model (this may take a moment on first run)..."):
            generator, tokenizer = load_gpt()

        c1, c2 = st.columns([2, 1])
        with c1:
            user_prompt = st.text_area(
                "Enter a story prompt:",
                value="The old clock tower struck midnight, and",
                height=100,
            )
        with c2:
            st.markdown("**Generation Parameters**")
            temperature = st.slider(
                "Temperature",
                0.1,
                2.0,
                0.8,
                0.1,
                help="Controls randomness. Lower = more deterministic, Higher = more creative.",
            )
            top_k = st.slider(
                "Top-K Sampling",
                10,
                100,
                50,
                5,
                help="Limits the next token choice to the K most likely options.",
            )
            max_length = st.slider("Max Length (tokens)", 30, 150, 80, 5)

            generate_btn = st.button(
                "✨ Generate Continuation", use_container_width=True, type="primary"
            )

        if generate_btn:
            with st.spinner("Generating text..."):
                # Generate
                output = generator(
                    user_prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    truncation=True,
                )

                generated_text = output[0]["generated_text"]

                # Split into prompt and continuation for visual distinction
                if generated_text.startswith(user_prompt):
                    continuation = generated_text[len(user_prompt) :]
                else:
                    continuation = generated_text

                st.markdown("### 📝 Generated Output")
                st.markdown(
                    f"""
                <div class="gen-box">
                  <span class="prompt-text">{user_prompt}</span><span class="continuation-text">{continuation}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Token probability visualization (conceptual)
                st.markdown("### 🧠 How GPT Chose the Next Words")
                st.markdown(
                    "At each step, GPT calculates a probability distribution over its entire vocabulary (e.g., 50,000+ words)."
                )

                # Mock next-token probabilities for the first generated word
                mock_probs = [
                    ("the", 0.45),
                    ("a", 0.25),
                    ("suddenly", 0.15),
                    ("it", 0.10),
                    ("...", 0.05),
                ]

                fig, ax = dfig(10, 4)
                words = [item[0] for item in mock_probs]
                probs = [item[1] for item in mock_probs]

                bars = ax.barh(
                    words[::-1], probs[::-1], color="#10b981", edgecolor="white"
                )
                ax.set_xlim(0, 0.6)
                ax.set_xlabel("Probability", color=TEXT)
                ax.set_title(
                    "Conceptual Next-Token Probability Distribution", color="#e2e8f0"
                )
                ax.set_facecolor(DARK_AX)
                ax.tick_params(colors=TEXT)

                # Add percentage labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(
                        width + 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{width * 100:.0f}%",
                        va="center",
                        color=TEXT,
                        fontsize=9,
                    )

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

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
    st.title("Inside the GPT Model")

    tab1, tab2 = st.tabs(["🔬 Decoder-Only & Causal Masking", "🌡️ Sampling Strategies"])

    with tab1:
        st.markdown("**The Decoder-Only Architecture**")
        st.markdown(
            '<div class="theory-box">Unlike BERT (which is encoder-only and bidirectional), GPT is a <b>decoder-only</b> Transformer. It is trained on <b>Causal Language Modeling</b>: predicting the next token given only the <i>past</i> tokens.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(r"""
        | Component | Purpose in GPT |
        |---|---|
        | **Causal Mask (Look-Ahead Mask)** | Ensures that the prediction for position $i$ can only depend on known outputs at positions $< i$. It masks out future tokens during training. |
        | **Self-Attention** | Allows each token to attend to all *previous* tokens, building rich contextual representations. |
        | **Feed-Forward Network** | Processes the attention outputs independently at each position to add non-linearity and capacity. |
        | **Autoregressive Loop** | During inference, the generated token is appended to the input, and the model runs again to predict the *next* token. |
        """)

        st.markdown("**Visualizing the Causal Mask**")
        fig, ax = dfig(8, 6)
        # Create a lower triangular matrix (causal mask)
        mask = np.tril(np.ones((5, 5)))
        sns.heatmap(
            mask,
            annot=True,
            fmt=".0f",
            cmap="Greens",
            ax=ax,
            cbar=False,
            linewidths=1,
            linecolor=DARK_AX,
            annot_kws={"color": "white", "size": 14, "weight": "bold"},
        )
        ax.set_title(
            "Causal (Look-Ahead) Mask", color="#e2e8f0", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(np.arange(5) + 0.5)
        ax.set_yticks(np.arange(5) + 0.5)
        ax.set_xticklabels(["t-4", "t-3", "t-2", "t-1", "t"])
        ax.set_yticklabels(["t-4", "t-3", "t-2", "t-1", "t"])
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors=TEXT)

        # Add 0s and 1s explanation
        ax.text(
            5.2, 2, "1 = Attend\n(Allowed)", color="#10b981", fontsize=10, va="center"
        )
        ax.text(
            5.2, 3.5, "0 = Masked\n(Future)", color="#f87171", fontsize=10, va="center"
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown("**Controlling Creativity: Sampling Strategies**")
        st.markdown(
            '<div class="theory-box">GPT doesn\'t just pick the #1 most likely word every time (that leads to repetitive, boring text). We use sampling strategies to balance coherence and creativity.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("""
        | Strategy | How it Works | Effect |
        |---|---|---|
        | **Greedy Search** | Always picks the token with the highest probability. | Highly repetitive, gets stuck in loops. |
        | **Temperature ($T$)** | Divides logits by $T$ before softmax. $T < 1$ sharpens distribution (more focused); $T > 1$ flattens it (more random). | Low $T$ = safe/boring. High $T$ = creative/nonsensical. |
        | **Top-K Sampling** | Filters the vocabulary to only the $K$ most likely tokens, then samples from that restricted set. | Prevents very low-probability "weird" words from being chosen. |
        | **Top-P (Nucleus)** | Filters tokens to the smallest set whose cumulative probability exceeds $P$ (e.g., 0.9), then samples. | Dynamic vocabulary size; adapts to the certainty of the model. |
        """)

        st.markdown("### 🎛️ Interactive Concept: Temperature Effect")
        st.markdown("Imagine the model is deciding the next word after *'The sky is'*:")

        temp = st.slider("Adjust Temperature", 0.1, 2.0, 1.0, 0.1, key="temp_slider")

        # Mock probabilities adjusting with temperature
        base_logits = np.array([2.0, 1.0, 0.5, -1.0])  # blue, clear, cloudy, green
        scaled_logits = base_logits / temp
        probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))

        words = ["blue", "clear", "cloudy", "green"]
        colors = ["#10b981", "#3b82f6", "#94a3b8", "#f87171"]

        fig, ax = dfig(10, 4)
        bars = ax.barh(words, probs, color=colors, edgecolor="white")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability", color=TEXT)
        ax.set_title(
            f"Next-Token Probabilities at Temperature = {temp}", color="#e2e8f0"
        )
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors=TEXT)

        for bar, p in zip(bars, probs):
            ax.text(
                p + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{p * 100:.1f}%",
                va="center",
                color=TEXT,
                fontsize=9,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.markdown("---")
st.markdown(
    "<center style='color:#4a5568;font-size:.78rem'>GPT (Generative Pre-trained Transformer) Explorer &nbsp;|&nbsp; Hugging Face · PyTorch</center>",
    unsafe_allow_html=True,
)
