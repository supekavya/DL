import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="BiLSTM Explorer",
    page_icon="🔮",
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
  background:#1a1f40;color:#a78bfa;border:1px solid #8b5cf6;}
.theory-box{background:linear-gradient(135deg,#1a1f35,#1e2540);border-left:4px solid #8b5cf6;
  border-radius:0 10px 10px 0;padding:1rem 1.4rem;margin:.8rem 0;color:#cbd5e1;
  line-height:1.7;font-size:.9rem;}
.mrow{display:flex;gap:.8rem;margin:.8rem 0;}
.mchip{background:#1e2130;border:1px solid #3a3f5c;border-radius:10px;padding:.7rem 1rem;text-align:center;flex:1;}
.mval{font-family:'DM Mono',monospace;font-size:1.4rem;color:#a78bfa;}
.mlbl{font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.1em;}
.pred-box{background:linear-gradient(135deg,#1a1040,#1e1560);border:2px solid #8b5cf6;
  border-radius:16px;padding:1.5rem;text-align:center;}
.pred-val{font-family:'DM Mono',monospace;font-size:2.8rem;font-weight:700;color:#a78bfa;}
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
    df = pd.read_csv("data/timeseries.csv")
    return df


df = load_data()


def create_sequences_with_labels(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        window = data[i : i + seq_length]
        X.append(window)
        # Label: 1 if the end of the window is higher than the start (Upward trend), else 0
        label = 1 if window[-1] > window[0] else 0
        y.append(label)
    return np.array(X), np.array(y)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 BiLSTM Explorer")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 EDA", "🏗️ Build & Train", "🔮 Classify Sequence", "📐 Architecture Study"],
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption(f"Synthetic Time Series")
    st.caption(f"Samples: **{df.shape[0]}** | Features: **1**")
    st.caption(f"Task: Sequence Classification (Trend Direction)")

# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 EDA":
    st.markdown(
        '<span class="badge">Exploratory Data Analysis</span>', unsafe_allow_html=True
    )
    st.title("Dataset Exploration")

    tab1, tab2 = st.tabs(["Overview", "Time Series Visualization"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Dataset Preview**")
            st.dataframe(df.head(10), use_container_width=True)
        with c2:
            st.markdown("**Statistics**")
            st.dataframe(df.describe().round(4), use_container_width=True)

    with tab2:
        fig, ax = dfig(14, 5)
        ax.plot(df["time"], df["value"], color="#a78bfa", lw=1.5, label="Value")
        ax.set_title(
            "Synthetic Time Series (Trend + Seasonality + Noise)",
            color="#e2e8f0",
            fontsize=13,
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(
            '<div class="theory-box">This dataset contains complex seasonal patterns, a slight upward trend, and Gaussian noise. The classification task is to determine if a given sliding window exhibits a net <b>Upward</b> or <b>Downward</b> trajectory, a task where seeing both past and future context within the window is highly beneficial.</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# BUILD & TRAIN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Build & Train":
    st.markdown(
        '<span class="badge">Build & Train BiLSTM</span>', unsafe_allow_html=True
    )
    st.title("Build & Train Bidirectional LSTM")
    st.markdown(
        '<div class="theory-box">Configure the BiLSTM architecture. By processing the sequence in both forward and backward directions, the model can learn patterns that depend on the entire context of the window, not just the past.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Architecture**")
        seq_length = st.slider("Sequence Length (Time Steps)", 10, 100, 30)
        units = st.select_slider(
            "BiLSTM Units (per direction)", options=[16, 32, 64, 128], value=32
        )
        layers_count = st.slider("Number of BiLSTM Layers", 1, 3, 1)
        dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
        st.markdown("**Training**")
        lr = st.select_slider(
            "Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005], value=0.001
        )
        batch_sz = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
        epochs = st.slider("Max Epochs", 20, 200, 100)
        run_btn = st.button("🚀 Train Model", use_container_width=True)

    with c2:
        if run_btn:
            try:
                import tensorflow as tf
                from tensorflow import keras
                from tensorflow.keras import layers
                from tensorflow.keras.callbacks import EarlyStopping

                values = df["value"].values.reshape(-1, 1)
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(values)

                X, y = create_sequences_with_labels(scaled_values.flatten(), seq_length)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))

                # Train/test split
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

                model = keras.Sequential()
                for i in range(layers_count):
                    return_seq = True if i < layers_count - 1 else False
                    # Bidirectional wrapper doubles the units internally, but we specify units per direction
                    model.add(
                        layers.Bidirectional(
                            layers.LSTM(units, return_sequences=return_seq),
                            input_shape=(seq_length, 1) if i == 0 else None,
                        )
                    )
                    if i < layers_count - 1:
                        model.add(layers.Dropout(dropout))

                model.add(layers.Dropout(dropout))
                model.add(layers.Dense(1, activation="sigmoid"))

                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr),
                    loss="binary_crossentropy",
                    metrics=["accuracy"],
                )

                cbs = [
                    EarlyStopping(
                        monitor="val_accuracy",
                        patience=15,
                        restore_best_weights=True,
                        mode="max",
                    )
                ]

                with st.spinner("Training…"):
                    hist = model.fit(
                        X_train,
                        y_train,
                        validation_split=0.15,
                        epochs=epochs,
                        batch_size=batch_sz,
                        callbacks=cbs,
                        verbose=0,
                    )

                y_pred_prob = model.predict(X_test, verbose=0).flatten()
                y_pred = (y_pred_prob >= 0.5).astype(int)

                acc = accuracy_score(y_test, y_pred)
                params = model.count_params()

                # Calculate precision, recall, f1
                from sklearn.metrics import precision_score, recall_score, f1_score

                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                mchips(acc, prec, rec, f1)

                hdf = pd.DataFrame(hist.history)
                fig, axes = dfig(14, 5, 2)
                axes = np.array(axes).flatten()

                axes[0].plot(hdf["loss"], color="#f87171", lw=2, label="Train")
                axes[0].plot(hdf["val_loss"], color="#a78bfa", lw=2, label="Val")
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

                # Confusion Matrix & ROC
                fig, axes2 = dfig(14, 5, 2)
                axes2 = np.array(axes2).flatten()

                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Purples",
                    ax=axes2[0],
                    xticklabels=["Downward (0)", "Upward (1)"],
                    yticklabels=["Downward (0)", "Upward (1)"],
                    linewidths=1,
                    linecolor="white",
                    annot_kws={"color": "white", "size": 14},
                )
                axes2[0].set_title("Confusion Matrix", color="#e2e8f0")
                axes2[0].set_facecolor(DARK_AX)
                axes2[0].tick_params(colors=TEXT)

                fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                roc_auc = auc(fpr, tpr)
                axes2[1].plot(
                    fpr,
                    tpr,
                    color="#a78bfa",
                    lw=2.5,
                    label=f"BiLSTM (AUC = {roc_auc:.3f})",
                )
                axes2[1].fill_between(fpr, tpr, alpha=0.1, color="#a78bfa")
                axes2[1].plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
                axes2[1].set_title("ROC Curve")
                axes2[1].set_xlabel("False Positive Rate")
                axes2[1].set_ylabel("True Positive Rate")
                axes2[1].legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            except ImportError:
                st.error("TensorFlow not installed. Run: `pip install tensorflow`")
        else:
            st.info("👈 Configure the network on the left and click **Train Model**.")
            st.markdown("**Planned Architecture:**")
            arch_lines = ["```", f"Input  (Sequence Length: {seq_length}, Features: 1)"]
            for i in range(layers_count):
                arch_lines += [f"  Bidirectional(LSTM({units}), merge_mode='concat')"]
                if i < layers_count - 1:
                    arch_lines += [f"  Dropout({dropout})"]
            arch_lines += [
                f"  Dropout({dropout})",
                "  Dense(1, activation='sigmoid')",
                "Output (Probability of Upward Trend)",
                "```",
            ]
            st.markdown("\n".join(arch_lines))

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Classify Sequence":
    st.markdown(
        '<span class="badge">Live Classification</span>', unsafe_allow_html=True
    )
    st.title("Classify a Time Series Window")
    st.markdown(
        '<div class="theory-box">Input a custom sequence or use a random window from the dataset. The BiLSTM will analyze the <b>entire</b> window (from start to end and end to start) to determine if the overall trend is Upward or Downward.</div>',
        unsafe_allow_html=True,
    )

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.callbacks import EarlyStopping

        @st.cache_resource
        def quick_model():
            values = df["value"].values.reshape(-1, 1)
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(values).flatten()
            seq_len = 30

            X, y = create_sequences_with_labels(scaled_values, seq_len)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            split_idx = int(len(X) * 0.8)
            X_train, y_train = X[:split_idx], y[:split_idx]

            model = keras.Sequential(
                [
                    layers.Bidirectional(
                        layers.LSTM(32, return_sequences=True), input_shape=(seq_len, 1)
                    ),
                    layers.Dropout(0.2),
                    layers.Bidirectional(layers.LSTM(16, return_sequences=False)),
                    layers.Dropout(0.2),
                    layers.Dense(1, activation="sigmoid"),
                ]
            )
            model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )
            model.fit(
                X_train,
                y_train,
                epochs=60,
                batch_size=32,
                verbose=0,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True, mode="max")
                ],
                validation_split=0.15,
            )
            return model, scaler, seq_len, scaled_values

        with st.spinner("Preparing model…"):
            model, scaler, seq_len, scaled_values = quick_model()

        use_random = st.checkbox("Use random window from dataset", value=True)

        if use_random:
            start_idx = np.random.randint(0, len(scaled_values) - seq_len - 100)
            window_scaled = scaled_values[start_idx : start_idx + seq_len]
            true_label = 1 if window_scaled[-1] > window_scaled[0] else 0
        else:
            st.markdown(
                "**Manually define the sequence (first 10 steps shown for brevity, rest are zero)**"
            )
            cols = st.columns(10)
            manual_vals = []
            for i in range(10):
                manual_vals.append(
                    cols[i].number_input(f"Step {i + 1}", value=0.0, step=0.1)
                )
            window_manual = np.array(manual_vals + [0.0] * (seq_len - 10))
            window_scaled = scaler.transform(window_manual.reshape(-1, 1)).flatten()
            true_label = 1 if window_scaled[-1] > window_scaled[0] else 0

        if st.button("🔮 Classify Window", use_container_width=True):
            X_input = window_scaled.reshape(1, seq_len, 1)
            prob = float(model.predict(X_input, verbose=0).flatten()[0])
            pred_class = "Upward Trend (1)" if prob >= 0.5 else "Downward Trend (0)"
            color = "#34d399" if prob >= 0.5 else "#f87171"

            st.markdown(
                f"""
            <div class="pred-box">
              <div style="color:#94a3b8;font-size:.85rem;text-transform:uppercase;letter-spacing:.15em">Prediction</div>
              <div class="pred-val" style="color:{color}">{pred_class}</div>
              <div style="color:#94a3b8;margin-top:.5rem">Probability: <b style="color:#a78bfa">{prob:.4f}</b></div>
            </div>""",
                unsafe_allow_html=True,
            )

            # Visualize the window
            fig, ax = dfig(12, 4)
            ax.plot(
                np.arange(seq_len),
                window_scaled,
                color="#a78bfa",
                lw=2,
                marker="o",
                markersize=4,
            )
            ax.axhline(0, color=TEXT, linestyle=":", alpha=0.5)
            ax.set_title("Input Sequence (Standardized)")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")

            # Add start and end markers
            ax.scatter(
                0, window_scaled[0], color="#f87171", s=100, zorder=5, label="Start"
            )
            ax.scatter(
                seq_len - 1,
                window_scaled[-1],
                color="#34d399",
                s=100,
                zorder=5,
                label="End",
            )
            ax.legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    except ImportError:
        st.error("TensorFlow required. Install with: `pip install tensorflow`")

# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE STUDY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Architecture Study":
    st.markdown('<span class="badge">Architecture Study</span>', unsafe_allow_html=True)
    st.title("Bidirectional LSTM Architecture & Concepts")

    tab1, tab2 = st.tabs(["🔬 The BiLSTM Concept", "📊 When to use BiLSTM vs LSTM"])

    with tab1:
        st.markdown("**How Bidirectional Processing Works**")
        st.markdown(
            '<div class="theory-box">A standard LSTM only looks at <b>past</b> context ($x_{t-1}, x_{t-2}, ...$). A <b>Bidirectional LSTM</b> consists of two separate LSTMs: one processes the sequence forward (past to future), and the other processes it backward (future to past). Their hidden states are then concatenated at each time step.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(r"""
        | Component | Direction | Purpose |
        |---|---|---|
        | **Forward LSTM** | $x_1 \rightarrow x_T$ | Captures dependencies from past to present. |
        | **Backward LSTM** | $x_T \rightarrow x_1$ | Captures dependencies from future to present. |
        | **Merge Mode** | Concatenate (default) | Combines both contexts: $h_t = [\overrightarrow{h_t}, \overleftarrow{h_t}]$ |
        """)

        st.markdown("**Visualizing the BiLSTM Unrolled**")
        fig, ax = dfig(14, 5)
        steps = 4
        x_pos = np.arange(steps)

        # Forward pass (bottom)
        for i in range(steps):
            ax.plot(
                [x_pos[i], x_pos[i]],
                [1, 1.3],
                color="#34d399",
                lw=3,
                solid_capstyle="round",
            )
            ax.text(
                x_pos[i],
                1.5,
                rf"$\overrightarrow{{h}}_{{{i}}}$",
                ha="center",
                color="#34d399",
                fontsize=12,
                fontweight="bold",
            )
            ax.text(x_pos[i], 0.8, f"$x_{{{i}}}$", ha="center", color=TEXT, fontsize=12)
            if i < steps - 1:
                ax.annotate(
                    "",
                    xy=(x_pos[i + 1], 1),
                    xytext=(x_pos[i], 1),
                    arrowprops=dict(arrowstyle="->", color="#34d399", lw=2),
                )

        # Backward pass (top)
        for i in range(steps - 1, -1, -1):
            ax.plot(
                [x_pos[i], x_pos[i]],
                [3, 2.7],
                color="#f87171",
                lw=3,
                solid_capstyle="round",
            )
            ax.text(
                x_pos[i],
                2.5,
                rf"$\overleftarrow{{h}}_{{{i}}}$",
                ha="center",
                color="#f87171",
                fontsize=12,
                fontweight="bold",
            )
            if i > 0:
                ax.annotate(
                    "",
                    xy=(x_pos[i - 1], 3),
                    xytext=(x_pos[i], 3),
                    arrowprops=dict(arrowstyle="->", color="#f87171", lw=2),
                )

        # Concatenation
        for i in range(steps):
            ax.plot(
                [x_pos[i], x_pos[i]],
                [1.6, 2.4],
                color="#a78bfa",
                lw=1.5,
                linestyle="--",
            )
            ax.text(
                x_pos[i],
                2.0,
                "Concat",
                ha="center",
                va="center",
                color="#a78bfa",
                fontsize=9,
                bbox=dict(
                    facecolor=DARK_AX, edgecolor="#a78bfa", boxstyle="round,pad=0.2"
                ),
            )

        ax.set_xlim(-0.5, steps - 0.5)
        ax.set_ylim(0.5, 3.5)
        ax.set_title(
            "Bidirectional LSTM Unrolled Through Time",
            color="#e2e8f0",
            fontsize=14,
            fontweight="bold",
        )
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown("**BiLSTM vs Standard LSTM: Use Cases**")
        st.markdown(
            '<div class="theory-box">BiLSTMs are <b>not</b> suitable for real-time future forecasting (since they require future data to process the backward pass). However, they are the gold standard for tasks where the full sequence is available at inference time.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("""
        | Task Type | Recommended Architecture | Why? |
        |---|---|---|
        | **Time Series Forecasting** | Standard LSTM / GRU | Future data is unknown; backward pass is impossible. |
        | **Sequence Classification** | **BiLSTM** | The entire window is available; context from both ends improves accuracy (e.g., trend detection). |
        | **Time Series Smoothing / Imputation** | **BiLSTM** | Missing values can be inferred from both past and future surrounding data. |
        | **Natural Language Processing** | **BiLSTM** | Word meaning depends on both preceding and succeeding words (e.g., Named Entity Recognition). |
        | **Anomaly Detection** | **BiLSTM** | Anomalies are often best identified by comparing a point to its full local context. |
        """)

st.markdown("---")
st.markdown(
    "<center style='color:#4a5568;font-size:.78rem'>Bidirectional LSTM (BiLSTM) Explorer &nbsp;|&nbsp; TensorFlow · Keras · Scikit-learn</center>",
    unsafe_allow_html=True,
)
