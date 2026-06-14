import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
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
        label = 1 if window[-1] > window[0] else 0
        y.append(label)
    return np.array(X), np.array(y)


# ── PyTorch Model ─────────────────────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.2):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 because bidirectional
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = out[:, -1, :]  # Take the last time step
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


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
    st.title("Build & Train Bidirectional LSTM (PyTorch)")
    st.markdown(
        '<div class="theory-box">Configure the BiLSTM architecture. By processing the sequence in both forward and backward directions, the model can learn patterns that depend on the entire context of the window.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Architecture**")
        seq_length = st.slider("Sequence Length (Time Steps)", 10, 100, 30)
        hidden_size = st.select_slider(
            "Hidden Size (per direction)", options=[16, 32, 64, 128], value=32
        )
        num_layers = st.slider("Number of BiLSTM Layers", 1, 3, 1)
        dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
        st.markdown("**Training**")
        lr = st.select_slider(
            "Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005], value=0.001
        )
        batch_sz = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
        epochs = st.slider("Max Epochs", 10, 100, 50)
        run_btn = st.button("🚀 Train Model", use_container_width=True)

    with c2:
        if run_btn:
            try:
                values = df["value"].values.reshape(-1, 1)
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(values)

                X, y = create_sequences_with_labels(scaled_values.flatten(), seq_length)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1)).astype(np.float32)
                y = y.astype(np.float32)

                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

                train_dataset = TensorDataset(
                    torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1)
                )
                test_dataset = TensorDataset(
                    torch.tensor(X_test), torch.tensor(y_test).unsqueeze(1)
                )

                train_loader = DataLoader(
                    train_dataset, batch_size=batch_sz, shuffle=True
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=batch_sz, shuffle=False
                )

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = BiLSTMClassifier(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                ).to(device)

                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                train_losses, val_losses, train_accs, val_accs = [], [], [], []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for epoch in range(epochs):
                    model.train()
                    epoch_loss, epoch_correct, epoch_total = 0, 0, 0
                    for bx, by in train_loader:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        outputs = model(bx)
                        loss = criterion(outputs, by)
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item() * bx.size(0)
                        preds = (outputs >= 0.5).float()
                        epoch_correct += (preds == by).sum().item()
                        epoch_total += bx.size(0)

                    train_losses.append(epoch_loss / epoch_total)
                    train_accs.append(epoch_correct / epoch_total)

                    # Validation
                    model.eval()
                    val_loss, val_correct, val_total = 0, 0, 0
                    with torch.no_grad():
                        for bx, by in test_loader:
                            bx, by = bx.to(device), by.to(device)
                            outputs = model(bx)
                            loss = criterion(outputs, by)
                            val_loss += loss.item() * bx.size(0)
                            preds = (outputs >= 0.5).float()
                            val_correct += (preds == by).sum().item()
                            val_total += bx.size(0)

                    val_losses.append(val_loss / val_total)
                    val_accs.append(val_correct / val_total)

                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(
                        f"Epoch {epoch + 1}/{epochs} | Train Acc: {train_accs[-1]:.3f} | Val Acc: {val_accs[-1]:.3f}"
                    )

                progress_bar.empty()
                status_text.empty()

                # Final Evaluation
                model.eval()
                all_preds, all_targets = [], []
                with torch.no_grad():
                    for bx, by in test_loader:
                        bx = bx.to(device)
                        outputs = model(bx)
                        preds = (outputs >= 0.5).float().cpu().numpy()
                        all_preds.extend(preds.flatten())
                        all_targets.extend(by.numpy().flatten())

                acc = accuracy_score(all_targets, all_preds)
                prec = precision_score(all_targets, all_preds, zero_division=0)
                rec = recall_score(all_targets, all_preds, zero_division=0)
                f1 = f1_score(all_targets, all_preds, zero_division=0)

                mchips(acc, prec, rec, f1)

                # Plotting
                fig, axes = dfig(14, 5, 2)
                axes = np.array(axes).flatten()

                axes[0].plot(train_losses, color="#f87171", lw=2, label="Train")
                axes[0].plot(val_losses, color="#a78bfa", lw=2, label="Val")
                axes[0].set_title("Training Loss (BCE)")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Loss")
                axes[0].legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)

                axes[1].plot(train_accs, color="#34d399", lw=2, label="Train")
                axes[1].plot(val_accs, color="#fbbf24", lw=2, label="Val")
                axes[1].set_title("Training Accuracy")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Accuracy")
                axes[1].legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Confusion Matrix
                fig, ax = dfig(8, 6)
                cm = confusion_matrix(all_targets, all_preds)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Purples",
                    ax=ax,
                    xticklabels=["Downward (0)", "Upward (1)"],
                    yticklabels=["Downward (0)", "Upward (1)"],
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

                # Save to session state
                st.session_state["pt_model"] = model.cpu()
                st.session_state["pt_scaler"] = scaler
                st.session_state["pt_seq_len"] = seq_length
                st.success("Model trained and saved to session!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.info("👈 Configure the network on the left and click **Train Model**.")
            st.markdown("**Planned Architecture:**")
            arch_lines = ["```", f"Input  (Sequence Length: {seq_length}, Features: 1)"]
            arch_lines += [
                f"  BiLSTM(hidden_size={hidden_size}, num_layers={num_layers}, bidirectional=True)"
            ]
            arch_lines += [f"  Dropout({dropout})"]
            arch_lines += [
                "  Linear(hidden_size * 2, 1)",
                "  Sigmoid()",
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
        '<div class="theory-box">Input a custom sequence or use a random window from the dataset. The BiLSTM will analyze the <b>entire</b> window to determine if the overall trend is Upward or Downward.</div>',
        unsafe_allow_html=True,
    )

    try:
        if "pt_model" not in st.session_state:
            with st.spinner("Training default model for prediction..."):
                values = df["value"].values.reshape(-1, 1)
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(values).flatten()
                seq_len = 30

                X, y = create_sequences_with_labels(scaled_values, seq_len)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1)).astype(np.float32)
                y = y.astype(np.float32)

                split_idx = int(len(X) * 0.8)
                X_train, y_train = X[:split_idx], y[:split_idx]

                train_dataset = TensorDataset(
                    torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1)
                )
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

                model = BiLSTMClassifier(
                    input_size=1, hidden_size=32, num_layers=1, dropout=0.2
                )
                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(40):
                    model.train()
                    for bx, by in train_loader:
                        optimizer.zero_grad()
                        outputs = model(bx)
                        loss = criterion(outputs, by)
                        loss.backward()
                        optimizer.step()

                st.session_state["pt_model"] = model
                st.session_state["pt_scaler"] = scaler
                st.session_state["pt_seq_len"] = seq_len

        model = st.session_state["pt_model"]
        scaler = st.session_state["pt_scaler"]
        seq_len = st.session_state["pt_seq_len"]

        use_random = st.checkbox("Use random window from dataset", value=True)

        if use_random:
            start_idx = np.random.randint(
                0,
                len(scaler.transform(df["value"].values.reshape(-1, 1)).flatten())
                - seq_len
                - 100,
            )
            window_scaled = scaler.transform(
                df["value"].values.reshape(-1, 1)
            ).flatten()[start_idx : start_idx + seq_len]
            true_label = 1 if window_scaled[-1] > window_scaled[0] else 0
        else:
            st.markdown("**Manually define the sequence (first 10 steps shown)**")
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
            X_input = torch.tensor(
                window_scaled.reshape(1, seq_len, 1), dtype=torch.float32
            )
            model.eval()
            with torch.no_grad():
                prob = float(model(X_input).item())

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

    except Exception as e:
        st.error(f"An error occurred: {e}")

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
            '<div class="theory-box">A standard LSTM only looks at <b>past</b> context. A <b>Bidirectional LSTM</b> consists of two separate LSTMs: one processes the sequence forward, and the other processes it backward. Their hidden states are then concatenated at each time step.</div>',
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
        | **Sequence Classification** | **BiLSTM** | The entire window is available; context from both ends improves accuracy. |
        | **Time Series Smoothing / Imputation** | **BiLSTM** | Missing values can be inferred from both past and future surrounding data. |
        | **Natural Language Processing** | **BiLSTM** | Word meaning depends on both preceding and succeeding words. |
        """)

st.markdown("---")
st.markdown(
    "<center style='color:#4a5568;font-size:.78rem'>Bidirectional LSTM (BiLSTM) Explorer &nbsp;|&nbsp; PyTorch · Scikit-learn</center>",
    unsafe_allow_html=True,
)
