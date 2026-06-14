import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="RNN Explorer",
    page_icon="🔄",
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
  background:#1a1f40;color:#818cf8;border:1px solid #6366f1;}
.theory-box{background:linear-gradient(135deg,#1a1f35,#1e2540);border-left:4px solid #6366f1;
  border-radius:0 10px 10px 0;padding:1rem 1.4rem;margin:.8rem 0;color:#cbd5e1;
  line-height:1.7;font-size:.9rem;}
.mrow{display:flex;gap:.8rem;margin:.8rem 0;}
.mchip{background:#1e2130;border:1px solid #3a3f5c;border-radius:10px;padding:.7rem 1rem;text-align:center;flex:1;}
.mval{font-family:'DM Mono',monospace;font-size:1.4rem;color:#818cf8;}
.mlbl{font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.1em;}
.pred-box{background:linear-gradient(135deg,#1a1040,#1e1560);border:2px solid #6366f1;
  border-radius:16px;padding:1.5rem;text-align:center;}
.pred-val{font-family:'DM Mono',monospace;font-size:2.8rem;font-weight:700;color:#818cf8;}
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


def mchips(mse, mae, loss, params):
    st.markdown(
        f"""<div class="mrow">
      <div class="mchip"><div class="mval">{mse:.4f}</div><div class="mlbl">MSE</div></div>
      <div class="mchip"><div class="mval">{mae:.4f}</div><div class="mlbl">MAE</div></div>
      <div class="mchip"><div class="mval">{loss:.4f}</div><div class="mlbl">Test Loss</div></div>
      <div class="mchip"><div class="mval">{params:,}</div><div class="mlbl">Parameters</div></div>
    </div>""",
        unsafe_allow_html=True,
    )


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/timeseries.csv")
    return df


df = load_data()


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# ── PyTorch Models ────────────────────────────────────────────────────────────
class RNNModel(nn.Module):
    def __init__(
        self, rnn_type, input_size=1, hidden_size=64, num_layers=1, dropout=0.2
    ):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            self.rnn = nn.RNN(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                nonlinearity="tanh",
            )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Take the last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔄 RNN Explorer")
    st.markdown("---")
    page = st.radio(
        "Navigate", ["📊 EDA", "🏗️ Build & Train", "🔮 Predict", "📐 Architecture Study"]
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption(f"Synthetic Time Series")
    st.caption(f"Samples: **{df.shape[0]}** | Features: **1**")
    st.caption(f"Task: Multi-step forecasting")

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
        ax.plot(df["time"], df["value"], color="#818cf8", lw=1.5, label="Value")
        ax.set_title(
            "Synthetic Time Series (Sine Waves + Noise)", color="#e2e8f0", fontsize=13
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(
            '<div class="theory-box">The dataset consists of two superimposed sine waves with different frequencies, plus Gaussian noise. This creates a non-trivial pattern that RNNs (especially LSTMs/GRUs) can learn to forecast.</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# BUILD & TRAIN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Build & Train":
    st.markdown('<span class="badge">Build & Train RNN</span>', unsafe_allow_html=True)
    st.title("Build & Train Recurrent Neural Network (PyTorch)")
    st.markdown(
        '<div class="theory-box">Configure the RNN architecture. The model will learn to predict the next value in the sequence based on a sliding window of past values.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Architecture**")
        rnn_type = st.selectbox("RNN Type", ["LSTM", "GRU", "SimpleRNN"])
        seq_length = st.slider("Sequence Length", 5, 50, 20)
        units = st.select_slider(
            "Hidden Units", options=[16, 32, 64, 128, 256], value=64
        )
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
                values = df["value"].values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_values = scaler.fit_transform(values)

                train_size = int(len(scaled_values) * 0.8)
                train_data = scaled_values[:train_size]
                test_data = scaled_values[train_size - seq_length :]

                X_train, y_train = create_sequences(train_data, seq_length)
                X_test, y_test = create_sequences(test_data, seq_length)

                X_train = np.reshape(
                    X_train, (X_train.shape[0], X_train.shape[1], 1)
                ).astype(np.float32)
                X_test = np.reshape(
                    X_test, (X_test.shape[0], X_test.shape[1], 1)
                ).astype(np.float32)
                y_train = y_train.astype(np.float32)
                y_test = y_test.astype(np.float32)

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
                model = RNNModel(
                    rnn_type=rnn_type,
                    input_size=1,
                    hidden_size=units,
                    num_layers=1,
                    dropout=dropout,
                ).to(device)

                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                train_losses, val_losses = [], []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for epoch in range(epochs):
                    model.train()
                    epoch_loss = 0
                    for bx, by in train_loader:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        outputs = model(bx)
                        loss = criterion(outputs, by)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item() * bx.size(0)

                    train_losses.append(epoch_loss / len(train_dataset))

                    # Validation
                    model.eval()
                    val_loss, val_total = 0, 0
                    with torch.no_grad():
                        for bx, by in test_loader:
                            bx, by = bx.to(device), by.to(device)
                            outputs = model(bx)
                            loss = criterion(outputs, by)
                            val_loss += loss.item() * bx.size(0)
                            val_total += bx.size(0)

                    val_losses.append(val_loss / val_total)
                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(
                        f"Epoch {epoch + 1}/{epochs} | Val Loss: {val_losses[-1]:.4f}"
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
                        all_preds.extend(outputs.cpu().numpy().flatten())
                        all_targets.extend(by.numpy().flatten())

                y_pred_inv = scaler.inverse_transform(
                    np.array(all_preds).reshape(-1, 1)
                )
                y_test_inv = scaler.inverse_transform(
                    np.array(all_targets).reshape(-1, 1)
                )

                mse = mean_squared_error(y_test_inv, y_pred_inv)
                mae = mean_absolute_error(y_test_inv, y_pred_inv)
                tloss = val_losses[-1]
                params = sum(p.numel() for p in model.parameters())
                mchips(mse, mae, tloss, params)

                fig, ax = dfig(12, 4)
                ax.plot(train_losses, color="#f87171", lw=2, label="Train")
                ax.plot(val_losses, color="#60a5fa", lw=2, label="Val")
                ax.set_title("Training Loss (MSE)")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                fig, ax = dfig(14, 5)
                test_indices = np.arange(len(y_test_inv))
                ax.plot(
                    test_indices, y_test_inv, color="#818cf8", lw=1.5, label="Actual"
                )
                ax.plot(
                    test_indices,
                    y_pred_inv,
                    color="#f87171",
                    lw=1.5,
                    linestyle="--",
                    label="Predicted",
                )
                ax.set_title("Test Set: Actual vs Predicted")
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Value")
                ax.legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.session_state["rnn_model"] = model.cpu()
                st.session_state["rnn_scaler"] = scaler
                st.session_state["rnn_seq_len"] = seq_length
                st.session_state["rnn_type"] = rnn_type
                st.success("Model trained and saved to session!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.info("👈 Configure the network on the left and click **Train Model**.")
            st.markdown("**Planned Architecture:**")
            arch_lines = ["```", f"Input  (Sequence Length: {seq_length}, Features: 1)"]
            arch_lines += [f"  {rnn_type}({units}) → Dropout({dropout})"]
            arch_lines += ["  Linear(1)", "Output (Next Value Prediction)", "```"]
            st.markdown("\n".join(arch_lines))

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown('<span class="badge">Live Forecasting</span>', unsafe_allow_html=True)
    st.title("Forecast Future Values")
    st.markdown(
        '<div class="theory-box">The trained model will forecast the next N steps based on the last known sequence. A quick model is trained automatically for demonstration.</div>',
        unsafe_allow_html=True,
    )

    try:
        if "rnn_model" not in st.session_state:
            with st.spinner("Training default model for prediction..."):
                values = df["value"].values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_values = scaler.fit_transform(values)
                seq_len = 20
                X, y = create_sequences(scaled_values, seq_len)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1)).astype(np.float32)
                y = y.astype(np.float32)

                train_dataset = TensorDataset(
                    torch.tensor(X), torch.tensor(y).unsqueeze(1)
                )
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

                model = RNNModel(
                    rnn_type="LSTM",
                    input_size=1,
                    hidden_size=64,
                    num_layers=1,
                    dropout=0.2,
                )
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(60):
                    model.train()
                    for bx, by in train_loader:
                        optimizer.zero_grad()
                        outputs = model(bx)
                        loss = criterion(outputs, by)
                        loss.backward()
                        optimizer.step()

                st.session_state["rnn_model"] = model
                st.session_state["rnn_scaler"] = scaler
                st.session_state["rnn_seq_len"] = seq_len

        model = st.session_state["rnn_model"]
        scaler = st.session_state["rnn_scaler"]
        seq_len = st.session_state["rnn_seq_len"]

        forecast_steps = st.slider("Forecast Steps", 10, 100, 50)

        if st.button("🔄 Generate Forecast", use_container_width=True):
            last_seq = (
                scaler.transform(df["value"].values.reshape(-1, 1))[-seq_len:]
                .reshape(1, seq_len, 1)
                .astype(np.float32)
            )
            forecast_scaled = []

            model.eval()
            with torch.no_grad():
                current_seq = torch.tensor(last_seq)
                for _ in range(forecast_steps):
                    pred = model(current_seq).item()
                    forecast_scaled.append(pred)
                    # Shift sequence and add new prediction
                    next_seq = torch.cat(
                        (current_seq[:, 1:, :], torch.tensor([[[pred]]])), dim=1
                    )
                    current_seq = next_seq

            forecast_inv = scaler.inverse_transform(
                np.array(forecast_scaled).reshape(-1, 1)
            )
            last_actual_inv = scaler.inverse_transform(last_seq.reshape(-1, 1))

            fig, ax = dfig(14, 5)
            hist_steps = len(last_actual_inv)

            ax.plot(
                np.arange(hist_steps),
                last_actual_inv.flatten(),
                color="#818cf8",
                lw=2,
                label="Historical",
            )

            forecast_y = np.concatenate(
                ([last_actual_inv[-1, 0]], forecast_inv.flatten())
            )
            forecast_x = np.arange(hist_steps - 1, hist_steps - 1 + forecast_steps + 1)

            ax.plot(
                forecast_x,
                forecast_y,
                color="#f87171",
                lw=2,
                linestyle="--",
                label="Forecast",
            )
            ax.axvline(
                hist_steps - 1,
                color=TEXT,
                linestyle=":",
                alpha=0.5,
                label="Forecast Start",
            )
            ax.set_title(f"{forecast_steps}-Step Ahead Forecast")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
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
    st.title("RNN Architecture & Concepts")

    tab1, tab2 = st.tabs(["🔬 RNN Variants", "📉 Activation Functions"])

    with tab1:
        st.markdown("**Compare Recurrent Architectures**")
        st.markdown("""
        | Architecture | Key Feature | Best For | Vanishing Gradient? |
        |---|---|---|---|
        | **SimpleRNN** | Basic feedback loop | Very short sequences | Yes, severe |
        | **LSTM** | Cell state + 3 gates (Forget, Input, Output) | Long-term dependencies | Mitigated |
        | **GRU** | 2 gates (Reset, Update), no cell state | Faster training, good performance | Mitigated |
        """)

        st.markdown(
            '<div class="theory-box"><b>LSTM Cell:</b> Maintains a "cell state" that acts as a conveyor belt, allowing gradients to flow unchanged. Gates decide what information to add or remove.<br><br><b>GRU Cell:</b> A simplified LSTM that combines the forget and input gates into a single "update gate", making it computationally cheaper while often matching LSTM performance.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Visualizing Unrolled RNN**")
        fig, ax = dfig(14, 4)
        steps = 4
        x_pos = np.arange(steps)
        y_pos = np.zeros(steps)

        for i in range(steps):
            ax.plot(
                [x_pos[i], x_pos[i]],
                [y_pos[i] - 0.3, y_pos[i] + 0.3],
                color="#818cf8",
                lw=3,
                solid_capstyle="round",
            )
            ax.text(
                x_pos[i],
                y_pos[i] + 0.5,
                f"$h_{{{i}}}$",
                ha="center",
                color=TEXT,
                fontsize=12,
                fontweight="bold",
            )
            ax.text(
                x_pos[i],
                y_pos[i] - 0.5,
                f"$x_{{{i}}}$",
                ha="center",
                color=TEXT,
                fontsize=12,
            )
            if i < steps - 1:
                ax.annotate(
                    "",
                    xy=(x_pos[i + 1], y_pos[i + 1]),
                    xytext=(x_pos[i], y_pos[i]),
                    arrowprops=dict(arrowstyle="->", color="#f87171", lw=2),
                )

        ax.set_xlim(-0.5, steps - 0.5)
        ax.set_ylim(-1, 1)
        ax.set_title(
            "RNN Unrolled Through Time", color="#e2e8f0", fontsize=14, fontweight="bold"
        )
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown(
            "**Visualise how different activation functions transform inputs (Common in RNN gates)**"
        )
        x = np.linspace(-5, 5, 300)
        funcs = {
            "Tanh": np.tanh(x),
            "Sigmoid": 1 / (1 + np.exp(-x)),
            "ReLU": np.maximum(0, x),
        }
        colors_act = ["#818cf8", "#f87171", "#34d399"]

        fig, axes = dfig(14, 5, 2)
        axes = np.array(axes).flatten()
        for (name, vals), color in zip(funcs.items(), colors_act):
            axes[0].plot(x, vals, lw=2.5, color=color, label=name)
        axes[0].axhline(0, color="white", lw=0.5, alpha=0.4)
        axes[0].axvline(0, color="white", lw=0.5, alpha=0.4)
        axes[0].set_title("Activation Functions")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("f(x)")
        axes[0].legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)

        derivs = {
            "Tanh": 1 - np.tanh(x) ** 2,
            "Sigmoid": (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x))),
            "ReLU": np.where(x > 0, 1, 0),
        }
        for (name, vals), color in zip(derivs.items(), colors_act):
            axes[1].plot(x, vals, lw=2.5, color=color, label=f"d/dx {name}")
        axes[1].axhline(0, color="white", lw=0.5, alpha=0.4)
        axes[1].set_title("Derivatives (Gradients)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("f'(x)")
        axes[1].legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.markdown("---")
st.markdown(
    "<center style='color:#4a5568;font-size:.78rem'>Recurrent Neural Network (RNN) Explorer &nbsp;|&nbsp; PyTorch · Scikit-learn</center>",
    unsafe_allow_html=True,
)
