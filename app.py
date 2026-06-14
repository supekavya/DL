import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="GRU Explorer",
    page_icon="⚡",
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
  background:#1a1f40;color:#fbbf24;border:1px solid #f59e0b;}
.theory-box{background:linear-gradient(135deg,#1a1f35,#1e2540);border-left:4px solid #f59e0b;
  border-radius:0 10px 10px 0;padding:1rem 1.4rem;margin:.8rem 0;color:#cbd5e1;
  line-height:1.7;font-size:.9rem;}
.mrow{display:flex;gap:.8rem;margin:.8rem 0;}
.mchip{background:#1e2130;border:1px solid #3a3f5c;border-radius:10px;padding:.7rem 1rem;text-align:center;flex:1;}
.mval{font-family:'DM Mono',monospace;font-size:1.4rem;color:#fbbf24;}
.mlbl{font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.1em;}
.pred-box{background:linear-gradient(135deg,#1a1040,#1e1560);border:2px solid #f59e0b;
  border-radius:16px;padding:1.5rem;text-align:center;}
.pred-val{font-family:'DM Mono',monospace;font-size:2.8rem;font-weight:700;color:#fbbf24;}
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ GRU Explorer")
    st.markdown("---")
    page = st.radio(
        "Navigate", ["📊 EDA", "🏗️ Build & Train", "🔮 Predict", "📐 Architecture Study"]
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption(f"Synthetic Time Series (Trend + Seasonality)")
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
        ax.plot(df["time"], df["value"], color="#fbbf24", lw=1.5, label="Value")
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
            '<div class="theory-box">This dataset combines a linear trend, multiple seasonal components, and Gaussian noise. GRUs excel at learning these temporal dependencies with fewer parameters and faster training times compared to LSTMs.</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# BUILD & TRAIN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Build & Train":
    st.markdown('<span class="badge">Build & Train GRU</span>', unsafe_allow_html=True)
    st.title("Build & Train Gated Recurrent Unit")
    st.markdown(
        '<div class="theory-box">Configure the GRU architecture. GRUs combine the forget and input gates of an LSTM into a single "update gate", and merge the cell state and hidden state, resulting in a simpler, faster model.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Architecture**")
        seq_length = st.slider("Sequence Length (Time Steps)", 10, 100, 30)
        units = st.select_slider("GRU Units", options=[16, 32, 64, 128, 256], value=64)
        layers_count = st.slider("Number of GRU Layers", 1, 3, 1)
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
                from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
                import time

                values = df["value"].values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_values = scaler.fit_transform(values)

                train_size = int(len(scaled_values) * 0.8)
                train_data = scaled_values[:train_size]
                test_data = scaled_values[train_size - seq_length :]

                X_train, y_train = create_sequences(train_data, seq_length)
                X_test, y_test = create_sequences(test_data, seq_length)

                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                model = keras.Sequential()
                for i in range(layers_count):
                    return_seq = True if i < layers_count - 1 else False
                    model.add(
                        layers.GRU(
                            units,
                            return_sequences=return_seq,
                            input_shape=(seq_length, 1) if i == 0 else None,
                        )
                    )
                    if i < layers_count - 1:
                        model.add(layers.Dropout(dropout))

                model.add(layers.Dropout(dropout))
                model.add(layers.Dense(1))

                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse"
                )

                cbs = [
                    EarlyStopping(
                        monitor="val_loss", patience=15, restore_best_weights=True
                    ),
                    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7),
                ]

                with st.spinner("Training…"):
                    start_time = time.time()
                    hist = model.fit(
                        X_train,
                        y_train,
                        validation_split=0.15,
                        epochs=epochs,
                        batch_size=batch_sz,
                        callbacks=cbs,
                        verbose=0,
                    )
                    train_time = time.time() - start_time
                    st.success(f"Training completed in {train_time:.2f} seconds!")

                y_pred = model.predict(X_test, verbose=0)
                y_pred_inv = scaler.inverse_transform(y_pred)
                y_test_inv = scaler.inverse_transform(y_test)

                mse = mean_squared_error(y_test_inv, y_pred_inv)
                mae = mean_absolute_error(y_test_inv, y_pred_inv)
                tloss = float(hist.history["val_loss"][-1])
                params = model.count_params()
                mchips(mse, mae, tloss, params)

                hdf = pd.DataFrame(hist.history)
                fig, ax = dfig(12, 4)
                ax.plot(hdf["loss"], color="#f87171", lw=2, label="Train")
                ax.plot(hdf["val_loss"], color="#fbbf24", lw=2, label="Val")
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

            except ImportError:
                st.error("TensorFlow not installed. Run: `pip install tensorflow`")
        else:
            st.info("👈 Configure the network on the left and click **Train Model**.")
            st.markdown("**Planned Architecture:**")
            arch_lines = ["```", f"Input  (Sequence Length: {seq_length}, Features: 1)"]
            for i in range(layers_count):
                arch_lines += [
                    f"  GRU({units}, return_sequences={'True' if i < layers_count - 1 else 'False'})"
                ]
                if i < layers_count - 1:
                    arch_lines += [f"  Dropout({dropout})"]
            arch_lines += [
                f"  Dropout({dropout})",
                "  Dense(1)",
                "Output (Next Value Prediction)",
                "```",
            ]
            st.markdown("\n".join(arch_lines))

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown('<span class="badge">Live Forecasting</span>', unsafe_allow_html=True)
    st.title("Forecast Future Values")
    st.markdown(
        '<div class="theory-box">The trained GRU will forecast the next N steps based on the last known sequence. GRUs often achieve comparable forecasting accuracy to LSTMs but with faster inference and fewer parameters.</div>',
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
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_values = scaler.fit_transform(values)
            seq_len = 30
            X, y = create_sequences(scaled_values, seq_len)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            model = keras.Sequential(
                [
                    layers.GRU(64, return_sequences=True, input_shape=(seq_len, 1)),
                    layers.Dropout(0.2),
                    layers.GRU(32, return_sequences=False),
                    layers.Dropout(0.2),
                    layers.Dense(1),
                ]
            )
            model.compile(optimizer="adam", loss="mse")
            model.fit(
                X,
                y,
                epochs=80,
                batch_size=32,
                verbose=0,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                validation_split=0.15,
            )
            return model, scaler, seq_len, scaled_values

        with st.spinner("Preparing model…"):
            model, scaler, seq_len, scaled_values = quick_model()

        forecast_steps = st.slider("Forecast Steps", 10, 100, 50)

        if st.button("🔄 Generate Forecast", use_container_width=True):
            last_seq = scaled_values[-seq_len:].reshape(1, seq_len, 1)
            forecast_scaled = []

            for _ in range(forecast_steps):
                pred = model.predict(last_seq, verbose=0)
                forecast_scaled.append(pred[0, 0])
                last_seq = np.append(
                    last_seq[:, 1:, :], np.reshape(pred, (1, 1, 1)), axis=1
                )

            forecast_inv = scaler.inverse_transform(
                np.array(forecast_scaled).reshape(-1, 1)
            )
            last_actual_inv = scaler.inverse_transform(scaled_values[-seq_len:])

            fig, ax = dfig(14, 5)
            hist_steps = len(last_actual_inv)

            # Plot historical data
            ax.plot(
                np.arange(hist_steps),
                last_actual_inv.flatten(),
                color="#818cf8",
                lw=2,
                label="Historical",
            )

            # Plot forecast data (connects smoothly from the last historical point)
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
            ax.set_title(f"{forecast_steps}-Step Ahead Forecast (GRU)")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
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
    st.title("GRU Architecture & Concepts")

    tab1, tab2 = st.tabs(["🔬 The GRU Cell", "⚡ GRU vs LSTM"])

    with tab1:
        st.markdown("**The Anatomy of a GRU Cell**")
        st.markdown(
            '<div class="theory-box">The Gated Recurrent Unit (GRU) is a streamlined version of the LSTM. It merges the cell state and hidden state into a single hidden state, and combines the forget and input gates into a single <b>Update Gate</b>. This reduces the number of parameters and speeds up training.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(r"""
        | Gate | Function | Mathematical Intuition |
        |---|---|---|
        | **Update Gate ($z_t$)** | Decides how much of the past information to keep and how much new information to let in. (Combines LSTM's forget and input gates). | $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$ |
        | **Reset Gate ($r_t$)** | Decides how much of the past information to forget when computing the candidate hidden state. | $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$ |
        | **Candidate Hidden State ($\tilde{h}_t$)** | Computes new memory content using the reset gate to drop irrelevant past info. | $\tilde{h}_t = \tanh(W \cdot [r_t * h_{t-1}, x_t] + b)$ |
        | **Final Hidden State ($h_t$)** | The final output, blending the previous hidden state and the candidate state using the update gate. | $h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$ |
        """)

        st.markdown("**Visualizing the GRU Cell Flow**")
        fig, ax = dfig(14, 6)

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        boxes = [
            (1, 7, 2, 1.5, "Input $x_t$"),
            (1, 4, 2, 1.5, "Prev Hidden $h_{t-1}$"),
            (4, 7, 2, 1.5, "Update Gate ($z_t$)"),
            (4, 4, 2, 1.5, "Reset Gate ($r_t$)"),
            (7, 5.5, 2, 1.5, "Candidate $\tilde{h}_t$"),
            (7, 2, 2, 1.5, "Final Hidden $h_t$"),
            (9, 7, 2, 1.5, "Output $h_t$"),
        ]

        for x, y, w, h, text in boxes:
            rect = plt.Rectangle(
                (x, y),
                w,
                h,
                facecolor="#1e2130",
                edgecolor="#f59e0b",
                linewidth=2,
                zorder=2,
            )
            ax.add_patch(rect)
            ax.text(
                x + w / 2,
                y + h / 2,
                text,
                ha="center",
                va="center",
                color="#e2e8f0",
                fontsize=10,
                fontweight="bold",
                zorder=3,
            )

        ax.plot(
            [0, 10],
            [2.75, 2.75],
            color="#fbbf24",
            lw=4,
            zorder=1,
            label="Hidden State (Memory & Output)",
        )
        ax.text(0.5, 3.2, "h", color="#fbbf24", fontsize=12, fontweight="bold")
        ax.text(9.5, 3.2, "h", color="#fbbf24", fontsize=12, fontweight="bold")

        ax.set_title(
            "Conceptual GRU Cell Architecture",
            color="#e2e8f0",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown("**GRU vs LSTM: When to use which?**")
        st.markdown(
            '<div class="theory-box">While LSTMs have a separate cell state allowing for more complex memory management, GRUs are often preferred in practice due to their computational efficiency and comparable performance on many tasks.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("""
        | Feature | LSTM | GRU |
        |---|---|---|
        | **Gates** | 3 (Forget, Input, Output) | 2 (Update, Reset) |
        | **States** | 2 (Cell State $C_t$, Hidden State $h_t$) | 1 (Hidden State $h_t$ only) |
        | **Parameters** | Higher (~4x hidden units) | Lower (~3x hidden units) |
        | **Training Speed** | Slower | **Faster** |
        | **Long-term Memory** | Excellent (dedicated cell state) | Very Good (sufficient for most tasks) |
        | **Best For** | Very long sequences, complex dependencies | Resource-constrained environments, general sequence tasks |
        """)

st.markdown("---")
st.markdown(
    "<center style='color:#4a5568;font-size:.78rem'>Gated Recurrent Unit (GRU) Explorer &nbsp;|&nbsp; TensorFlow · Keras · Scikit-learn</center>",
    unsafe_allow_html=True,
)
