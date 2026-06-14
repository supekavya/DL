import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ANN Explorer",
    page_icon="🧠",
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
CLASSES = ["Malignant (0)", "Benign (1)"]


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
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, data


df, raw_data = load_data()


# ── PyTorch Model ─────────────────────────────────────────────────────────────
class ANNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super(ANNClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 ANN Explorer")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 EDA", "🏗️ Build & Train", "🔮 Live Prediction", "📐 Architecture Study"],
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption(f"Breast Cancer Wisconsin")
    st.caption(f"Samples: **{df.shape[0]}** | Features: **{df.shape[1] - 1}**")
    st.caption(f"Task: Binary Classification")

# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 EDA":
    st.markdown(
        '<span class="badge">Exploratory Data Analysis</span>', unsafe_allow_html=True
    )
    st.title("Dataset Exploration")

    tab1, tab2 = st.tabs(["Overview", "Feature Distributions"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Dataset Preview**")
            st.dataframe(df.head(10), use_container_width=True)
        with c2:
            st.markdown("**Class Distribution**")
            counts = df["target"].value_counts()
            fig, ax = dfig(8, 4)
            colors = ["#f87171", "#60a5fa"]
            bars = ax.bar(
                ["Malignant", "Benign"],
                counts.values,
                color=colors,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.set_title("Class Distribution", color="#e2e8f0")
            ax.set_ylabel("Count")
            for bar in bars:
                yval = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval + 5,
                    int(yval),
                    ha="center",
                    color="white",
                    fontweight="bold",
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab2:
        st.markdown("**Top Discriminative Features**")
        # Calculate feature importance by mean difference
        feature_scores = (
            df.drop("target", axis=1)
            .apply(
                lambda col: (
                    abs(col[df.target == 0].mean() - col[df.target == 1].mean())
                    / col.std()
                )
            )
            .nlargest(6)
            .index.tolist()
        )

        fig, axes = dfig(16, 8, 3, 2)
        axes = axes.flatten()
        for i, feat in enumerate(feature_scores):
            axes[i].hist(
                df[df.target == 0][feat],
                bins=20,
                alpha=0.7,
                color="#f87171",
                label="Malignant",
                edgecolor="none",
            )
            axes[i].hist(
                df[df.target == 1][feat],
                bins=20,
                alpha=0.7,
                color="#60a5fa",
                label="Benign",
                edgecolor="none",
            )
            axes[i].set_title(
                feat[:25], color="#e2e8f0", fontsize=10, fontweight="bold"
            )
            axes[i].set_facecolor(DARK_AX)
            axes[i].tick_params(colors=TEXT)
            axes[i].legend(
                fontsize=8, labelcolor="white", facecolor=DARK_AX, edgecolor=GRID
            )
        plt.suptitle(
            "Top 6 Discriminative Features",
            color="#e2e8f0",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# BUILD & TRAIN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Build & Train":
    st.markdown('<span class="badge">Build & Train ANN</span>', unsafe_allow_html=True)
    st.title("Build & Train Artificial Neural Network")
    st.markdown(
        '<div class="theory-box">Configure the ANN architecture. The model will learn non-linear decision boundaries using stacked Dense layers, Batch Normalization, and Dropout for robust generalization.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Architecture**")
        n_layers = st.slider("Hidden Layers", 1, 4, 2)
        hidden_dims = []
        for i in range(n_layers):
            dim = st.select_slider(
                f"Layer {i + 1} Neurons",
                options=[8, 16, 32, 64, 128, 256],
                value=[64, 32, 16, 8][i],
            )
            hidden_dims.append(dim)

        dropout = st.slider("Dropout Rate", 0.0, 0.6, 0.3, 0.05)
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
                X = df.drop("target", axis=1).values.astype(np.float32)
                y = df["target"].values.astype(np.float32)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

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
                model = ANNClassifier(
                    input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout
                ).to(device)

                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                train_losses, val_accs = [], []
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

                    model.eval()
                    val_correct, val_total = 0, 0
                    with torch.no_grad():
                        for bx, by in test_loader:
                            bx, by = bx.to(device), by.to(device)
                            outputs = model(bx)
                            preds = (outputs >= 0.5).float()
                            val_correct += (preds == by).sum().item()
                            val_total += bx.size(0)

                    val_accs.append(val_correct / val_total)
                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(
                        f"Epoch {epoch + 1}/{epochs} | Val Acc: {val_accs[-1]:.3f}"
                    )

                progress_bar.empty()
                status_text.empty()

                # Final Evaluation
                model.eval()
                all_preds, all_targets, all_probs = [], [], []
                with torch.no_grad():
                    for bx, by in test_loader:
                        bx = bx.to(device)
                        outputs = model(bx)
                        probs = outputs.cpu().numpy().flatten()
                        preds = (outputs >= 0.5).float().cpu().numpy().flatten()
                        all_preds.extend(preds)
                        all_targets.extend(by.numpy().flatten())
                        all_probs.extend(probs)

                acc = accuracy_score(all_targets, all_preds)
                prec = precision_score(all_targets, all_preds, zero_division=0)
                rec = recall_score(all_targets, all_preds, zero_division=0)
                f1 = f1_score(all_targets, all_preds, zero_division=0)

                mchips(acc, prec, rec, f1)

                fig, axes = dfig(14, 5, 2)
                axes = np.array(axes).flatten()

                axes[0].plot(train_losses, color="#f87171", lw=2, label="Train Loss")
                axes[0].set_title("Training Loss (Binary Crossentropy)")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Loss")
                axes[0].legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)

                axes[1].plot(val_accs, color="#60a5fa", lw=2, label="Val Accuracy")
                axes[1].set_title("Validation Accuracy")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Accuracy")
                axes[1].legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                fig, ax = dfig(8, 6)
                cm = confusion_matrix(all_targets, all_preds)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=ax,
                    xticklabels=CLASSES,
                    yticklabels=CLASSES,
                    linewidths=1,
                    linecolor="white",
                    annot_kws={"color": "white", "size": 12},
                )
                ax.set_title("Confusion Matrix", color="#e2e8f0")
                ax.set_facecolor(DARK_AX)
                ax.tick_params(colors=TEXT)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.session_state["ann_model"] = model.cpu()
                st.session_state["ann_scaler"] = scaler
                st.session_state["ann_feature_names"] = list(raw_data.feature_names)
                st.success("Model trained and saved to session!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.info("👈 Configure the network on the left and click **Train Model**.")
            st.markdown("**Planned Architecture:**")
            arch_lines = ["```", f"Input  ({df.shape[1] - 1} features)"]
            for dim in hidden_dims:
                arch_lines += [
                    f"  Linear({dim}) → BatchNorm → ReLU → Dropout({dropout})"
                ]
            arch_lines += [
                "  Linear(1)",
                "  Sigmoid()",
                "Output (Probability of Benign)",
                "```",
            ]
            st.markdown("\n".join(arch_lines))

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Prediction":
    st.markdown('<span class="badge">Live Prediction</span>', unsafe_allow_html=True)
    st.title("Predict — Single Sample")
    st.markdown(
        '<div class="theory-box">Adjust feature sliders to simulate a patient sample. The ANN will output the probability of the tumor being Benign.</div>',
        unsafe_allow_html=True,
    )

    try:
        if "ann_model" not in st.session_state:
            with st.spinner("Training default model for prediction..."):
                X = df.drop("target", axis=1).values.astype(np.float32)
                y = df["target"].values.astype(np.float32)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                train_dataset = TensorDataset(
                    torch.tensor(X_scaled), torch.tensor(y).unsqueeze(1)
                )
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

                model = ANNClassifier(
                    input_dim=X.shape[1], hidden_dims=[64, 32], dropout=0.3
                )
                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(80):
                    model.train()
                    for bx, by in train_loader:
                        optimizer.zero_grad()
                        outputs = model(bx)
                        loss = criterion(outputs, by)
                        loss.backward()
                        optimizer.step()

                st.session_state["ann_model"] = model
                st.session_state["ann_scaler"] = scaler
                st.session_state["ann_feature_names"] = list(raw_data.feature_names)

        model = st.session_state["ann_model"]
        scaler = st.session_state["ann_scaler"]
        feature_names = st.session_state["ann_feature_names"]

        if st.button("🎲 Load Random Test Sample", use_container_width=True):
            idx = np.random.randint(0, len(df))
            row = df.iloc[idx]
            st.session_state["sample_idx"] = idx
            st.session_state["sample_true"] = (
                "Benign" if row["target"] == 1 else "Malignant"
            )
            st.session_state["sample_features"] = row[feature_names].to_dict()

        if "sample_features" in st.session_state:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("**Feature Values**")
                input_data = {}
                for feat in feature_names[
                    :10
                ]:  # Show first 10 for brevity, rest default to median
                    val = st.session_state["sample_features"][feat]
                    mn, mx = float(df[feat].min()), float(df[feat].max())
                    input_data[feat] = st.slider(
                        feat, mn, mx, val, step=(mx - mn) / 100, format="%.4f"
                    )

                st.caption(
                    "*Showing first 10 features. Remaining features use sample defaults.*"
                )
                for feat in feature_names[10:]:
                    input_data[feat] = st.session_state["sample_features"][feat]

            with c2:
                if st.button("🧠 Run Prediction", use_container_width=True):
                    inp = np.array([[input_data[f] for f in feature_names]]).astype(
                        np.float32
                    )
                    inp_scaled = scaler.transform(inp)

                    model.eval()
                    with torch.no_grad():
                        prob = float(model(torch.tensor(inp_scaled)).item())

                    cls = "Benign" if prob >= 0.5 else "Malignant"
                    color = "#60a5fa" if cls == "Benign" else "#f87171"
                    true_color = (
                        "#34d399"
                        if st.session_state["sample_true"] == cls
                        else "#f87171"
                    )

                    st.markdown(
                        f"""
                    <div class="pred-box">
                      <div style="color:#94a3b8;font-size:.85rem;text-transform:uppercase;letter-spacing:.15em">Prediction</div>
                      <div class="pred-val" style="color:{color}">{cls}</div>
                      <div style="color:#94a3b8;margin-top:.5rem">Benign Probability: <b style="color:#818cf8">{prob:.4f}</b></div>
                      <div style="color:{true_color};margin-top:.5rem;font-size:0.9rem">True Label: {st.session_state["sample_true"]}</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )

                    # Probability gauge
                    fig, ax = dfig(10, 2.5)
                    ax.barh(
                        [0], [prob], color="#60a5fa", height=0.4, label="Benign Prob"
                    )
                    ax.barh(
                        [0],
                        [1 - prob],
                        left=[prob],
                        color="#f87171",
                        height=0.4,
                        label="Malignant Prob",
                    )
                    ax.axvline(0.5, color="white", lw=1.5, linestyle="--", alpha=0.7)
                    ax.set_xlim(0, 1)
                    ax.set_yticks([])
                    ax.set_xlabel("Probability")
                    ax.set_title(f"Prediction: {cls} (p={prob:.4f})")
                    ax.legend(
                        labelcolor="white",
                        facecolor=DARK_AX,
                        edgecolor=GRID,
                        loc="lower right",
                    )
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
    st.title("ANN Architecture & Concepts")

    tab1, tab2 = st.tabs(["🔬 Dense Layer Mechanics", "📉 Activation Functions"])

    with tab1:
        st.markdown("**How a Dense (Fully Connected) Layer Works**")
        st.markdown(
            '<div class="theory-box">Every neuron in a Dense layer is connected to every neuron in the previous layer. It computes a weighted sum of its inputs, adds a bias, and passes the result through an activation function: <b>output = activation(W·x + b)</b>.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Visualizing a Dense Layer Connection**")
        fig, ax = dfig(12, 6)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Input nodes
        for i in range(3):
            ax.add_patch(
                plt.Circle(
                    (2, 7 - i * 2), 0.4, facecolor="#1e293b", edgecolor="#60a5fa", lw=2
                )
            )
            ax.text(
                2,
                7 - i * 2,
                f"$x_{i + 1}$",
                ha="center",
                va="center",
                color="white",
                fontsize=12,
                fontweight="bold",
            )

        # Output nodes
        for i in range(2):
            ax.add_patch(
                plt.Circle(
                    (8, 6 - i * 3), 0.4, facecolor="#1e293b", edgecolor="#f87171", lw=2
                )
            )
            ax.text(
                8,
                6 - i * 3,
                f"$h_{i + 1}$",
                ha="center",
                va="center",
                color="white",
                fontsize=12,
                fontweight="bold",
            )

        # Connections
        for i in range(3):
            for j in range(2):
                ax.plot(
                    [2.4, 7.6], [7 - i * 2, 6 - j * 3], color="#3a3f5c", lw=1, alpha=0.6
                )

        ax.text(
            5,
            8,
            "Weights (W) & Bias (b)",
            ha="center",
            color="#e2e8f0",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            5,
            2,
            "Dense (Fully Connected) Layer",
            ha="center",
            color="#e2e8f0",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown("**Visualise how different activation functions transform inputs**")
        x = np.linspace(-5, 5, 300)
        funcs = {
            "ReLU": np.maximum(0, x),
            "Sigmoid": 1 / (1 + np.exp(-x)),
            "Tanh": np.tanh(x),
            "Leaky ReLU": np.where(x > 0, x, 0.01 * x),
        }
        colors_act = ["#818cf8", "#f87171", "#34d399", "#fbbf24"]

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
            "ReLU": np.where(x > 0, 1, 0),
            "Sigmoid": (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x))),
            "Tanh": 1 - np.tanh(x) ** 2,
            "Leaky ReLU": np.where(x > 0, 1, 0.01),
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
    "<center style='color:#4a5568;font-size:.78rem'>Artificial Neural Network (ANN) Explorer &nbsp;|&nbsp; PyTorch · Scikit-learn</center>",
    unsafe_allow_html=True,
)
