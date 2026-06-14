import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CNN Explorer",
    page_icon="👁️",
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
CLASSES = ["Circle", "Square", "Triangle"]


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
    df = pd.read_csv("data/shapes_dataset.csv")
    return df


df = load_data()


# ── PyTorch Model ─────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👁️ CNN Explorer")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 EDA", "🏗️ Build & Train", "🔮 Live Prediction", "📐 Architecture Study"],
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption(f"Synthetic 28x28 Grayscale Images")
    st.caption(f"Samples: **{df.shape[0]}** | Classes: **3**")
    st.caption(f"Task: Image Classification")

# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 EDA":
    st.markdown(
        '<span class="badge">Exploratory Data Analysis</span>', unsafe_allow_html=True
    )
    st.title("Dataset Exploration")

    tab1, tab2 = st.tabs(["Overview", "Sample Images"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Class Distribution**")
            counts = df["label"].value_counts().sort_index()
            fig, ax = dfig(8, 4)
            colors = ["#06b6d4", "#f59e0b", "#ec4899"]
            bars = ax.bar(
                [CLASSES[i] for i in counts.index],
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
                    yval + 10,
                    int(yval),
                    ha="center",
                    color="white",
                    fontweight="bold",
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with c2:
            st.markdown("**Dataset Statistics**")
            pixel_cols = [c for c in df.columns if c.startswith("pixel_")]
            st.dataframe(df[pixel_cols].describe().round(2).T, use_container_width=True)

    with tab2:
        st.markdown("**Random Sample Images**")
        st.markdown(
            '<div class="theory-box">Each image is a 28x28 pixel grid. CNNs excel at detecting local patterns (like edges and corners) in these grids using convolutional filters, making them vastly superior to standard dense networks for image tasks.</div>',
            unsafe_allow_html=True,
        )

        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        fig.patch.set_facecolor(DARK_FIG)
        axes = axes.flatten()
        sample_idx = np.random.choice(df.index, 10, replace=False)

        for i, idx in enumerate(sample_idx):
            row = df.iloc[idx]
            img = row[pixel_cols].values.reshape(28, 28)
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(CLASSES[row["label"]], color="#e2e8f0", fontsize=10)
            axes[i].axis("off")
            for sp in axes[i].spines.values():
                sp.set_edgecolor(GRID)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# BUILD & TRAIN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Build & Train":
    st.markdown('<span class="badge">Build & Train CNN</span>', unsafe_allow_html=True)
    st.title("Build & Train Convolutional Neural Network")
    st.markdown(
        '<div class="theory-box">Configure the CNN architecture. Convolutional layers extract spatial features, while pooling layers reduce dimensionality and provide translation invariance.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Architecture**")
        filters1 = st.select_slider("Conv1 Filters", options=[8, 16, 32, 64], value=16)
        filters2 = st.select_slider(
            "Conv2 Filters", options=[16, 32, 64, 128], value=32
        )
        dense_units = st.select_slider(
            "Dense Layer Units", options=[32, 64, 128, 256], value=64
        )
        dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.3, 0.05)
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
                pixel_cols = [c for c in df.columns if c.startswith("pixel_")]
                X = (
                    df[pixel_cols].values.reshape(-1, 1, 28, 28).astype(np.float32)
                    / 255.0
                )
                y = df["label"].values

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                train_dataset = TensorDataset(
                    torch.tensor(X_train), torch.tensor(y_train)
                )
                test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

                train_loader = DataLoader(
                    train_dataset, batch_size=batch_sz, shuffle=True
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=batch_sz, shuffle=False
                )

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Dynamic model building based on user input
                class DynamicCNN(nn.Module):
                    def __init__(self, f1, f2, d_units, drop):
                        super(DynamicCNN, self).__init__()
                        self.conv1 = nn.Conv2d(1, f1, kernel_size=3, padding=1)
                        self.pool1 = nn.MaxPool2d(2, 2)
                        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
                        self.pool2 = nn.MaxPool2d(2, 2)
                        self.flatten = nn.Flatten()
                        self.fc1 = nn.Linear(f2 * 7 * 7, d_units)
                        self.dropout = nn.Dropout(drop)
                        self.fc2 = nn.Linear(d_units, 3)

                    def forward(self, x):
                        x = self.pool1(torch.relu(self.conv1(x)))
                        x = self.pool2(torch.relu(self.conv2(x)))
                        x = self.flatten(x)
                        x = torch.relu(self.fc1(x))
                        x = self.dropout(x)
                        return self.fc2(x)

                model = DynamicCNN(filters1, filters2, dense_units, dropout).to(device)
                criterion = nn.CrossEntropyLoss()
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
                            _, preds = torch.max(outputs, 1)
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
                all_preds, all_targets = [], []
                with torch.no_grad():
                    for bx, by in test_loader:
                        bx = bx.to(device)
                        outputs = model(bx)
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(by.cpu().numpy())

                acc = accuracy_score(all_targets, all_preds)
                report = classification_report(
                    all_targets, all_preds, target_names=CLASSES, output_dict=True
                )
                prec = report["macro avg"]["precision"]
                rec = report["macro avg"]["recall"]
                f1 = report["macro avg"]["f1-score"]

                mchips(acc, prec, rec, f1)

                fig, axes = dfig(14, 5, 2)
                axes = np.array(axes).flatten()
                axes[0].plot(train_losses, color="#f87171", lw=2, label="Train Loss")
                axes[0].set_title("Training Loss (Cross Entropy)")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Loss")
                axes[0].legend(labelcolor="white", facecolor=DARK_AX, edgecolor=GRID)

                axes[1].plot(val_accs, color="#06b6d4", lw=2, label="Val Accuracy")
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

                st.session_state["cnn_model"] = model.cpu()
                st.session_state["cnn_f1"] = filters1
                st.session_state["cnn_f2"] = filters2
                st.session_state["cnn_d"] = dense_units
                st.session_state["cnn_drop"] = dropout
                st.success("Model trained and saved to session!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.info("👈 Configure the network on the left and click **Train Model**.")

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Prediction":
    st.markdown(
        '<span class="badge">Live Image Classification</span>', unsafe_allow_html=True
    )
    st.title("Classify an Image")
    st.markdown(
        '<div class="theory-box">The trained CNN will analyze the spatial hierarchies of features in the image to predict its class.</div>',
        unsafe_allow_html=True,
    )

    try:
        if "cnn_model" not in st.session_state:
            with st.spinner("Training default model for prediction..."):
                pixel_cols = [c for c in df.columns if c.startswith("pixel_")]
                X = (
                    df[pixel_cols].values.reshape(-1, 1, 28, 28).astype(np.float32)
                    / 255.0
                )
                y = df["label"].values
                X_train, _, y_train, _ = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                train_dataset = TensorDataset(
                    torch.tensor(X_train), torch.tensor(y_train)
                )
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

                model = SimpleCNN().to("cpu")
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(25):
                    model.train()
                    for bx, by in train_loader:
                        optimizer.zero_grad()
                        outputs = model(bx)
                        loss = criterion(outputs, by)
                        loss.backward()
                        optimizer.step()

                st.session_state["cnn_model"] = model
                st.session_state["cnn_f1"] = 16
                st.session_state["cnn_f2"] = 32
                st.session_state["cnn_d"] = 64
                st.session_state["cnn_drop"] = 0.3

        model = st.session_state["cnn_model"]
        pixel_cols = [c for c in df.columns if c.startswith("pixel_")]

        if st.button("🎲 Load Random Test Image", use_container_width=True):
            idx = np.random.randint(0, len(df))
            row = df.iloc[idx]
            true_label = CLASSES[row["label"]]
            img = (
                row[pixel_cols].values.reshape(1, 1, 28, 28).astype(np.float32) / 255.0
            )

            st.session_state["test_img"] = img
            st.session_state["test_true"] = true_label
            st.session_state["test_display"] = row[pixel_cols].values.reshape(28, 28)

        if "test_img" in st.session_state:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("**Input Image**")
                fig, ax = plt.subplots(figsize=(4, 4))
                fig.patch.set_facecolor(DARK_FIG)
                ax.imshow(st.session_state["test_display"], cmap="gray")
                ax.set_title(
                    f"True Label: {st.session_state['test_true']}", color="#e2e8f0"
                )
                ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with c2:
                if st.button("🔮 Predict Class", use_container_width=True):
                    model.eval()
                    with torch.no_grad():
                        tensor_img = torch.tensor(st.session_state["test_img"])
                        outputs = model(tensor_img)
                        probs = torch.softmax(outputs, dim=1).numpy().flatten()
                        pred_idx = np.argmax(probs)
                        pred_label = CLASSES[pred_idx]
                        conf = probs[pred_idx]

                    st.markdown(
                        f"""
                    <div class="pred-box">
                      <div style="color:#94a3b8;font-size:.85rem;text-transform:uppercase;letter-spacing:.15em">Prediction</div>
                      <div class="pred-val" style="color:#06b6d4">{pred_label}</div>
                      <div style="color:#94a3b8;margin-top:.5rem">Confidence: <b style="color:#06b6d4">{conf:.4f}</b></div>
                    </div>""",
                        unsafe_allow_html=True,
                    )

                    fig, ax = dfig(8, 4)
                    colors = [
                        "#06b6d4" if i == pred_idx else "#3a3f5c" for i in range(3)
                    ]
                    bars = ax.barh(
                        CLASSES[::-1],
                        probs[::-1],
                        color=colors[::-1],
                        edgecolor="white",
                    )
                    ax.set_xlim(0, 1)
                    ax.set_title("Class Probabilities", color="#e2e8f0")
                    ax.set_facecolor(DARK_AX)
                    ax.tick_params(colors=TEXT)
                    for bar, p in zip(bars, probs[::-1]):
                        ax.text(
                            p + 0.02,
                            bar.get_y() + bar.get_height() / 2,
                            f"{p * 100:.1f}%",
                            va="center",
                            color=TEXT,
                            fontsize=9,
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
    st.markdown('<span class="badge">CNN Architecture</span>', unsafe_allow_html=True)
    st.title("How Convolutional Neural Networks Work")

    tab1, tab2 = st.tabs(["🔬 Convolution & Pooling", "📊 CNN vs Dense Networks"])

    with tab1:
        st.markdown("**The Core Operations of a CNN**")
        st.markdown(
            '<div class="theory-box">Unlike dense networks that treat every pixel independently, CNNs use <b>Convolutional Filters</b> to scan the image, detecting local patterns like edges, corners, and textures. <b>Pooling</b> then shrinks the feature maps, making the model robust to small shifts in the image.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("""
        | Operation | Purpose | Visual Effect |
        |---|---|---|
        | **Convolution (Conv2D)** | Applies learnable filters (kernels) to extract features (edges, shapes). | Produces "Feature Maps" highlighting specific patterns. |
        | **Activation (ReLU)** | Introduces non-linearity, allowing the network to learn complex boundaries. | Sets all negative pixel values to zero. |
        | **Max Pooling** | Downsamples the feature map by taking the maximum value in a window (e.g., 2x2). | Reduces spatial dimensions by half, retaining the strongest signals. |
        """)

        st.markdown("**Visualizing a 3x3 Convolution Step**")
        fig, ax = dfig(10, 4)
        ax.axis("off")

        # Input patch
        input_patch = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        kernel = np.array(
            [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
        )  # Vertical edge detector

        ax.text(1, 3, "Input Patch", ha="center", color="#e2e8f0", fontweight="bold")
        for i in range(3):
            for j in range(3):
                ax.add_patch(
                    plt.Rectangle(
                        (j, 2 - i), 1, 1, facecolor="#1e293b", edgecolor="#06b6d4", lw=2
                    )
                )
                ax.text(
                    j + 0.5,
                    2.5 - i,
                    str(input_patch[i, j]),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                )

        ax.text(
            4.5,
            2.5,
            "×",
            ha="center",
            va="center",
            color="#f59e0b",
            fontsize=24,
            fontweight="bold",
        )

        ax.text(
            6, 3, "Filter (Kernel)", ha="center", color="#e2e8f0", fontweight="bold"
        )
        for i in range(3):
            for j in range(3):
                ax.add_patch(
                    plt.Rectangle(
                        (j + 5, 2 - i),
                        1,
                        1,
                        facecolor="#1e293b",
                        edgecolor="#f59e0b",
                        lw=2,
                    )
                )
                ax.text(
                    j + 5.5,
                    2.5 - i,
                    str(kernel[i, j]),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                )

        ax.text(
            9.5,
            2.5,
            "=",
            ha="center",
            va="center",
            color="#e2e8f0",
            fontsize=24,
            fontweight="bold",
        )

        result = np.sum(input_patch * kernel)
        ax.add_patch(
            plt.Rectangle(
                (10, 2), 1.5, 1.5, facecolor="#1e293b", edgecolor="#10b981", lw=2
            )
        )
        ax.text(
            10.75,
            2.75,
            str(result),
            ha="center",
            va="center",
            color="#10b981",
            fontsize=14,
            fontweight="bold",
        )
        ax.text(10.75, 4, "Output", ha="center", color="#e2e8f0", fontweight="bold")

        ax.set_xlim(-0.5, 12)
        ax.set_ylim(0, 4.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown("**Why CNNs Dominate Image Tasks**")
        st.markdown(
            '<div class="theory-box">A standard Dense (Fully Connected) network flattens the image, destroying all 2D spatial relationships. A 28x28 image becomes 784 independent inputs. CNNs preserve the 2D structure.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("""
        | Feature | Dense (MLP) Network | Convolutional Neural Network (CNN) |
        |---|---|---|
        | **Input Structure** | Flattened 1D array (loses spatial info) | 2D/3D Grid (preserves spatial relationships) |
        | **Parameters** | Massive (e.g., 784 × 128 = 100k+ for first layer) | Shared weights (e.g., 3×3 filter = 9 weights per channel) |
        | **Translation Invariance** | Poor (must relearn patterns in new locations) | **Excellent** (Pooling and sliding filters handle shifts) |
        | **Feature Learning** | Learns global pixel combinations | Learns hierarchical local features (edges → shapes → objects) |
        """)

st.markdown("---")
st.markdown(
    "<center style='color:#4a5568;font-size:.78rem'>Convolutional Neural Network (CNN) Explorer &nbsp;|&nbsp; PyTorch · Scikit-learn</center>",
    unsafe_allow_html=True,
)
