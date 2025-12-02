import os
import gzip
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from config import WINDOW_SIZE
from build_datasets import read_fasta_sequences, one_hot_encode



# Configuration (easy to change)


# Meta and fasta come from legacy windows (coordinates and sequences are correct)
META_PATH = os.path.join("legacy", "output_dnaShapeR", "output", "ctcf_windows_meta.tsv")
FASTA_PATH = os.path.join("legacy", "output_dnaShapeR", "output", "ctcf_windows.fa")

# chr1 wig files (MGW, ProT, Roll, Buckle, Opening)
WIG_DIR = os.path.join("data", "download")
WIG_FILES = {
    "MGW": "hg19.chr1.MGW.2nd.wig.gz",
    "ProT": "hg19.chr1.ProT.2nd.wig.gz",
    "Roll": "hg19.chr1.Roll.2nd.wig.gz",
    "Buckle": "hg19.chr1.Buckle.wig.gz",
    "Opening": "hg19.chr1.Opening.wig.gz",
}

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Small test mode: limit number of windows per class (before motif matching)
USE_SMALL_TEST = False
MAX_POS = 1500
MAX_NEG = 1500

# Optional: use motif-matched negatives (consensus-based CTCF motif)
USE_MOTIF_MATCHED = True

# Plot behavior
SAVE_PLOTS = False
SHOW_PLOTS = True



# Load windows and subset chr1


def load_chr1_windows() -> Tuple[pd.DataFrame, List[int]]:
    print("Loading window metadata from:", META_PATH)
    meta = pd.read_csv(META_PATH, sep="\t")
    meta["row"] = np.arange(len(meta), dtype=np.int64)
    chr1_meta = meta[meta["chrom"] == "chr1"].copy()
    chr1_meta = chr1_meta.reset_index(drop=True)
    print("Total windows on chr1:", len(chr1_meta))
    return chr1_meta, meta["row"].tolist()


def load_chr1_sequences(chr1_meta: pd.DataFrame) -> List[str]:
    print("Loading all sequences from fasta:", FASTA_PATH)
    seqs = read_fasta_sequences(FASTA_PATH)
    if len(seqs) != chr1_meta["row"].max() + 1:
        print("Warning: unexpected number of sequences vs meta rows.")
    idx = chr1_meta["row"].values.astype(int)
    chr1_seqs = [seqs[i] for i in idx]
    return chr1_seqs


def subsample_balanced(
    chr1_meta: pd.DataFrame,
) -> pd.DataFrame:
    if not USE_SMALL_TEST:
        return chr1_meta
    pos = chr1_meta[chr1_meta["label"] == 1]
    neg = chr1_meta[chr1_meta["label"] == 0]
    n_pos = min(MAX_POS, len(pos))
    n_neg = min(MAX_NEG, len(neg))
    pos_sample = pos.sample(n=n_pos, random_state=0)
    neg_sample = neg.sample(n=n_neg, random_state=1)
    subset = pd.concat([pos_sample, neg_sample], axis=0).sort_values("start").reset_index(drop=True)
    print("Subsampled windows (small test):", len(subset))
    print("  pos:", (subset["label"] == 1).sum(), "neg:", (subset["label"] == 0).sum())
    return subset


def consensus_motif_score(seq: str, motif: str) -> int:
    """Return the maximum consensus match score of motif on seq.

    motif may contain 'N' characters that match any base.
    Score is the number of matching positions in the best alignment.
    """
    L = len(motif)
    if len(seq) < L:
        return 0
    best = 0
    for i in range(len(seq) - L + 1):
        score = 0
        for j, m in enumerate(motif):
            b = seq[i + j]
            if m == "N":
                score += 1
            elif b == m:
                score += 1
        if score > best:
            best = score
    return best


def apply_motif_matched_filter(
    chr1_meta: pd.DataFrame,
    chr1_seqs: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Filter windows to keep high-scoring motif-like positives and negatives.

    This approximates motif-matched negatives using a simple CTCF consensus motif.
    """
    print("Applying motif-matched filter (consensus-based)")
    motif = "CCGCGNGGNGGCAG"  # CTCF consensus, N matches any base
    scores = np.array([consensus_motif_score(s, motif) for s in chr1_seqs], dtype=np.int16)

    chr1_meta = chr1_meta.copy()
    chr1_meta["motif_score"] = scores

    labels = chr1_meta["label"].values.astype(int)
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_scores = scores[pos_mask]
    if pos_scores.size == 0:
        print("No positive windows available for motif scoring; skipping filter.")
        return chr1_meta, chr1_seqs

    # Use 80th percentile of positive scores as threshold for "motif-like"
    thresh = float(np.percentile(pos_scores, 80.0))
    thresh_int = int(np.floor(thresh))
    print("Motif score threshold (80th percentile of positives):", thresh_int)

    keep_pos = pos_mask & (scores >= thresh_int)
    keep_neg = neg_mask & (scores >= thresh_int)

    n_pos = int(keep_pos.sum())
    n_neg = int(keep_neg.sum())
    print("Windows passing motif filter:")
    print("  positives:", n_pos)
    print("  negatives:", n_neg)

    if n_pos == 0 or n_neg == 0:
        print("Not enough motif-like positives or negatives; skipping filter.")
        return chr1_meta, chr1_seqs

    # Balance negatives to have similar count as positives
    if n_neg > n_pos:
        neg_idx_all = np.where(keep_neg)[0]
        rng = np.random.RandomState(0)
        chosen_neg_idx = rng.choice(neg_idx_all, size=n_pos, replace=False)
        keep_balanced_neg = np.zeros_like(keep_neg)
        keep_balanced_neg[chosen_neg_idx] = True
        keep_mask = keep_pos | keep_balanced_neg
        print("Downsampled negatives to match positives:", int(keep_balanced_neg.sum()))
    else:
        keep_mask = keep_pos | keep_neg

    chr1_meta_filtered = chr1_meta.loc[keep_mask].reset_index(drop=True)
    chr1_seqs_filtered = [s for i, s in enumerate(chr1_seqs) if keep_mask[i]]

    print("Motif-matched dataset size:", len(chr1_meta_filtered))
    print("  positives:", int((chr1_meta_filtered["label"] == 1).sum()))
    print("  negatives:", int((chr1_meta_filtered["label"] == 0).sum()))

    return chr1_meta_filtered, chr1_seqs_filtered



# Read wig files only at needed positions


def collect_needed_positions(meta: pd.DataFrame) -> List[int]:
    positions: List[int] = []
    for start in meta["start"].tolist():
        s = int(start)
        for j in range(WINDOW_SIZE):
            positions.append(s + 1 + j)  # wig is 1-based
    return sorted(set(positions))


def load_wig_values(path: str, needed_positions: List[int]) -> Dict[int, float]:
    needed = set(needed_positions)
    values: Dict[int, float] = {}
    print("  Reading wig:", path)
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in ("t", "v", "#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            pos = int(parts[0])
            if pos not in needed:
                continue
            try:
                val = float(parts[1])
            except ValueError:
                continue
            values[pos] = val
    print("  Collected positions:", len(values))
    return values


def build_shape_matrix(
    meta: pd.DataFrame,
    wig_maps: Dict[str, Dict[int, float]],
) -> np.ndarray:
    n = len(meta)
    feature_names = sorted(wig_maps.keys())
    n_feat = len(feature_names)
    shape = np.zeros((n, n_feat * WINDOW_SIZE), dtype=np.float32)
    for i, row in tqdm(meta.iterrows(), total=n, desc="Building shape matrix"):
        start = int(row["start"])
        vec_parts: List[float] = []
        for fname in feature_names:
            fmap = wig_maps[fname]
            for j in range(WINDOW_SIZE):
                pos = start + 1 + j
                v = fmap.get(pos, 0.0)
                vec_parts.append(float(v))
        shape[i, :] = np.asarray(vec_parts, dtype=np.float32)
    return shape



# Split, standardize, and simple plots


def split_by_coordinate(meta: pd.DataFrame) -> Dict[str, np.ndarray]:
    starts = meta["start"].values
    order = np.argsort(starts)
    n = len(meta)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    idx_train = order[:n_train]
    idx_val = order[n_train : n_train + n_val]
    idx_test = order[n_train + n_val :]
    return {"train": idx_train, "val": idx_val, "test": idx_test}


def standardize_shape(x_shape: np.ndarray, idx_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_vals = x_shape[idx_train]
    mean = train_vals.mean(axis=0)
    std = train_vals.std(axis=0)
    std[std == 0.0] = 1.0
    x_norm = (x_shape - mean) / std
    return x_norm, mean, std


def plot_gc_hist(gc: np.ndarray, labels: np.ndarray) -> None:
    pos_gc = gc[labels == 1]
    neg_gc = gc[labels == 0]

    print("GC content summary (wig dataset, chr1):")
    print(f"  positives: n={len(pos_gc)}, mean={pos_gc.mean():.4f}, std={pos_gc.std():.4f}")
    print(f"  negatives: n={len(neg_gc)}, mean={neg_gc.mean():.4f}, std={neg_gc.std():.4f}")
    print(f"  mean difference (pos - neg): {pos_gc.mean() - neg_gc.mean():.4f}")

    plt.figure(figsize=(5, 4))
    plt.hist(pos_gc, bins=40, alpha=0.5, label="positive", density=True)
    plt.hist(neg_gc, bins=40, alpha=0.5, label="negative", density=True)
    plt.xlabel("GC content")
    plt.ylabel("Density")
    plt.title("GC content (wig dataset, chr1)")
    plt.legend()
    if 1:
        path = os.path.join(OUTPUT_DIR, "wig_gc_hist.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        print("Saved GC histogram to:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def compute_gc(x_seq: np.ndarray) -> np.ndarray:
    c = x_seq[:, 1, :].sum(axis=1)
    g = x_seq[:, 2, :].sum(axis=1)
    total = x_seq.sum(axis=(1, 2)) + 1e-8
    return (c + g) / total



# Models and evaluation


def train_shape_baselines(x_shape: np.ndarray, y: np.ndarray, splits: Dict[str, np.ndarray]) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    idx_train = splits["train"]
    idx_val = splits["val"]
    idx_test = splits["test"]

    x_train = x_shape[idx_train]
    y_train = y[idx_train]
    x_test = x_shape[idx_test]
    y_test = y[idx_test]

    print("Training Logistic Regression (shape-only, wig)")
    t0 = time.perf_counter()
    logreg = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
    logreg.fit(x_train, y_train)
    proba_log = logreg.predict_proba(x_test)[:, 1]
    t1 = time.perf_counter()
    logreg_time = t1 - t0
    print(f"  LogReg training + inference time: {logreg_time:.2f} s")

    print("Training SVM RBF (shape-only, wig)")
    t0 = time.perf_counter()
    svm = SVC(kernel="rbf", probability=True)
    svm.fit(x_train, y_train)
    proba_svm = svm.predict_proba(x_test)[:, 1]
    t1 = time.perf_counter()
    svm_time = t1 - t0
    print(f"  SVM RBF training + inference time: {svm_time:.2f} s")

    def eval_model(name: str, proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        auroc = auc(fpr, tpr)
        auprc = auc(rec, prec)
        print(f"{name:10s} AUROC={auroc:.6f}  AUPRC={auprc:.6f}")
        return fpr, tpr, auroc, rec, prec, auprc

    fpr_l, tpr_l, auroc_l, rec_l, prec_l, auprc_l = eval_model("LogReg", proba_log)
    fpr_s, tpr_s, auroc_s, rec_s, prec_s, auprc_s = eval_model("SVM RBF", proba_svm)

    # ROC
    plt.figure(figsize=(5, 4))
    plt.plot(fpr_l, tpr_l, label=f"LogReg (AUROC={auroc_l:.3f})")
    plt.plot(fpr_s, tpr_s, label=f"SVM RBF (AUROC={auroc_s:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Shape-only models (wig, chr1) ROC")
    plt.legend()
    if SAVE_PLOTS:
        path = os.path.join(OUTPUT_DIR, "wig_shape_only_roc.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        print("Saved shape-only ROC to:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    print("Shape-only models summary (wig, chr1 test):")
    print(f"  LogReg  : AUROC={auroc_l:.4f}, AUPRC={auprc_l:.4f}, time={logreg_time:.2f}s")
    print(f"  SVM RBF : AUROC={auroc_s:.4f}, AUPRC={auprc_s:.4f}, time={svm_time:.2f}s")


def train_seq_flat_baselines(x_seq: np.ndarray, y: np.ndarray, splits: Dict[str, np.ndarray]) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    idx_train = splits["train"]
    idx_val = splits["val"]
    idx_test = splits["test"]

    # Flatten one-hot sequence: (N, 4, L) -> (N, 4L)
    x_flat = x_seq.reshape(x_seq.shape[0], -1)

    x_train = x_flat[idx_train]
    y_train = y[idx_train]
    x_test = x_flat[idx_test]
    y_test = y[idx_test]

    print("Training Logistic Regression (sequence-only, flat)")
    t0 = time.perf_counter()
    logreg = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
    logreg.fit(x_train, y_train)
    proba_log = logreg.predict_proba(x_test)[:, 1]
    t1 = time.perf_counter()
    logreg_time = t1 - t0
    print(f"  Seq-LogReg training + inference time: {logreg_time:.2f} s")

    print("Training SVM RBF (sequence-only, flat)")
    t0 = time.perf_counter()
    svm = SVC(kernel="rbf", probability=True)
    svm.fit(x_train, y_train)
    proba_svm = svm.predict_proba(x_test)[:, 1]
    t1 = time.perf_counter()
    svm_time = t1 - t0
    print(f"  Seq-SVM training + inference time: {svm_time:.2f} s")

    def eval_model(name: str, proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        auroc = auc(fpr, tpr)
        auprc = auc(rec, prec)
        print(f"{name:10s} AUROC={auroc:.6f}  AUPRC={auprc:.6f}")
        return fpr, tpr, auroc, rec, prec, auprc

    fpr_l, tpr_l, auroc_l, rec_l, prec_l, auprc_l = eval_model("Seq-LogReg", proba_log)
    fpr_s, tpr_s, auroc_s, rec_s, prec_s, auprc_s = eval_model("Seq-SVM", proba_svm)

    # ROC
    plt.figure(figsize=(5, 4))
    plt.plot(fpr_l, tpr_l, label=f"Seq-LogReg (AUROC={auroc_l:.3f})")
    plt.plot(fpr_s, tpr_s, label=f"Seq-SVM (AUROC={auroc_s:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Sequence-only flat models (wig, chr1) ROC")
    plt.legend()
    if SAVE_PLOTS:
        path = os.path.join(OUTPUT_DIR, "wig_seq_flat_roc.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        print("Saved seq-flat ROC to:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # PR
    plt.figure(figsize=(5, 4))
    plt.plot(rec_l, prec_l, label=f"Seq-LogReg (AUPRC={auprc_l:.3f})")
    plt.plot(rec_s, prec_s, label=f"Seq-SVM (AUPRC={auprc_s:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Sequence-only flat models (wig, chr1) PR")
    plt.legend()
    if SAVE_PLOTS:
        path = os.path.join(OUTPUT_DIR, "wig_seq_flat_pr.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        print("Saved seq-flat PR to:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    print("Sequence-only flat models summary (wig, chr1 test):")
    print(f"  Seq-LogReg : AUROC={auroc_l:.4f}, AUPRC={auprc_l:.4f}, time={logreg_time:.2f}s")
    print(f"  Seq-SVM    : AUROC={auroc_s:.4f}, AUPRC={auprc_s:.4f}, time={svm_time:.2f}s")

    # PR
    plt.figure(figsize=(5, 4))
    plt.plot(rec_l, prec_l, label=f"LogReg (AUPRC={auprc_l:.3f})")
    plt.plot(rec_s, prec_s, label=f"SVM RBF (AUPRC={auprc_s:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Shape-only models (wig, chr1) PR")
    plt.legend()
    if SAVE_PLOTS:
        path = os.path.join(OUTPUT_DIR, "wig_shape_only_pr.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        print("Saved shape-only PR to:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def train_cnn_models(
    x_seq: np.ndarray,
    x_shape: np.ndarray,
    y: np.ndarray,
    splits: Dict[str, np.ndarray],
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> None:
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    idx_train = splits["train"]
    idx_val = splits["val"]
    idx_test = splits["test"]

    x_seq_train = torch.tensor(x_seq[idx_train], dtype=torch.float32)
    y_train = torch.tensor(y[idx_train], dtype=torch.float32)
    x_seq_val = torch.tensor(x_seq[idx_val], dtype=torch.float32)
    y_val = torch.tensor(y[idx_val], dtype=torch.float32)
    x_seq_test = torch.tensor(x_seq[idx_test], dtype=torch.float32)
    y_test = torch.tensor(y[idx_test], dtype=torch.float32)

    x_shape_train = torch.tensor(x_shape[idx_train], dtype=torch.float32)
    x_shape_val = torch.tensor(x_shape[idx_val], dtype=torch.float32)
    x_shape_test = torch.tensor(x_shape[idx_test], dtype=torch.float32)

    train_seq_ds = TensorDataset(x_seq_train, y_train)
    val_seq_ds = TensorDataset(x_seq_val, y_val)
    test_seq_ds = TensorDataset(x_seq_test, y_test)

    train_seq_loader = DataLoader(train_seq_ds, batch_size=batch_size, shuffle=True)
    val_seq_loader = DataLoader(val_seq_ds, batch_size=batch_size, shuffle=False)
    test_seq_loader = DataLoader(test_seq_ds, batch_size=batch_size, shuffle=False)

    train_fusion_ds = TensorDataset(x_seq_train, x_shape_train, y_train)
    val_fusion_ds = TensorDataset(x_seq_val, x_shape_val, y_val)
    test_fusion_ds = TensorDataset(x_seq_test, x_shape_test, y_test)

    train_fusion_loader = DataLoader(train_fusion_ds, batch_size=batch_size, shuffle=True)
    val_fusion_loader = DataLoader(val_fusion_ds, batch_size=batch_size, shuffle=False)
    test_fusion_loader = DataLoader(test_fusion_ds, batch_size=batch_size, shuffle=False)

    input_length = x_seq_train.shape[2]
    shape_dim = x_shape_train.shape[1]

    class SeqCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(4, 32, kernel_size=8)
            self.pool = nn.MaxPool1d(kernel_size=4)
            conv_out_len = input_length - 8 + 1
            pool_out_len = conv_out_len // 4
            self.fc = nn.Linear(32 * pool_out_len, 64)
            self.out = nn.Linear(64, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = x.flatten(1)
            x = torch.relu(self.fc(x))
            x = self.out(x)
            return x.squeeze(-1)

    class SeqShapeCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seq_cnn = SeqCNN()
            self.fusion = nn.Sequential(
                nn.Linear(64 + shape_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x_seq: torch.Tensor, x_shape: torch.Tensor) -> torch.Tensor:
            x = self.seq_cnn.conv(x_seq)
            x = torch.relu(x)
            x = self.seq_cnn.pool(x)
            x = x.flatten(1)
            x = torch.relu(self.seq_cnn.fc(x))
            x = torch.cat([x, x_shape], dim=1)
            x = self.fusion(x)
            return x.squeeze(-1)

    def train_seq() -> Tuple[np.ndarray, np.ndarray, float]:
        model = SeqCNN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        best_state = None
        best_val_auc = -1.0

        from sklearn.metrics import roc_auc_score

        t0 = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            pbar = tqdm(train_seq_loader, desc=f"SeqCNN epoch {epoch+1}/{epochs}")
            for xb, yb in pbar:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=float(loss.item()))

            model.eval()
            with torch.no_grad():
                all_logits = []
                all_y = []
                for xb, yb in val_seq_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    all_logits.extend(logits.detach().cpu().tolist())
                    all_y.extend(yb.detach().cpu().tolist())
            logits_val = np.asarray(all_logits, dtype=np.float32)
            y_val_np = np.asarray(all_y, dtype=np.float32)
            proba_val = 1.0 / (1.0 + np.exp(-logits_val))
            try:
                auc_val = roc_auc_score(y_val_np, proba_val)
            except ValueError:
                auc_val = 0.0
            if auc_val > best_val_auc:
                best_val_auc = auc_val
                best_state = model.state_dict()

        if best_state is not None:
            model.load_state_dict(best_state)
        t1 = time.perf_counter()
        train_time = t1 - t0

        model.eval()
        with torch.no_grad():
            all_logits = []
            all_y = []
            for xb, yb in test_seq_loader:
                xb = xb.to(device)
                logits = model(xb)
                all_logits.extend(logits.detach().cpu().tolist())
                all_y.extend(yb.detach().cpu().tolist())
        logits_test = np.asarray(all_logits, dtype=np.float32)
        y_test_np = np.asarray(all_y, dtype=np.float32)
        proba_test = 1.0 / (1.0 + np.exp(-logits_test))
        return y_test_np, proba_test, train_time

    def train_fusion() -> Tuple[np.ndarray, np.ndarray, float]:
        model = SeqShapeCNN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        best_state = None
        best_val_auc = -1.0

        from sklearn.metrics import roc_auc_score

        t0 = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            pbar = tqdm(train_fusion_loader, desc=f"Seq+Shape epoch {epoch+1}/{epochs}")
            for xs, xf, yb in pbar:
                xs = xs.to(device)
                xf = xf.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = model(xs, xf)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=float(loss.item()))

            model.eval()
            with torch.no_grad():
                all_logits = []
                all_y = []
                for xs, xf, yb in val_fusion_loader:
                    xs = xs.to(device)
                    xf = xf.to(device)
                    logits = model(xs, xf)
                    all_logits.extend(logits.detach().cpu().tolist())
                    all_y.extend(yb.detach().cpu().tolist())
            logits_val = np.asarray(all_logits, dtype=np.float32)
            y_val_np = np.asarray(all_y, dtype=np.float32)
            proba_val = 1.0 / (1.0 + np.exp(-logits_val))
            try:
                auc_val = roc_auc_score(y_val_np, proba_val)
            except ValueError:
                auc_val = 0.0
            if auc_val > best_val_auc:
                best_val_auc = auc_val
                best_state = model.state_dict()

        if best_state is not None:
            model.load_state_dict(best_state)
        t1 = time.perf_counter()
        train_time = t1 - t0

        model.eval()
        with torch.no_grad():
            all_logits = []
            all_y = []
            for xs, xf, yb in test_fusion_loader:
                xs = xs.to(device)
                xf = xf.to(device)
                logits = model(xs, xf)
                all_logits.extend(logits.detach().cpu().tolist())
                all_y.extend(yb.detach().cpu().tolist())
        logits_test = np.asarray(all_logits, dtype=np.float32)
        y_test_np = np.asarray(all_y, dtype=np.float32)
        proba_test = 1.0 / (1.0 + np.exp(-logits_test))
        return y_test_np, proba_test, train_time

    print("Training sequence-only CNN (wig, small test)")
    y_test_seq, proba_seq, time_seq = train_seq()

    print("Training sequence+shape CNN (wig, small test)")
    y_test_fusion, proba_fusion, time_fusion = train_fusion()

    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    fpr_s, tpr_s, _ = roc_curve(y_test_seq, proba_seq)
    prec_s, rec_s, _ = precision_recall_curve(y_test_seq, proba_seq)
    auroc_s = auc(fpr_s, tpr_s)
    auprc_s = auc(rec_s, prec_s)

    fpr_f, tpr_f, _ = roc_curve(y_test_fusion, proba_fusion)
    prec_f, rec_f, _ = precision_recall_curve(y_test_fusion, proba_fusion)
    auroc_f = auc(fpr_f, tpr_f)
    auprc_f = auc(rec_f, prec_f)

    print(f"SeqCNN       AUROC={auroc_s:.6f}  AUPRC={auprc_s:.6f}  time={time_seq:.2f}s")
    print(f"Seq+ShapeCNN AUROC={auroc_f:.6f}  AUPRC={auprc_f:.6f}  time={time_fusion:.2f}s")

    # ROC
    plt.figure(figsize=(5, 4))
    plt.plot(fpr_s, tpr_s, label=f"SeqCNN (AUROC={auroc_s:.3f})")
    plt.plot(fpr_f, tpr_f, label=f"Seq+Shape (AUROC={auroc_f:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("CNN models (wig, chr1) ROC")
    plt.legend()
    if SAVE_PLOTS:
        path = os.path.join(OUTPUT_DIR, "wig_cnn_roc.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        print("Saved CNN ROC to:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # PR
    plt.figure(figsize=(5, 4))
    plt.plot(rec_s, prec_s, label=f"SeqCNN (AUPRC={auprc_s:.3f})")
    plt.plot(rec_f, prec_f, label=f"Seq+Shape (AUPRC={auprc_f:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("CNN models (wig, chr1) PR")
    plt.legend()
    if SAVE_PLOTS:
        path = os.path.join(OUTPUT_DIR, "wig_cnn_pr.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        print("Saved CNN PR to:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    print("CNN models summary (wig, chr1 test):")
    print(f"  SeqCNN       : AUROC={auroc_s:.4f}, AUPRC={auprc_s:.4f}, time={time_seq:.2f}s")
    print(f"  Seq+ShapeCNN : AUROC={auroc_f:.4f}, AUPRC={auprc_f:.4f}, time={time_fusion:.2f}s")



# 5. Main


def main() -> None:
    print("=== WIG-BASED CTCF PIPELINE (chr1, small test) ===")
    chr1_meta, _ = load_chr1_windows()
    chr1_meta = subsample_balanced(chr1_meta)

    chr1_seqs = load_chr1_sequences(chr1_meta)
    if USE_MOTIF_MATCHED:
        chr1_meta, chr1_seqs = apply_motif_matched_filter(chr1_meta, chr1_seqs)
    print("One-hot encoding sequences")
    x_seq = one_hot_encode(chr1_seqs, WINDOW_SIZE)

    print("Computing GC content and plotting histogram")
    gc = compute_gc(x_seq)
    plot_gc_hist(gc, chr1_meta["label"].values.astype(int))

    print("Collecting needed genomic positions for wig")
    needed_positions = collect_needed_positions(chr1_meta)
    print("Total unique positions needed:", len(needed_positions))

    wig_maps: Dict[str, Dict[int, float]] = {}
    for name, fname in WIG_FILES.items():
        path = os.path.join(WIG_DIR, fname)
        wig_maps[name] = load_wig_values(path, needed_positions)

    print("Building shape matrix from wig tracks")
    x_shape_raw = build_shape_matrix(chr1_meta, wig_maps)
    print("Shape matrix shape:", x_shape_raw.shape)

    print("Splitting into train/val/test by genomic coordinate")
    splits = split_by_coordinate(chr1_meta)
    x_shape, mean_shape, std_shape = standardize_shape(x_shape_raw, splits["train"])
    y = chr1_meta["label"].values.astype(int)

    print("Training shape-only baselines")
    train_shape_baselines(x_shape, y, splits)

    print("Training sequence-only flat baselines (LogReg / SVM)")
    train_seq_flat_baselines(x_seq, y, splits)

    print("Training CNN models (sequence-only and sequence+shape)")
    train_cnn_models(x_seq, x_shape, y, splits, epochs=2, batch_size=128, lr=1e-3)


if __name__ == "__main__":
    main()


