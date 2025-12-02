import os
import gzip
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from config import WINDOW_SIZE
from build_datasets import one_hot_encode
from legacy_pipeline import (
    collect_needed_positions,
    load_wig_values,
    build_shape_matrix,
    split_by_coordinate,
    standardize_shape,
    compute_gc,
    plot_gc_hist,
)



# Configuration


# Choose a TF that is expected to be more shape-dependent.
# If this TF has too few sites on chr1 in your Factorbook file,
# change TF_NAME to another factor (check data/factorbookMotifPos.txt.gz).
# TF_NAME = "BHLHE40"
TF_NAME = "CTCF"

CHR = "chr1"

FASTA_PATH = os.path.join("data", f"{CHR}.fa")
FACTORBOOK_PATH = os.path.join("data", "factorbookMotifPos.txt.gz")
GM_BED_PATH = os.path.join("data", "wgEncodeRegTfbsClusteredV3.GM12878.merged.bed.gz")

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


# Utilities for reading genome and windows



def load_chr_sequence(path: str) -> str:
    seq_chunks: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                continue
            seq_chunks.append(line)
    return "".join(seq_chunks).upper()


def reverse_complement(seq: str) -> str:
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    return "".join(comp.get(b, "N") for b in reversed(seq))


def read_factorbook_tf_chr1(tf_name: str) -> pd.DataFrame:
    rows = []
    with gzip.open(FACTORBOOK_PATH, "rt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            chrom = parts[1]
            if chrom != CHR:
                continue
            start = int(parts[2])
            end = int(parts[3])
            name = parts[4]
            strand = parts[6]
            if name != tf_name:
                continue
            rows.append((chrom, start, end, strand))
    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "strand"])
    return df


def build_positive_windows(fb_df: pd.DataFrame, chrom_len: int) -> pd.DataFrame:
    half = WINDOW_SIZE // 2
    starts: List[int] = []
    ends: List[int] = []
    strands: List[str] = []
    for _, row in fb_df.iterrows():
        c = (int(row["start"]) + int(row["end"])) // 2
        s = c - half
        e = s + WINDOW_SIZE
        if s < 0 or e > chrom_len:
            continue
        starts.append(s)
        ends.append(e)
        strands.append(row["strand"])
    df = pd.DataFrame(
        {
            "chrom": [CHR] * len(starts),
            "start": starts,
            "end": ends,
            "strand": strands,
            "label": [1] * len(starts),
        }
    )
    return df


def read_active_regions_chr1() -> List[Tuple[int, int]]:
    regions: List[Tuple[int, int]] = []
    with gzip.open(GM_BED_PATH, "rt") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            chrom = parts[0]
            if chrom != CHR:
                continue
            start = int(parts[1])
            end = int(parts[2])
            regions.append((start, end))
    return regions


def build_negative_windows(
    pos_windows: pd.DataFrame,
    chrom_len: int,
    step_factor: int = 5,
) -> pd.DataFrame:
    pos_intervals = pos_windows[["start", "end"]].values
    pos_intervals = pos_intervals[np.argsort(pos_intervals[:, 0])]

    def has_overlap(s: int, e: int) -> bool:
        # simple linear scan with early break; pos_windows count is moderate on chr1
        for ps, pe in pos_intervals:
            if pe <= s:
                continue
            if ps >= e:
                break
            if not (pe <= s or ps >= e):
                return True
        return False

    regions = read_active_regions_chr1()
    neg_starts: List[int] = []
    neg_ends: List[int] = []
    step = WINDOW_SIZE * step_factor
    for reg_start, reg_end in regions:
        cursor = reg_start
        while cursor + WINDOW_SIZE <= reg_end and cursor + WINDOW_SIZE <= chrom_len:
            s = cursor
            e = s + WINDOW_SIZE
            if not has_overlap(s, e):
                neg_starts.append(s)
                neg_ends.append(e)
            cursor += step

    df = pd.DataFrame(
        {
            "chrom": [CHR] * len(neg_starts),
            "start": neg_starts,
            "end": neg_ends,
            "strand": ["+"] * len(neg_starts),
            "label": [0] * len(neg_starts),
        }
    )
    return df


def extract_sequences(
    windows: pd.DataFrame,
    chrom_seq: str,
) -> List[str]:
    seqs: List[str] = []
    for _, row in windows.iterrows():
        s = int(row["start"])
        e = int(row["end"])
        seq = chrom_seq[s:e]
        if row["strand"] == "-":
            seq = reverse_complement(seq)
        seqs.append(seq)
    return seqs



# Models: LogReg and SVM for shape / seq / seq+shape



def train_lr_svm_triplet(
    name_prefix: str,
    x_shape: np.ndarray,
    x_seq: np.ndarray,
    y: np.ndarray,
    splits: Dict[str, np.ndarray],
) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    idx_train = splits["train"]
    idx_val = splits["val"]
    idx_test = splits["test"]

    # Flatten sequence
    x_seq_flat = x_seq.reshape(x_seq.shape[0], -1)
    # Concatenate
    x_concat = np.concatenate([x_seq_flat, x_shape], axis=1)

    feature_sets = {
        "shape": x_shape,
        "seq": x_seq_flat,
        "seq+shape": x_concat,
    }

    def run_algorithm(
        algo_name: str,
        make_model,
    ) -> None:
        results = {}
        times = {}
        for feat_name, X in feature_sets.items():
            x_train = X[idx_train]
            y_train = y[idx_train]
            x_test = X[idx_test]
            y_test = y[idx_test]

            print(f"{name_prefix} {algo_name} on {feat_name} features")
            t0 = time.perf_counter()
            model = make_model()
            model.fit(x_train, y_train)
            proba = model.predict_proba(x_test)[:, 1]
            t1 = time.perf_counter()
            times[feat_name] = t1 - t0

            fpr, tpr, _ = roc_curve(y_test, proba)
            prec, rec, _ = precision_recall_curve(y_test, proba)
            auroc = auc(fpr, tpr)
            auprc = auc(rec, prec)
            results[feat_name] = (fpr, tpr, auroc, rec, prec, auprc)
            print(
                f"  {algo_name}-{feat_name:9s}: AUROC={auroc:.4f}, "
                f"AUPRC={auprc:.4f}, time={times[feat_name]:.2f}s"
            )

        # ROC plot
        plt.figure(figsize=(5, 4))
        for feat_name, (fpr, tpr, auroc, _, _, _) in results.items():
            plt.plot(fpr, tpr, label=f"{feat_name} (AUROC={auroc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name_prefix} {algo_name}: ROC (chr1)")
        plt.legend()
        if True:
            out_path = os.path.join(
                OUTPUT_DIR, f"{name_prefix.lower()}_{algo_name.lower()}_roc.png"
            )
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            print("Saved ROC to:", out_path)
        plt.show()
        plt.close()

        # PR plot
        plt.figure(figsize=(5, 4))
        for feat_name, (_, _, _, rec, prec, auprc) in results.items():
            plt.plot(rec, prec, label=f"{feat_name} (AUPRC={auprc:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{name_prefix} {algo_name}: PR (chr1)")
        plt.legend()
        if True:
            out_path = os.path.join(
                OUTPUT_DIR, f"{name_prefix.lower()}_{algo_name.lower()}_pr.png"
            )
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            print("Saved PR to:", out_path)
        plt.show()
        plt.close()

        print(f"{name_prefix} {algo_name} summary:")
        for feat_name, (_, _, auroc, _, _, auprc) in results.items():
            print(
                f"  {feat_name:9s}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, "
                f"time={times[feat_name]:.2f}s"
            )
        print()

    # Logistic Regression
    def make_logreg() -> LogisticRegression:
        return LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)

    # SVM RBF
    def make_svm() -> SVC:
        return SVC(kernel="rbf", probability=True)

    run_algorithm("LogReg", make_logreg)
    run_algorithm("SVM", make_svm)



# Main pipeline



def main() -> None:
    print("=== WIG-BASED PIPELINE FOR SHAPE-SENSITIVE TF (chr1) ===")
    print("TF_NAME:", TF_NAME)

    print("Loading chr1 sequence")
    chr_seq = load_chr_sequence(FASTA_PATH)
    chrom_len = len(chr_seq)
    print("chr1 length:", chrom_len)

    print("Reading Factorbook positions for TF on chr1")
    fb_df = read_factorbook_tf_chr1(TF_NAME)
    print("Raw Factorbook sites on chr1:", len(fb_df))
    if len(fb_df) == 0:
        print("No sites found for this TF on chr1. Please choose another TF_NAME.")
        return

    print("Building positive windows")
    pos_windows = build_positive_windows(fb_df, chrom_len)
    print("Positive windows:", len(pos_windows))

    print("Building negative windows from GM12878 active regions")
    neg_windows = build_negative_windows(pos_windows, chrom_len)
    print("Negative candidate windows:", len(neg_windows))

    all_windows = pd.concat([pos_windows, neg_windows], axis=0).reset_index(drop=True)
    print(
        "Total windows:",
        len(all_windows),
        "positives:",
        int(all_windows["label"].sum()),
        "negatives:",
        int((1 - all_windows["label"]).sum()),
    )

    print("Extracting sequences for all windows")
    seqs = extract_sequences(all_windows, chr_seq)
    print("One-hot encoding sequences")
    x_seq = one_hot_encode(seqs, WINDOW_SIZE)

    print("Computing GC content and plotting histogram")
    gc = compute_gc(x_seq)
    plot_gc_hist(gc, all_windows["label"].values.astype(int))

    print("Collecting wig positions for all windows")
    needed_positions = collect_needed_positions(all_windows)
    print("Total unique positions needed:", len(needed_positions))

    wig_maps: Dict[str, Dict[int, float]] = {}
    for name, fname in WIG_FILES.items():
        path = os.path.join(WIG_DIR, fname)
        wig_maps[name] = load_wig_values(path, needed_positions)

    print("Building shape matrix from wig tracks")
    x_shape_raw = build_shape_matrix(all_windows, wig_maps)
    print("Shape matrix shape:", x_shape_raw.shape)

    print("Splitting into train/val/test by genomic coordinate")
    splits = split_by_coordinate(all_windows)
    x_shape, mean_shape, std_shape = standardize_shape(x_shape_raw, splits["train"])
    y = all_windows["label"].values.astype(int)

    print("Training LogReg/SVM on shape / seq / seq+shape features")
    train_lr_svm_triplet(TF_NAME, x_shape, x_seq, y, splits)



if __name__ == "__main__":
    main()


