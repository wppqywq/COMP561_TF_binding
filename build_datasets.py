import os
from typing import List, Tuple

import numpy as np

from config import (
    TF_NAME,
    WINDOW_SIZE,
    TRAIN_CHROMS,
    VAL_CHROMS,
    TEST_CHROMS,
    OUTPUT_DIR,
    WINDOW_FASTA_PATH,
    WINDOW_META_PATH,
)


SHAPE_TSV_PATH = os.path.join(OUTPUT_DIR, f"{TF_NAME.lower()}_shape.tsv")


def read_meta(path: str) -> Tuple[List[str], np.ndarray, List[str]]:
    ids: List[str] = []
    labels_list: List[int] = []
    chroms: List[str] = []
    with open(path, "r") as f:
        header = next(f, None)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            ids.append(parts[0])
            labels_list.append(int(parts[1]))
            chroms.append(parts[2])
    labels = np.array(labels_list, dtype=np.int64)
    return ids, labels, chroms


def read_fasta_sequences(path: str) -> List[str]:
    seqs: List[str] = []
    with open(path, "r") as f:
        current_seq: List[str] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    seqs.append("".join(current_seq).upper())
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            seqs.append("".join(current_seq).upper())
    return seqs


def one_hot_encode(seqs: List[str], window_size: int) -> np.ndarray:
    n = len(seqs)
    x = np.zeros((n, 4, window_size), dtype=np.float32)
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, seq in enumerate(seqs):
        if len(seq) != window_size:
            continue
        for j, base in enumerate(seq):
            k = base_to_idx.get(base)
            if k is not None:
                x[i, k, j] = 1.0
    return x


def read_shape_matrix(path: str) -> np.ndarray:
    data = np.genfromtxt(path, dtype=np.float32, filling_values=0.0)
    return data


def assign_split(chrom: str) -> str:
    chrom_clean = chrom.replace("chr", "")
    if chrom_clean in TRAIN_CHROMS:
        return "train"
    if chrom_clean in VAL_CHROMS:
        return "val"
    if chrom_clean in TEST_CHROMS:
        return "test"
    return "ignore"


def standardize_shape(
    x_shape: np.ndarray,
    split_tags: List[str],
) -> np.ndarray:
    train_mask = np.array([t == "train" for t in split_tags], dtype=bool)
    train_vals = x_shape[train_mask]
    mean = train_vals.mean(axis=0)
    std = train_vals.std(axis=0)
    std[std == 0.0] = 1.0
    x_norm = (x_shape - mean) / std
    scaler_path = os.path.join(OUTPUT_DIR, f"{TF_NAME.lower()}_shape_scaler.npz")
    np.savez_compressed(scaler_path, mean=mean, std=std)
    return x_norm


def save_split(
    name: str,
    x_seq: np.ndarray,
    x_shape: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> None:
    out_path = os.path.join(OUTPUT_DIR, f"{TF_NAME.lower()}_{name}.npz")
    np.savez_compressed(
        out_path,
        x_seq=x_seq[mask],
        x_shape=x_shape[mask],
        y=y[mask],
    )


def main() -> None:
    ids, labels, chroms = read_meta(WINDOW_META_PATH)
    seqs = read_fasta_sequences(WINDOW_FASTA_PATH)
    if len(seqs) != len(ids):
        raise ValueError("FASTA and meta have different numbers of sequences")
    x_seq = one_hot_encode(seqs, WINDOW_SIZE)
    x_shape = read_shape_matrix(SHAPE_TSV_PATH)
    if x_shape.shape[0] != len(ids):
        raise ValueError("Shape matrix row count does not match number of windows")
    split_tags = [assign_split(c) for c in chroms]
    x_shape_norm = standardize_shape(x_shape, split_tags)
    y = labels
    mask_train = np.array([t == "train" for t in split_tags], dtype=bool)
    mask_val = np.array([t == "val" for t in split_tags], dtype=bool)
    mask_test = np.array([t == "test" for t in split_tags], dtype=bool)
    save_split("train", x_seq, x_shape_norm, y, mask_train)
    save_split("val", x_seq, x_shape_norm, y, mask_val)
    save_split("test", x_seq, x_shape_norm, y, mask_test)


if __name__ == "__main__":
    main()


