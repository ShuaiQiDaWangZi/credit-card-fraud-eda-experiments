"""实验公共工具：路径、划分、预处理、指标与阈值选择。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

SEEDS = [42, 123, 456]
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
TARGET_COL = "Class"

EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from paths import resolve_data_csv

DATA_PATH = resolve_data_csv()
SPLIT_DIR = EXP_ROOT / "00_数据划分与协议" / "输出" / "splits"
SPLIT_SUMMARY = EXP_ROOT / "00_数据划分与协议" / "输出" / "tables" / "划分摘要.csv"

LR_KWARGS = dict(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=5000,
    class_weight=None,
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_full_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def split_indices(
    y: np.ndarray, seed: int, test_size: float = 0.2, val_size: float = 0.2
) -> dict[str, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    train_pool, test_idx = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_pool,
        test_size=val_size,
        stratify=y[train_pool],
        random_state=seed,
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_splits(seed: int, splits: dict[str, np.ndarray]) -> None:
    ensure_dir(SPLIT_DIR)
    for name, indices in splits.items():
        pd.Series(indices, name="index").to_csv(
            SPLIT_DIR / f"seed{seed}_{name}_index.csv", index=False
        )


def load_splits(seed: int) -> dict[str, np.ndarray]:
    out = {}
    for name in ("train", "val", "test"):
        path = SPLIT_DIR / f"seed{seed}_{name}_index.csv"
        out[name] = pd.read_csv(path)["index"].to_numpy()
    return out


def get_xy(df: pd.DataFrame, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    part = df.iloc[indices]
    X = part[FEATURE_COLS].to_numpy(dtype=float)
    y = part[TARGET_COL].to_numpy(dtype=int)
    return X, y


def fit_scaler_transform(
    X_train: np.ndarray, *others: np.ndarray
) -> tuple[StandardScaler, tuple[np.ndarray, ...]]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    transformed = [scaler.transform(X) for X in others]
    return scaler, (X_train_s, *transformed)


def undersample_by_r(
    X: np.ndarray, y: np.ndarray, r: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """保留全部欺诈，随机抽取 r * n_fraud 条正常样本。"""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_pos = len(pos)
    n_neg_sample = r * n_pos
    if n_neg_sample > len(neg):
        raise ValueError(f"正常样本不足：需要 {n_neg_sample}，仅有 {len(neg)}")
    neg_pick = rng.choice(neg, size=n_neg_sample, replace=False)
    keep = np.concatenate([pos, neg_pick])
    rng.shuffle(keep)
    return X[keep], y[keep]


def select_threshold_f1(
    y_true: np.ndarray, y_prob: np.ndarray, n_grid: int = 500
) -> float:
    thresholds = np.linspace(0.001, 0.5, n_grid)
    best_t, best_f1, best_prec = thresholds[0], -1.0, -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        if f1 > best_f1 or (f1 == best_f1 and prec > best_prec):
            best_f1, best_prec, best_t = f1, prec, t
    return float(best_t)


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, Any]:
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }


def evaluate_split(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, Any]:
    out = probability_metrics(y_true, y_prob)
    out.update(classification_metrics(y_true, y_prob, threshold))
    return out


def save_metrics_csv(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def output_dir(script_dir: Path, sub: str = "tables") -> Path:
    return ensure_dir(script_dir / "输出" / sub)
