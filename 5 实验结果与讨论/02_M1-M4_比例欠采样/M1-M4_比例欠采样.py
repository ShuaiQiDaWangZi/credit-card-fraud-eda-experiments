"""M1–M4：比例控制欠采样 + Logistic Regression。"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "公共"))
from 实验工具 import (  # noqa: E402
    LR_KWARGS,
    SEEDS,
    evaluate_split,
    fit_scaler_transform,
    get_xy,
    load_full_data,
    load_splits,
    output_dir,
    save_metrics_csv,
    select_threshold_f1,
    undersample_by_r,
)

HERE = Path(__file__).resolve().parent
OUT = output_dir(HERE)
RATIOS = {1: "M1", 2: "M2", 5: "M3", 10: "M4"}


def run_method(seed: int, r: int, method: str) -> dict:
    df = load_full_data()
    splits = load_splits(seed)
    X_tr, y_tr = get_xy(df, splits["train"])
    X_va, y_va = get_xy(df, splits["val"])
    X_te, y_te = get_xy(df, splits["test"])
    _, (X_tr, X_va, X_te) = fit_scaler_transform(X_tr, X_va, X_te)
    X_tr, y_tr = undersample_by_r(X_tr, y_tr, r=r, seed=seed)

    clf = LogisticRegression(**LR_KWARGS)
    clf.fit(X_tr, y_tr)
    p_va = clf.predict_proba(X_va)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]
    thr = select_threshold_f1(y_va, p_va)
    metrics = evaluate_split(y_te, p_te, thr)
    n_fraud = int((y_tr == 1).sum())
    n_normal = int((y_tr == 0).sum())
    metrics.update(
        {
            "method": method,
            "seed": seed,
            "route": "数据层",
            "r": r,
            "train_n": len(y_tr),
            "train_n_fraud": n_fraud,
            "train_n_normal": n_normal,
        }
    )
    return metrics


def main() -> None:
    rows = []
    for seed in SEEDS:
        for r, method in RATIOS.items():
            rows.append(run_method(seed, r, method))
    save_metrics_csv(rows, OUT / "M1-M4_各seed指标.csv")
    df = pd.DataFrame(rows)
    summary = df.groupby("method")[["pr_auc", "brier", "precision", "recall", "f1"]].agg(
        ["mean", "std"]
    )
    summary.to_csv(OUT / "M1-M4_汇总_mean_std.csv", encoding="utf-8-sig")
    print(summary)


if __name__ == "__main__":
    main()
