"""M7：加权 Logistic Regression。"""

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
)

HERE = Path(__file__).resolve().parent
OUT = output_dir(HERE)


def run_seed(seed: int) -> dict:
    df = load_full_data()
    splits = load_splits(seed)
    X_tr, y_tr = get_xy(df, splits["train"])
    X_va, y_va = get_xy(df, splits["val"])
    X_te, y_te = get_xy(df, splits["test"])
    _, (X_tr, X_va, X_te) = fit_scaler_transform(X_tr, X_va, X_te)

    kw = {**LR_KWARGS, "class_weight": "balanced"}
    clf = LogisticRegression(**kw)
    clf.fit(X_tr, y_tr)
    p_va = clf.predict_proba(X_va)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]
    thr = select_threshold_f1(y_va, p_va)
    metrics = evaluate_split(y_te, p_te, thr)
    metrics.update({"method": "M7", "seed": seed, "route": "数据层"})
    return metrics


def main() -> None:
    rows = [run_seed(s) for s in SEEDS]
    save_metrics_csv(rows, OUT / "M7_各seed指标.csv")
    summary = (
        pd.DataFrame(rows)
        .groupby("method")[["pr_auc", "brier", "precision", "recall", "f1"]]
        .agg(["mean", "std"])
    )
    summary.to_csv(OUT / "M7_汇总_mean_std.csv", encoding="utf-8-sig")
    print(summary)


if __name__ == "__main__":
    main()
