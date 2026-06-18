"""M8：Rare Events Logistic Regression（King–Zeng 修正）。"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "公共"))
from 实验工具 import (  # noqa: E402
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
from kz_logistic import kz_predict_proba  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = output_dir(HERE)


def run_seed(seed: int) -> dict:
    df = load_full_data()
    splits = load_splits(seed)
    X_tr, y_tr = get_xy(df, splits["train"])
    X_va, y_va = get_xy(df, splits["val"])
    X_te, y_te = get_xy(df, splits["test"])
    _, (X_tr, X_va, X_te) = fit_scaler_transform(X_tr, X_va, X_te)

    p_va = kz_predict_proba(X_tr, y_tr, X_va, ridge=1e-6)
    p_te = kz_predict_proba(X_tr, y_tr, X_te, ridge=1e-6)
    thr = select_threshold_f1(y_va, p_va)
    metrics = evaluate_split(y_te, p_te, thr)
    metrics.update({"method": "M8", "seed": seed, "route": "统计层"})
    return metrics


def main() -> None:
    rows = [run_seed(s) for s in SEEDS]
    save_metrics_csv(rows, OUT / "M8_各seed指标.csv")
    summary = (
        pd.DataFrame(rows)
        .groupby("method")[["pr_auc", "brier", "precision", "recall", "f1"]]
        .agg(["mean", "std"])
    )
    summary.to_csv(OUT / "M8_汇总_mean_std.csv", encoding="utf-8-sig")
    print(summary)


if __name__ == "__main__":
    main()
