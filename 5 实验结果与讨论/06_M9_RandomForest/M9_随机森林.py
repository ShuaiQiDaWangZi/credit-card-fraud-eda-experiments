"""M9：Random Forest（轻量固定超参，不做网格搜索）。"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "公共"))
from 实验工具 import (  # noqa: E402
    SEEDS,
    evaluate_split,
    get_xy,
    load_full_data,
    load_splits,
    output_dir,
    save_metrics_csv,
    select_threshold_f1,
)

HERE = Path(__file__).resolve().parent
OUT = output_dir(HERE)

# 轻量固定配置：少树 + 浅深度 + bootstrap 子采样，目标单次 seed 约 30–60 秒
RF_PARAMS = dict(
    n_estimators=50,
    max_depth=8,
    min_samples_leaf=10,
    max_samples=0.1,
    class_weight="balanced_subsample",
)


def run_seed(seed: int) -> dict:
    df = load_full_data()
    splits = load_splits(seed)
    X_tr, y_tr = get_xy(df, splits["train"])
    X_va, y_va = get_xy(df, splits["val"])
    X_te, y_te = get_xy(df, splits["test"])

    clf = RandomForestClassifier(**RF_PARAMS, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    p_va = clf.predict_proba(X_va)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]
    thr = select_threshold_f1(y_va, p_va)
    metrics = evaluate_split(y_te, p_te, thr)
    metrics.update({"method": "M9", "seed": seed, "route": "容量层"})
    return metrics


def main() -> None:
    rows = [run_seed(s) for s in SEEDS]
    save_metrics_csv(rows, OUT / "M9_各seed指标.csv")
    summary = (
        pd.DataFrame(rows)
        .groupby("method")[["pr_auc", "brier", "precision", "recall", "f1"]]
        .agg(["mean", "std"])
    )
    summary.to_csv(OUT / "M9_汇总_mean_std.csv", encoding="utf-8-sig")
    print(summary)


if __name__ == "__main__":
    main()
