"""根据已保存 CSV 生成第四章柱状图与混淆矩阵。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
import sys
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from paths import CHAPTER4_FIG_DIR

FIG = CHAPTER4_FIG_DIR
TAB = HERE / "输出" / "tables"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REP_SEED = 42
METHOD_ORDER = [f"M{i}" for i in range(11)]

# 与 §4 三条技术路线一致：蓝=基准，橙=数据层，绿=统计层，红=容量层
ROUTE_COLOR = {
    "M0": "#4C72B0",
    "M1": "#DD8452", "M2": "#DD8452", "M3": "#DD8452", "M4": "#DD8452",
    "M5": "#DD8452", "M6": "#DD8452", "M7": "#DD8452",
    "M8": "#55A868",
    "M9": "#C44E52", "M10": "#C44E52",
}

LEGEND = [
    Patch(facecolor="#4C72B0", label="M0 基准"),
    Patch(facecolor="#DD8452", label="M1–M7 数据层"),
    Patch(facecolor="#55A868", label="M8 统计层"),
    Patch(facecolor="#C44E52", label="M9–M10 容量层"),
]

CM_METHODS = ["M0", "M1", "M4", "M9", "M10"]


def load_summary() -> pd.DataFrame:
    raw = pd.read_csv(TAB / "表_核心指标对比.csv", header=[0, 1], index_col=0)
    rows = []
    for m in raw.index:
        row = {"method": m}
        for metric in ("pr_auc", "brier", "precision", "recall", "f1"):
            row[f"{metric}_mean"] = raw.loc[m, (metric, "mean")]
            row[f"{metric}_std"] = raw.loc[m, (metric, "std")]
        rows.append(row)
    return pd.DataFrame(rows).set_index("method").loc[METHOD_ORDER]


def plot_metric_bars(
    df: pd.DataFrame,
    col: str,
    ylabel: str,
    fname: str,
    *,
    higher_better: bool = True,
    ylim=None,
    logy=False,
):
    x = np.arange(len(df))
    means = df[f"{col}_mean"].values
    stds = df[f"{col}_std"].values
    colors = [ROUTE_COLOR[m] for m in df.index]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(
        x, means, yerr=stds, capsize=3,
        color=colors, alpha=0.9, edgecolor="white", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=0)
    ax.set_ylabel(ylabel)
    direction = "越高越好" if higher_better else "越低越好"
    ax.set_title(f"{ylabel} 对比（三 seed 均值 ± 标准差，{direction}）")
    if logy:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(ylim)

    for bar, val in zip(bars, means):
        fmt = f"{val:.4f}" if val < 0.01 else f"{val:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(), fmt,
            ha="center", va="bottom", fontsize=7, rotation=90,
        )

    ax.legend(handles=LEGEND, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_grids():
    detail = pd.read_csv(TAB / "全部方法_各seed明细.csv")
    sub = detail[(detail.seed == REP_SEED) & (detail.method.isin(CM_METHODS))]
    sub = sub.set_index("method").loc[CM_METHODS].reset_index()
    n = len(sub)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, sub.iterrows()):
        cm = np.array([[row.tn, row.fp], [row.fn, row.tp]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", color="black", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["判正常", "判欺诈"])
        ax.set_yticklabels(["真正常", "真欺诈"])
        ax.set_title(
            f"{row.method}\nP={row.precision:.2f} R={row.recall:.2f} F1={row.f1:.2f}",
            fontsize=9,
        )
    fig.suptitle(f"混淆矩阵（seed={REP_SEED}，测试集）", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "图4-6_代表方法混淆矩阵.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def remove_stale_figures():
    stale_names = [
        "图4-2_代表方法PR曲线.png",
        "图4-3_Calibration曲线.png",
        "图4-4_代表方法混淆矩阵.png",
        "图4-5_F1对比.png",
        "图4-6_Brier对比.png",
        "图4-6_代表方法PR曲线.png",
        "图4-7_Calibration曲线.png",
        "图4-8_代表方法混淆矩阵.png",
    ]
    for folder in (FIG, HERE / "输出" / "figures"):
        if not folder.exists():
            continue
        for name in stale_names:
            p = folder / name
            if p.exists():
                p.unlink()


def main():
    remove_stale_figures()
    df = load_summary()
    plot_metric_bars(df, "pr_auc", "PR-AUC", "图4-1_PR-AUC对比.png", higher_better=True, ylim=(0.5, 0.82))
    plot_metric_bars(df, "f1", "F1-score", "图4-2_F1对比.png", higher_better=True, ylim=(0, 0.9))
    plot_metric_bars(df, "precision", "Precision", "图4-3_Precision对比.png", higher_better=True, ylim=(0, 0.85))
    plot_metric_bars(df, "recall", "Recall", "图4-4_Recall对比.png", higher_better=True, ylim=(0, 1.0))
    plot_metric_bars(df, "brier", "Brier Score", "图4-5_Brier对比.png", higher_better=False, logy=True)
    plot_confusion_grids()
    print(f"全部配图已写入 {FIG}")


if __name__ == "__main__":
    main()
