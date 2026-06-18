"""
2.3 类别分布与稀有事件分析 — 单文件脚本

运行：py 类别分布与稀有事件分析.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
import sys
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from paths import resolve_data_csv

DATA_PATH = resolve_data_csv()
TABLE_DIR = HERE / "输出" / "tables"
FIG_DIR = HERE / "输出" / "figures"

CLASS_LABELS = {0: "正常", 1: "欺诈"}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def setup_plot() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 150,
        }
    )
    sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到数据: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def task1_class_frequency(df: pd.DataFrame) -> pd.DataFrame:
    vc = df["Class"].value_counts().sort_index()
    n = len(df)
    freq = pd.DataFrame(
        {
            "Class": vc.index.astype(int),
            "Label": [CLASS_LABELS[i] for i in vc.index],
            "Count": vc.values,
            "Percentage": (vc.values / n * 100).round(4),
        }
    )
    freq.to_csv(TABLE_DIR / "表2.3_类别频数统计.csv", index=False, encoding="utf-8-sig")
    return freq, n


def task2_ratio(freq: pd.DataFrame) -> pd.DataFrame:
    n0 = int(freq.loc[freq["Class"] == 0, "Count"].iloc[0])
    n1 = int(freq.loc[freq["Class"] == 1, "Count"].iloc[0])
    ratio = n0 / n1
    ratio_int = round(ratio)
    summary = pd.DataFrame(
        [
            {"指标": "N0（正常）", "值": n0},
            {"指标": "N1（欺诈）", "值": n1},
            {"指标": "N0/N1", "值": round(ratio, 4)},
            {"指标": "约简比例（正常:欺诈）", "值": f"{ratio_int}:1"},
        ]
    )
    summary.to_csv(TABLE_DIR / "正负样本比例.csv", index=False, encoding="utf-8-sig")
    return summary, ratio, ratio_int


def task3_plots(freq: pd.DataFrame) -> None:
    labels = freq["Label"].tolist()
    counts = freq["Count"].tolist()
    pcts = freq["Percentage"].tolist()
    colors = ["#4C78A8", "#E45756"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    axes[0].bar(labels, counts, color=colors)
    axes[0].set_title("Class 频数柱状图")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(labels, pcts, color=colors)
    axes[1].set_title("Class 占比（%）")
    axes[1].set_ylabel("Percentage (%)")
    for i, p in enumerate(pcts):
        axes[1].text(i, p, f"{p:.4f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"类别分布（欺诈率 {pcts[1]:.4f}%）", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "Figure1_类别分布柱状图.png", bbox_inches="tight")
    plt.close(fig)

    # 饼图（可选，展示欺诈极少）
    fig2, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        counts,
        labels=[f"{l}\n({p:.4f}%)" for l, p in zip(labels, pcts)],
        colors=colors,
        startangle=90,
        explode=(0, 0.08),
    )
    ax.set_title("Class 占比饼图（欺诈占比极小）")
    fig2.savefig(FIG_DIR / "Figure2_类别占比饼图.png", bbox_inches="tight")
    plt.close(fig2)


def task4_accuracy_trap(n: int, n0: int, n1: int) -> pd.DataFrame:
    """永远预测正常：Accuracy 高但欺诈 Recall=0。"""
    accuracy = n0 / n
    recall_fraud = 0.0
    precision_fraud = 0.0  # 无预测为欺诈

    baseline = pd.DataFrame(
        [
            {"指标": "朴素规则", "值": "全部预测为正常（Class=0）"},
            {"指标": "Accuracy", "值": f"{accuracy * 100:.4f}%"},
            {"指标": "欺诈类 Recall", "值": recall_fraud},
            {"指标": "欺诈类 Precision", "值": precision_fraud},
        ]
    )
    baseline.to_csv(TABLE_DIR / "Accuracy基准陷阱.csv", index=False, encoding="utf-8-sig")
    return baseline, accuracy


def task5_rare_events(n1: int, n: int) -> pd.DataFrame:
    p_hat = n1 / n
    judgment = pd.DataFrame(
        [
            {"项目": "欺诈概率 P(Y=1)", "值": f"{p_hat * 100:.4f}%"},
            {"项目": "正常概率 P(Y=0)", "值": f"{(1 - p_hat) * 100:.4f}%"},
            {"项目": "是否 Rare Events", "值": "是（P(Y=1) 远小于 P(Y=0)）"},
        ]
    )
    judgment.to_csv(TABLE_DIR / "RareEvents判断.csv", index=False, encoding="utf-8-sig")
    return judgment, p_hat


def write_report_md(
    freq: pd.DataFrame,
    ratio_int: int,
    accuracy: float,
    p_hat: float,
) -> None:
    n1 = int(freq.loc[freq["Class"] == 1, "Count"].iloc[0])
    n0 = int(freq.loc[freq["Class"] == 0, "Count"].iloc[0])
    n = n0 + n1
    pct1 = freq.loc[freq["Class"] == 1, "Percentage"].iloc[0]

    md = f"""# 2.3 类别分布与稀有事件分析

## 表 2.3 类别频数

| Class | Count | Percentage |
| ----- | ----- | ---------- |
| 0 正常 | {n0:,} | {freq.loc[freq['Class']==0,'Percentage'].iloc[0]:.4f}% |
| 1 欺诈 | {n1:,} | {pct1:.4f}% |

- 欺诈率：{n1}/{n} = {p_hat*100:.4f}%
- 正常:欺诈 ≈ {ratio_int}:1
- 全预测正常的 Accuracy：{accuracy*100:.4f}%，欺诈 Recall = 0

## 输出文件

- 表：`输出/tables/`
- 图：`输出/figures/Figure1_类别分布柱状图.png`
"""
    (HERE / "2.3_类别分布与稀有事件分析报告.md").write_text(md, encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ensure_dirs()
    setup_plot()
    print("=" * 60)
    print("2.3 类别分布与稀有事件分析")
    print("=" * 60)

    df = load_data()
    freq, n = task1_class_frequency(df)
    print("\n【Task 1】类别频数")
    print(freq.to_string(index=False))
    print(f"\n欺诈率 p̂ = {freq.loc[freq['Class']==1,'Count'].iloc[0]}/{n} "
          f"= {freq.loc[freq['Class']==1,'Percentage'].iloc[0]:.4f}%")

    ratio_df, ratio, ratio_int = task2_ratio(freq)
    print("\n【Task 2】正负样本比例")
    print(ratio_df.to_string(index=False))
    print(f"正常:欺诈 ≈ {ratio_int}:1（精确比值 {ratio:.4f}）")

    task3_plots(freq)
    print("\n【Task 3】已保存类别分布图")

    n0 = int(freq.loc[freq["Class"] == 0, "Count"].iloc[0])
    n1 = int(freq.loc[freq["Class"] == 1, "Count"].iloc[0])
    baseline, acc = task4_accuracy_trap(n, n0, n1)
    print("\n【Task 4】Accuracy 基准陷阱")
    print(baseline.to_string(index=False))

    judgment, p_hat = task5_rare_events(n1, n)
    print("\n【Task 5】Rare Events 判断")
    print(judgment.to_string(index=False))

    write_report_md(freq, ratio_int, acc, p_hat)

    print("\n【Task 6】本节核心结论")
    print("1. 欺诈样本约占 0.1727%")
    print(f"2. 正负样本比例约为 {ratio_int}:1")
    print("3. 存在极端类别失衡")
    print("4. Accuracy 不适合作为唯一评价指标")
    print("5. 后续需关注 Recall、F1、PR-AUC 等")
    print("6. 为 Rare Events Logistic 提供依据")
    print(f"\n输出: {HERE / '输出'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
