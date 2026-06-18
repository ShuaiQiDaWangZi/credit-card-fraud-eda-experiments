"""
2.4 特征与类别关系分析 — 单文件脚本

研究 X ↔ Y：Amount、Time、V1–V28 与 Class 的关系。
运行：py 特征与类别关系分析.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
import sys
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from paths import resolve_data_csv

DATA_PATH = resolve_data_csv()
TABLE_DIR = HERE / "输出" / "tables"
FIG_DIR = HERE / "输出" / "figures"

CLASS_LABELS = {0: "Normal", 1: "Fraud"}
CLASS_LABELS_CN = {0: "正常", 1: "欺诈"}
V_COLS = [f"V{i}" for i in range(1, 29)]
TOP_N_CORR = 10
TOP_N_PLOT = 5


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
    df = pd.read_csv(DATA_PATH)
    df["ClassLabel"] = df["Class"].map(CLASS_LABELS_CN)
    return df


def grouped_desc(df: pd.DataFrame, var: str, stats: list[str]) -> pd.DataFrame:
    rows = []
    for c in [0, 1]:
        s = df.loc[df["Class"] == c, var]
        row = {"Class": CLASS_LABELS[c], "ClassCode": c}
        if "mean" in stats:
            row["Mean"] = s.mean()
        if "median" in stats:
            row["Median"] = s.median()
        if "std" in stats:
            row["SD"] = s.std()
        if "q1" in stats:
            row["Q1"] = s.quantile(0.25)
        if "q3" in stats:
            row["Q3"] = s.quantile(0.75)
        if "max" in stats:
            row["Max"] = s.max()
        rows.append(row)
    return pd.DataFrame(rows)


def task1_amount(df: pd.DataFrame) -> pd.DataFrame:
    tbl = grouped_desc(df, "Amount", ["mean", "median", "q1", "q3", "max"])
    out_cols = ["Class", "Mean", "Median", "Q1", "Q3", "Max"]
    tbl[out_cols].to_csv(TABLE_DIR / "Amount按类别统计.csv", index=False, encoding="utf-8-sig")

    palette = {"正常": "#4C78A8", "欺诈": "#E45756"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    sns.boxplot(
        data=df, x="ClassLabel", y="Amount", hue="ClassLabel",
        order=["正常", "欺诈"], hue_order=["正常", "欺诈"],
        palette=palette, legend=False, ax=axes[0],
    )
    axes[0].set_title("Amount 箱线图")

    sns.violinplot(
        data=df, x="ClassLabel", y="Amount", hue="ClassLabel",
        order=["正常", "欺诈"], hue_order=["正常", "欺诈"],
        palette=palette, legend=False, inner="quartile", ax=axes[1],
    )
    axes[1].set_title("Amount 小提琴图")

    for label, color in zip(["正常", "欺诈"], ["#4C78A8", "#E45756"]):
        sub = df.loc[df["ClassLabel"] == label, "Amount"]
        sns.kdeplot(sub, label=label, color=color, ax=axes[2], common_norm=False)
    axes[2].set_title("Amount KDE")
    axes[2].legend()

    fig.suptitle("Figure 1  Amount vs Class", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "Figure1_Amount_vs_Class.png", bbox_inches="tight")
    plt.close(fig)

    a0 = df.loc[df["Class"] == 0, "Amount"]
    a1 = df.loc[df["Class"] == 1, "Amount"]
    _, p = mannwhitneyu(a1, a0, alternative="two-sided")
    print(f"  Amount Mann-Whitney p = {p:.4e}")
    return tbl


def task2_time(df: pd.DataFrame) -> pd.DataFrame:
    tbl = grouped_desc(df, "Time", ["mean", "median", "std"])
    tbl[["Class", "Mean", "Median", "SD"]].to_csv(
        TABLE_DIR / "Time按类别统计.csv", index=False, encoding="utf-8-sig"
    )

    palette = {"正常": "#4C78A8", "欺诈": "#E45756"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sns.boxplot(
        data=df, x="ClassLabel", y="Time", hue="ClassLabel",
        order=["正常", "欺诈"], hue_order=["正常", "欺诈"],
        palette=palette, legend=False, ax=axes[0],
    )
    axes[0].set_title("Time 箱线图")
    axes[0].set_ylabel("Time (秒)")

    for label, color in zip(["正常", "欺诈"], ["#4C78A8", "#E45756"]):
        sub = df.loc[df["ClassLabel"] == label, "Time"]
        sns.histplot(sub, label=label, color=color, ax=axes[1], bins=40, alpha=0.55, stat="density")
    axes[1].set_title("Time 密度直方图（按类别）")
    axes[1].legend()

    fig.suptitle("Figure 2  Time vs Class", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "Figure2_Time_vs_Class.png", bbox_inches="tight")
    plt.close(fig)

    t0 = df.loc[df["Class"] == 0, "Time"]
    t1 = df.loc[df["Class"] == 1, "Time"]
    _, p = mannwhitneyu(t1, t0, alternative="two-sided")
    print(f"  Time Mann-Whitney p = {p:.4e}")
    return tbl


def task3_correlation(df: pd.DataFrame) -> pd.DataFrame:
    corr = df.corr(numeric_only=True)["Class"].drop("Class")
    corr = corr.drop("ClassLabel", errors="ignore")
    ranking = corr.sort_values(key=lambda s: s.abs(), ascending=False).reset_index()
    ranking.columns = ["Variable", "Corr_Class"]
    ranking["Direction"] = ranking["Corr_Class"].apply(
        lambda x: "Negative" if x < 0 else "Positive"
    )

    ranking.to_csv(TABLE_DIR / "Correlation_Ranking.csv", index=False, encoding="utf-8-sig")
    top10 = ranking.head(TOP_N_CORR).copy()
    top10.to_csv(TABLE_DIR / "Top10_与Class相关.csv", index=False, encoding="utf-8-sig")
    return top10, ranking


def task4_top_variables(df: pd.DataFrame, top_vars: list[str]) -> None:
    n = len(top_vars)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    palette = {"正常": "#4C78A8", "欺诈": "#E45756"}

    # KDE
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, var in enumerate(top_vars):
        sns.kdeplot(
            data=df, x=var, hue="ClassLabel", hue_order=["正常", "欺诈"],
            palette=palette, ax=axes[i], common_norm=False,
        )
        axes[i].set_title(f"{var} KDE")
    for j in range(len(top_vars), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Figure 3  Top PCA Variables KDE vs Class", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "Figure3_Top变量KDE_vs_Class.png", bbox_inches="tight")
    plt.close(fig)

    # Boxplot
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes2 = np.atleast_1d(axes2).flatten()
    for i, var in enumerate(top_vars):
        sns.boxplot(
            data=df, x="ClassLabel", y=var, hue="ClassLabel",
            order=["正常", "欺诈"], hue_order=["正常", "欺诈"],
            palette=palette, legend=False, ax=axes2[i],
        )
        axes2[i].set_title(f"{var} 箱线图")
    for j in range(len(top_vars), len(axes2)):
        axes2[j].axis("off")
    fig2.suptitle("Figure 4  Top PCA Variables Boxplot vs Class", fontsize=12)
    plt.tight_layout()
    fig2.savefig(FIG_DIR / "Figure4_Top变量Boxplot_vs_Class.png", bbox_inches="tight")
    plt.close(fig2)


def task5_summary(top10: pd.DataFrame) -> pd.DataFrame:
    summary = top10[["Variable", "Corr_Class", "Direction"]].copy()
    summary.columns = ["Variable", "Corr(Class)", "Direction"]
    summary.to_csv(TABLE_DIR / "Top变量候选总结.csv", index=False, encoding="utf-8-sig")
    return summary


def write_report_md(
    amt_tbl: pd.DataFrame,
    time_tbl: pd.DataFrame,
    summary: pd.DataFrame,
    top_vars: list[str],
) -> None:
    lines = [
        "# 2.4 特征与类别关系分析报告",
        "",
        "## 输出",
        "- Figure1_Amount_vs_Class.png",
        "- Figure2_Time_vs_Class.png",
        "- Figure3_Top变量KDE_vs_Class.png",
        "- Figure4_Top变量Boxplot_vs_Class.png",
        "",
        "## Top 5 候选变量",
        ", ".join(top_vars),
    ]
    (HERE / "2.4_特征与类别关系分析报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ensure_dirs()
    setup_plot()
    print("=" * 60)
    print("2.4 特征与类别关系分析")
    print("=" * 60)

    df = load_data()

    print("\n【Task 1】Amount 按类别比较")
    amt_tbl = task1_amount(df)
    print(amt_tbl[["Class", "Mean", "Median", "Q1", "Q3", "Max"]].to_string(index=False))

    print("\n【Task 2】Time 按类别比较")
    time_tbl = task2_time(df)
    print(time_tbl[["Class", "Mean", "Median", "SD"]].to_string(index=False))

    print("\n【Task 3】与 Class 相关性 Top 10")
    top10, full_rank = task3_correlation(df)
    print(top10.to_string(index=False))

    top_vars = top10["Variable"].head(TOP_N_PLOT).tolist()
    print(f"\n【Task 4】Top {TOP_N_PLOT} 变量可视化: {', '.join(top_vars)}")
    task4_top_variables(df, top_vars)

    print("\n【Task 5】特征重要性候选")
    summary = task5_summary(top10)
    print(summary.to_string(index=False))

    write_report_md(amt_tbl, time_tbl, summary, top_vars)

    print("\n【本节结论】")
    print("- Amount、Time 与 Class 存在可检出的组间差异")
    print(f"- 与 Class 相关最强: {', '.join(top_vars)}")
    print("- 上述变量可作为 Rare Events Logistic 的重要候选解释变量")
    print(f"\n输出: {HERE / '输出'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
