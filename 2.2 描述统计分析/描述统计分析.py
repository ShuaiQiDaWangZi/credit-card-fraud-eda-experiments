"""
2.2 描述统计分析 — 单文件脚本

重点变量：Time、Amount（原始业务变量）
不涉及 Fraud vs Non-Fraud（属 2.4）

运行：py 描述统计分析.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
import sys
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from paths import resolve_data_csv

DATA_PATH = resolve_data_csv()
OUT_DIR = HERE / "输出"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

VARS = ["Time", "Amount"]
OBSERVATION_HOURS = 48  # 数据集为两天交易


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


def describe_time_amount(df: pd.DataFrame) -> pd.DataFrame:
    """表：Time & Amount 描述统计（Mean, Median, SD, Min, Q1, Q3, Max）。"""
    rows = []
    for var in VARS:
        s = df[var]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows.append(
            {
                "Variable": var,
                "Mean": s.mean(),
                "Median": s.median(),
                "SD": s.std(),
                "Min": s.min(),
                "Q1": q1,
                "Q3": q3,
                "Max": s.max(),
            }
        )
    return pd.DataFrame(rows)


def extra_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """IQR、箱线图上界、超上界数量、偏度、均值/中位数比等。"""
    rows = []
    for var in VARS:
        s = df[var]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        n_above = int((s > upper).sum())
        median = s.median()
        mean_med_ratio = s.mean() / median if median != 0 else np.nan
        rows.append(
            {
                "Variable": var,
                "Skewness": s.skew(),
                "IQR": iqr,
                "IQR上界_Q3+1.5IQR": upper,
                "超出上界观测数": n_above,
                "超出上界占比_%": round(n_above / len(s) * 100, 4),
                "Mean_Median_Ratio": round(mean_med_ratio, 4),
                "Max_Q3_倍数": round(s.max() / q3, 4) if q3 != 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_amount_histogram(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df["Amount"], bins=50, kde=False, color="#4C78A8", ax=ax)
    ax.set_title("Figure 1  Amount 直方图（原始尺度）")
    ax.set_xlabel("Amount")
    ax.set_ylabel("频数")
    path = FIG_DIR / "Figure1_Amount直方图.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_log_amount_histogram(df: pd.DataFrame) -> Path:
    log_amt = np.log1p(df["Amount"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(log_amt, bins=50, kde=False, color="#72B7B2", ax=ax)
    ax.set_title("Figure 2  log(Amount+1) 直方图")
    ax.set_xlabel("log(Amount+1)")
    ax.set_ylabel("频数")
    path = FIG_DIR / "Figure2_LogAmount直方图.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_amount_boxplot(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(x=df["Amount"], color="#4C78A8", ax=ax)
    ax.set_title("Figure 3  Amount 箱线图")
    ax.set_xlabel("Amount")
    path = FIG_DIR / "Figure3_Amount箱线图.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_time_histogram(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(df["Time"], bins=60, color="#4C78A8", ax=ax)
    ax.set_title("Figure 4  Time 直方图")
    ax.set_xlabel("Time（距首笔交易的秒数）")
    ax.set_ylabel("频数")
    span_h = (df["Time"].max() - df["Time"].min()) / 3600
    ax.text(
        0.98,
        0.95,
        f"跨度约 {span_h:.1f} 小时",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    path = FIG_DIR / "Figure4_Time直方图.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_time_line_optional(df: pd.DataFrame, n_bins: int = 48) -> Path:
    """按时间分箱统计交易笔数（折线，可选）。"""
    counts, edges = np.histogram(df["Time"], bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2 / 3600  # 转为小时

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(centers, counts, color="#E45756", linewidth=1.2)
    ax.set_title("Figure 5（可选）按时间窗口的交易笔数")
    ax.set_xlabel("Time（小时，自首笔交易起）")
    ax.set_ylabel("交易笔数")
    path = FIG_DIR / "Figure5_Time交易笔数折线.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_pca_note() -> None:
    text = (
        "2.2.4 PCA 匿名变量说明\n"
        "────────────────────\n"
        "V1–V28 为经 PCA 处理后的匿名主成分变量，已在构造过程中标准化，"
        "不具有 Time、Amount 那样的直接业务含义，故本节不对其做描述统计与业务解释；"
        "相关分析留待后续特征关系与建模章节。\n"
    )
    (TABLE_DIR / "2.2.4_PCA变量说明.txt").write_text(text, encoding="utf-8")


def build_report_md(
    desc: pd.DataFrame, extra: pd.DataFrame, time_span_h: float
) -> str:
    t = desc[desc["Variable"] == "Time"].iloc[0]
    a = desc[desc["Variable"] == "Amount"].iloc[0]
    te = extra[extra["Variable"] == "Time"].iloc[0]
    ae = extra[extra["Variable"] == "Amount"].iloc[0]

    md = f"""# 2.2 描述统计分析（Descriptive Statistics）

> 由 `描述统计分析.py` 自动生成。重点：Time、Amount。

## 目标

研究 Time、Amount 的分布特征、偏态、极端值与数值跨度，为 Rare Events 分析、特征关系分析与 Logistic 建模做准备。

---

## 2.2.1 主要变量描述统计

### Table 1  Time & Amount describe

| Variable | Mean | Median | SD | Min | Q1 | Q3 | Max |
| -------- | ---- | ------ | -- | --- | -- | -- | --- |
| Time | {t['Mean']:.2f} | {t['Median']:.2f} | {t['SD']:.2f} | {t['Min']:.2f} | {t['Q1']:.2f} | {t['Q3']:.2f} | {t['Max']:.2f} |
| Amount | {a['Mean']:.2f} | {a['Median']:.2f} | {a['SD']:.2f} | {a['Min']:.2f} | {a['Q1']:.2f} | {a['Q3']:.2f} | {a['Max']:.2f} |

### 补充统计量

| Variable | Skewness | IQR | IQR上界 | 超出上界数 | 超出上界% | Mean/Median |
| -------- | -------- | --- | ------- | ---------- | --------- | ----------- |
| Time | {te['Skewness']:.4f} | {te['IQR']:.2f} | {te['IQR上界_Q3+1.5IQR']:.2f} | {int(te['超出上界观测数'])} | {te['超出上界占比_%']}% | {te['Mean_Median_Ratio']} |
| Amount | {ae['Skewness']:.4f} | {ae['IQR']:.2f} | {ae['IQR上界_Q3+1.5IQR']:.2f} | {int(ae['超出上界观测数'])} | {ae['超出上界占比_%']}% | {ae['Mean_Median_Ratio']} |

#### （1）Time

- 观测窗口：约 **{time_span_h:.1f} 小时**（Time 从 {t['Min']:.0f}s 到 {t['Max']:.0f}s），覆盖完整约 {OBSERVATION_HOURS} 小时观测期。
- 均值 ({t['Mean']:.0f}s) 与中位数 ({t['Median']:.0f}s) {'接近' if abs(t['Mean']-t['Median'])/t['Median']<0.15 else '存在一定差异'}。
- 偏度 {te['Skewness']:.4f}，{'无明显强偏' if abs(te['Skewness'])<1 else '存在一定偏态'}。

#### （2）Amount

- Mean ({a['Mean']:.2f}) **大于** Median ({a['Median']:.2f})，Mean/Median = {ae['Mean_Median_Ratio']}，呈**右偏**。
- 偏度 {ae['Skewness']:.4f}，右偏明显。
- Max ({a['Max']:.2f}) 远高于 Q3 ({a['Q3']:.2f})，约为 Q3 的 {ae['Max_Q3_倍数']} 倍，存在**大量极端高额交易**。
- IQR 上界约 {ae['IQR上界_Q3+1.5IQR']:.2f}，超出上界 **{int(ae['超出上界观测数']):,}** 笔（{ae['超出上界占比_%']}%）。

---

## 2.2.2 Amount 分布可视化

| 图 | 文件 |
|----|------|
| Figure 1 | `输出/figures/Figure1_Amount直方图.png` |
| Figure 2 | `输出/figures/Figure2_LogAmount直方图.png` |
| Figure 3 | `输出/figures/Figure3_Amount箱线图.png` |

**观察要点：**

- 原始直方图：右偏、长右尾，绝大多数交易集中在小金额区域。
- 对数直方图：log(Amount+1) 后分布更平滑，但仍可见长尾。
- 箱线图：离群点众多，与 IQR 上界统计一致。

---

## 2.2.3 Time 分布可视化

| 图 | 文件 |
|----|------|
| Figure 4 | `输出/figures/Figure4_Time直方图.png` |
| Figure 5（可选） | `输出/figures/Figure5_Time交易笔数折线.png` |

**观察要点：** 交易在约 48 小时内非均匀分布，存在活跃高峰时段（见折线图）。

---

## 2.2.4 PCA 匿名变量说明

V1–V28 为 PCA 匿名变量，已标准化，不具有直接业务解释意义；本节不做描述统计。

---

## 2.2 核心结论

1. **Time** 覆盖完整观测周期（约 {time_span_h:.0f} 小时）。
2. **Amount** 明显右偏（偏度 ≈ {ae['Skewness']:.1f}）。
3. **Amount** 存在大量极端值（约 {int(ae['超出上界观测数']):,} 笔超出箱线图 IQR 上界）。
4. **Amount** 金额跨度极大（{a['Min']:.2f} ~ {a['Max']:.2f}）。
5. **V1–V28** 为已标准化 PCA 变量，业务解释留待后续章节。

**说明：** 本节未进行 Fraud vs Non-Fraud 比较（属 2.4）。
"""
    return md


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ensure_dirs()
    setup_plot()
    print("=" * 60)
    print("2.2 描述统计分析（Time & Amount）")
    print("=" * 60)

    df = load_data()
    time_span_h = (df["Time"].max() - df["Time"].min()) / 3600

    # 表格
    desc = describe_time_amount(df)
    extra = extra_statistics(df)
    desc.to_csv(TABLE_DIR / "Table1_Time_Amount描述统计.csv", index=False, encoding="utf-8-sig")
    extra.to_csv(TABLE_DIR / "补充统计量_IQR与偏态.csv", index=False, encoding="utf-8-sig")

    print("\n【Table 1】Time & Amount describe")
    print(desc.to_string(index=False))
    print("\n【补充统计量】")
    print(extra.to_string(index=False))

    # 图形
    p1 = plot_amount_histogram(df)
    p2 = plot_log_amount_histogram(df)
    p3 = plot_amount_boxplot(df)
    p4 = plot_time_histogram(df)
    p5 = plot_time_line_optional(df)
    print("\n【图形已保存】")
    for p in (p1, p2, p3, p4, p5):
        print(f"  {p.name}")

    write_pca_note()

    report = build_report_md(desc, extra, time_span_h)
    report_path = HERE / "2.2_描述统计分析报告.md"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print("完成。报告:", report_path.name)
    print("表格:", TABLE_DIR)
    print("图形:", FIG_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
