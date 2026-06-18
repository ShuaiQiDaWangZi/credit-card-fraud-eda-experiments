"""
数据质量评估 — 单文件脚本

用途：复现上一组实验中的 2.1 结论，并逐项核对是否正确。
运行：在本目录执行  py 数据质量评估.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

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

# ---------------------------------------------------------------------------
# 上一组实验报告中的「预期结论」（用于核对）
# ---------------------------------------------------------------------------
EXPECTED = {
    "样本量": 284_807,
    "变量数量": 31,
    "解释变量数": 30,
    "全部为数值型": True,
    "缺失值总数": 0,
    "完全重复行数": 1_081,
    "重复率_百分比_约": 0.38,  # 报告中写 0.38%
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def check_pass(label: str, actual, expected, tol: float | None = None) -> bool:
    if tol is not None:
        ok = abs(float(actual) - float(expected)) <= tol
    else:
        ok = actual == expected
    mark = "通过" if ok else "不一致"
    print(f"  [{mark}] {label}: 实际={actual!r}  预期={expected!r}")
    return ok


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ensure_dirs()
    print("=" * 60)
    print("数据质量评估 — 核对上一组实验结论")
    print("=" * 60)
    print(f"数据: {DATA_PATH}\n")

    df = load_data()
    n, p = df.shape
    n_feat = p - 1
    all_numeric = bool(df.dtypes.apply(pd.api.types.is_numeric_dtype).all())
    missing_total = int(df.isnull().sum().sum())
    n_dup = int(df.duplicated().sum())
    dup_pct = n_dup / n * 100

    actual = {
        "样本量": n,
        "变量数量": p,
        "解释变量数": n_feat,
        "全部为数值型": all_numeric,
        "缺失值总数": missing_total,
        "完全重复行数": n_dup,
        "重复率_百分比": round(dup_pct, 4),
    }

    # ----- （1）基本信息 -----
    print("【（1）数据基本信息】")
    lines_basic = [
        f"样本量: {n:,}",
        f"变量数量: {p}（解释变量 {n_feat} + Class）",
        f"全部为数值型: {all_numeric}",
        "",
        "各列类型:",
        df.dtypes.to_string(),
    ]
    text_basic = "\n".join(lines_basic)
    (TABLE_DIR / "01_数据基本信息.txt").write_text(text_basic, encoding="utf-8")
    pd.DataFrame({"列名": df.columns, "dtype": df.dtypes.astype(str)}).to_csv(
        TABLE_DIR / "01_各列数据类型.csv", index=False, encoding="utf-8-sig"
    )
    print(text_basic[:200] + "...\n")

    # ----- （2）缺失值 -----
    print("【（2）缺失值检查】")
    miss_df = (
        df.isnull()
        .sum()
        .rename("Missing Count")
        .to_frame()
        .assign(**{"Missing %": (df.isnull().mean() * 100).round(4)})
        .reset_index()
        .rename(columns={"index": "Variable"})
    )
    miss_df.to_csv(TABLE_DIR / "02_缺失值统计.csv", index=False, encoding="utf-8-sig")
    print(f"  总缺失: {missing_total}，各列均为 0\n")

    # ----- （3）重复值 -----
    print("【（3）重复值检查】")
    dup_text = (
        f"完全重复行数: {n_dup:,}\n"
        f"重复率: {dup_pct:.4f}%（报告约 {EXPECTED['重复率_百分比_约']}%）\n"
        f"EDA 建议: 保留全部样本"
    )
    (TABLE_DIR / "03_重复值检查.txt").write_text(dup_text, encoding="utf-8")
    pd.DataFrame(
        [{"总样本量": n, "完全重复行数": n_dup, "重复率_百分比": round(dup_pct, 4)}]
    ).to_csv(TABLE_DIR / "03_重复值统计.csv", index=False, encoding="utf-8-sig")
    print(f"  {dup_text}\n")

    # ----- 表 2.1 -----
    table21 = pd.DataFrame(
        [
            {"检查项目": "样本量", "结果": f"{n:,}", "说明": "数据规模较大，适合统计分析"},
            {
                "检查项目": "变量数量",
                "结果": str(p),
                "说明": f"{n_feat}个解释变量 + 1个响应变量Class",
            },
            {"检查项目": "数据类型", "结果": "全部为数值型", "说明": "无需类别编码或文本处理"},
            {"检查项目": "缺失值", "结果": str(missing_total), "说明": "数据完整性良好"},
            {
                "检查项目": "重复记录",
                "结果": f"{n_dup:,}条（{dup_pct:.2f}%）",
                "说明": "占比低，EDA阶段保留",
            },
        ]
    )
    table21.to_csv(TABLE_DIR / "表2.1_数据质量检查汇总.csv", index=False, encoding="utf-8-sig")
    md_rows = [
        "## 表 2.1 数据质量检查汇总",
        "",
        "| 检查项目 | 结果 | 说明 |",
        "| -------- | ---- | ---- |",
    ]
    for _, row in table21.iterrows():
        md_rows.append(f"| {row['检查项目']} | {row['结果']} | {row['说明']} |")
    (TABLE_DIR / "表2.1_数据质量检查汇总.md").write_text("\n".join(md_rows), encoding="utf-8")

    # ----- 核对上一组结论 -----
    print("=" * 60)
    print("【结论核对】与上一组实验报告对照")
    print("=" * 60)
    results = []
    results.append(check_pass("样本量", actual["样本量"], EXPECTED["样本量"]))
    results.append(check_pass("变量数量", actual["变量数量"], EXPECTED["变量数量"]))
    results.append(check_pass("解释变量数", actual["解释变量数"], EXPECTED["解释变量数"]))
    results.append(check_pass("全部为数值型", actual["全部为数值型"], EXPECTED["全部为数值型"]))
    results.append(check_pass("缺失值总数", actual["缺失值总数"], EXPECTED["缺失值总数"]))
    results.append(check_pass("完全重复行数", actual["完全重复行数"], EXPECTED["完全重复行数"]))
    results.append(
        check_pass(
            "重复率(%)",
            round(dup_pct, 2),
            EXPECTED["重复率_百分比_约"],
            tol=0.01,
        )
    )

    verify_df = pd.DataFrame(
        [
            {
                "检查项": "样本量",
                "上一组结论": EXPECTED["样本量"],
                "本次结果": actual["样本量"],
                "是否一致": actual["样本量"] == EXPECTED["样本量"],
            },
            {
                "检查项": "变量数量",
                "上一组结论": EXPECTED["变量数量"],
                "本次结果": actual["变量数量"],
                "是否一致": actual["变量数量"] == EXPECTED["变量数量"],
            },
            {
                "检查项": "解释变量数",
                "上一组结论": EXPECTED["解释变量数"],
                "本次结果": actual["解释变量数"],
                "是否一致": actual["解释变量数"] == EXPECTED["解释变量数"],
            },
            {
                "检查项": "全部为数值型",
                "上一组结论": EXPECTED["全部为数值型"],
                "本次结果": actual["全部为数值型"],
                "是否一致": actual["全部为数值型"] == EXPECTED["全部为数值型"],
            },
            {
                "检查项": "缺失值总数",
                "上一组结论": EXPECTED["缺失值总数"],
                "本次结果": actual["缺失值总数"],
                "是否一致": actual["缺失值总数"] == EXPECTED["缺失值总数"],
            },
            {
                "检查项": "完全重复行数",
                "上一组结论": EXPECTED["完全重复行数"],
                "本次结果": actual["完全重复行数"],
                "是否一致": actual["完全重复行数"] == EXPECTED["完全重复行数"],
            },
            {
                "检查项": "重复率(%)",
                "上一组结论": EXPECTED["重复率_百分比_约"],
                "本次结果": round(dup_pct, 2),
                "是否一致": abs(round(dup_pct, 2) - EXPECTED["重复率_百分比_约"]) <= 0.01,
            },
        ]
    )
    verify_path = TABLE_DIR / "结论核对结果.csv"
    verify_df.to_csv(verify_path, index=False, encoding="utf-8-sig")

    all_ok = all(results)
    print("\n" + "=" * 60)
    if all_ok:
        print("总评: 上一组实验关于数据质量的结论 — 全部验证通过。")
    else:
        print("总评: 存在与上一组不一致的项目，请查看上方 [不一致] 项。")
    print(f"\n输出目录: {TABLE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
