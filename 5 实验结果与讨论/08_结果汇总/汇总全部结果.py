"""合并 M0–M10 全部结果表。"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "公共"))
from 实验工具 import ensure_dir  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ensure_dir(HERE / "输出" / "tables")

SOURCES = [
    ROOT / "01_M0_基准Logistic" / "输出" / "tables" / "M0_各seed指标.csv",
    ROOT / "02_M1-M4_比例欠采样" / "输出" / "tables" / "M1-M4_各seed指标.csv",
    ROOT / "03_M5-M6_合成增强" / "输出" / "tables" / "M5-M6_各seed指标.csv",
    ROOT / "04_M7_加权Logistic" / "输出" / "tables" / "M7_各seed指标.csv",
    ROOT / "05_M8_稀有事件Logistic" / "输出" / "tables" / "M8_各seed指标.csv",
    ROOT / "06_M9_RandomForest" / "输出" / "tables" / "M9_各seed指标.csv",
    ROOT / "07_M10_XGBoost" / "输出" / "tables" / "M10_各seed指标.csv",
]

CORE = ["pr_auc", "brier", "precision", "recall", "f1"]


def main() -> None:
    frames = [pd.read_csv(p) for p in SOURCES if p.exists()]
    missing = [p for p in SOURCES if not p.exists()]
    if missing:
        print("缺少以下结果文件（请先运行对应脚本）：")
        for p in missing:
            print(" ", p.relative_to(ROOT))
    if not frames:
        raise FileNotFoundError("未找到任何模型结果 CSV。")
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUT / "全部方法_各seed明细.csv", index=False, encoding="utf-8-sig")

    summary = all_df.groupby(["method", "route"])[CORE].agg(["mean", "std"])
    summary.to_csv(OUT / "全部方法_汇总_mean_std.csv", encoding="utf-8-sig")

    core = all_df.groupby("method")[CORE].agg(["mean", "std"]).sort_index()
    core.to_csv(OUT / "表_核心指标对比.csv", encoding="utf-8-sig")
    print(core)
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
