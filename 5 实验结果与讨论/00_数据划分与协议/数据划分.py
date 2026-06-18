"""按附录协议生成 train / val / test 划分（三 seed）。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "公共"))
from 实验工具 import (  # noqa: E402
    SEEDS,
    TARGET_COL,
    ensure_dir,
    load_full_data,
    save_splits,
    split_indices,
    SPLIT_SUMMARY,
)

OUT_TABLE = Path(__file__).resolve().parent / "输出" / "tables"


def main() -> None:
    df = load_full_data()
    y = df[TARGET_COL].to_numpy()
    rows = []
    for seed in SEEDS:
        splits = split_indices(y, seed)
        save_splits(seed, splits)
        for name, idx in splits.items():
            sub_y = y[idx]
            rows.append(
                {
                    "seed": seed,
                    "split": name,
                    "n": len(idx),
                    "n_fraud": int(sub_y.sum()),
                    "n_normal": int((sub_y == 0).sum()),
                    "fraud_pct": float(sub_y.mean() * 100),
                }
            )
    ensure_dir(OUT_TABLE)
    pd.DataFrame(rows).to_csv(SPLIT_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"已保存划分至 {SPLIT_SUMMARY}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
