"""仓库路径：克隆后请将 creditcard.csv 放入 data/ 目录。"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
CHAPTER4_FIG_DIR = OUTPUT_DIR / "第四章配图"


def resolve_data_csv() -> Path:
    """优先 data/creditcard.csv；兼容本地开发时上级目录的旧路径。"""
    candidates = [
        DATA_DIR / "creditcard.csv",
        REPO_ROOT.parent / "creditcard.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]
