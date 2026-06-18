# 5 实验结果与讨论

实验协议见主文件 `信用卡欺诈数据介绍.md` **附录：实验协议**。

## 执行顺序

| 步骤 | 文件夹 | 命令 |
|------|--------|------|
| 00 | `00_数据划分与协议/` | `py 数据划分.py` |
| 01 | `01_M0_基准Logistic/` | `py M0_基准Logistic.py` |
| 02 | `02_M1-M4_比例欠采样/` | `py M1-M4_比例欠采样.py` |
| 03 | `03_M5-M6_合成增强/` | `py M5-M6_合成增强.py` |
| 04 | `04_M7_加权Logistic/` | `py M7_加权Logistic.py` |
| 05 | `05_M8_稀有事件Logistic/` | `py M8_稀有事件Logistic.py` |
| 06 | `06_M9_RandomForest/` | `py M9_随机森林.py`（轻量固定超参） |
| 07 | `07_M10_XGBoost/` | `py M10_XGBoost.py`（轻量固定超参） |
| 08 | `08_结果汇总/` | `py 汇总全部结果.py`；`py 生成实验图表.py` → `output/第四章配图/` |

配图输出目录：仓库根目录 `output/第四章配图/`（图 4-1–图 4-6）。**不生成** PR 曲线图与 Calibration 曲线图；概率校准在正文中以 Brier（图 4-5）讨论。

## 依赖

```powershell
py -m pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib
```
