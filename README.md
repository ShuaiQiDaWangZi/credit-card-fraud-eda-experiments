# 信用卡欺诈检测：探索性分析与建模实验

[![GitHub](https://img.shields.io/badge/GitHub-ShuaiQiDaWangZi%2Fcredit--card--fraud--eda--experiments-blue?logo=github)](https://github.com/ShuaiQiDaWangZi/credit-card-fraud-eda-experiments)

**作者：** 帅气大王子  
**单位：** 中国科学技术大学（USTC）  
**日期：** 2026 年 6 月 11 日

本仓库为 USTC 课程大作业的**实验源码与复现脚本**（Python），涵盖 Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 数据集上的探索性分析与十一组建模对比（M0–M10）。

> 汇报用 Beamer 幻灯片、完整文字报告等**不在本仓库**；此处仅供老师审阅代码与复现实验。

---

## 数据准备

原始数据**未随仓库上传**（约 144 MB）。请先下载并放置：

```text
data/creditcard.csv
```

说明见 [data/README.md](./data/README.md)。  
Kaggle：[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

本地若在上级目录已有 `creditcard.csv`，脚本也会自动兼容。

---

## 环境

```powershell
py -m pip install -r requirements.txt
```

依赖：`pandas`、`numpy`、`scikit-learn`、`imbalanced-learn`、`xgboost`、`matplotlib`

---

## 目录结构

| 目录 | 内容 |
|------|------|
| `2.1 数据质量评估/` | 缺失值、重复记录、变量类型 |
| `2.2 描述统计分析/` | Time、Amount 分布与图表 |
| `2.3 类别分布与稀有事件分析/` | 578:1 失衡、Accuracy 陷阱 |
| `2.4 特征与类别关系分析/` | Amount/Time/V 与 Class |
| `5 实验结果与讨论/` | M0–M10 实验、划分协议、结果汇总 |
| `output/第四章配图/` | 运行 `生成实验图表.py` 后生成的图 4-1–4-6 |

各子目录均有 `README.md` 与 `输出/`（已包含本次运行的 CSV 结果，便于直接查看）。

---

## 运行顺序

### 1. 探索性分析（§2，可独立运行）

```powershell
cd "2.1 数据质量评估"
py 数据质量评估.py

cd "../2.2 描述统计分析"
py 描述统计分析.py

cd "../2.3 类别分布与稀有事件分析"
py 类别分布与稀有事件分析.py

cd "../2.4 特征与类别关系分析"
py 特征与类别关系分析.py
```

### 2. 建模实验（§4，须按序）

```powershell
cd "5 实验结果与讨论"

cd "00_数据划分与协议"
py 数据划分.py

cd "../01_M0_基准Logistic"
py M0_基准Logistic.py

cd "../02_M1-M4_比例欠采样"
py M1-M4_比例欠采样.py

cd "../03_M5-M6_合成增强"
py M5-M6_合成增强.py

cd "../04_M7_加权Logistic"
py M7_加权Logistic.py

cd "../05_M8_稀有事件Logistic"
py M8_稀有事件Logistic.py

cd "../06_M9_RandomForest"
py M9_随机森林.py

cd "../07_M10_XGBoost"
py M10_XGBoost.py

cd "../08_结果汇总"
py 汇总全部结果.py
py 生成实验图表.py
```

详细协议见 [5 实验结果与讨论/README.md](./5%20实验结果与讨论/README.md)。

---

## 实验设计摘要

- **划分：** 训练 64% / 验证 16% / 测试 20%；随机种子 42、123、456  
- **路线：** M0 基准 → M1–M7 数据层 → M8 统计层（KZ 稀有事件 LR）→ M9–M10 容量层（RF、XGBoost）  
- **评价：** PR-AUC、Precision、Recall、F1、Brier；阈值在验证集选取  

---

## 许可与引用

代码仅供课程作业与学术交流。数据版权归 Kaggle / 原发布者所有。  
使用本仓库请注明「帅气大王子 / USTC 课程作业」出处。
