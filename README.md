# StockFactor\_code

基于证券交易大数据的 AI 模型选股因子多路径挖掘探索项目代码库。该项目综合运用遗传规划（GP）、随机森林、以及大语言模型（LLM）等多种路径，自动化挖掘出具有预测价值的量化选股因子，辅助构建多因子策略组合并实现实盘级别的因子验证与回测。

## 📌 项目亮点

* 🧠 多路径因子挖掘：整合 GP、RF、LLM 等方法，提升因子多样性与预测能力
* 🧮 自定义函数集拓展 gplearn，用于表达复杂的非线性量价关系
* 🤖 使用 GPT/Kimi 大模型生成因子表达式，探索 NLP 在金融因子构建的应用
* 📊 支持完整因子回测分析：IC、IR、收益率、行业分布、换手率
* 📈 可视化热力图展示因子表现，并支持多因子组合策略设计与验证

---

## 🗂️ 项目结构

```
StockFactor_code/
├── examples/               # 示例因子构建流程
├── gp/                    # 基于遗传规划的因子挖掘模块
│   ├── function_set.py    # 自定义函数集
│   ├── get_data.py        # 数据读取与预处理
│   ├── new_gp.py          # 遗传规划主函数
├── strategy/              # 多因子策略构建与回测模块
│   └── factor_strategy.py
├── main.py                # 主运行入口
├── other.py               # 辅助函数集合
└── README.md              # 项目说明
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/LeonnelTang/StockFactor_code.git
cd StockFactor_code
```

### 2. 安装依赖

建议使用虚拟环境（如 `venv` 或 `conda`）：

```bash
pip install -r requirements.txt
```

依赖包括但不限于：

* `pandas`, `numpy`
* `scikit-learn`
* `gplearn`
* `matplotlib`, `seaborn`
* `shap`

### 3. 运行示例

```bash
python main.py
```

或运行具体模块：

```bash
# 运行遗传规划因子挖掘
python gp/new_gp.py

# 回测策略构建与结果分析
python strategy/factor_strategy.py
```

---

## 📘 示例因子表达式（GP 生成）

```python
Alpha1 = mul(div(log(abs(total_turnover_Lag1)), SIGN(volume_Lag20)),
             EMA(div(high, high_Lag20)))
```

**年化收益率**：19.72%
**多空累计收益**：105.41%

---

## 📊 示例可视化

* IC 时间序列图
* 多空组合收益曲线
* 因子暴露热力图

示例文件输出：

```
factor_0.0001_HS300.csv
heatmap_0.0001_HS300.png
```

---

## 🧠 项目方法概述

| 方法               | 描述                                   |
| ---------------- | ------------------------------------ |
| 遗传规划 (gplearn)   | 自动生成因子表达式，优化 RankIC，支持函数扩展           |
| 随机森林 (RF)        | 合成基本因子进行涨跌预测，用预测值作为因子                |
| 大语言模型 (GPT/Kimi) | 利用 Prompt Engineering 生成逻辑合理的选股因子表达式 |

---

## 🤝 致谢

本项目由华东师范大学创新创业培育项目支持，感谢米筐数据平台教育版支持。感谢宫峰飞副教授与韩莉副教授在技术指导方面提供的重要帮助。

