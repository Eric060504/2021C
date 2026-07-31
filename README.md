# 2021 C 题：生产企业原材料订购与运输（Python 求解）

本项目依据 `2021_C.docx` 的建模框架，对供应商重要性评价、最少供应商选择、24 周订购与转运、成本压缩及产能提升四个问题进行可复现求解。原始 Excel 附件不会被修改，生成的订购与转运结果统一写入 `outputs/`。

## 当前实现状态

- 问题 1：六项供货指标、极差标准化、熵权法综合评分，并输出前 50 家重要供应商。
- 问题 2：在前 50 家重要供应商中选择满足 A/B/C 材料覆盖与目标产能的最少供应商，并生成 24 周订购、库存和转运方案。
- 问题 3：在 A/C 材料偏好、采购奖惩和低损耗率硬阈值不变的前提下，使用未来 24 周动态供给上限与跨周库存调节编制方案；损耗率超过阈值的转运商不会被使用。
- 问题 4：使用全部 402 家供应商，以未来 24 周动态供给预测、逐周转运能力和两周安全库存为约束，计算最大可持续周产能并给出动态订购、转运方案。

## 运行环境

推荐使用项目当前的 Conda 环境：

```powershell
& 'D:\anaconda\envs\LLM_Classification\python.exe' -X utf8 -m pytest -v
```

主要依赖：`numpy`、`pandas`、`scipy`（HiGHS 线性规划）、`openpyxl`、`pytest`。

## 运行各问题

```powershell
& 'D:\anaconda\envs\LLM_Classification\python.exe' -X utf8 q1_supplier_importance.py
& 'D:\anaconda\envs\LLM_Classification\python.exe' -X utf8 q2_economic_order_transport.py
& 'D:\anaconda\envs\LLM_Classification\python.exe' -X utf8 q3_cost_compression_plan.py
& 'D:\anaconda\envs\LLM_Classification\python.exe' -X utf8 q4_capacity_expansion.py
```

| 题目 | 程序入口 | 核心模型 | 主要输出 |
|---|---|---|---|
| 问题 1 | `q1_supplier_importance.py` | 六指标评价、极差标准化、熵权法 | `outputs/q1/supplier_ranking.xlsx`、`outputs/q1/top_50_suppliers.xlsx` |
| 问题 2 | `q2_economic_order_transport.py` | 最少供应商筛选、24 周库存线性规划、低损耗转运 | 最终附件 A/B 的“问题 2”工作表 |
| 问题 3 | `q3_cost_compression_plan.py` | 动态供给预测、A/C 材料偏好、采购奖惩、低损耗率硬约束、跨周库存和控制变量敏感性分析 | 最终附件 A/B 的“问题 3”工作表、`outputs/q3/supplier_weekly_capacity_forecast.xlsx`、`outputs/q3/sensitivity_analysis.csv` |
| 问题 4 | `q4_capacity_expansion.py` | 全供应商动态最大可持续产能、跨周库存调节、逐周供给与转运运力 | 最终附件 A/B 的“问题 4”工作表、`outputs/q4/summary.xlsx` |

## 建模口径

### 1. 供应商预测

1. 供应商供货率取“有订货记录周”的实际供货率中位数，并按配置上下界截断。
2. 问题 2 使用历史正供货量的 90% 分位数作为静态供给能力。
3. 问题 3、4 在稳健能力基础上，采用最近 24 周实际供给的相对起伏构造未来 24 周逐周供货上限，并以 50% 收缩系数向稳健能力回归，避免直接复制偶然异常周。
4. 附件 2 中的零损耗率代表该周未发生运输，不作为真实零损耗样本参与中位数预测。
5. A/B/C 三类原料统一折算为产品当量；生产 1 m³ 产品分别消耗 0.60/0.66/0.72 m³ 的 A/B/C 类原料。

### 2. 问题 2：最少供应商与订购转运

1. 在问题 1 的前 50 家重要供应商中，保证 A、B、C 三类材料均被覆盖。
2. 基于预测供货能力和产品当量筛选供应商，再利用 24 周线性规划确定订单、预计发运量及库存。
3. 每家转运商每周运力上限为 6,000 m³；按预测损耗率由低到高分配，优先不拆分同一供应商的周发运量。
4. 当前真实附件运行结果中，问题 2 的转运方案未发生供应商拆分，且所有转运商均未超出单周运力。

### 3. 问题 3：动态降本与低损耗转运

1. 供应商集合仍为问题 1 排名前 50 的重要供应商。对每家供应商，以历史稳健能力为基准，结合最近 24 周供给起伏生成未来逐周供货上限。
2. 每周均硬性约束 A 类产品当量不低于 `Q3_A_MIN_SHARE`，C 类产品当量不高于 `Q3_C_MAX_SHARE`；`Q3_A_REWARD` 和 `Q3_C_PENALTY` 仍用于调整采购成本。
3. `Q3_LOW_LOSS_THRESHOLD = 0.02` 是硬约束：预测损耗率超过 2% 的转运商不参与分配；如果合格运力不足，模型会判定方案不可行。
4. 题设要求保留不少于两周的安全库存。为在供给偏紧的前几周仍保持生产，动态情景将期初库存设为 4 周产品当量（`Q3_INITIAL_INVENTORY_WEEKS = 4.0`），并始终强制期末库存不低于 2 周安全库存。这是对期初存量的情景假设，不放宽题设安全库存约束。
5. 敏感性分析仍采用控制变量法，且与基准方案使用同一动态供给、期初库存、逐周运力与低损耗筛选规则。

当前真实附件运行结果：

| 指标 | 数值 |
|---|---:|
| 预测周供货能力范围 | 21,467.31--25,289.56 m³ 原料/周 |
| 计划周发运量范围 | 15,088.30--19,373.92 m³ 原料/周 |
| 期初库存 | 112,800.00 m³ 产品当量（4 周） |
| 实际最低库存 | 89,815.70 m³ 产品当量（高于 56,400.00 m³ 安全线） |
| 实际 A 类平均占比 | 57.69% |
| 实际 C 类平均占比 | 13.23% |

动态供给使问题 3 的 24 周订购、发运、实际到厂量和库存均产生正常的周度变化；变化来自供应商逐周上限和成本、材料偏好及库存约束的联合作用，而非对原方案的人为扰动。

### 4. 问题 4：产能提升

1. 最大产能模型使用全部 402 家供应商，而不是仅使用问题 1 的前 50 家。
2. `build_supplier_weekly_capacity_forecast()` 将最近 24 周的供货起伏转化为“供应商 × 周次”可供货上限；订购模型在每周分别受该上限约束。
3. `build_carrier_capacity_forecast()` 返回“转运商 × 周次”的运力矩阵；当前附件中每家转运商每周运力均为 6,000 m³，但优化接口保留逐周约束。
4. `solve_dynamic_sustainable_capacity()` 仍以单一的最大可持续周产能为目标；它通过库存平衡方程和两周安全库存约束，使供货充足周可以为供货紧张周提前备货。
5. `Q4_MIN_PRODUCT_SHARES` 可设置 A/B/C 三类材料产品当量最低比例。题目未提供具体数值时，默认配置为 `{"A": 0.0, "B": 0.0, "C": 0.0}`，即不额外施加比例限制。

当前真实附件运行结果：

| 指标 | 数值 |
|---|---:|
| 候选供应商数 | 402 |
| 最大可持续周产能 | 35,653.96 m³ 产品/周 |
| 相对当前 28,200 m³/周的提升 | 7,453.96 m³/周（26.43%） |
| 两周安全库存 | 71,307.92 m³ 产品当量 |
| 实际最低库存 | 75,495.43 m³ 产品当量 |
| 预测周供货能力范围 | 23,344.09--27,347.66 m³ 原料/周 |
| 计划周发运量范围 | 23,344.09--26,003.07 m³ 原料/周 |


## 标准化成本与关键配置

题目只说明三类原料的单位运输费和仓储费相同，但未给出绝对金额。因此 `config.py` 使用以 C 类原料单价为 1 的相对价格尺度：

- `RAW_PRICE = {"A": 1.20, "B": 1.10, "C": 1.00}`：三类原料相对采购价格；
- `UNIT_TRANSPORT_COST = 0.02`：单位原料运输的标准化成本；
- `UNIT_HOLDING_COST = 0.005`：单位产品当量、每周的标准化库存持有成本；
- `TRANSPORT_LOSS_BUFFER_RATE = 0.05`：订购阶段使用的保守损耗缓冲率；
- `Q3_LOSS_WEIGHT = 0.65`：问题 3 中用于调整保守损耗缓冲程度的参数；
- `Q3_INITIAL_INVENTORY_WEEKS = 4.0`：问题 3 动态供给情景下的期初库存假设，不改变 2 周安全库存下限；
- `Q4_MIN_PRODUCT_SHARES`：问题 4 的材料最低比例参数；
- `Q4_SUPPLY_FORECAST_SHRINKAGE = 0.50`：问题 3、4 将近期供货起伏向长期稳健能力收缩的系数；
- `Q4_SUPPLY_FORECAST_LOWER_MULTIPLIER = 0.50`、`Q4_SUPPLY_FORECAST_UPPER_MULTIPLIER = 1.50`：逐周预测供给相对近期均值的截断范围。

## 校验与测试

测试覆盖数据读写、单位换算、供给与损耗预测、熵权排序、供应商筛选、库存安全约束、转运容量、低损耗率阈值、材料比例约束、问题 3/4 动态供给计划、时变运力接口、模板保护和真实附件结果复核。

最近一次完整测试结果：**25 passed**。

```powershell
& 'D:\anaconda\envs\LLM_Classification\python.exe' -X utf8 -m pytest -v
```

## 输出文件说明

- `outputs/附件A_最终结果.xlsx`：问题 2、3、4 的订购方案；
- `outputs/附件B_最终结果.xlsx`：问题 2、3、4 的转运方案；
- `outputs/q3/summary.xlsx`、`outputs/q3/inventory_trace.xlsx`、`outputs/q3/supplier_weekly_capacity_forecast.xlsx`、`outputs/q3/sensitivity_analysis.csv`：问题 3 的汇总、动态供给预测、库存轨迹与敏感性分析；
- `outputs/q4/summary.xlsx`、`outputs/q4/inventory_trace.xlsx`、`outputs/q4/supplier_weekly_capacity_forecast.xlsx`：问题 4 的扩产结果、周度库存轨迹与供应商逐周供给预测。

> 注意：请勿将 `outputs/` 中的结果文件覆盖回原始附件。结果工作簿由模板副本生成，仅写入各问题对应的结果区域；分别运行问题 2、3、4 不会清除已写入的其他问题工作表。

## 使用边界

1. 供应能力和损耗率均由历史数据的稳健统计量预测，属于建模假设，应在论文中说明。
2. 问题 3、4 的未来逐周供给是基于近期历史波动的稳健外推，而非真实订单承诺；其结果应解释为情景预测下的有安全库存保障的可行方案。当前附件中的转运能力为常数，但模型已经按周施加转运能力约束，并可直接接入未来时变运力情景。
3. 若要求严格禁止供应商周发运量拆分，应在后续模型中加入供应商—转运商二元分配变量；当前真实附件结果没有发生拆分。

## 自动化审计、可视化与一键运行

每次运行问题 2、3、4 时，程序会在对应 `outputs/q*` 目录额外写出：

- `solution_audit.xlsx`：库存安全、转运运力、预测损耗、供应商拆分和材料比例约束的统一审计表；
- `carrier_utilization.xlsx`：转运商逐周运输量、运力上限与利用率；
- `supplier_split_report.xlsx`：供应商单周发运量是否拆分给多个转运商；
- `material_share.xlsx`：A/B/C 类材料的周度产品当量与占比。

默认还会在 `outputs/figures/` 下生成论文可用的 PNG 图表：

| 问题 | 自动生成的主要图表 |
|---|---|
| 问题 1 | 熵权柱状图、前 20 家供应商得分图、前 50 家指标热力图 |
| 问题 2 | 库存与到厂量图、转运商利用率热力图、入选供应商能力图、材料结构图 |
| 问题 3 | 库存图、动态供给与发运图、材料结构图、低损耗阈值图、转运商利用率图、参数敏感性热力图、问题 2/3 材料结构对比图 |
| 问题 4 | 当前与最大可持续产能对比图、库存图、转运商利用率图、材料结构图、未来24周供给预测与计划发运动态图 |

可一键完成四问求解、审计和制图：

```powershell
& 'D:\anaconda\envs\LLM_Classification\python.exe' -X utf8 run_all.py
```

如只需重新求解而不生成图片：

```powershell
& 'D:\anaconda\envs\LLM_Classification\python.exe' -X utf8 run_all.py --no-plot
```

各问题脚本同样支持 `--output-dir` 与 `--no-plot` 参数，例如：

```powershell
& 'D:\anaconda\envs\LLM_Classification\python.exe' -X utf8 q3_cost_compression_plan.py --no-plot
```

新增模块：

- `reporting.py`：统一生成可行性审计表和转运利用率等明细；
- `visualization.py`：统一生成适合论文使用的静态图表；
- `run_all.py`：按问题 1 至问题 4 的顺序运行模型，并输出 `outputs/overall_summary.xlsx`。

