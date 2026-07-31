# 2021 C 题：生产企业原材料订购与运输（Python 求解）

本项目依据 `2021_C.docx` 的建模框架，对供应商重要性评价、最少供应商选择、24 周订购与转运、成本压缩及产能提升四个问题进行可复现求解。原始 Excel 附件不会被修改，生成的订购与转运结果统一写入 `outputs/`。

## 当前实现状态

- 问题 1：六项供货指标、极差标准化、熵权法综合评分，并输出前 50 家重要供应商。
- 问题 2：在前 50 家重要供应商中选择满足 A/B/C 材料覆盖与目标产能的最少供应商，并生成 24 周订购、库存和转运方案。
- 问题 3：加入 A 类最低比例、C 类最高比例、采购奖惩和低损耗率阈值；损耗率超过阈值的转运商不会被使用。
- 问题 4：使用全部 402 家供应商，联合供应能力、转运能力、库存约束和可选材料比例约束，计算最大可持续周产能。

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
| 问题 3 | `q3_cost_compression_plan.py` | A/C 材料偏好、采购奖惩、低损耗率硬约束、控制变量敏感性分析 | 最终附件 A/B 的“问题 3”工作表、`outputs/q3/sensitivity_analysis.csv` |
| 问题 4 | `q4_capacity_expansion.py` | 全供应商最大可持续产能、库存可行性、周度转运运力 | 最终附件 A/B 的“问题 4”工作表、`outputs/q4/summary.xlsx` |

## 建模口径

### 1. 供应商预测

1. 供应商供货率取“有订货记录周”的实际供货率中位数，并按配置上下界截断。
2. 供应商供货能力取历史正供货量的 90% 分位数。
3. 附件 2 中的零损耗率代表该周未发生运输，不作为真实零损耗样本参与中位数预测。
4. A/B/C 三类原料统一折算为产品当量；生产 1 m³ 产品分别消耗 0.60/0.66/0.72 m³ 的 A/B/C 类原料。

### 2. 问题 2：最少供应商与订购转运

1. 在问题 1 的前 50 家重要供应商中，保证 A、B、C 三类材料均被覆盖。
2. 基于预测供货能力和产品当量筛选供应商，再利用 24 周线性规划确定订单、预计发运量及库存。
3. 每家转运商每周运力上限为 6,000 m³；按预测损耗率由低到高分配，优先不拆分同一供应商的周发运量。
4. 当前真实附件运行结果中，问题 2 的转运方案未发生供应商拆分，且所有转运商均未超出单周运力。

### 3. 问题 3：成本压缩与低损耗转运

1. 通过 `Q3_A_MIN_SHARE` 和 `Q3_C_MAX_SHARE` 约束每周产品当量中的 A/C 材料比例。
2. 通过 `Q3_A_REWARD` 和 `Q3_C_PENALTY` 调整 A/C 类原料的标准化采购成本。
3. `Q3_LOW_LOSS_THRESHOLD = 0.02` 是硬约束：预测损耗率超过 2% 的转运商不参与分配；如果合格运力不足，模型会判定方案不可行。
4. 敏感性分析采用控制变量法，并同时复核订购、低损耗转运、实际到厂量和库存安全：
   - 材料比例试验固定 2% 的低损耗率阈值，令 A 类最低占比在 50%--90%、C 类最高占比在 0.10%--15% 之间变化；取值覆盖最优解附近的临界区间，而非只使用宽松约束。
   - 阈值试验固定基准 A/C 比例，令低损耗率阈值在 0.5%--2.0% 之间变化，记录合格转运商数量、实际使用转运商数量、损耗产品当量、最低库存和安全库存裕度。
   - `outputs/q3/sensitivity_analysis.csv` 以“试验类型”区分两组控制变量试验；`outputs/figures/q3/sensitivity/` 输出材料比例热力图，`outputs/figures/q3/loss_threshold_sensitivity.png` 输出阈值敏感性图。

当前结果中，问题 3 使用 T3、T4、T6、T8 四家转运商，已使用转运商的最大预测损耗率为 0.678%，低于 2% 阈值。新的敏感性分析显示：当低损耗率阈值降至 0.5% 时，合格运力不足而不可行；阈值达到 0.7% 后方案可行。A 类最低占比提高到 85% 时订购问题不可行，说明原方案在 A 类偏好上具有明确的可行性边界。

### 4. 问题 4：产能提升

1. 最大产能模型使用全部 402 家供应商，而不是仅使用问题 1 的前 50 家。
2. `build_carrier_capacity_forecast()` 返回“转运商 × 周次”的运力矩阵；当前附件中每家转运商每周运力均为 6,000 m³。
3. 最大可持续周产能同时受供应商预测能力、材料产品当量、规划期内最小周总转运能力和库存可行性约束。
4. `Q4_MIN_PRODUCT_SHARES` 可设置 A/B/C 三类材料产品当量最低比例。题目未提供具体数值时，默认配置为 `{"A": 0.0, "B": 0.0, "C": 0.0}`，即不额外施加比例限制。

当前真实附件运行结果：

| 指标 | 数值 |
|---|---:|
| 候选供应商数 | 402 |
| 最大可持续周产能 | 49,972.16 m³ 产品/周 |
| 相对当前 28,200 m³/周的提升 | 21,772.16 m³/周（77.21%） |
| 两周安全库存 | 99,944.33 m³ 产品当量 |
| 实际最低库存 | 102,294.27 m³ 产品当量 |

## 标准化成本与关键配置

题目只说明三类原料的单位运输费和仓储费相同，但未给出绝对金额。因此 `config.py` 使用以 C 类原料单价为 1 的相对价格尺度：

- `RAW_PRICE = {"A": 1.20, "B": 1.10, "C": 1.00}`：三类原料相对采购价格；
- `UNIT_TRANSPORT_COST = 0.02`：单位原料运输的标准化成本；
- `UNIT_HOLDING_COST = 0.005`：单位产品当量、每周的标准化库存持有成本；
- `TRANSPORT_LOSS_BUFFER_RATE = 0.05`：订购阶段使用的保守损耗缓冲率；
- `Q3_LOSS_WEIGHT = 0.65`：问题 3 中用于调整保守损耗缓冲程度的参数；
- `Q4_MIN_PRODUCT_SHARES`：问题 4 的材料最低比例参数。

## 校验与测试

测试覆盖数据读写、单位换算、供货与损耗预测、熵权排序、供应商筛选、库存安全约束、转运容量、低损耗率阈值、材料比例约束、时变运力接口、模板保护和真实附件结果复核。

最近一次完整测试结果：**19 passed**。

```powershell
& 'D:\anaconda\envs\LLM_Classification\python.exe' -X utf8 -m pytest -v
```

## 输出文件说明

- `outputs/附件A_最终结果.xlsx`：问题 2、3、4 的订购方案；
- `outputs/附件B_最终结果.xlsx`：问题 2、3、4 的转运方案；
- `outputs/q3/summary.xlsx`、`outputs/q3/inventory_trace.xlsx`、`outputs/q3/sensitivity_analysis.csv`：问题 3 的汇总、库存轨迹与敏感性分析；
- `outputs/q4/summary.xlsx`、`outputs/q4/inventory_trace.xlsx`：问题 4 的扩产结果与库存轨迹。

> 注意：请勿将 `outputs/` 中的结果文件覆盖回原始附件。结果工作簿由模板副本生成，仅写入各问题对应的结果区域；分别运行问题 2、3、4 不会清除已写入的其他问题工作表。

## 使用边界

1. 供应能力和损耗率均由历史数据的稳健统计量预测，属于建模假设，应在论文中说明。
2. 当前附件中的转运能力为常数，因此按最小周总运力处理与逐周运力约束等价；若未来输入存在明显时变运力，当前模型可保证保守可行，但未利用跨周库存完全挖掘动态运力。
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
| 问题 3 | 库存图、材料结构图、低损耗阈值图、转运商利用率图、参数敏感性热力图、问题 2/3 材料结构对比图 |
| 问题 4 | 当前与最大可持续产能对比图、库存图、转运商利用率图、材料结构图 |

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

