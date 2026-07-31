"""项目的路径、生产约束与模型默认参数配置。"""
from pathlib import Path

# 项目根目录及题目附件、最终结果文件的位置。
ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"
SUPPLIER_XLSX = ROOT / "附件1 近5年402家供应商的相关数据.xlsx"
CARRIER_XLSX = ROOT / "附件2 近5年8家转运商的相关数据.xlsx"
ORDER_TEMPLATE_XLSX = ROOT / "附件A 订购方案数据结果.xlsx"
TRANSPORT_TEMPLATE_XLSX = ROOT / "附件B 转运方案数据结果.xlsx"
FINAL_ORDER_RESULT_XLSX = OUTPUT_ROOT / "附件A_最终结果.xlsx"
FINAL_TRANSPORT_RESULT_XLSX = OUTPUT_ROOT / "附件B_最终结果.xlsx"

# 历史数据长度、规划期长度，以及题设中的生产和转运能力约束。
WEEKS_HISTORY = 240
WEEKS_PLAN = 24
PRODUCT_CAPACITY = 28_200.0
SAFETY_WEEKS = 2
CARRIER_WEEKLY_CAPACITY = 6_000.0

# 单位产品的原料消耗量，以及 A/B/C 三类原料的相对采购单价。
MATERIAL_CONSUMPTION = {"A": 0.60, "B": 0.66, "C": 0.72}
RAW_PRICE = {"A": 1.20, "B": 1.10, "C": 1.00}

# 稳健预测参数：供货率截断区间、供货能力分位数和缺失损耗的回退值。
DELIVERY_RATE_LOWER = 0.10
DELIVERY_RATE_UPPER = 1.50
SUPPLIER_CAPACITY_QUANTILE = 0.90
LOSS_RATE_FALLBACK = 0.03

# 题目未给出绝对运输费和库存持有费，故以 C 类原料单价为基准设置标准化系数。
UNIT_TRANSPORT_COST = 0.02
UNIT_HOLDING_COST = 0.005
TRANSPORT_LOSS_BUFFER_RATE = 0.05

# 问题 3 的材料偏好约束、奖惩系数和敏感性分析取值网格。
Q3_A_MIN_SHARE = 0.50
Q3_C_MAX_SHARE = 0.15
Q3_A_REWARD = 0.08
Q3_C_PENALTY = 0.12
Q3_LOW_LOSS_THRESHOLD = 0.02
Q3_LOSS_WEIGHT = 0.65
Q3_SENSITIVITY_A_MIN = (0.45, 0.50, 0.55)
Q3_SENSITIVITY_C_MAX = (0.10, 0.15, 0.20)
Q3_SENSITIVITY_LOSS_WEIGHT = (0.50, 0.65, 0.80)

# 问题 4 的各材料产品当量最低占比约束。
# 题目未给出具体最低占比，因此默认取 0，不额外限制基础模型。
Q4_MIN_PRODUCT_SHARES = {"A": 0.0, "B": 0.0, "C": 0.0}

# 数值计算中用于判定零值和比较约束是否满足的容差。
EPSILON = 1e-7
