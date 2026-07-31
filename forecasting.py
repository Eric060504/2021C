"""基于历史订单、供货和损耗记录构建未来周度预测。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DELIVERY_RATE_LOWER,
    DELIVERY_RATE_UPPER,
    LOSS_RATE_FALLBACK,
    SUPPLIER_CAPACITY_QUANTILE,
    CARRIER_WEEKLY_CAPACITY,
    Q4_SUPPLY_FORECAST_LOWER_MULTIPLIER,
    Q4_SUPPLY_FORECAST_SHRINKAGE,
    Q4_SUPPLY_FORECAST_UPPER_MULTIPLIER,
    WEEKS_PLAN,
)
from data_io import MATERIAL, SUPPLIER_ID, week_columns


# 将不同形式的历史输入统一为数值序列，非法值按零处理。
def _numeric_series(values: pd.Series | np.ndarray | list[float]) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0)


# 用有订货周的供货率中位数估计履约水平，再按上下限抑制异常值。
def estimate_delivery_rate(
    order: pd.Series | np.ndarray | list[float],
    supply: pd.Series | np.ndarray | list[float],
    lower: float = DELIVERY_RATE_LOWER,
    upper: float = DELIVERY_RATE_UPPER,
) -> float:
    """以截断后的稳健中位数估计供应商的供货率。

    未订货周无法提供履约可靠性信息，故不参与统计；截断上下限统一
    在 ``config`` 中配置，用于抑制极端历史记录。
    """
    if lower <= 0 or upper < lower:
        raise ValueError("供货率截断区间必须满足 0 < lower <= upper")
    order_series, supply_series = _numeric_series(order), _numeric_series(supply)
    # 未订货周没有履约信息，不能参与供货率统计。
    valid = order_series > 1e-8
    if not valid.any():
        return float(lower)
    ratios = supply_series.loc[valid] / order_series.loc[valid]
    ratio = float(np.nanmedian(ratios.to_numpy(dtype=float)))
    if not np.isfinite(ratio):
        ratio = lower
    return float(np.clip(ratio, lower, upper))


# 以正供货历史的高分位数估计稳定可用能力，忽略未合作周的零值。
def estimate_supplier_capacity(
    supply: pd.Series | np.ndarray | list[float],
    quantile: float = SUPPLIER_CAPACITY_QUANTILE,
) -> float:
    """以正供货历史记录的高分位数预测可用供货能力。"""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("供货能力分位数必须在 [0, 1] 内")
    values = _numeric_series(supply)
    values = values.loc[values > 1e-8]
    if values.empty:
        return 0.0
    return float(np.quantile(values.to_numpy(dtype=float), quantile))


# 将每家转运商的非零历史损耗中位数平铺到未来规划期。
def forecast_loss_rates(
    history: pd.DataFrame,
    horizon: int = WEEKS_PLAN,
    fallback: float = LOSS_RATE_FALLBACK,
) -> pd.DataFrame:
    """以各转运商非零历史损耗率的中位数预测未来各周损耗率。

    附件 2 的零值表示该周未发生运输，而不是真实的零损耗运输，
    因此不参与损耗率估计。
    """
    if horizon <= 0:
        raise ValueError("预测周数必须为正")
    if not 0 <= fallback < 1:
        raise ValueError("损耗率回退值必须在 [0,1) 内")
    numeric = history.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    future = pd.DataFrame(index=history.index, columns=range(1, horizon + 1), dtype=float)
    for carrier, row in numeric.iterrows():
        # 附件损耗表的零值代表未运输，不作为真实零损耗样本。
        observed = row.loc[row > 1e-8].to_numpy(dtype=float)
        rate = float(np.median(observed)) if observed.size else float(fallback)
        future.loc[carrier, :] = rate
    return future


# 同时输出供货率和物理能力上限，作为后续订购优化的供应商参数。
def build_supplier_forecasts(
    order: pd.DataFrame,
    supply: pd.DataFrame,
    weeks: int = WEEKS_PLAN,
) -> pd.DataFrame:
    """为供应商构建规划期内恒定的供货能力预测。

    结果保存预期供货率和独立于订货量的物理能力上限；优化模型通过
    线性发运变量表示 ``min(供货率 × 订货量, 能力上限)``。
    """
    if order[SUPPLIER_ID].tolist() != supply[SUPPLIER_ID].tolist():
        raise ValueError("订货和供货数据的供应商行顺序不一致")
    columns = [col for col in week_columns() if col in order.columns and col in supply.columns]
    records: list[dict[str, object]] = []
    for idx, supplier_id in enumerate(order[SUPPLIER_ID]):
        eta = estimate_delivery_rate(order.loc[idx, columns], supply.loc[idx, columns])
        capacity = estimate_supplier_capacity(supply.loc[idx, columns])
        record: dict[str, object] = {
            SUPPLIER_ID: supplier_id,
            MATERIAL: order.loc[idx, MATERIAL],
            "delivery_rate": eta,
            "capacity": capacity,
        }
        for week in range(1, weeks + 1):
            record[week] = capacity
        records.append(record)
    return pd.DataFrame.from_records(records).set_index(SUPPLIER_ID, drop=False)


# 将近期供货的周度起伏收缩到稳健能力附近，生成“供应商 × 未来周次”的动态可供货上限。
def build_supplier_weekly_capacity_forecast(
    order: pd.DataFrame,
    supply: pd.DataFrame,
    weeks: int = WEEKS_PLAN,
    shrinkage: float = Q4_SUPPLY_FORECAST_SHRINKAGE,
    lower_multiplier: float = Q4_SUPPLY_FORECAST_LOWER_MULTIPLIER,
    upper_multiplier: float = Q4_SUPPLY_FORECAST_UPPER_MULTIPLIER,
) -> pd.DataFrame:
    """预测规划期内每家供应商的逐周可供货上限。

    以最近 ``weeks`` 周的实际供货波动作为短期预测轮廓，并向历史高分位稳健能力收缩，
    从而同时保留周度差异和避免直接复制个别异常周。第 ``t`` 周预测为
    ``capacity × [shrinkage + (1 - shrinkage) × profile_t]``；其中 ``profile_t``
    是近期第 ``t`` 周供货量相对于近期均值的比值，并经过上下界截断。
    """
    if weeks <= 0:
        raise ValueError("预测周数必须为正")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("收缩系数必须在 [0, 1] 内")
    if lower_multiplier <= 0.0 or upper_multiplier < lower_multiplier:
        raise ValueError("周度能力倍数上下界不合法")
    if order[SUPPLIER_ID].tolist() != supply[SUPPLIER_ID].tolist():
        raise ValueError("订货和供货数据的供应商行顺序不一致")

    columns = [col for col in week_columns() if col in supply.columns]
    if not columns:
        raise ValueError("供货数据缺少历史周列")
    records: list[dict[str, object]] = []
    for idx, supplier_id in enumerate(supply[SUPPLIER_ID]):
        history = _numeric_series(supply.loc[idx, columns]).to_numpy(dtype=float)
        capacity = estimate_supplier_capacity(history)
        recent = history[-min(weeks, len(history)) :]
        if len(recent) < weeks:
            recent = np.pad(recent, (weeks - len(recent), 0), mode="edge")
        recent_mean = float(np.mean(recent))
        if capacity <= 1e-9 or recent_mean <= 1e-9:
            forecast = np.zeros(weeks, dtype=float)
        else:
            profile = np.clip(recent / recent_mean, lower_multiplier, upper_multiplier)
            multiplier = shrinkage + (1.0 - shrinkage) * profile
            forecast = capacity * multiplier
        record: dict[str, object] = {SUPPLIER_ID: str(supplier_id)}
        record.update({week: float(value) for week, value in enumerate(forecast, start=1)})
        records.append(record)
    return pd.DataFrame.from_records(records).set_index(SUPPLIER_ID, drop=False)


# 将附件中的百分比损耗转换为小数，并生成“转运商 × 周次”预测矩阵。
def build_carrier_loss_forecast(loss_history: pd.DataFrame, weeks: int = WEEKS_PLAN) -> pd.DataFrame:
    """返回以转运商为行、规划周次为列的损耗率预测矩阵。"""
    id_column = "转运商ID"
    if id_column not in loss_history.columns:
        raise ValueError("损耗历史表缺少转运商ID列")
    historical_columns = [col for col in week_columns() if col in loss_history.columns]
    # 附件 2 中的损耗率以百分数记录（如 1.2 表示 1.2%），优化函数统一使用小数。
    # 附件使用百分数，优化模型统一使用 0--1 的小数损耗率。
    history = loss_history.set_index(id_column)[historical_columns] / 100.0
    return forecast_loss_rates(history, horizon=weeks)

# 构建“转运商 × 周次”运力矩阵，兼容当前常量运力与未来时变运力情景。
def build_carrier_capacity_forecast(
    loss_history: pd.DataFrame,
    weeks: int = WEEKS_PLAN,
    default_capacity: float = CARRIER_WEEKLY_CAPACITY,
) -> pd.DataFrame:
    """返回规划期内的转运商周度运力矩阵。

    附件 2 规定每家转运商的运力为常数。以“转运商 × 周次”矩阵返回，
    可保持当前设定，并支持未来情景中替换为逐周变化的运力。
    """
    if weeks <= 0:
        raise ValueError("Forecast horizon must be positive")
    if not np.isfinite(default_capacity) or default_capacity <= 0:
        raise ValueError("Carrier capacity must be a positive finite number")
    id_column = "\u8f6c\u8fd0\u5546ID"
    if id_column not in loss_history.columns:
        raise ValueError("Carrier loss history is missing the carrier-ID column")
    ids = loss_history[id_column].dropna().astype(str).str.strip()
    if ids.empty or ids.duplicated().any() or (ids == "").any():
        raise ValueError("Carrier IDs must be nonempty and unique")
    return pd.DataFrame(float(default_capacity), index=ids.tolist(), columns=range(1, weeks + 1))

