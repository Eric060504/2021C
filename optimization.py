"""问题 2--4 的线性订购优化、材料偏好约束与低损耗转运分配。"""
from __future__ import annotations

from itertools import product
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from config import (
    CARRIER_WEEKLY_CAPACITY,
    MATERIAL_CONSUMPTION,
    RAW_PRICE,
    UNIT_HOLDING_COST,
    UNIT_TRANSPORT_COST,
)
from data_io import MATERIAL


# 统一表示模型不可行或转运能力不足等优化层面的业务错误。
class OptimizationError(RuntimeError):
    """当给定生产计划在模型约束下不可行时抛出。"""


# 校验预测数据并过滤供货率或能力为零的供应商，避免无效决策变量。
def _ensure_supplier_frame(suppliers: pd.DataFrame) -> pd.DataFrame:
    required = {MATERIAL, "delivery_rate", "capacity"}
    missing = required - set(suppliers.columns)
    if missing:
        raise ValueError(f"供应商预测数据缺少列: {sorted(missing)}")
    data = suppliers.copy()
    data.index = data.index.astype(str)
    data[MATERIAL] = data[MATERIAL].astype(str)
    data["delivery_rate"] = pd.to_numeric(data["delivery_rate"], errors="coerce").fillna(0.0)
    data["capacity"] = pd.to_numeric(data["capacity"], errors="coerce").fillna(0.0)
    return data.loc[(data["delivery_rate"] > 1e-9) & (data["capacity"] > 1e-9)].copy()


# 将供应商原料供给能力换算为产品当量，便于跨材料比较。
def _product_equivalent_capacity(data: pd.DataFrame) -> pd.Series:
    return data.apply(lambda row: float(row["capacity"]) / MATERIAL_CONSUMPTION[row[MATERIAL]], axis=1)


# 先保证 A/B/C 三类材料均被覆盖，再按能力从高到低补足目标产能。
def select_minimum_suppliers(suppliers: pd.DataFrame, demand: float, transport_loss_rate: float = 0.0) -> list[str]:
    """在保留 A/B/C 三类材料的前提下，贪心选择尽可能少的高能力供应商。"""
    data = _ensure_supplier_frame(suppliers)
    if demand <= 0:
        raise ValueError("周产能需求必须为正")
    if not 0 <= transport_loss_rate < 1:
        raise ValueError("运输损耗率必须在 [0,1) 内")
    selected: list[str] = []
    for material in ("A", "B", "C"):
        subset = data.loc[data[MATERIAL] == material]
        if subset.empty:
            raise OptimizationError(f"没有可用的{material}类供应商")
        best = _product_equivalent_capacity(subset).sort_values(ascending=False).index[0]
        selected.append(str(best))
    target = demand / (1.0 - transport_loss_rate)
    capacity = _product_equivalent_capacity(data)
    running = float(capacity.loc[selected].sum())
    for supplier_id in capacity.sort_values(ascending=False).index:
        supplier_id = str(supplier_id)
        if running >= target - 1e-8:
            break
        if supplier_id not in selected:
            selected.append(supplier_id)
            running += float(capacity.loc[supplier_id])
    if running < target - 1e-8:
        raise OptimizationError("全部供应商预测能力仍不足以满足目标产能")
    return selected


# 将标量、序列或“转运商 × 周次”矩阵统一为规划期内的周度总量序列。
def _weekly_total_vector(
    value: float | Sequence[float] | pd.Series | pd.DataFrame,
    weeks: int,
    name: str,
) -> np.ndarray:
    if np.isscalar(value):
        result = np.full(weeks, float(value), dtype=float)
    elif isinstance(value, pd.DataFrame):
        missing = set(range(1, weeks + 1)) - set(value.columns)
        if missing:
            raise ValueError(f"{name}缺少周列: {sorted(missing)}")
        result = value.loc[:, range(1, weeks + 1)].apply(pd.to_numeric, errors="coerce").sum(axis=0).to_numpy(float)
    else:
        series = pd.to_numeric(pd.Series(value), errors="coerce")
        if len(series) != weeks:
            raise ValueError(f"{name}长度必须等于规划周数 {weeks}")
        result = series.to_numpy(float)
    if not np.isfinite(result).all() or (result < 0.0).any():
        raise ValueError(f"{name}必须是非负有限值")
    return result


# 将供应商静态能力或“供应商 × 周次”预测矩阵统一为逐周发运上限。
def _supplier_capacity_matrix(data: pd.DataFrame, weeks: int) -> np.ndarray:
    week_columns = list(range(1, weeks + 1))
    if set(week_columns).issubset(data.columns):
        matrix = data.loc[:, week_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    else:
        matrix = np.repeat(data["capacity"].to_numpy(float)[:, None], weeks, axis=1)
    if not np.isfinite(matrix).all() or (matrix < 0.0).any():
        raise ValueError("供应商周度能力必须是非负有限值")
    return matrix


# 建立跨周的线性规划：决策发运量和期末库存，最小化采购、运输与库存成本。
def _solve_shipments(
    suppliers: pd.DataFrame,
    demand: float | Sequence[float] | pd.Series,
    weeks: int,
    initial_inventory: float,
    safety_inventory: float,
    carrier_capacity_total: float | Sequence[float] | pd.Series | pd.DataFrame,
    transport_loss_rate: float,
    preference: Mapping[str, float] | None,
) -> dict[str, object]:
    data = _ensure_supplier_frame(suppliers)
    n = len(data)
    if n == 0:
        raise OptimizationError("不存在可用供应商")
    if weeks <= 0:
        raise ValueError("计划期必须为正")
    demand_by_week = _weekly_total_vector(demand, weeks, "周需求")
    if (demand_by_week <= 0.0).any():
        raise ValueError("周需求必须为正")
    if initial_inventory < 0.0 or safety_inventory < 0.0:
        raise ValueError("初始库存和安全库存必须非负")
    if not 0 <= transport_loss_rate < 1:
        raise ValueError("运输损耗率必须在 [0,1) 内")

    materials = data[MATERIAL].tolist()
    eta = data["delivery_rate"].to_numpy(float)
    capacity_by_week = _supplier_capacity_matrix(data, weeks)
    carrier_by_week = _weekly_total_vector(carrier_capacity_total, weeks, "周转运总能力")
    # 发运 1 m³ 各类原料在扣除规划损耗后对应的产品当量。
    product_factor = np.array([(1.0 - transport_loss_rate) / MATERIAL_CONSUMPTION[m] for m in materials])
    # 问题 3 通过 A 类奖励和 C 类惩罚修正采购成本。
    modifiers = np.ones(n)
    preference = dict(preference or {})
    modifiers[np.array(materials) == "A"] *= 1.0 - float(preference.get("a_reward", 0.0))
    modifiers[np.array(materials) == "C"] *= 1.0 + float(preference.get("c_penalty", 0.0))
    purchase_cost = np.array([RAW_PRICE[m] for m in materials]) / eta * modifiers
    shipment_cost = purchase_cost + UNIT_TRANSPORT_COST

    # 决策变量依次为“供应商-周发运量”和“每周末库存”。
    shipment_variables = n * weeks
    inventory_offset = shipment_variables
    total_variables = shipment_variables + weeks
    objective = np.zeros(total_variables)
    for t in range(weeks):
        objective[t * n : (t + 1) * n] = shipment_cost
        objective[inventory_offset + t] = UNIT_HOLDING_COST

    # 等式约束为逐周库存平衡：期末库存 = 上期库存 + 到厂量 - 当周需求。
    a_eq, b_eq = [], []
    for t in range(weeks):
        row = np.zeros(total_variables)
        row[t * n : (t + 1) * n] = -product_factor
        row[inventory_offset + t] = 1.0
        if t == 0:
            b_eq.append(initial_inventory - demand_by_week[t])
        else:
            row[inventory_offset + t - 1] = -1.0
            b_eq.append(-demand_by_week[t])
        a_eq.append(row)

    # 每周总发运量不超过相应周次的总转运能力。
    a_ub, b_ub = [], []
    for t in range(weeks):
        row = np.zeros(total_variables)
        row[t * n : (t + 1) * n] = 1.0
        a_ub.append(row)
        b_ub.append(carrier_by_week[t])

    # 仅问题 3 启用 A/C 材料构成比例约束。
    a_min = preference.get("a_min_share")
    c_max = preference.get("c_max_share")
    for t in range(weeks):
        if a_min is not None:
            row = np.zeros(total_variables)
            row[t * n : (t + 1) * n] = float(a_min) * product_factor - np.where(np.array(materials) == "A", product_factor, 0.0)
            a_ub.append(row)
            b_ub.append(0.0)
        if c_max is not None:
            row = np.zeros(total_variables)
            row[t * n : (t + 1) * n] = np.where(np.array(materials) == "C", product_factor, 0.0) - float(c_max) * product_factor
            a_ub.append(row)
            b_ub.append(0.0)

    # 发运量受供应商逐周供货能力约束；所有期末库存不得低于安全库存。
    bounds = [(0.0, float(capacity_by_week[i, t])) for t in range(weeks) for i in range(n)] + [(float(safety_inventory), None)] * weeks
    result = linprog(
        c=objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise OptimizationError(f"订购优化不可行：{result.message}")
    shipments = pd.DataFrame(
        result.x[:shipment_variables].reshape(weeks, n).T,
        index=data.index,
        columns=range(1, weeks + 1),
    )
    # 以“发运量 ÷ 预期供货率”反推需要向供应商下达的订货量。
    orders = shipments.div(data["delivery_rate"], axis=0)
    inventory = pd.Series(result.x[inventory_offset:], index=range(1, weeks + 1), name="库存产品当量")
    received_product = shipments.mul(product_factor, axis=0)
    material_equiv = pd.DataFrame(
        {material: received_product.loc[data[MATERIAL] == material].sum(axis=0) for material in ("A", "B", "C")}
    ).fillna(0.0)
    material_equiv.index = range(1, weeks + 1)
    return {
        "orders": orders,
        "shipments": shipments,
        "inventory": inventory,
        "material_product_equivalent": material_equiv,
        "demand_by_week": pd.Series(demand_by_week, index=range(1, weeks + 1), name="周生产需求"),
        "supplier_capacity_by_week": pd.DataFrame(capacity_by_week, index=data.index, columns=range(1, weeks + 1)),
        "carrier_capacity_by_week": pd.Series(carrier_by_week, index=range(1, weeks + 1), name="周转运总能力"),
        "objective": float(result.fun),
        "selected_suppliers": list(data.index),
        "transport_loss_rate": transport_loss_rate,
    }


# 问题 2/4 的基础订购模型：不附加材料偏好比例约束。
def solve_order_plan(
    suppliers: pd.DataFrame,
    demand: float | Sequence[float] | pd.Series,
    weeks: int = 24,
    initial_inventory: float | None = None,
    safety_inventory: float | None = None,
    carrier_capacity_total: float | Sequence[float] | pd.Series | pd.DataFrame = 8 * CARRIER_WEEKLY_CAPACITY,
    transport_loss_rate: float = 0.0,
) -> dict[str, object]:
    """在给定供应商集合下，最小化标准化采购、运输和库存持有成本。"""
    demand_by_week = _weekly_total_vector(demand, weeks, "周需求")
    safety = float(2.0 * demand_by_week.max() if safety_inventory is None else safety_inventory)
    initial = float(safety if initial_inventory is None else initial_inventory)
    return _solve_shipments(suppliers, demand_by_week, weeks, initial, safety, carrier_capacity_total, transport_loss_rate, None)


# 问题 3 的订购模型：加入 A 类最低占比、C 类最高占比及采购奖惩。
def solve_preference_order_plan(
    suppliers: pd.DataFrame,
    demand: float | Sequence[float] | pd.Series,
    *,
    a_min_share: float,
    c_max_share: float,
    a_reward: float,
    c_penalty: float,
    weeks: int = 24,
    initial_inventory: float | None = None,
    safety_inventory: float | None = None,
    carrier_capacity_total: float | Sequence[float] | pd.Series | pd.DataFrame = 8 * CARRIER_WEEKLY_CAPACITY,
    transport_loss_rate: float = 0.0,
) -> dict[str, object]:
    # A 类为最低占比、C 类为最高占比，二者不是两个最低占比，因而不要求参数之和不超过 1。
    if not 0 <= a_min_share <= 1 or not 0 <= c_max_share <= 1:
        raise ValueError("A类下限和C类上限参数必须在 [0, 1] 内")
    demand_by_week = _weekly_total_vector(demand, weeks, "周需求")
    safety = float(2.0 * demand_by_week.max() if safety_inventory is None else safety_inventory)
    initial = float(safety if initial_inventory is None else initial_inventory)
    preference = {
        "a_min_share": a_min_share,
        "c_max_share": c_max_share,
        "a_reward": a_reward,
        "c_penalty": c_penalty,
    }
    return _solve_shipments(suppliers, demand_by_week, weeks, initial, safety, carrier_capacity_total, transport_loss_rate, preference)


# 独立复核单周产品当量的材料构成是否满足问题 3 的比例边界。
def check_material_share_constraints(
    product_equivalent: Mapping[str, float], a_min_share: float, c_max_share: float, tolerance: float = 1e-8
) -> bool:
    total = float(sum(float(product_equivalent.get(m, 0.0)) for m in ("A", "B", "C")))
    if total <= tolerance:
        return False
    return (
        float(product_equivalent.get("A", 0.0)) / total >= a_min_share - tolerance
        and float(product_equivalent.get("C", 0.0)) / total <= c_max_share + tolerance
    )


# 每周按预测损耗率由低到高分配发运量，并受每家转运商运力限制。
# 将标量、按转运商给定的运力或“转运商 × 周次”运力矩阵统一为单周运力序列。
def _carrier_capacity_for_week(
    carrier_capacity: float | pd.Series | pd.DataFrame | Mapping[str, float],
    carrier_ids: Sequence[str],
    week: int,
) -> pd.Series:
    """返回指定周次、并按 ``carrier_ids`` 对齐的各转运商运力。"""
    ids = [str(carrier) for carrier in carrier_ids]
    if np.isscalar(carrier_capacity):
        values = pd.Series(float(carrier_capacity), index=ids, dtype=float)
    elif isinstance(carrier_capacity, pd.DataFrame):
        if week not in carrier_capacity.columns:
            raise ValueError(f"Carrier-capacity matrix has no column for week {week}")
        values = pd.to_numeric(carrier_capacity[week].copy(), errors="coerce")
        values.index = values.index.astype(str)
        values = values.reindex(ids)
    else:
        values = pd.to_numeric(pd.Series(carrier_capacity), errors="coerce")
        values.index = values.index.astype(str)
        values = values.reindex(ids)
    if values.isna().any() or (~np.isfinite(values)).any() or (values <= 0).any():
        raise ValueError("Every selected carrier must have a positive finite capacity")
    return values.astype(float)

def assign_carriers(
    shipments: pd.DataFrame,
    loss_rates: pd.DataFrame,
    carrier_capacity: float | pd.Series | pd.DataFrame | Mapping[str, float] = CARRIER_WEEKLY_CAPACITY,
    max_loss_rate: float | None = None,
) -> dict[int, pd.DataFrame]:
    """在运力与损耗约束下，将每周发运量分配给转运商。

    指定 ``max_loss_rate`` 后，预测损耗率高于阈值的转运商视为不可用。
    合格转运商运力不足时抛出 ``OptimizationError``，不会回退使用高损耗率
    转运商。可行时优先保持单个供应商的发运量完整，仅在运力需要时拆分。
    """
    if max_loss_rate is not None and not 0.0 <= max_loss_rate < 1.0:
        raise ValueError("Maximum loss rate must be in [0, 1)")
    normalized_loss = loss_rates.copy()
    normalized_loss.index = normalized_loss.index.astype(str)
    carrier_ids = normalized_loss.index.tolist()
    if not carrier_ids or len(set(carrier_ids)) != len(carrier_ids):
        raise ValueError("Carrier loss-rate data must have unique carrier IDs")
    allocations: dict[int, pd.DataFrame] = {}
    for week in shipments.columns:
        if week not in normalized_loss.columns:
            raise ValueError(f"Carrier loss-rate data has no column for week {week}")
        rates = pd.to_numeric(normalized_loss[week].reindex(carrier_ids), errors="coerce")
        if rates.isna().any() or (~np.isfinite(rates)).any() or ((rates < 0.0) | (rates >= 1.0)).any():
            raise ValueError(f"Invalid carrier loss rates in week {week}")
        eligible = rates.index if max_loss_rate is None else rates.index[rates <= max_loss_rate + 1e-12]
        if len(eligible) == 0:
            raise OptimizationError(f"No carrier meets the loss threshold in week {week}")
        remaining = _carrier_capacity_for_week(carrier_capacity, carrier_ids, int(week)).reindex(eligible)
        frame = pd.DataFrame(0.0, index=shipments.index.astype(str), columns=carrier_ids)
        ranked_carriers = rates.reindex(eligible).sort_values(kind="stable").index.tolist()
        for supplier_id, volume in shipments[week].items():
            outstanding = max(float(volume), 0.0)
            candidates = [carrier for carrier in ranked_carriers if remaining[carrier] >= outstanding - 1e-8]
            if candidates:
                carrier = candidates[0]
                frame.loc[str(supplier_id), carrier] += outstanding
                remaining[carrier] -= outstanding
                continue
            for carrier in ranked_carriers:
                if outstanding <= 1e-8:
                    break
                amount = min(outstanding, float(remaining[carrier]))
                if amount > 0.0:
                    frame.loc[str(supplier_id), carrier] += amount
                    remaining[carrier] -= amount
                    outstanding -= amount
            if outstanding > 1e-6:
                scope = "qualified " if max_loss_rate is not None else ""
                raise OptimizationError(
                    f"Week {week} has insufficient {scope}carrier capacity; "
                    f"unallocated volume is {outstanding:.3f} m3"
                )
        allocations[int(week)] = frame
    return allocations

def allocation_loss(allocation: Mapping[int, pd.DataFrame], loss_rates: pd.DataFrame) -> tuple[float, pd.Series]:
    """计算总原料损耗，并返回按周汇总的到厂产品当量占位序列。"""
    total_loss = 0.0
    received_product: dict[int, float] = {}
    for week, matrix in allocation.items():
        rates = loss_rates[week].reindex(matrix.columns).astype(float)
        total_loss += float(matrix.mul(rates, axis=1).to_numpy().sum())
        received_product[int(week)] = 0.0
    return total_loss, pd.Series(received_product, name="到厂产品当量")


# 按转运商扣损后，再按供应商材料类型换算每周实际到厂产品当量。
def post_loss_product_equivalent_by_week(
    allocation: Mapping[int, pd.DataFrame],
    loss_rates: pd.DataFrame,
    supplier_materials: Mapping[str, str] | pd.Series,
) -> pd.Series:
    """根据转运方案和损耗率计算每周实际到厂的产品当量。"""
    material_map = {str(key): str(value) for key, value in dict(supplier_materials).items()}
    receipts: dict[int, float] = {}
    for week, matrix in allocation.items():
        rates = loss_rates[int(week)].reindex(matrix.columns).astype(float)
        # 先按转运商损耗扣减，再按供应商材料类别换算产品当量。
        after_loss = matrix.mul(1.0 - rates, axis=1).sum(axis=1)
        total = 0.0
        for supplier_id, volume in after_loss.items():
            total += float(volume) / MATERIAL_CONSUMPTION[material_map[str(supplier_id)]]
        receipts[int(week)] = total
    return pd.Series(receipts, name="received_product_equivalent").sort_index()


# 依据库存平衡公式递推实际库存，用于核验安全库存约束。
def inventory_from_receipts(
    receipts: pd.Series,
    demand: float | Sequence[float] | pd.Series,
    initial_inventory: float,
) -> pd.Series:
    """按产品当量库存平衡关系递推库存序列，兼容恒定或逐周生产需求。"""
    ordered_receipts = receipts.sort_index()
    demand_by_week = _weekly_total_vector(demand, len(ordered_receipts), "周需求")
    inventory: dict[int, float] = {}
    current = float(initial_inventory)
    for position, (week, received) in enumerate(ordered_receipts.items()):
        current += float(received) - float(demand_by_week[position])
        inventory[int(week)] = current
    return pd.Series(inventory, name="inventory_product_equivalent")


# 在供应商供货上限和总转运能力约束下，求最大可持续周产能。
# 校验各类材料产品当量的最低占比约束。
def _material_min_shares(material_min_shares: Mapping[str, float] | None) -> dict[str, float]:
    """校验 A、B、C 三类材料产品当量最低占比的参数合法性。"""
    shares = {material: 0.0 for material in ("A", "B", "C")}
    if material_min_shares is not None:
        unknown = set(material_min_shares) - set(shares)
        if unknown:
            raise ValueError(f"Unknown material share keys: {sorted(unknown)}")
        shares.update({material: float(value) for material, value in material_min_shares.items()})
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in shares.values()):
        raise ValueError("Material minimum shares must be finite values in [0, 1]")
    if sum(shares.values()) > 1.0 + 1e-8:
        raise ValueError("The sum of material minimum shares cannot exceed 1")
    return shares

def _minimum_total_carrier_capacity(
    carrier_capacity_total: float | Sequence[float] | pd.Series | pd.DataFrame,
) -> float:
    """返回规划期内限制最严格的周总转运能力。"""
    if np.isscalar(carrier_capacity_total):
        value = float(carrier_capacity_total)
    elif isinstance(carrier_capacity_total, pd.DataFrame):
        if carrier_capacity_total.empty:
            raise ValueError("Carrier-capacity matrix cannot be empty")
        numeric = carrier_capacity_total.apply(pd.to_numeric, errors="coerce")
        weekly_totals = numeric.sum(axis=0, min_count=1)
        value = float(weekly_totals.min())
    else:
        values = pd.to_numeric(pd.Series(carrier_capacity_total), errors="coerce")
        if values.empty:
            raise ValueError("Carrier-capacity sequence cannot be empty")
        value = float(values.min())
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Carrier capacity must be a positive finite number")
    return value

def solve_max_sustainable_capacity(
    suppliers: pd.DataFrame | None = None,
    *,
    max_received_volume: float | None = None,
    material: str | None = None,
    carrier_capacity_total: float | Sequence[float] | pd.Series | pd.DataFrame = 8 * CARRIER_WEEKLY_CAPACITY,
    transport_loss_rate: float = 0.0,
    material_min_shares: Mapping[str, float] | None = None,
) -> float:
    """在供应与转运能力约束下最大化可持续周产能。

    对运力矩阵采用保守解释：取周总运力的最小值，保证返回的产能在规划期
    每一周均可持续。``material_min_shares`` 约束各材料的产品当量产出，
    即“某材料产品当量不少于其最低占比乘以总产品当量”。
    """
    if not 0.0 <= transport_loss_rate < 1.0:
        raise ValueError("Transport loss rate must be in [0, 1)")
    shares = _material_min_shares(material_min_shares)
    total_capacity = _minimum_total_carrier_capacity(carrier_capacity_total)
    if suppliers is None:
        if max_received_volume is None or material is None:
            raise ValueError("Either suppliers or both max_received_volume and material are required")
        if material not in MATERIAL_CONSUMPTION:
            raise ValueError("Material must be A, B, or C")
        if any(shares[name] > 1e-12 for name in shares if name != material):
            raise OptimizationError("Single-material capacity cannot satisfy the requested material shares")
        raw_limit = min(float(max_received_volume), total_capacity)
        return raw_limit * (1.0 - transport_loss_rate) / MATERIAL_CONSUMPTION[material]

    data = _ensure_supplier_frame(suppliers)
    if data.empty:
        raise OptimizationError("No supplier has usable delivery capacity")
    factors = np.array([(1.0 - transport_loss_rate) / MATERIAL_CONSUMPTION[item] for item in data[MATERIAL]])
    materials = data[MATERIAL].to_numpy(str)
    a_ub: list[np.ndarray] = [np.ones(len(data))]
    b_ub: list[float] = [total_capacity]
    for name, share in shares.items():
        if share <= 1e-12:
            continue
        # 最低占比乘以总产品当量减去该材料产品当量不大于 0。
        row = share * factors - np.where(materials == name, factors, 0.0)
        a_ub.append(row)
        b_ub.append(0.0)
    result = linprog(
        c=-factors,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        bounds=[(0.0, float(value)) for value in data["capacity"]],
        method="highs",
    )
    if not result.success:
        raise OptimizationError(f"Maximum sustainable capacity optimization failed: {result.message}")
    return float(-result.fun)


# 在逐周供应、转运和安全库存约束下最大化整个规划期均可保持的固定周产能。
def solve_dynamic_sustainable_capacity(
    suppliers: pd.DataFrame,
    *,
    weeks: int,
    carrier_capacity_total: float | Sequence[float] | pd.Series | pd.DataFrame = 8 * CARRIER_WEEKLY_CAPACITY,
    transport_loss_rate: float = 0.0,
    safety_weeks: float = 2.0,
    material_min_shares: Mapping[str, float] | None = None,
) -> float:
    """求解具有周度供给波动时的最大可持续周产能。

    将初始库存和每周期末安全库存均设为 ``safety_weeks × p``，其中 ``p`` 是待求的
    固定周产能。该形式保留“长期稳定生产”的问题 4 口径，同时让供应商逐周能力和
    转运能力通过库存跨周调节发挥作用。
    """
    if weeks <= 0:
        raise ValueError("计划期必须为正")
    if not 0.0 <= transport_loss_rate < 1.0:
        raise ValueError("运输损耗率必须在 [0,1) 内")
    if not np.isfinite(safety_weeks) or safety_weeks < 0.0:
        raise ValueError("安全库存周数必须为非负有限值")
    data = _ensure_supplier_frame(suppliers)
    if data.empty:
        raise OptimizationError("不存在可用供应商")
    shares = _material_min_shares(material_min_shares)
    n = len(data)
    materials = data[MATERIAL].to_numpy(str)
    factors = np.array([(1.0 - transport_loss_rate) / MATERIAL_CONSUMPTION[item] for item in materials])
    supplier_capacity = _supplier_capacity_matrix(data, weeks)
    carrier_capacity = _weekly_total_vector(carrier_capacity_total, weeks, "周转运总能力")

    # 变量顺序为：固定周产能 p、供应商-周发运量、每周期末库存。
    capacity_index = 0
    shipment_offset = 1
    shipment_variables = n * weeks
    inventory_offset = shipment_offset + shipment_variables
    total_variables = inventory_offset + weeks
    objective = np.zeros(total_variables)
    objective[capacity_index] = -1.0

    # 逐周库存平衡。第 1 周的初始库存为 safety_weeks × p。
    a_eq: list[np.ndarray] = []
    b_eq: list[float] = []
    for t in range(weeks):
        row = np.zeros(total_variables)
        row[shipment_offset + t * n : shipment_offset + (t + 1) * n] = -factors
        row[inventory_offset + t] = 1.0
        row[capacity_index] = 1.0 - safety_weeks if t == 0 else 1.0
        if t > 0:
            row[inventory_offset + t - 1] = -1.0
        a_eq.append(row)
        b_eq.append(0.0)

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for t in range(weeks):
        # 每周总发运量不超过当周总转运能力。
        row = np.zeros(total_variables)
        row[shipment_offset + t * n : shipment_offset + (t + 1) * n] = 1.0
        a_ub.append(row)
        b_ub.append(carrier_capacity[t])
        # 期末库存不低于 safety_weeks × p。
        safety_row = np.zeros(total_variables)
        safety_row[capacity_index] = safety_weeks
        safety_row[inventory_offset + t] = -1.0
        a_ub.append(safety_row)
        b_ub.append(0.0)
        for name, share in shares.items():
            if share <= 1e-12:
                continue
            material_row = np.zeros(total_variables)
            material_row[shipment_offset + t * n : shipment_offset + (t + 1) * n] = share * factors - np.where(materials == name, factors, 0.0)
            a_ub.append(material_row)
            b_ub.append(0.0)

    bounds = [(0.0, None)]
    bounds.extend((0.0, float(supplier_capacity[i, t])) for t in range(weeks) for i in range(n))
    bounds.extend((0.0, None) for _ in range(weeks))
    result = linprog(
        c=objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise OptimizationError(f"动态最大可持续产能优化不可行：{result.message}")
    return float(result.x[capacity_index])

def preference_loss_buffer(base_loss_rate: float, loss_weight: float) -> float:
    """将问题 3 的低损耗偏好权重映射为保守的规划损耗缓冲率。

    权重为 0.50 时保持基础缓冲率；权重越高，缓冲率线性增加，
    以体现对转运损耗更强的规避。
    """
    if not 0.0 <= loss_weight <= 1.0:
        raise ValueError("Loss preference weight must be in [0, 1]")
    return min(0.95, float(base_loss_rate) * (0.5 + float(loss_weight)))


# 将发运量按原料类别换算为未扣损的产品当量，用于敏感性审计。
def shipment_product_equivalent_by_week(
    shipments: pd.DataFrame,
    supplier_materials: Mapping[str, str] | pd.Series,
) -> pd.Series:
    """将每周原料发运量换算为未扣运输损耗的产品当量。"""
    material_map = {str(key): str(value) for key, value in dict(supplier_materials).items()}
    values: dict[int, float] = {}
    for week in shipments.columns:
        total = 0.0
        for supplier_id, volume in shipments[week].items():
            total += float(volume) / MATERIAL_CONSUMPTION[material_map[str(supplier_id)]]
        values[int(week)] = total
    return pd.Series(values, name="planned_product_equivalent").sort_index()


# 采用控制变量法开展两类问题 3 敏感性试验：材料比例网格与低损耗阈值网格。
def run_sensitivity_grid(
    suppliers: pd.DataFrame,
    demand: float,
    a_min_values: Sequence[float],
    c_max_values: Sequence[float],
    loss_weights: Sequence[float],
    a_reward: float,
    c_penalty: float,
    transport_loss_rate: float,
    weeks: int = 24,
    initial_inventory: float | None = None,
    carrier_loss_rates: pd.DataFrame | None = None,
    max_loss_rate: float | None = None,
    carrier_capacity: float | pd.Series | pd.DataFrame | Mapping[str, float] = CARRIER_WEEKLY_CAPACITY,
    carrier_capacity_total: float | Sequence[float] | pd.Series | pd.DataFrame = 8 * CARRIER_WEEKLY_CAPACITY,
    loss_thresholds: Sequence[float] | None = None,
    baseline_a_min: float | None = None,
    baseline_c_max: float | None = None,
) -> pd.DataFrame:
    """运行问题 3 的控制变量敏感性试验并记录采购、转运和库存结果。

    材料比例试验固定低损耗率阈值；阈值试验固定基准 A/C 比例。
    两类试验分开记录，避免四个参数同时变化造成结果难以解释。
    """
    if max_loss_rate is not None and not 0.0 <= float(max_loss_rate) < 1.0:
        raise ValueError("低损耗率阈值必须在 [0, 1) 内")
    threshold_values = tuple(float(value) for value in (loss_thresholds or ()))
    if any(not 0.0 <= value < 1.0 for value in threshold_values):
        raise ValueError("敏感性低损耗率阈值必须在 [0, 1) 内")
    if threshold_values and max_loss_rate is None:
        raise ValueError("提供阈值敏感性网格时必须同时提供基准低损耗率阈值")

    base_a_min = float(a_min_values[0] if baseline_a_min is None else baseline_a_min)
    base_c_max = float(c_max_values[-1] if baseline_c_max is None else baseline_c_max)
    records: list[dict[str, object]] = []
    supplier_materials = suppliers[MATERIAL]
    safety_inventory = 2.0 * float(demand)
    base_initial_inventory = safety_inventory if initial_inventory is None else float(initial_inventory)
    if base_initial_inventory < safety_inventory - 1e-8:
        raise ValueError("期初库存不得低于安全库存")

    def evaluate(a_min: float, c_max: float, loss_weight: float, threshold: float | None, experiment: str) -> None:
        record: dict[str, object] = {
            "试验类型": experiment,
            "A类最低占比": float(a_min),
            "C类最高占比": float(c_max),
            "低损耗权重": float(loss_weight),
            "低损耗率阈值": np.nan if threshold is None else float(threshold),
            "规划损耗缓冲率": preference_loss_buffer(transport_loss_rate, float(loss_weight)),
        }
        try:
            result = solve_preference_order_plan(
                suppliers,
                demand,
                a_min_share=float(a_min),
                c_max_share=float(c_max),
                a_reward=a_reward,
                c_penalty=c_penalty,
                weeks=weeks,
                initial_inventory=base_initial_inventory,
                safety_inventory=safety_inventory,
                carrier_capacity_total=carrier_capacity_total,
                transport_loss_rate=float(record["规划损耗缓冲率"]),
            )
            material = result["material_product_equivalent"]
            total = material.sum(axis=1)
            record.update(
                {
                    "可行": True,
                    "目标成本": float(result["objective"]),
                    "A类实际占比": float((material["A"] / total).mean()),
                    "C类实际占比": float((material["C"] / total).mean()),
                }
            )
            if carrier_loss_rates is None:
                records.append(record)
                return

            allocation = assign_carriers(
                result["shipments"],
                carrier_loss_rates,
                carrier_capacity=carrier_capacity,
                max_loss_rate=threshold,
            )
            planned_receipts = shipment_product_equivalent_by_week(result["shipments"], supplier_materials)
            actual_receipts = post_loss_product_equivalent_by_week(allocation, carrier_loss_rates, supplier_materials)
            actual_inventory = inventory_from_receipts(actual_receipts, demand, safety_inventory)
            eligible_counts = (
                pd.Series(len(carrier_loss_rates), index=carrier_loss_rates.columns)
                if threshold is None
                else (carrier_loss_rates <= threshold + 1e-12).sum(axis=0)
            )
            used_rates: list[float] = []
            used_carriers: set[str] = set()
            for week, matrix in allocation.items():
                positive = matrix.sum(axis=0) > 1e-8
                carriers = matrix.columns[positive].astype(str).tolist()
                used_carriers.update(carriers)
                used_rates.extend(carrier_loss_rates[int(week)].reindex(carriers).astype(float).tolist())
            record.update(
                {
                    "最少合格转运商数": int(eligible_counts.min()),
                    "实际使用转运商数": len(used_carriers),
                    "已使用最大预测损耗率": float(max(used_rates)) if used_rates else np.nan,
                    "实际运输损耗产品当量": float((planned_receipts - actual_receipts).sum()),
                    "实际最低库存": float(actual_inventory.min()),
                    "实际安全库存裕度": float(actual_inventory.min() - safety_inventory),
                }
            )
        except OptimizationError as exc:
            record.update({"可行": False, "原因": str(exc)})
        records.append(record)

    # 材料比例试验：固定题设对应的基准低损耗率阈值，比较 A/C 偏好约束的临界影响。
    for a_min, c_max, loss_weight in product(a_min_values, c_max_values, loss_weights):
        evaluate(float(a_min), float(c_max), float(loss_weight), max_loss_rate, "材料比例敏感性")

    # 阈值试验：固定基准材料偏好，只改变允许使用的低损耗转运商集合。
    for threshold, loss_weight in product(threshold_values, loss_weights):
        evaluate(base_a_min, base_c_max, float(loss_weight), float(threshold), "低损耗率阈值敏感性")
    return pd.DataFrame.from_records(records)