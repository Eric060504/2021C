"""求解结果审计、汇总和表格导出工具。"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from config import EPSILON


def _capacity_for_week(
    carrier_capacity: float | pd.Series | pd.DataFrame | Mapping[str, float],
    carriers: list[str],
    week: int,
) -> pd.Series:
    """将标量、按转运商、按周或转运商×周次运力统一为单周序列。"""
    if np.isscalar(carrier_capacity):
        return pd.Series(float(carrier_capacity), index=carriers, dtype=float)
    if isinstance(carrier_capacity, pd.DataFrame):
        if week not in carrier_capacity.columns:
            raise ValueError(f"转运能力表缺少第 {week} 周")
        return pd.to_numeric(carrier_capacity[week].reindex(carriers), errors="coerce").fillna(0.0)
    if isinstance(carrier_capacity, pd.Series):
        return pd.to_numeric(carrier_capacity.reindex(carriers), errors="coerce").fillna(0.0)
    values = pd.Series(dict(carrier_capacity), dtype=float)
    return values.reindex(carriers).fillna(0.0)


def build_carrier_utilization_report(
    allocation: Mapping[int, pd.DataFrame],
    carrier_capacity: float | pd.Series | pd.DataFrame | Mapping[str, float],
) -> pd.DataFrame:
    """汇总每周各转运商运输量、运力和利用率。"""
    records: list[dict[str, float | int | str]] = []
    for week, matrix in sorted(allocation.items()):
        carriers = [str(carrier) for carrier in matrix.columns]
        loads = matrix.copy()
        loads.columns = carriers
        weekly_load = loads.sum(axis=0).reindex(carriers).astype(float)
        weekly_capacity = _capacity_for_week(carrier_capacity, carriers, int(week))
        for carrier in carriers:
            capacity = float(weekly_capacity[carrier])
            transported = float(weekly_load[carrier])
            records.append(
                {
                    "周": int(week),
                    "转运商ID": carrier,
                    "运输量": transported,
                    "运力上限": capacity,
                    "利用率": transported / capacity if capacity > EPSILON else np.nan,
                }
            )
    return pd.DataFrame(records).sort_values(["周", "转运商ID"]).reset_index(drop=True)


def build_material_share_report(material_product_equivalent: pd.DataFrame) -> pd.DataFrame:
    """将各类材料产品当量转换为周度占比表。"""
    materials = ("A", "B", "C")
    frame = material_product_equivalent.copy()
    for material in materials:
        if material not in frame.columns:
            frame[material] = 0.0
    frame = frame.loc[:, list(materials)].astype(float)
    total = frame.sum(axis=1)
    result = pd.DataFrame({"周": frame.index.astype(int), "总产品当量": total.values})
    for material in materials:
        result[f"{material}类产品当量"] = frame[material].values
        result[f"{material}类占比"] = np.divide(
            frame[material].to_numpy(), total.to_numpy(), out=np.zeros(len(frame)), where=total.to_numpy() > EPSILON
        )
    return result


def build_supplier_split_report(allocation: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """统计供应商单周发运量是否被拆分至多个转运商。"""
    records: list[dict[str, float | int | str]] = []
    for week, matrix in sorted(allocation.items()):
        for supplier_id, row in matrix.iterrows():
            used_count = int((row.astype(float) > EPSILON).sum())
            transported = float(row.astype(float).sum())
            if transported > EPSILON:
                records.append(
                    {
                        "周": int(week),
                        "供应商ID": str(supplier_id),
                        "发运量": transported,
                        "使用转运商数": used_count,
                        "是否拆分": "是" if used_count > 1 else "否",
                    }
                )
    return pd.DataFrame(records)


def build_solution_audit(
    *,
    actual_inventory: pd.Series,
    safety_inventory: float,
    allocation: Mapping[int, pd.DataFrame],
    carrier_capacity: float | pd.Series | pd.DataFrame | Mapping[str, float],
    loss_rates: pd.DataFrame,
    max_loss_rate: float | None = None,
    material_share_report: pd.DataFrame | None = None,
    material_constraints: Mapping[str, float] | None = None,
    material_upper_constraints: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成库存、运力、损耗、拆分和材料比例的统一可行性审计结果。"""
    utilization = build_carrier_utilization_report(allocation, carrier_capacity)
    split_report = build_supplier_split_report(allocation)
    min_inventory = float(actual_inventory.min())
    max_utilization = float(utilization["利用率"].max()) if not utilization.empty else 0.0
    total_shipment = float(utilization["运输量"].sum()) if not utilization.empty else 0.0
    total_loss = 0.0
    used_loss_rates: list[float] = []
    for week, matrix in allocation.items():
        rates = pd.to_numeric(loss_rates[int(week)].reindex(matrix.columns), errors="coerce").fillna(0.0)
        total_loss += float(matrix.mul(rates, axis=1).to_numpy().sum())
        active = matrix.columns[matrix.sum(axis=0) > EPSILON]
        used_loss_rates.extend(float(rates[carrier]) for carrier in active)

    records: list[dict[str, float | str]] = [
        {"校验项": "实际最低库存", "数值": min_inventory, "判定": "通过" if min_inventory >= safety_inventory - EPSILON else "不通过", "说明": "实际库存不低于安全库存"},
        {"校验项": "安全库存", "数值": float(safety_inventory), "判定": "基准", "说明": "两周需求对应的安全库存"},
        {"校验项": "最大转运商利用率", "数值": max_utilization, "判定": "通过" if max_utilization <= 1.0 + EPSILON else "不通过", "说明": "任一转运商任一周不得超运力"},
        {"校验项": "总转运量", "数值": total_shipment, "判定": "统计", "说明": "规划期所有转运商运输量之和"},
        {"校验项": "预测总损耗", "数值": total_loss, "判定": "统计", "说明": "按转运商周损耗率计算"},
        {"校验项": "发生拆分的供应商周数", "数值": int((split_report["是否拆分"] == "是").sum()) if not split_report.empty else 0, "判定": "统计", "说明": "单周发运量分配给多个转运商的次数"},
    ]
    if max_loss_rate is not None:
        max_used_loss = max(used_loss_rates, default=0.0)
        records.append(
            {
                "校验项": "已使用转运商最大预测损耗率",
                "数值": max_used_loss,
                "判定": "通过" if max_used_loss <= max_loss_rate + EPSILON else "不通过",
                "说明": f"不高于低损耗率阈值 {max_loss_rate:.2%}",
            }
        )
    if material_share_report is not None and material_constraints is not None:
        for material, lower in material_constraints.items():
            if material not in {"A", "B", "C"} or lower <= 0.0:
                continue
            value = float(material_share_report[f"{material}类占比"].min())
            records.append(
                {
                    "校验项": f"{material}类最低周占比",
                    "数值": value,
                    "判定": "通过" if value >= lower - EPSILON else "不通过",
                    "说明": f"不低于设定下限 {lower:.2%}",
                }
            )
    if material_share_report is not None and material_upper_constraints is not None:
        for material, upper in material_upper_constraints.items():
            if material not in {"A", "B", "C"} or upper >= 1.0:
                continue
            value = float(material_share_report[f"{material}类占比"].max())
            records.append(
                {
                    "校验项": f"{material}类最高周占比",
                    "数值": value,
                    "判定": "通过" if value <= upper + EPSILON else "不通过",
                    "说明": f"不高于设定上限 {upper:.2%}",
                }
            )
    return pd.DataFrame(records), utilization, split_report


def export_solution_reports(
    output_dir: Path,
    *,
    audit: pd.DataFrame,
    utilization: pd.DataFrame,
    split_report: pd.DataFrame,
    material_share_report: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """写出审计明细，供论文制图、结果复核和附录引用。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "audit": output_dir / "solution_audit.xlsx",
        "carrier_utilization": output_dir / "carrier_utilization.xlsx",
        "supplier_split": output_dir / "supplier_split_report.xlsx",
    }
    audit.to_excel(paths["audit"], index=False)
    utilization.to_excel(paths["carrier_utilization"], index=False)
    split_report.to_excel(paths["supplier_split"], index=False)
    if material_share_report is not None:
        paths["material_share"] = output_dir / "material_share.xlsx"
        material_share_report.to_excel(paths["material_share"], index=False)
    return paths


