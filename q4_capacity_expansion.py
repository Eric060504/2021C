"""问题 4：最大可持续周产能及对应的 24 周订购、转运方案。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import OUTPUT_ROOT, PRODUCT_CAPACITY, Q4_MIN_PRODUCT_SHARES, SAFETY_WEEKS, TRANSPORT_LOSS_BUFFER_RATE, WEEKS_PLAN
from data_io import MATERIAL, SUPPLIER_ID, build_order_result_frame, build_transport_result_frame, read_input_data, write_combined_question_workbooks
from forecasting import build_carrier_capacity_forecast, build_carrier_loss_forecast, build_supplier_forecasts
from optimization import assign_carriers, inventory_from_receipts, post_loss_product_equivalent_by_week, solve_max_sustainable_capacity, solve_order_plan


# 求重要供应商条件下的最大可持续产能，并生成配套的 24 周计划。
def run_question_4(output_dir: Path | None = None) -> dict[str, object]:
    output_dir = Path(output_dir or OUTPUT_ROOT / "q4")
    output_dir.mkdir(parents=True, exist_ok=True)
    order, supply, loss = read_input_data()
    all_supplier_ids = order[SUPPLIER_ID].astype(str).tolist()
    # 问题 4 要求在全部供应商范围内寻求最大可持续产能，
    # 不能仅使用问题 1 中综合评分前 50 的重要供应商。
    forecasts = build_supplier_forecasts(order, supply)
    carrier_loss = build_carrier_loss_forecast(loss)
    carrier_capacity = build_carrier_capacity_forecast(loss)
    weekly_total_capacity = float(carrier_capacity.sum(axis=0).min())
    max_capacity = solve_max_sustainable_capacity(
        forecasts,
        carrier_capacity_total=carrier_capacity,
        transport_loss_rate=TRANSPORT_LOSS_BUFFER_RATE,
        material_min_shares=Q4_MIN_PRODUCT_SHARES,
    )
    safety = SAFETY_WEEKS * max_capacity
    plan = solve_order_plan(
        forecasts,
        max_capacity,
        weeks=WEEKS_PLAN,
        initial_inventory=safety,
        safety_inventory=safety,
        carrier_capacity_total=weekly_total_capacity,
        transport_loss_rate=TRANSPORT_LOSS_BUFFER_RATE,
    )
    # 依据实际转运损耗重新计算到厂量，检验安全库存是否仍满足。
    allocation = assign_carriers(
        plan["shipments"],
        carrier_loss,
        carrier_capacity=carrier_capacity,
    )
    actual_receipts = post_loss_product_equivalent_by_week(allocation, carrier_loss, forecasts[MATERIAL])
    actual_inventory = inventory_from_receipts(actual_receipts, max_capacity, safety)
    if (actual_inventory < safety - 1e-6).any():
        raise AssertionError("实际损耗后的库存违反两周安全库存约束")
    material_product = plan["material_product_equivalent"]
    total_product = material_product.sum(axis=1)
    for material, minimum_share in Q4_MIN_PRODUCT_SHARES.items():
        if minimum_share > 0.0 and (material_product[material] < minimum_share * total_product - 1e-6).any():
            raise AssertionError(f"Question 4 minimum share for material {material} is violated")
    order_result = build_order_result_frame(plan["orders"], all_supplier_ids)
    transport_result = build_transport_result_frame(allocation, all_supplier_ids, carrier_loss.index.astype(str).tolist())
    order_path, transport_path = write_combined_question_workbooks(4, order_result, transport_result)
    summary = pd.DataFrame(
        {
            "指标": ["候选供应商数", "最大可持续周产能", "相对当前产能提升", "提升比例", "安全库存", "标准化总成本", "实际最低库存", "总转运量"],
            "数值": [len(forecasts), max_capacity, max_capacity - PRODUCT_CAPACITY, max_capacity / PRODUCT_CAPACITY - 1.0, safety, plan["objective"], float(actual_inventory.min()), float(plan["shipments"].to_numpy().sum())],
        }
    )
    summary.to_excel(output_dir / "summary.xlsx", index=False)
    pd.DataFrame({"周": actual_inventory.index, "实际到厂产品当量": actual_receipts.values, "实际库存产品当量": actual_inventory.values}).to_excel(
        output_dir / "inventory_trace.xlsx", index=False
    )
    return {"candidate_supplier_count": len(forecasts), "carrier_capacity": carrier_capacity, "max_capacity": max_capacity, "plan": plan, "allocation": allocation, "actual_inventory": actual_inventory, "order_path": order_path, "transport_path": transport_path}


if __name__ == "__main__":
    result = run_question_4()
    print("问题4完成：已生成最大可持续产能下的订购与转运方案。")
    print(f"最大可持续周产能：{result['max_capacity']:.2f} m³ 产品")
    print(f"附件A：{result['order_path']}")
    print(f"附件B：{result['transport_path']}")
