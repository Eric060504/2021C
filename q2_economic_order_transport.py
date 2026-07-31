"""问题 2：最少供应商的经济订购与最小损耗转运方案。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import OUTPUT_ROOT, PRODUCT_CAPACITY, SAFETY_WEEKS, TRANSPORT_LOSS_BUFFER_RATE, WEEKS_PLAN
from data_io import (
    MATERIAL,
    SUPPLIER_ID,
    build_order_result_frame,
    build_transport_result_frame,
    read_input_data,
    write_combined_question_workbooks,
)
from forecasting import build_carrier_loss_forecast, build_supplier_forecasts
from optimization import (
    assign_carriers,
    inventory_from_receipts,
    post_loss_product_equivalent_by_week,
    select_minimum_suppliers,
    solve_order_plan,
)
from q1_supplier_importance import compute_supplier_indicators, entropy_weight_score


# 复用问题 1 的前 50 名供应商，并构造供货、转运损耗预测。
def _important_forecasts() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    order, supply, loss = read_input_data()
    ranking, _ = entropy_weight_score(compute_supplier_indicators(order, supply))
    supplier_forecast = build_supplier_forecasts(order, supply)
    important_ids = ranking.head(50)[SUPPLIER_ID].astype(str).tolist()
    carrier_loss = build_carrier_loss_forecast(loss)
    return supplier_forecast.loc[important_ids], carrier_loss, order[SUPPLIER_ID].astype(str).tolist(), important_ids


# 选择满足目标产能的最少供应商，生成 24 周经济订购和低损耗转运方案。
def run_question_2(output_dir: Path | None = None) -> dict[str, object]:
    output_dir = Path(output_dir or OUTPUT_ROOT / "q2")
    output_dir.mkdir(parents=True, exist_ok=True)
    forecasts, carrier_loss, all_supplier_ids, _ = _important_forecasts()
    # 先按产品当量能力选最少供应商，再在该集合内做成本优化。
    selected_ids = select_minimum_suppliers(forecasts, PRODUCT_CAPACITY, TRANSPORT_LOSS_BUFFER_RATE)
    selected = forecasts.loc[selected_ids]
    safety = SAFETY_WEEKS * PRODUCT_CAPACITY
    plan = solve_order_plan(
        selected,
        PRODUCT_CAPACITY,
        weeks=WEEKS_PLAN,
        initial_inventory=safety,
        safety_inventory=safety,
        transport_loss_rate=TRANSPORT_LOSS_BUFFER_RATE,
    )
    # 用逐家转运商的预测损耗进行低损耗分配，并复核实际库存。
    allocation = assign_carriers(plan["shipments"], carrier_loss)
    actual_receipts = post_loss_product_equivalent_by_week(allocation, carrier_loss, selected[MATERIAL])
    actual_inventory = inventory_from_receipts(actual_receipts, PRODUCT_CAPACITY, safety)
    if (actual_inventory < safety - 1e-6).any():
        raise AssertionError("实际损耗后的库存违反两周安全库存约束")
    order_result = build_order_result_frame(plan["orders"], all_supplier_ids)
    transport_result = build_transport_result_frame(allocation, all_supplier_ids, carrier_loss.index.astype(str).tolist())
    order_path, transport_path = write_combined_question_workbooks(2, order_result, transport_result)
    total_shipped = float(plan["shipments"].to_numpy().sum())
    total_loss = float(sum(matrix.mul(carrier_loss[week], axis=1).to_numpy().sum() for week, matrix in allocation.items()))
    summary = pd.DataFrame(
        {
            "指标": ["计划周数", "选定供应商数量", "目标周产能", "安全库存", "标准化总成本", "总转运量", "预测总损耗", "实际最低库存"],
            "数值": [WEEKS_PLAN, len(selected_ids), PRODUCT_CAPACITY, safety, plan["objective"], total_shipped, total_loss, float(actual_inventory.min())],
        }
    )
    summary.to_excel(output_dir / "summary.xlsx", index=False)
    pd.DataFrame({"供应商ID": selected_ids}).to_excel(output_dir / "selected_suppliers.xlsx", index=False)
    pd.DataFrame({"周": actual_inventory.index, "实际到厂产品当量": actual_receipts.values, "实际库存产品当量": actual_inventory.values}).to_excel(
        output_dir / "inventory_trace.xlsx", index=False
    )
    return {
        "plan": plan,
        "allocation": allocation,
        "actual_inventory": actual_inventory,
        "selected_ids": selected_ids,
        "order_path": order_path,
        "transport_path": transport_path,
    }


if __name__ == "__main__":
    result = run_question_2()
    print("问题2完成：已生成最少供应商订购与最小损耗转运方案。")
    print(f"选定供应商数：{len(result['selected_ids'])}")
    print(f"附件A：{result['order_path']}")
    print(f"附件B：{result['transport_path']}")
