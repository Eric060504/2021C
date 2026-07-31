"""问题 3：带材料偏好约束的降本方案与敏感性分析。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    OUTPUT_ROOT,
    PRODUCT_CAPACITY,
    Q3_A_MIN_SHARE,
    Q3_A_REWARD,
    Q3_C_MAX_SHARE,
    Q3_C_PENALTY,
    Q3_LOSS_WEIGHT,
    Q3_LOW_LOSS_THRESHOLD,
    Q3_SENSITIVITY_A_MIN,
    Q3_SENSITIVITY_C_MAX,
    Q3_SENSITIVITY_LOSS_WEIGHT,
    SAFETY_WEEKS,
    TRANSPORT_LOSS_BUFFER_RATE,
    WEEKS_PLAN,
)
from data_io import MATERIAL, SUPPLIER_ID, build_order_result_frame, build_transport_result_frame, read_input_data, write_combined_question_workbooks
from forecasting import build_carrier_loss_forecast, build_supplier_forecasts
from optimization import (
    assign_carriers,
    check_material_share_constraints,
    inventory_from_receipts,
    post_loss_product_equivalent_by_week,
    preference_loss_buffer,
    run_sensitivity_grid,
    solve_preference_order_plan,
)
from q1_supplier_importance import compute_supplier_indicators, entropy_weight_score


# 在重要供应商范围内实施材料偏好约束的降本优化，并进行敏感性分析。
def run_question_3(output_dir: Path | None = None) -> dict[str, object]:
    output_dir = Path(output_dir or OUTPUT_ROOT / "q3")
    output_dir.mkdir(parents=True, exist_ok=True)
    order, supply, loss = read_input_data()
    ranking, _ = entropy_weight_score(compute_supplier_indicators(order, supply))
    all_supplier_ids = order[SUPPLIER_ID].astype(str).tolist()
    important_ids = ranking.head(50)[SUPPLIER_ID].astype(str).tolist()
    forecasts = build_supplier_forecasts(order, supply).loc[important_ids]
    carrier_loss = build_carrier_loss_forecast(loss)
    safety = SAFETY_WEEKS * PRODUCT_CAPACITY
    # 成本目标之外，同时施加 A 类下限、C 类上限及采购奖惩。
    plan = solve_preference_order_plan(
        forecasts,
        PRODUCT_CAPACITY,
        a_min_share=Q3_A_MIN_SHARE,
        c_max_share=Q3_C_MAX_SHARE,
        a_reward=Q3_A_REWARD,
        c_penalty=Q3_C_PENALTY,
        weeks=WEEKS_PLAN,
        initial_inventory=safety,
        safety_inventory=safety,
        transport_loss_rate=preference_loss_buffer(TRANSPORT_LOSS_BUFFER_RATE, Q3_LOSS_WEIGHT),
    )
    allocation = assign_carriers(
        plan["shipments"],
        carrier_loss,
        max_loss_rate=Q3_LOW_LOSS_THRESHOLD,
    )
    actual_receipts = post_loss_product_equivalent_by_week(allocation, carrier_loss, forecasts[MATERIAL])
    actual_inventory = inventory_from_receipts(actual_receipts, PRODUCT_CAPACITY, safety)
    if (actual_inventory < safety - 1e-6).any():
        raise AssertionError("实际损耗后的库存违反两周安全库存约束")
    # 逐周复核材料比例，避免仅满足全期平均比例。
    for _, row in plan["material_product_equivalent"].iterrows():
        if not check_material_share_constraints(row.to_dict(), Q3_A_MIN_SHARE, Q3_C_MAX_SHARE):
            raise AssertionError("问题3材料偏好比例未满足")
    order_result = build_order_result_frame(plan["orders"], all_supplier_ids)
    transport_result = build_transport_result_frame(allocation, all_supplier_ids, carrier_loss.index.astype(str).tolist())
    order_path, transport_path = write_combined_question_workbooks(3, order_result, transport_result)
    # 对三类关键偏好参数进行全组合敏感性分析。
    sensitivity = run_sensitivity_grid(
        forecasts,
        PRODUCT_CAPACITY,
        Q3_SENSITIVITY_A_MIN,
        Q3_SENSITIVITY_C_MAX,
        Q3_SENSITIVITY_LOSS_WEIGHT,
        Q3_A_REWARD,
        Q3_C_PENALTY,
        TRANSPORT_LOSS_BUFFER_RATE,
        weeks=WEEKS_PLAN,
        carrier_loss_rates=carrier_loss,
        max_loss_rate=Q3_LOW_LOSS_THRESHOLD,
    )
    sensitivity.to_csv(output_dir / "sensitivity_analysis.csv", index=False, encoding="utf-8-sig")
    material = plan["material_product_equivalent"]
    total_product = material.sum(axis=1)
    summary = pd.DataFrame(
        {
            "指标": ["A类最低占比", "C类最高占比", "A类采购奖励", "C类采购惩罚", "低损耗率阈值", "低损耗偏好权重", "标准化总成本", "实际A类平均占比", "实际C类平均占比", "实际最低库存"],
            "数值": [Q3_A_MIN_SHARE, Q3_C_MAX_SHARE, Q3_A_REWARD, Q3_C_PENALTY, Q3_LOW_LOSS_THRESHOLD, Q3_LOSS_WEIGHT, plan["objective"], float((material["A"] / total_product).mean()), float((material["C"] / total_product).mean()), float(actual_inventory.min())],
        }
    )
    summary.to_excel(output_dir / "summary.xlsx", index=False)
    pd.DataFrame({"周": actual_inventory.index, "实际到厂产品当量": actual_receipts.values, "实际库存产品当量": actual_inventory.values}).to_excel(
        output_dir / "inventory_trace.xlsx", index=False
    )
    return {"plan": plan, "allocation": allocation, "actual_inventory": actual_inventory, "order_path": order_path, "transport_path": transport_path, "sensitivity": sensitivity}


if __name__ == "__main__":
    result = run_question_3()
    print("问题3完成：已生成偏好约束的降本方案与敏感性分析。")
    print(f"附件A：{result['order_path']}")
    print(f"附件B：{result['transport_path']}")
