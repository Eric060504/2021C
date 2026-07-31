"""问题 3：带材料偏好约束的降本方案与敏感性分析。"""
from __future__ import annotations

import argparse
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
    Q3_INITIAL_INVENTORY_WEEKS,
    Q3_LOW_LOSS_THRESHOLD,
    Q3_SENSITIVITY_A_MIN,
    Q3_SENSITIVITY_C_MAX,
    Q3_SENSITIVITY_LOSS_WEIGHT,
    Q3_SENSITIVITY_LOSS_THRESHOLD,
    SAFETY_WEEKS,
    TRANSPORT_LOSS_BUFFER_RATE,
    WEEKS_PLAN,
)
from data_io import MATERIAL, SUPPLIER_ID, build_order_result_frame, build_transport_result_frame, read_input_data, write_combined_question_workbooks
from forecasting import (
    build_carrier_capacity_forecast,
    build_carrier_loss_forecast,
    build_supplier_forecasts,
    build_supplier_weekly_capacity_forecast,
)
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
from reporting import build_material_share_report, build_solution_audit, export_solution_reports
from visualization import (
    plot_carrier_loss_threshold,
    plot_carrier_utilization,
    plot_inventory_trace,
    plot_material_comparison,
    plot_material_structure,
    plot_loss_threshold_sensitivity,
    plot_sensitivity_heatmaps,
    plot_weekly_supply_and_shipment,
)


# 在重要供应商范围内实施材料偏好约束的降本优化，并进行敏感性分析。
def run_question_3(output_dir: Path | None = None, generate_plots: bool = True) -> dict[str, object]:
    output_dir = Path(output_dir or OUTPUT_ROOT / "q3")
    output_dir.mkdir(parents=True, exist_ok=True)
    order, supply, loss = read_input_data()
    ranking, _ = entropy_weight_score(compute_supplier_indicators(order, supply))
    all_supplier_ids = order[SUPPLIER_ID].astype(str).tolist()
    important_ids = ranking.head(50)[SUPPLIER_ID].astype(str).tolist()
    forecasts = build_supplier_forecasts(order, supply).loc[important_ids]
    # 以近期实际供给的波动轮廓构造未来 24 周能力上限，避免稳态假设导致周计划机械重复。
    supplier_weekly_capacity = build_supplier_weekly_capacity_forecast(order, supply, weeks=WEEKS_PLAN)
    forecasts.loc[:, range(1, WEEKS_PLAN + 1)] = supplier_weekly_capacity.loc[
        forecasts.index, range(1, WEEKS_PLAN + 1)
    ]
    carrier_loss = build_carrier_loss_forecast(loss)
    # 当前附件的运力为常量；使用矩阵接口可保证与未来时变运力情景保持一致。
    carrier_capacity = build_carrier_capacity_forecast(loss, weeks=WEEKS_PLAN)
    safety = SAFETY_WEEKS * PRODUCT_CAPACITY
    initial_inventory = Q3_INITIAL_INVENTORY_WEEKS * PRODUCT_CAPACITY
    # 成本目标之外，同时施加 A 类下限、C 类上限及采购奖惩。
    plan = solve_preference_order_plan(
        forecasts,
        PRODUCT_CAPACITY,
        a_min_share=Q3_A_MIN_SHARE,
        c_max_share=Q3_C_MAX_SHARE,
        a_reward=Q3_A_REWARD,
        c_penalty=Q3_C_PENALTY,
        weeks=WEEKS_PLAN,
        initial_inventory=initial_inventory,
        safety_inventory=safety,
        carrier_capacity_total=carrier_capacity,
        transport_loss_rate=preference_loss_buffer(TRANSPORT_LOSS_BUFFER_RATE, Q3_LOSS_WEIGHT),
    )
    allocation = assign_carriers(
        plan["shipments"],
        carrier_loss,
        carrier_capacity=carrier_capacity,
        max_loss_rate=Q3_LOW_LOSS_THRESHOLD,
    )
    actual_receipts = post_loss_product_equivalent_by_week(allocation, carrier_loss, forecasts[MATERIAL])
    actual_inventory = inventory_from_receipts(actual_receipts, PRODUCT_CAPACITY, initial_inventory)
    if (actual_inventory < safety - 1e-6).any():
        raise AssertionError("实际损耗后的库存违反两周安全库存约束")
    # 逐周复核材料比例，避免仅满足全期平均比例。
    for _, row in plan["material_product_equivalent"].iterrows():
        if not check_material_share_constraints(row.to_dict(), Q3_A_MIN_SHARE, Q3_C_MAX_SHARE):
            raise AssertionError("问题3材料偏好比例未满足")
    material_share = build_material_share_report(plan["material_product_equivalent"])
    audit, utilization, split_report = build_solution_audit(
        actual_inventory=actual_inventory,
        safety_inventory=safety,
        allocation=allocation,
        carrier_capacity=carrier_capacity,
        loss_rates=carrier_loss,
        max_loss_rate=Q3_LOW_LOSS_THRESHOLD,
        material_share_report=material_share,
        material_constraints={"A": Q3_A_MIN_SHARE},
        material_upper_constraints={"C": Q3_C_MAX_SHARE},
    )
    report_paths = export_solution_reports(
        output_dir,
        audit=audit,
        utilization=utilization,
        split_report=split_report,
        material_share_report=material_share,
    )
    if not (audit["判定"] != "不通过").all():
        raise AssertionError("问题3方案审计未通过")
    order_result = build_order_result_frame(plan["orders"], all_supplier_ids)
    transport_result = build_transport_result_frame(allocation, all_supplier_ids, carrier_loss.index.astype(str).tolist())
    order_path, transport_path = write_combined_question_workbooks(3, order_result, transport_result)
    # 采用控制变量法分别分析材料偏好参数和低损耗率阈值。
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
        initial_inventory=initial_inventory,
        carrier_loss_rates=carrier_loss,
        carrier_capacity=carrier_capacity,
        carrier_capacity_total=carrier_capacity,
        max_loss_rate=Q3_LOW_LOSS_THRESHOLD,
        loss_thresholds=Q3_SENSITIVITY_LOSS_THRESHOLD,
        baseline_a_min=Q3_A_MIN_SHARE,
        baseline_c_max=Q3_C_MAX_SHARE,
    )
    sensitivity.to_csv(output_dir / "sensitivity_analysis.csv", index=False, encoding="utf-8-sig")
    material = plan["material_product_equivalent"]
    total_product = material.sum(axis=1)
    weekly_supply = forecasts.loc[:, range(1, WEEKS_PLAN + 1)].sum(axis=0).astype(float)
    weekly_shipments = plan["shipments"].sum(axis=0).astype(float)
    weekly_carrier_capacity = carrier_capacity.loc[:, range(1, WEEKS_PLAN + 1)].sum(axis=0).astype(float)
    summary = pd.DataFrame(
        {
            "指标": [
                "A类最低占比", "C类最高占比", "A类采购奖励", "C类采购惩罚", "低损耗率阈值",
                "低损耗偏好权重", "期初库存产品当量", "标准化总成本", "实际A类平均占比", "实际C类平均占比",
                "预测周供货能力最小值", "预测周供货能力最大值", "计划周发运量最小值",
                "计划周发运量最大值", "实际最低库存",
            ],
            "数值": [
                Q3_A_MIN_SHARE, Q3_C_MAX_SHARE, Q3_A_REWARD, Q3_C_PENALTY, Q3_LOW_LOSS_THRESHOLD,
                Q3_LOSS_WEIGHT, initial_inventory, plan["objective"], float((material["A"] / total_product).mean()),
                float((material["C"] / total_product).mean()), float(weekly_supply.min()),
                float(weekly_supply.max()), float(weekly_shipments.min()), float(weekly_shipments.max()),
                float(actual_inventory.min()),
            ],
        }
    )
    summary.to_excel(output_dir / "summary.xlsx", index=False)
    supplier_weekly_capacity.to_excel(output_dir / "supplier_weekly_capacity_forecast.xlsx", index=False)
    pd.DataFrame(
        {
            "周": actual_inventory.index,
            "预测可供货量": weekly_supply.reindex(actual_inventory.index).values,
            "计划发运量": weekly_shipments.reindex(actual_inventory.index).values,
            "周转运总能力": weekly_carrier_capacity.reindex(actual_inventory.index).values,
            "生产需求产品当量": PRODUCT_CAPACITY,
            "实际到厂产品当量": actual_receipts.reindex(actual_inventory.index).values,
            "实际库存产品当量": actual_inventory.values,
        }
    ).to_excel(output_dir / "inventory_trace.xlsx", index=False)
    figure_paths: list[Path] = []
    if generate_plots:
        figure_dir = output_dir.parent / "figures" / output_dir.name
        figure_paths.extend(
            [
                plot_inventory_trace(actual_inventory, safety, PRODUCT_CAPACITY, figure_dir / "inventory_trace.png", "问题3：库存与实际到厂量", actual_receipts),
                plot_carrier_utilization(utilization, figure_dir / "carrier_utilization_heatmap.png", "问题3：低损耗转运商周利用率"),
                plot_material_structure(material_share, figure_dir / "material_structure_by_week.png", "问题3：周度材料产品当量结构", Q3_A_MIN_SHARE, Q3_C_MAX_SHARE),
                plot_carrier_loss_threshold(carrier_loss, figure_dir / "carrier_loss_threshold.png", Q3_LOW_LOSS_THRESHOLD, allocation),
                plot_weekly_supply_and_shipment(
                    forecasts.loc[:, range(1, WEEKS_PLAN + 1)],
                    plan["shipments"],
                    weekly_carrier_capacity,
                    figure_dir / "weekly_supply_and_shipment.png",
                    "问题3：未来24周供给预测与偏好约束发运方案",
                ),
            ]
        )
        figure_paths.extend(plot_sensitivity_heatmaps(sensitivity, figure_dir / "sensitivity"))
        threshold_figure = plot_loss_threshold_sensitivity(sensitivity, figure_dir / "loss_threshold_sensitivity.png")
        if threshold_figure is not None:
            figure_paths.append(threshold_figure)
        q2_material_share_path = output_dir.parent / "q2" / "material_share.xlsx"
        if q2_material_share_path.exists():
            q2_material_share = pd.read_excel(q2_material_share_path)
            figure_paths.append(plot_material_comparison(q2_material_share, material_share, figure_dir / "q2_q3_material_comparison.png"))
    return {
        "plan": plan,
        "allocation": allocation,
        "actual_receipts": actual_receipts,
        "actual_inventory": actual_inventory,
        "audit": audit,
        "utilization": utilization,
        "material_share": material_share,
        "report_paths": report_paths,
        "figure_paths": figure_paths,
        "order_path": order_path,
        "transport_path": transport_path,
        "sensitivity": sensitivity,
        "supplier_weekly_capacity": supplier_weekly_capacity,
        "carrier_capacity": carrier_capacity,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行问题3成本压缩订购与转运模型")
    parser.add_argument("--output-dir", type=Path, default=None, help="结果输出目录")
    parser.add_argument("--no-plot", action="store_true", help="不生成图表")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_question_3(args.output_dir, generate_plots=not args.no_plot)
    print("问题3完成：已生成偏好约束的降本方案、审计、敏感性分析和论文图表。")
    print(f"附件A：{result['order_path']}")
    print(f"附件B：{result['transport_path']}")

