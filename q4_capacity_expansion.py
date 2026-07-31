"""问题 4：最大可持续周产能及对应的 24 周订购、转运方案。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import OUTPUT_ROOT, PRODUCT_CAPACITY, Q4_MIN_PRODUCT_SHARES, SAFETY_WEEKS, TRANSPORT_LOSS_BUFFER_RATE, WEEKS_PLAN
from data_io import MATERIAL, SUPPLIER_ID, build_order_result_frame, build_transport_result_frame, read_input_data, write_combined_question_workbooks
from forecasting import build_carrier_capacity_forecast, build_carrier_loss_forecast, build_supplier_forecasts, build_supplier_weekly_capacity_forecast
from optimization import assign_carriers, inventory_from_receipts, post_loss_product_equivalent_by_week, solve_dynamic_sustainable_capacity, solve_order_plan
from reporting import build_material_share_report, build_solution_audit, export_solution_reports
from visualization import plot_capacity_comparison, plot_carrier_utilization, plot_inventory_trace, plot_material_structure, plot_q4_weekly_supply_plan


# 在全部供应商范围内求最大可持续产能，并生成配套的订购、转运、审计和图表结果。
def run_question_4(output_dir: Path | None = None, generate_plots: bool = True) -> dict[str, object]:
    output_dir = Path(output_dir or OUTPUT_ROOT / "q4")
    output_dir.mkdir(parents=True, exist_ok=True)
    order, supply, loss = read_input_data()
    all_supplier_ids = order[SUPPLIER_ID].astype(str).tolist()
    # 问题 4 要求在全部供应商范围内寻求最大可持续产能，不能仅使用问题 1 的前 50 家。
    forecasts = build_supplier_forecasts(order, supply)
    # 保留稳健供货率和长期能力，同时以近期 24 周供货起伏构建逐周可供货上限。
    supplier_weekly_capacity = build_supplier_weekly_capacity_forecast(order, supply, weeks=WEEKS_PLAN)
    forecasts.loc[:, range(1, WEEKS_PLAN + 1)] = supplier_weekly_capacity.loc[forecasts.index, range(1, WEEKS_PLAN + 1)]
    carrier_loss = build_carrier_loss_forecast(loss)
    carrier_capacity = build_carrier_capacity_forecast(loss)
    # 在逐周供给、逐周转运能力与两周安全库存约束下，最大化整个规划期均可维持的固定周产能。
    max_capacity = solve_dynamic_sustainable_capacity(
        forecasts,
        weeks=WEEKS_PLAN,
        carrier_capacity_total=carrier_capacity,
        transport_loss_rate=TRANSPORT_LOSS_BUFFER_RATE,
        safety_weeks=SAFETY_WEEKS,
        material_min_shares=Q4_MIN_PRODUCT_SHARES,
    )
    safety = SAFETY_WEEKS * max_capacity
    plan = solve_order_plan(
        forecasts,
        max_capacity,
        weeks=WEEKS_PLAN,
        initial_inventory=safety,
        safety_inventory=safety,
        carrier_capacity_total=carrier_capacity,
        transport_loss_rate=TRANSPORT_LOSS_BUFFER_RATE,
    )
    # 依据实际转运损耗重新计算到厂量，检验安全库存是否仍满足。
    allocation = assign_carriers(
        plan["shipments"],
        carrier_loss,
        carrier_capacity=carrier_capacity,
    )
    actual_receipts = post_loss_product_equivalent_by_week(allocation, carrier_loss, forecasts[MATERIAL])
    actual_inventory = inventory_from_receipts(actual_receipts, plan["demand_by_week"], safety)
    if (actual_inventory < safety - 1e-6).any():
        raise AssertionError("实际损耗后的库存违反两周安全库存约束")
    material_product = plan["material_product_equivalent"]
    total_product = material_product.sum(axis=1)
    for material, minimum_share in Q4_MIN_PRODUCT_SHARES.items():
        if minimum_share > 0.0 and (material_product[material] < minimum_share * total_product - 1e-6).any():
            raise AssertionError(f"问题4材料 {material} 的最低占比约束未满足")
    material_share = build_material_share_report(material_product)
    audit, utilization, split_report = build_solution_audit(
        actual_inventory=actual_inventory,
        safety_inventory=safety,
        allocation=allocation,
        carrier_capacity=carrier_capacity,
        loss_rates=carrier_loss,
        material_share_report=material_share,
        material_constraints=Q4_MIN_PRODUCT_SHARES,
    )
    report_paths = export_solution_reports(
        output_dir,
        audit=audit,
        utilization=utilization,
        split_report=split_report,
        material_share_report=material_share,
    )
    if not (audit["判定"] != "不通过").all():
        raise AssertionError("问题4方案审计未通过")
    order_result = build_order_result_frame(plan["orders"], all_supplier_ids)
    transport_result = build_transport_result_frame(allocation, all_supplier_ids, carrier_loss.index.astype(str).tolist())
    order_path, transport_path = write_combined_question_workbooks(4, order_result, transport_result)
    summary = pd.DataFrame(
        {
            "指标": ["候选供应商数", "最大可持续周产能", "相对当前产能提升", "提升比例", "安全库存", "标准化总成本", "实际最低库存", "总转运量", "预测周供货能力最小值", "预测周供货能力最大值", "周发运量最小值", "周发运量最大值"],
            "数值": [len(forecasts), max_capacity, max_capacity - PRODUCT_CAPACITY, max_capacity / PRODUCT_CAPACITY - 1.0, safety, plan["objective"], float(actual_inventory.min()), float(plan["shipments"].to_numpy().sum()), float(plan["supplier_capacity_by_week"].sum(axis=0).min()), float(plan["supplier_capacity_by_week"].sum(axis=0).max()), float(plan["shipments"].sum(axis=0).min()), float(plan["shipments"].sum(axis=0).max())],
        }
    )
    summary.to_excel(output_dir / "summary.xlsx", index=False)
    pd.DataFrame({
        "周": actual_inventory.index,
        "生产需求产品当量": plan["demand_by_week"].values,
        "预测可供货量": plan["supplier_capacity_by_week"].sum(axis=0).values,
        "计划发运量": plan["shipments"].sum(axis=0).values,
        "周转运总能力": plan["carrier_capacity_by_week"].values,
        "实际到厂产品当量": actual_receipts.values,
        "实际库存产品当量": actual_inventory.values,
    }).to_excel(output_dir / "inventory_trace.xlsx", index=False)
    plan["supplier_capacity_by_week"].to_excel(output_dir / "supplier_weekly_capacity_forecast.xlsx")
    figure_paths: list[Path] = []
    if generate_plots:
        figure_dir = output_dir.parent / "figures" / output_dir.name
        figure_paths = [
            plot_capacity_comparison(PRODUCT_CAPACITY, max_capacity, figure_dir / "capacity_comparison.png"),
            plot_inventory_trace(actual_inventory, safety, max_capacity, figure_dir / "inventory_trace.png", "问题4：扩产方案库存与实际到厂量", actual_receipts),
            plot_carrier_utilization(utilization, figure_dir / "carrier_utilization_heatmap.png", "问题4：扩产方案转运商周利用率"),
            plot_material_structure(material_share, figure_dir / "material_structure_by_week.png", "问题4：扩产方案周度材料产品当量结构"),
            plot_q4_weekly_supply_plan(
                plan["supplier_capacity_by_week"],
                plan["shipments"],
                plan["carrier_capacity_by_week"],
                figure_dir / "weekly_supply_and_shipment.png",
            ),
        ]
    return {
        "candidate_supplier_count": len(forecasts),
        "carrier_capacity": carrier_capacity,
        "supplier_weekly_capacity": plan["supplier_capacity_by_week"],
        "max_capacity": max_capacity,
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
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行问题4最大可持续产能模型")
    parser.add_argument("--output-dir", type=Path, default=None, help="结果输出目录")
    parser.add_argument("--no-plot", action="store_true", help="不生成图表")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_question_4(args.output_dir, generate_plots=not args.no_plot)
    print("问题4完成：已生成最大可持续产能、订购转运方案、审计和论文图表。")
    print(f"最大可持续周产能：{result['max_capacity']:.2f} m³ 产品")
    print(f"附件A：{result['order_path']}")
    print(f"附件B：{result['transport_path']}")
