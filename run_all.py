"""一键运行问题 1 至问题 4，并汇总核心结果。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import OUTPUT_ROOT, PRODUCT_CAPACITY
from q1_supplier_importance import run_question_1
from q2_economic_order_transport import run_question_2
from q3_cost_compression_plan import run_question_3
from q4_capacity_expansion import run_question_4


def run_all(output_root: Path | None = None, generate_plots: bool = True) -> dict[str, object]:
    """按题号顺序运行全部模型，并导出跨问题核心指标汇总表。"""
    output_root = Path(output_root or OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    ranking, weights = run_question_1(output_root / "q1", generate_plots=generate_plots)
    q2 = run_question_2(output_root / "q2", generate_plots=generate_plots)
    q3 = run_question_3(output_root / "q3", generate_plots=generate_plots)
    q4 = run_question_4(output_root / "q4", generate_plots=generate_plots)
    q3_material = q3["material_share"]
    summary = pd.DataFrame(
        [
            {"问题": "问题1", "核心指标": "重要供应商数量", "数值": 50, "单位": "家"},
            {"问题": "问题1", "核心指标": "最高综合得分", "数值": float(ranking["综合得分"].max()), "单位": "无量纲"},
            {"问题": "问题2", "核心指标": "最少可行供应商数", "数值": len(q2["selected_ids"]), "单位": "家"},
            {"问题": "问题2", "核心指标": "实际最低库存", "数值": float(q2["actual_inventory"].min()), "单位": "m³产品当量"},
            {"问题": "问题3", "核心指标": "实际A类平均占比", "数值": float(q3_material["A类占比"].mean()), "单位": "比例"},
            {"问题": "问题3", "核心指标": "实际C类平均占比", "数值": float(q3_material["C类占比"].mean()), "单位": "比例"},
            {"问题": "问题3", "核心指标": "实际最低库存", "数值": float(q3["actual_inventory"].min()), "单位": "m³产品当量"},
            {"问题": "问题4", "核心指标": "最大可持续周产能", "数值": float(q4["max_capacity"]), "单位": "m³产品/周"},
            {"问题": "问题4", "核心指标": "相对当前产能提升", "数值": float(q4["max_capacity"] - PRODUCT_CAPACITY), "单位": "m³产品/周"},
            {"问题": "问题4", "核心指标": "提升比例", "数值": float(q4["max_capacity"] / PRODUCT_CAPACITY - 1.0), "单位": "比例"},
        ]
    )
    summary_path = output_root / "overall_summary.xlsx"
    summary.to_excel(summary_path, index=False)
    return {"ranking": ranking, "weights": weights, "q2": q2, "q3": q3, "q4": q4, "summary": summary, "summary_path": summary_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键运行2021年C题四个问题")
    parser.add_argument("--output-root", type=Path, default=None, help="统一结果输出根目录")
    parser.add_argument("--no-plot", action="store_true", help="不生成图表")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_all(args.output_root, generate_plots=not args.no_plot)
    print("问题1至问题4均已完成。")
    print(f"汇总结果：{result['summary_path']}")
