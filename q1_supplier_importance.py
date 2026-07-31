"""问题 1：基于熵权法的供应商重要性评价。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_ROOT
from data_io import MATERIAL, SUPPLIER_ID, read_input_data, week_columns
from visualization import plot_entropy_weights, plot_supplier_indicator_heatmap, plot_top_supplier_scores

INDICATOR_COLUMNS = [
    "供货率标准差",
    "90%供货可靠性",
    "平均供货能力",
    "供货充足率",
    "最大供货潜力",
    "合作持续性",
]


# 从订货与实际供货历史提取可靠性、能力和合作持续性等六项评价指标。
def compute_supplier_indicators(order: pd.DataFrame, supply: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    columns = [column for column in week_columns() if column in order.columns and column in supply.columns]
    for row_index, supplier_id in enumerate(order[SUPPLIER_ID]):
        order_values = order.loc[row_index, columns].astype(float)
        supply_values = supply.loc[row_index, columns].astype(float)
        ordered = order_values > 1e-8
        delivery_rates = supply_values.loc[ordered] / order_values.loc[ordered] if ordered.any() else pd.Series(dtype=float)
        records.append(
            {
                SUPPLIER_ID: str(supplier_id),
                MATERIAL: str(order.loc[row_index, MATERIAL]),
                "供货率标准差": float(delivery_rates.std(ddof=0)) if not delivery_rates.empty else 1.0,
                "90%供货可靠性": float((delivery_rates >= 0.9).mean()) if not delivery_rates.empty else 0.0,
                "平均供货能力": float(supply_values.mean()),
                "供货充足率": float(supply_values.sum() / order_values.sum()) if order_values.sum() > 1e-8 else 0.0,
                "最大供货潜力": float(supply_values.max()),
                "合作持续性": float((supply_values > 1e-8).mean()),
            }
        )
    return pd.DataFrame(records).set_index(SUPPLIER_ID, drop=False)


# 对指标进行极差标准化，计算熵权并返回综合得分排序。
def entropy_weight_score(
    indicators: pd.DataFrame,
    negative_columns: list[str] | tuple[str, ...] = ("供货率标准差",),
) -> tuple[pd.DataFrame, pd.Series]:
    values = indicators.loc[:, INDICATOR_COLUMNS].astype(float).copy()
    standardized = pd.DataFrame(index=values.index, columns=values.columns, dtype=float)
    for column in values.columns:
        lower, upper = float(values[column].min()), float(values[column].max())
        if np.isclose(upper, lower):
            standardized[column] = 1.0
        elif column in negative_columns:
            standardized[column] = (upper - values[column]) / (upper - lower)
        else:
            standardized[column] = (values[column] - lower) / (upper - lower)
    # 正向平移避免熵值计算出现 log(0)，不改变各列相对大小。
    positive = standardized + 1e-12
    proportions = positive.div(positive.sum(axis=0), axis=1)
    n = len(indicators)
    if n <= 1:
        entropy = pd.Series(0.0, index=values.columns)
    else:
        entropy = -(proportions * np.log(proportions)).sum(axis=0) / np.log(n)
    diversification = (1.0 - entropy).clip(lower=0.0)
    weights = diversification / diversification.sum() if diversification.sum() > 1e-12 else pd.Series(1 / len(values.columns), index=values.columns)
    ranked = indicators.copy()
    ranked["综合得分"] = standardized.mul(weights, axis=1).sum(axis=1)
    ranked["排名"] = ranked["综合得分"].rank(method="first", ascending=False).astype(int)
    if SUPPLIER_ID in ranked.columns:
        ranked = ranked.assign(_supplier_sort=ranked[SUPPLIER_ID].astype(str)).sort_values(
            ["综合得分", "_supplier_sort"], ascending=[False, True]
        ).drop(columns="_supplier_sort")
    else:
        ranked = ranked.sort_values("综合得分", ascending=False)
    return ranked, weights


# 运行问题 1，并导出完整排名、前 50 家重要供应商、指标权重和论文图表。
def run_question_1(output_dir: Path | None = None, generate_plots: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    output_dir = Path(output_dir or OUTPUT_ROOT / "q1")
    output_dir.mkdir(parents=True, exist_ok=True)
    order, supply, _ = read_input_data()
    indicators = compute_supplier_indicators(order, supply)
    ranking, weights = entropy_weight_score(indicators)
    ranking.to_excel(output_dir / "supplier_ranking.xlsx", index=False)
    ranking.head(50).to_excel(output_dir / "top_50_suppliers.xlsx", index=False)
    pd.DataFrame({"指标": weights.index, "熵权": weights.values}).to_excel(output_dir / "entropy_weights.xlsx", index=False)
    summary = pd.DataFrame(
        {
            "项目": ["供应商总数", "重要供应商数", "最高综合得分", "最低综合得分"],
            "数值": [len(ranking), 50, float(ranking["综合得分"].max()), float(ranking["综合得分"].min())],
        }
    )
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    if generate_plots:
        figure_dir = output_dir.parent / "figures" / output_dir.name
        plot_entropy_weights(weights, figure_dir / "entropy_weights.png")
        plot_top_supplier_scores(ranking, figure_dir / "top20_supplier_scores.png")
        plot_supplier_indicator_heatmap(ranking, figure_dir / "top50_indicator_heatmap.png")
    return ranking, weights


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行问题1供应商重要性评价")
    parser.add_argument("--output-dir", type=Path, default=None, help="结果输出目录")
    parser.add_argument("--no-plot", action="store_true", help="不生成图表")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ranking, weights = run_question_1(args.output_dir, generate_plots=not args.no_plot)
    print("问题1完成：已生成供应商熵权评价、前50名结果和论文图表。")
    print("熵权：")
    print(weights.round(6).to_string())
    print("前10名：")
    print(ranking[[SUPPLIER_ID, MATERIAL, "综合得分", "排名"]].head(10).to_string(index=False))
