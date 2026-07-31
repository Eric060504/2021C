"""问题 1：基于熵权法的供应商重要性评价。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_ROOT
from data_io import MATERIAL, SUPPLIER_ID, read_input_data, week_columns

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
    """根据历史订货和供货数据计算供应商评价所需的六项指标。"""
    weeks = [column for column in week_columns() if column in order.columns]
    records: list[dict[str, object]] = []
    for idx, supplier_id in enumerate(order[SUPPLIER_ID]):
        order_values = pd.to_numeric(order.loc[idx, weeks], errors="coerce").fillna(0.0).to_numpy(float)
        supply_values = pd.to_numeric(supply.loc[idx, weeks], errors="coerce").fillna(0.0).to_numpy(float)
        # 供货率仅在企业确实下单的周次中定义。
        active_orders = order_values > 1e-8
        rates = supply_values[active_orders] / order_values[active_orders] if active_orders.any() else np.array([0.0])
        # 90% 供货可靠性：实际供货量不低于订货量 90% 的经验频率。
        reliability = float(np.mean(rates >= 0.90)) if rates.size else 0.0
        total_order = float(order_values.sum())
        total_supply = float(supply_values.sum())
        records.append(
            {
                SUPPLIER_ID: supplier_id,
                MATERIAL: order.loc[idx, MATERIAL],
                "供货率标准差": float(np.std(rates, ddof=0)),
                "90%供货可靠性": reliability,
                "平均供货能力": float(np.mean(supply_values)),
                "供货充足率": total_supply / total_order if total_order > 1e-8 else 0.0,
                "最大供货潜力": float(np.max(supply_values)),
                # 合作持续性：全部观测周中实际供货量为正的周次占比。
                "合作持续性": float(np.mean(supply_values > 1e-8)),
            }
        )
    return pd.DataFrame(records).set_index(SUPPLIER_ID, drop=False)


# 先区分正向/逆向指标进行标准化，再按指标离散程度计算熵权和综合得分。
def entropy_weight_score(
    indicators: pd.DataFrame,
    negative_columns: list[str] | tuple[str, ...] = ("供货率标准差",),
) -> tuple[pd.DataFrame, pd.Series]:
    """对指标进行极差标准化，计算熵权并返回综合得分排序。"""
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
    score_column = ranked.columns[-2]
    if SUPPLIER_ID in ranked.columns:
        ranked = ranked.assign(_supplier_sort=ranked[SUPPLIER_ID].astype(str)).sort_values(
            [score_column, "_supplier_sort"], ascending=[False, True]
        ).drop(columns="_supplier_sort")
    else:
        ranked = ranked.sort_values(score_column, ascending=False)
    return ranked, weights


# 运行问题 1，并导出完整排名、前 50 家重要供应商及各指标熵权。
def run_question_1(output_dir: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
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
    return ranking, weights


if __name__ == "__main__":
    ranking, weights = run_question_1()
    print("问题1完成：已生成供应商熵权评价与前50名结果。")
    print("熵权：")
    print(weights.round(6).to_string())
    print("前10名：")
    print(ranking[[SUPPLIER_ID, MATERIAL, "综合得分", "排名"]].head(10).to_string(index=False))


