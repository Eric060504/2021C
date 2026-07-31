"""数学建模结果的统一绘图工具。"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import EPSILON


# 统一使用中文字体和非交互式绘图后端，保证命令行环境也能稳定生成论文图片。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 240


def _save(fig: plt.Figure, path: Path) -> Path:
    """保存图片并释放图形资源。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_entropy_weights(weights: pd.Series, path: Path) -> Path:
    """绘制问题 1 的指标熵权柱状图。"""
    values = weights.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(values.index, values.values, color="#4C78A8")
    ax.set_title("问题1：供应商评价指标熵权")
    ax.set_xlabel("熵权")
    ax.set_xlim(0, max(float(values.max()) * 1.18, 0.05))
    for bar, value in zip(bars, values.values):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    return _save(fig, path)


def plot_top_supplier_scores(ranking: pd.DataFrame, path: Path, top_n: int = 20) -> Path:
    """绘制前若干家供应商综合得分排序图。"""
    data = ranking.head(top_n).copy().sort_values("综合得分", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.32)))
    labels = data["供应商ID"].astype(str) if "供应商ID" in data.columns else data.index.astype(str)
    bars = ax.barh(labels, data["综合得分"], color="#59A14F")
    ax.set_title(f"问题1：前 {top_n} 家供应商综合得分")
    ax.set_xlabel("综合得分")
    for bar, value in zip(bars, data["综合得分"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    return _save(fig, path)


def plot_supplier_indicator_heatmap(
    ranking: pd.DataFrame,
    path: Path,
    top_n: int = 50,
    negative_columns: tuple[str, ...] = ("供货率标准差",),
) -> Path:
    """绘制前若干家供应商的标准化指标热力图。"""
    columns = [column for column in ranking.columns if column not in {"供应商ID", "材料分类", "综合得分", "排名"}]
    data = ranking.head(top_n).loc[:, columns].astype(float).copy()
    for column in columns:
        lower, upper = data[column].min(), data[column].max()
        if upper - lower <= EPSILON:
            data[column] = 1.0
        elif column in negative_columns:
            data[column] = (upper - data[column]) / (upper - lower)
        else:
            data[column] = (data[column] - lower) / (upper - lower)
    fig, ax = plt.subplots(figsize=(10, max(7, top_n * 0.18)))
    image = ax.imshow(data.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    labels = ranking.head(top_n)["供应商ID"].astype(str).tolist() if "供应商ID" in ranking.columns else list(map(str, data.index))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=25, ha="right")
    ax.set_title(f"问题1：前 {top_n} 家供应商指标标准化热力图")
    fig.colorbar(image, ax=ax, label="标准化指标值")
    return _save(fig, path)


def plot_inventory_trace(
    actual_inventory: pd.Series,
    safety_inventory: float,
    demand: float,
    path: Path,
    title: str,
    actual_receipts: pd.Series | None = None,
) -> Path:
    """绘制库存轨迹、安全库存线和可选的周到厂量。"""
    weeks = actual_inventory.index.astype(int)
    if actual_receipts is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
        ax_receipt, ax = axes
        receipt = actual_receipts.reindex(actual_inventory.index).astype(float)
        ax_receipt.bar(weeks, receipt.values, color="#76B7B2", label="实际到厂产品当量")
        ax_receipt.axhline(demand, color="#E15759", linestyle="--", linewidth=1.5, label="周需求量")
        ax_receipt.set_ylabel("产品当量（m³）")
        ax_receipt.set_title(title)
        ax_receipt.legend(loc="best")
        ax_receipt.grid(axis="y", alpha=0.25)
    ax.plot(weeks, actual_inventory.values, marker="o", color="#4C78A8", linewidth=2, label="实际库存")
    ax.axhline(safety_inventory, color="#E15759", linestyle="--", linewidth=1.5, label="两周安全库存")
    min_week = int(actual_inventory.idxmin())
    min_value = float(actual_inventory.min())
    ax.scatter([min_week], [min_value], color="#F28E2B", zorder=3)
    ax.annotate(f"最低库存\n{min_value:.1f}", (min_week, min_value), xytext=(8, 10), textcoords="offset points", fontsize=8)
    ax.set_xlabel("周次")
    ax.set_ylabel("库存产品当量（m³）")
    if actual_receipts is None:
        ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    return _save(fig, path)


def plot_carrier_utilization(utilization: pd.DataFrame, path: Path, title: str) -> Path:
    """绘制转运商周利用率热力图。"""
    pivot = utilization.pivot(index="转运商ID", columns="周", values="利用率").sort_index()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=max(1.0, float(np.nanmax(pivot.to_numpy()))))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel("周次")
    ax.set_ylabel("转运商")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="运力利用率")
    return _save(fig, path)


def plot_material_structure(material_share: pd.DataFrame, path: Path, title: str, a_min_share: float | None = None, c_max_share: float | None = None) -> Path:
    """绘制 A/B/C 三类材料的周度产品当量占比堆叠图。"""
    weeks = material_share["周"].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(material_share))
    colors = {"A": "#4C78A8", "B": "#F28E2B", "C": "#59A14F"}
    for material in ("A", "B", "C"):
        values = material_share[f"{material}类占比"].to_numpy()
        ax.bar(weeks, values, bottom=bottom, color=colors[material], label=f"{material}类占比")
        bottom += values
    if a_min_share is not None:
        ax.axhline(a_min_share, color="#4C78A8", linestyle="--", linewidth=1.4, label=f"A类下限 {a_min_share:.0%}")
    if c_max_share is not None:
        ax.axhline(1.0 - c_max_share, color="#59A14F", linestyle=":", linewidth=1.4, label=f"C类上限 {c_max_share:.0%}")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("周次")
    ax.set_ylabel("产品当量占比")
    ax.set_title(title)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def plot_material_comparison(q2_material_share: pd.DataFrame, q3_material_share: pd.DataFrame, path: Path) -> Path:
    """绘制问题 2 与问题 3 的平均材料结构对比图。"""
    materials = ("A", "B", "C")
    q2_values = [float(q2_material_share[f"{material}类占比"].mean()) for material in materials]
    q3_values = [float(q3_material_share[f"{material}类占比"].mean()) for material in materials]
    x = np.arange(len(materials))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, q2_values, width, label="问题2基础方案", color="#9C755F")
    bars2 = ax.bar(x + width / 2, q3_values, width, label="问题3成本压缩方案", color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{material}类" for material in materials])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("平均产品当量占比")
    ax.set_title("问题2与问题3的材料结构对比")
    ax.legend()
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.1%}", ha="center", fontsize=8)
    return _save(fig, path)


def plot_carrier_loss_threshold(loss_rates: pd.DataFrame, path: Path, threshold: float, allocation: Mapping[int, pd.DataFrame] | None = None) -> Path:
    """绘制转运商历史预测损耗率与低损耗阈值。"""
    mean_rates = loss_rates.mean(axis=1).sort_values()
    used: set[str] = set()
    if allocation is not None:
        for matrix in allocation.values():
            used.update(str(carrier) for carrier in matrix.columns[matrix.sum(axis=0) > EPSILON])
    colors = ["#4C78A8" if str(carrier) in used else ("#B7B7B7" if value <= threshold else "#E15759") for carrier, value in mean_rates.items()]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(mean_rates.index.astype(str), mean_rates.values, color=colors)
    ax.axhline(threshold, color="#E15759", linestyle="--", label=f"低损耗率阈值 {threshold:.0%}")
    ax.set_title("问题3：转运商预测损耗率与低损耗阈值")
    ax.set_ylabel("预测损耗率")
    ax.set_ylim(0.0, max(float(mean_rates.max()) * 1.25, threshold * 1.3))
    ax.legend()
    for bar, value in zip(bars, mean_rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0003, f"{value:.2%}", ha="center", fontsize=8)
    return _save(fig, path)


def plot_sensitivity_heatmaps(sensitivity: pd.DataFrame, output_dir: Path) -> list[Path]:
    """针对材料比例试验，按低损耗权重绘制成本和可行性热力图。"""
    required = {"A类最低占比", "C类最高占比", "低损耗权重", "可行", "目标成本"}
    missing = required - set(sensitivity.columns)
    if missing:
        raise ValueError(f"敏感性结果缺少列: {sorted(missing)}")
    output_dir = Path(output_dir)
    data = sensitivity.copy()
    if "试验类型" in data.columns:
        data = data.loc[data["试验类型"] == "材料比例敏感性"].copy()
    if data.empty:
        return []

    all_costs = pd.to_numeric(data["目标成本"], errors="coerce").dropna() / 10_000.0
    cost_min = float(all_costs.min()) if not all_costs.empty else 0.0
    cost_max = float(all_costs.max()) if not all_costs.empty else 1.0
    if cost_max - cost_min <= EPSILON:
        cost_min -= 0.01
        cost_max += 0.01

    paths: list[Path] = []
    for loss_weight, group in data.groupby("低损耗权重", sort=True):
        cost = group.pivot(index="C类最高占比", columns="A类最低占比", values="目标成本") / 10_000.0
        feasible = group.pivot(index="C类最高占比", columns="A类最低占比", values="可行").astype(float)
        for label, matrix, cmap, color_label in (
            ("成本", cost, "YlOrRd", "标准化目标成本（万元）"),
            ("可行性", feasible, "YlGn", "可行（1）/不可行（0）"),
        ):
            fig, ax = plt.subplots(figsize=(8.4, 5.8))
            if label == "成本":
                image = ax.imshow(matrix.to_numpy(), origin="lower", aspect="auto", cmap=cmap, vmin=cost_min, vmax=cost_max)
            else:
                image = ax.imshow(matrix.to_numpy(), origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
            ax.set_xticks(np.arange(len(matrix.columns)))
            ax.set_xticklabels([f"{value:.1%}" for value in matrix.columns])
            ax.set_yticks(np.arange(len(matrix.index)))
            ax.set_yticklabels([f"{value:.2%}" for value in matrix.index])
            ax.set_xlabel("A类最低占比")
            ax.set_ylabel("C类最高占比")
            ax.set_title(f"问题3：{label}敏感性（低损耗权重={loss_weight:.2f}，阈值固定）")
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    value = matrix.iloc[row, column]
                    text = "不可行" if pd.isna(value) else (f"{value:.4f}" if label == "成本" else ("可行" if value >= 0.5 else "不可行"))
                    ax.text(column, row, text, ha="center", va="center", fontsize=7)
            fig.colorbar(image, ax=ax, label=color_label)
            filename = f"sensitivity_{label}_loss_weight_{loss_weight:.2f}.png"
            paths.append(_save(fig, output_dir / filename))
    return paths


def plot_loss_threshold_sensitivity(sensitivity: pd.DataFrame, path: Path) -> Path | None:
    """绘制固定材料偏好下低损耗率阈值对可行性和合格运力的影响。"""
    if "试验类型" not in sensitivity.columns:
        return None
    data = sensitivity.loc[sensitivity["试验类型"] == "低损耗率阈值敏感性"].copy()
    if data.empty:
        return None
    data["低损耗率阈值"] = pd.to_numeric(data["低损耗率阈值"], errors="coerce")
    data["目标成本"] = pd.to_numeric(data["目标成本"], errors="coerce")
    data["最少合格转运商数"] = pd.to_numeric(data.get("最少合格转运商数"), errors="coerce")
    data["可行"] = data["可行"].astype(bool)

    fig, (ax_cost, ax_carrier) = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    colors = plt.cm.Set2(np.linspace(0.0, 1.0, data["低损耗权重"].nunique()))
    for color, (loss_weight, group) in zip(colors, data.groupby("低损耗权重", sort=True)):
        group = group.sort_values("低损耗率阈值")
        feasible = group["可行"] & group["目标成本"].notna()
        ax_cost.plot(
            group.loc[feasible, "低损耗率阈值"] * 100.0,
            group.loc[feasible, "目标成本"] / 10_000.0,
            marker="o",
            color=color,
            label=f"权重={loss_weight:.2f}",
        )
        ax_carrier.plot(
            group["低损耗率阈值"] * 100.0,
            group["最少合格转运商数"],
            marker="o",
            color=color,
            label=f"权重={loss_weight:.2f}",
        )
        if (~feasible).any():
            ax_carrier.scatter(
                group.loc[~feasible, "低损耗率阈值"] * 100.0,
                np.zeros((~feasible).sum()),
                marker="x",
                color="#C00000",
                s=60,
                zorder=3,
            )
    ax_cost.set_title("问题3：阈值对成本的影响")
    ax_cost.set_xlabel("低损耗率阈值（%）")
    ax_cost.set_ylabel("标准化目标成本（万元）")
    ax_cost.grid(alpha=0.25)
    ax_cost.legend(title="低损耗权重")
    ax_carrier.set_title("问题3：阈值对合格运力的影响")
    ax_carrier.set_xlabel("低损耗率阈值（%）")
    ax_carrier.set_ylabel("每周最少合格转运商数")
    ax_carrier.grid(alpha=0.25)
    ax_carrier.legend(title="低损耗权重")
    ax_carrier.text(0.02, 0.03, "红色叉号：该阈值下转运方案不可行", transform=ax_carrier.transAxes, color="#C00000", fontsize=8)
    return _save(fig, path)

def plot_capacity_comparison(current_capacity: float, maximum_capacity: float, path: Path) -> Path:
    """绘制当前产能与最大可持续产能对比图。"""
    labels = ["当前周产能", "最大可持续周产能"]
    values = [current_capacity, maximum_capacity]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#9C755F", "#59A14F"], width=0.55)
    ax.set_ylabel("产品产能（m³/周）")
    ax.set_title("问题4：扩产前后周产能对比")
    ax.set_ylim(0.0, max(values) * 1.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.02, f"{value:,.2f}", ha="center")
    increase = maximum_capacity / current_capacity - 1.0
    ax.text(0.5, max(values) * 1.13, f"提升 {maximum_capacity - current_capacity:,.2f} m³/周（{increase:.2%}）", ha="center", color="#2F5597")
    return _save(fig, path)


# 汇总逐周预测供给、订购发运与转运能力，直观展示跨周库存调节过程。
def plot_weekly_supply_and_shipment(
    supplier_capacity_by_week: pd.DataFrame,
    shipments: pd.DataFrame,
    carrier_capacity_by_week: pd.Series,
    path: Path,
    title: str,
) -> Path:
    """绘制预测可供货量、计划发运量和逐周转运能力。"""
    weeks = pd.Index(shipments.columns, dtype=int)
    available = supplier_capacity_by_week.reindex(columns=weeks, fill_value=0.0).sum(axis=0)
    planned = shipments.reindex(columns=weeks, fill_value=0.0).sum(axis=0)
    carrier = carrier_capacity_by_week.reindex(weeks).astype(float)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(weeks, available.values, marker="o", linewidth=2, color="#4C78A8", label="预测可供货量")
    ax.plot(weeks, planned.values, marker="s", linewidth=2, color="#F28E2B", label="计划发运量")
    ax.plot(weeks, carrier.values, linestyle="--", linewidth=2, color="#59A14F", label="总转运能力")
    ax.fill_between(weeks, planned.values, available.values, where=available.values >= planned.values, color="#4C78A8", alpha=0.12, label="未使用供给余量")
    ax.set_title(title)
    ax.set_xlabel("周次")
    ax.set_ylabel("原料体积（m³）")
    ax.set_xticks(weeks)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", ncol=2)
    return _save(fig, path)


# 保留问题4专用入口，兼容既有调用代码。
def plot_q4_weekly_supply_plan(
    supplier_capacity_by_week: pd.DataFrame,
    shipments: pd.DataFrame,
    carrier_capacity_by_week: pd.Series,
    path: Path,
) -> Path:
    """绘制问题4的预测可供货量、计划发运量和逐周转运能力。"""
    return plot_weekly_supply_and_shipment(
        supplier_capacity_by_week,
        shipments,
        carrier_capacity_by_week,
        path,
        "问题4：未来24周供给预测与订购发运动态方案",
    )

def plot_selected_supplier_capacity(selected_forecasts: pd.DataFrame, path: Path) -> Path:
    """绘制问题2入选供应商的预测产品当量能力。"""
    data = selected_forecasts.copy()
    consumption = {"A": 0.60, "B": 0.66, "C": 0.72}
    data["产品当量能力"] = data["capacity"].astype(float) / data["材料分类"].map(consumption).astype(float)
    data = data.sort_values("产品当量能力", ascending=True)
    colors = {"A": "#4C78A8", "B": "#F28E2B", "C": "#59A14F"}
    fig, ax = plt.subplots(figsize=(9, max(5, len(data) * 0.42)))
    bars = ax.barh(data["供应商ID"].astype(str), data["产品当量能力"], color=[colors.get(value, "#9C755F") for value in data["材料分类"]])
    ax.set_xlabel("预测产品当量能力（m³/周）")
    ax.set_title("问题2：入选供应商的预测供给能力")
    for bar, value in zip(bars, data["产品当量能力"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"{value:.0f}", va="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors.values()]
    ax.legend(handles, [f"{material}类" for material in colors], title="材料类别", loc="lower right")
    return _save(fig, path)
