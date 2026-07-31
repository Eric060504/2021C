"""原材料体积、产品当量和相对采购成本的换算函数。"""
from __future__ import annotations

from typing import Mapping

from config import MATERIAL_CONSUMPTION, RAW_PRICE


# 根据材料消耗系数，把到厂原料体积换算为可生产的产品当量。
def material_to_product_equivalent(volume: float, material: str) -> float:
    """将到厂原材料体积换算为可生产的产品当量。"""
    return float(volume) / MATERIAL_CONSUMPTION[material]


# 根据材料消耗系数，将产品需求反推为所需原料体积。
def product_to_material_volume(product_volume: float, material: str) -> float:
    """将产品需求量换算为所需原材料体积。"""
    return float(product_volume) * MATERIAL_CONSUMPTION[material]


# 按题设给定的相对单价汇总各类原料的标准化采购成本。
def raw_purchase_cost(material_volumes: Mapping[str, float]) -> float:
    """按题设 A:B:C 相对价格计算标准化采购成本。"""
    return sum(float(material_volumes.get(material, 0.0)) * RAW_PRICE[material] for material in RAW_PRICE)
