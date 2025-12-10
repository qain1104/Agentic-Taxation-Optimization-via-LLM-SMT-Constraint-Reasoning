# -*- coding: utf-8 -*-
"""
estate_tax.py ── 遺產稅最適化模組

變更重點：
- 移除本地的 _apply_binary_constraints / RHS 正規化邏輯
- 統一改用 constraint_utils.apply_linear_constraints 解析 constraints
- 其他稅額公式與 Public API 結構維持不變
"""

from z3 import Optimize, Real, Int, If, And, ToInt, sat
from typing import Any, Dict, List, Optional, Tuple
import time

from .constraint_utils import apply_linear_constraints  # ★ 新增：共用線性約束工具

# ─── 共用錯誤型別 ────────────────────────────────────────────────
class UnsatError(Exception):
    """Constraint set is UNSAT but we want to report gracefully."""
    pass


def _norm_period(x) -> int:
    s = str(x).strip()
    # 全形→半形
    s = s.translate(str.maketrans("０１２３４５", "012345"))
    cn = {"一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5}
    n = None
    if s.isdigit():
        n = int(s)
    elif s in cn:
        n = cn[s]
    # 像 "113/05/01" 或 "113年" 這種不要硬轉，交給上游處理；這邊只接受 1~5
    if n not in (1, 2, 3, 4, 5):
        raise ValueError(f"death_period must be 1..5 (got {x!r})")
    return n


# ───────────────────────────────────────────────────────────────
# 1. 核心純函式 _calculate_estate_tax_internal
#    回傳 (final_tax, net_taxable, final_params, diff)
# ───────────────────────────────────────────────────────────────
def _calculate_estate_tax_internal(
    *,
    death_period: int = 1,
    is_military_police: bool = False,
    land_value: float = 0.0,
    building_value: float = 0.0,
    house_value: float = 0.0,
    deposit_bonds_value: float = 0.0,
    stock_invest_value: float = 0.0,
    cash_gold_jewelry_value: float = 0.0,
    gift_in_2yrs_value: float = 0.0,
    spouse_count: float = 0,
    lineal_descendant_count: float = 0,
    father_mother_count: float = 0,
    disability_count: float = 0,
    dependent_count: float = 0,
    farmland_val: float = 0.0,
    inheritance_6to9_val: float = 0.0,
    unpaid_tax_fines_val: float = 0.0,
    unpaid_debts_val: float = 0.0,
    will_management_fee: float = 0.0,
    public_facility_retention_val: float = 0.0,
    spouse_surplus_right_val: float = 0.0,
    gift_tax_offset: float = 0.0,
    foreign_tax_offset: float = 0.0,
    free_vars: Optional[List[str]] = None,
    constraints: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[int, int, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Pure Z3 函式：遺產稅最小化
    Inputs:
      free_vars     可優化的欄位清單
      constraints   全部使用線性比較式條件（由 constraint_utils.apply_linear_constraints 處理）
    Returns:
      final_tax     最終稅額 (Int)
      net_taxable   淨課稅遺產淨額 (Int)
      final_params  各欄位最終值與 fixed/free 標記
      diff          有變動欄位的 original/optimized/difference
    """
    free_vars = set(free_vars or [])
    constraints = constraints or {}
    death_period = _norm_period(death_period)

    # 1. 計算免稅額與喪葬費
    if death_period in (1, 2):
        base_ex_amt, funeral_amt = 12_000_000, 1_230_000
    else:
        base_ex_amt = 13_330_000
        funeral_amt = 1_230_000 if death_period in (1, 2, 3) else 1_380_000
    if is_military_police:
        base_ex_amt *= 2

    opt = Optimize()

    # 2. 宣告所有 Z3 變數
    def R(n: str):
        return Real(n)

    # 每個欄位：(z3var, 原始值, [內建檢查])
    params: Dict[str, Tuple[Any, float, List[Any]]] = {
        "land_value": (R("land_value"), land_value, [lambda v: v >= 0]),
        "building_value": (R("building_value"), building_value, [lambda v: v >= 0]),
        "house_value": (R("house_value"), house_value, [lambda v: v >= 0]),
        "deposit_bonds_value": (R("deposit_bonds_value"), deposit_bonds_value, [lambda v: v >= 0]),
        "stock_invest_value": (R("stock_invest_value"), stock_invest_value, [lambda v: v >= 0]),
        "cash_gold_jewelry_value": (R("cash_gold_jewelry_value"), cash_gold_jewelry_value, [lambda v: v >= 0]),
        "gift_in_2yrs_value": (R("gift_in_2yrs_value"), gift_in_2yrs_value, [lambda v: v >= 0]),
        "spouse_count": (R("spouse_count"), spouse_count, [lambda v: v >= 0]),
        "lineal_descendant_count": (
            R("lineal_descendant_count"),
            lineal_descendant_count,
            [lambda v: v >= 0],
        ),
        "father_mother_count": (R("father_mother_count"), father_mother_count, [lambda v: v >= 0]),
        "disability_count": (R("disability_count"), disability_count, [lambda v: v >= 0]),
        "dependent_count": (R("dependent_count"), dependent_count, [lambda v: v >= 0]),
        "farmland_val": (R("farmland_val"), farmland_val, [lambda v: v >= 0]),
        "inheritance_6to9_val": (R("inheritance_6to9_val"), inheritance_6to9_val, [lambda v: v >= 0]),
        "unpaid_tax_fines_val": (R("unpaid_tax_fines_val"), unpaid_tax_fines_val, [lambda v: v >= 0]),
        "unpaid_debts_val": (R("unpaid_debts_val"), unpaid_debts_val, [lambda v: v >= 0]),
        "will_management_fee": (R("will_management_fee"), will_management_fee, [lambda v: v >= 0]),
        "public_facility_retention_val": (
            R("public_facility_retention_val"),
            public_facility_retention_val,
            [lambda v: v >= 0],
        ),
        "spouse_surplus_right_val": (R("spouse_surplus_right_val"), spouse_surplus_right_val, [lambda v: v >= 0]),
        "gift_tax_offset": (R("gift_tax_offset"), gift_tax_offset, [lambda v: v >= 0]),
        "foreign_tax_offset": (R("foreign_tax_offset"), foreign_tax_offset, [lambda v: v >= 0]),
    }

    # 3. 綁定固定或自由變數（僅加內建檢查）
    for name, (zv, orig, checks) in params.items():
        if name not in free_vars:
            opt.add(zv == orig)
        for c in checks:
            opt.add(c(zv))

    # 4. 套用所有比較式條件（委由 constraint_utils）
    #    constraint_utils.apply_linear_constraints 期待 mapping: name -> (z3_var,)
    params_for_constraints = {name: (zv,) for name, (zv, _, _) in params.items()}
    apply_linear_constraints(opt, params_for_constraints, constraints, debug=False)

    # 5. 建立稅額模型
    # 資產總額
    asset_sum = Real("asset_sum")
    opt.add(
        asset_sum
        == sum(
            params[n][0]
            for n in [
                "land_value",
                "building_value",
                "house_value",
                "deposit_bonds_value",
                "stock_invest_value",
                "cash_gold_jewelry_value",
                "gift_in_2yrs_value",
            ]
        )
    )

    # 免稅額與喪葬費
    base_ex_z = Real("base_ex_z")
    opt.add(base_ex_z == base_ex_amt)
    funeral_z = Real("funeral_z")
    opt.add(funeral_z == funeral_amt)

    # 各項扣除
    ded_spouse = Real("ded_spouse")
    opt.add(
        ded_spouse
        == params["spouse_count"][0] * (5_530_000 if death_period > 3 else 4_930_000)
    )

    ded_lineal = Real("ded_lineal")
    opt.add(
        ded_lineal
        == params["lineal_descendant_count"][0]
        * (560_000 if death_period > 3 else 500_000)
    )

    ded_fm = Real("ded_fm")
    opt.add(
        ded_fm
        == params["father_mother_count"][0]
        * (1_380_000 if death_period > 3 else 1_230_000)
    )

    ded_dis = Real("ded_dis")
    opt.add(
        ded_dis
        == params["disability_count"][0]
        * (6_930_000 if death_period > 3 else 6_180_000)
    )

    ded_dep = Real("ded_dep")
    opt.add(
        ded_dep
        == params["dependent_count"][0] * (560_000 if death_period > 3 else 500_000)
    )

    ded_other = Real("ded_other")
    opt.add(
        ded_other
        == params["farmland_val"][0]
        + params["inheritance_6to9_val"][0]
        + params["unpaid_tax_fines_val"][0]
        + params["unpaid_debts_val"][0]
        + funeral_z
        + params["will_management_fee"][0]
        + params["public_facility_retention_val"][0]
        + params["spouse_surplus_right_val"][0]
    )

    total_ded = Real("total_ded")
    opt.add(
        total_ded
        == ded_spouse
        + ded_lineal
        + ded_fm
        + ded_dis
        + ded_dep
        + ded_other
    )

    tax_inherit = Real("tax_inherit")
    opt.add(tax_inherit == asset_sum - base_ex_z - total_ded)

    # 級距與差額
    bracket_r = Real("bracket_r")
    bracket_d = Real("bracket_d")
    opt.add(
        If(
            death_period == 1,
            And(bracket_r == 0.10, bracket_d == 0),
            If(
                death_period > 4,
                If(
                    tax_inherit <= 56_210_000,
                    And(bracket_r == 0.10, bracket_d == 0),
                    If(
                        tax_inherit <= 112_420_000,
                        And(bracket_r == 0.15, bracket_d == 2_810_500),
                        And(bracket_r == 0.20, bracket_d == 8_431_500),
                    ),
                ),
                If(
                    tax_inherit <= 50_000_000,
                    And(bracket_r == 0.10, bracket_d == 0),
                    If(
                        tax_inherit <= 100_000_000,
                        And(bracket_r == 0.15, bracket_d == 2_500_000),
                        And(bracket_r == 0.20, bracket_d == 7_500_000),
                    ),
                ),
            ),
        )
    )

    offsets = Real("offsets")
    opt.add(
        offsets
        == params["gift_tax_offset"][0] + params["foreign_tax_offset"][0]
    )

    est_tax = Real("est_tax")
    opt.add(
        est_tax
        == If(
            tax_inherit <= 0,
            0,
            If(
                tax_inherit * bracket_r - bracket_d - offsets < 0,
                0,
                tax_inherit * bracket_r - bracket_d - offsets,
            ),
        )
    )

    # 取整輸出 + 兩段式最小化
    final_tax_int = Int("final_tax_int")
    opt.add(final_tax_int == ToInt(est_tax))

    est_tax_scaled = Int("est_tax_scaled")
    opt.add(est_tax_scaled == ToInt(est_tax * 1_000))  # 0.001 元精度

    net_tax_int = Int("net_tax_int")
    opt.add(net_tax_int == ToInt(tax_inherit))

    # 兩段式目標：先整數稅額，再比較小數千分位
    opt.minimize(final_tax_int)
    opt.minimize(est_tax_scaled)

    if opt.check() != sat:
        raise UnsatError("constraint set unsat")
    m = opt.model()

    # 6. 收集結果
    ft = m[final_tax_int].as_long()
    nt = m[net_tax_int].as_long()
    final_params: Dict[str, Dict[str, Any]] = {}
    diff: Dict[str, Dict[str, Any]] = {}

    for name, (zv, orig, _) in params.items():
        model_val = m.evaluate(zv)
        if model_val.is_int_value():
            val = model_val.as_long()
        else:
            s = model_val.as_decimal(20).rstrip("?")
            val = float(s)
        final_params[name] = {
            "value": val,
            "type": "free" if name in free_vars else "fixed",
        }
        if val != orig:
            diff[name] = {
                "original": orig,
                "optimized": val,
                "difference": val - orig,
            }

    return ft, nt, final_params, diff


# ───────────────────────────────────────────────────────────────
# 2. Public API: calculate_estate_tax
# ───────────────────────────────────────────────────────────────
def calculate_estate_tax(
    *,
    death_period: int = 1,
    is_military_police: bool = False,
    land_value: float = 0.0,
    building_value: float = 0.0,
    house_value: float = 0.0,
    deposit_bonds_value: float = 0.0,
    stock_invest_value: float = 0.0,
    cash_gold_jewelry_value: float = 0.0,
    gift_in_2yrs_value: float = 0.0,
    spouse_count: float = 0,
    lineal_descendant_count: float = 0,
    father_mother_count: float = 0,
    disability_count: float = 0,
    dependent_count: float = 0,
    farmland_val: float = 0.0,
    inheritance_6to9_val: float = 0.0,
    unpaid_tax_fines_val: float = 0.0,
    unpaid_debts_val: float = 0.0,
    will_management_fee: float = 0.0,
    public_facility_retention_val: float = 0.0,
    spouse_surplus_right_val: float = 0.0,
    gift_tax_offset: float = 0.0,
    foreign_tax_offset: float = 0.0,
    free_vars: Optional[List[str]] = None,
    constraints: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Public API for estate tax.
    回傳結構：
    {
        mode,
        input_params,
        baseline,
        baseline_status,
        baseline_with_constraints,
        optimized,
        status,
        diff,
        final_params,
        constraints
    }
    """
    kwargs = {
        **{k: v for k, v in locals().items() if k not in ("free_vars", "constraints")},
        "free_vars": free_vars or [],
    }
    cons = constraints or {}
    started_time = time.perf_counter()

    # 1) baseline（純公式，不帶任何 constraints）
    base_tax, _, _, _ = _calculate_estate_tax_internal(
        **{k: v for k, v in kwargs.items() if k != "free_vars"},
        constraints={},
    )

    # 2) baseline_status：固定值 + constraints 可行性檢查
    try:
        _calculate_estate_tax_internal(
            **{k: v for k, v in kwargs.items() if k != "free_vars"},
            constraints=cons,
        )
        baseline_status = "sat"
        baseline_with_constraints = base_tax
    except UnsatError:
        baseline_status = "unsat"
        baseline_with_constraints = None

    # 3) manual_free 分支
    mode, optimized, status = "baseline", base_tax, baseline_status
    params_out: Dict[str, Any] = {}
    diff_out: Dict[str, Any] = {}
    if kwargs["free_vars"]:
        mode = "manual_free"
        try:
            optimized, _, params_out, diff_out = _calculate_estate_tax_internal(
                **kwargs,
                constraints=cons,
            )
            status = "sat"
        except UnsatError:
            optimized = None
            status = "unsat"

    print(
        f"Calculation time estate: {time.perf_counter() - started_time:.3f} seconds"
    )
    return {
        "mode": mode,
        "input_params": kwargs,
        "baseline": base_tax,
        "baseline_status": baseline_status,
        "baseline_with_constraints": baseline_with_constraints,
        "optimized": optimized,
        "status": status,
        "diff": diff_out,
        "final_params": params_out,
        "constraints": cons,
    }