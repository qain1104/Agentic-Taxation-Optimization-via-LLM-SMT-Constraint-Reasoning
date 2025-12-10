# -*- coding: utf-8 -*-
"""
business_tax_optimizer.py

Pure Z3-based optimiser for Taiwan corporate income tax (營利事業所得稅)。
* 無 auto-search；僅在 caller 傳入 free_vars 時進行最佳化。
* 支援 variable_constraints：交由 constraint_utils.apply_linear_constraints 處理
  （單變數比較、線性表達式、比例條件等）。
* 全程使用 Optimize.minimize，一步求解。
* 修正 Real.as_decimal("?") 轉型錯誤，透過 _to_number() 安全取值。

回傳格式與所得稅一致：
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
from __future__ import annotations
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from z3 import Optimize, Real, If, sat

from .constraint_utils import apply_linear_constraints


class UnsatError(Exception):
    """Raised when the constraint set is UNSAT but we want to report gracefully."""
    pass


# ---------------------------------------------------------------
# 小工具：安全將 Z3 數值轉為 Python int/float。
# ---------------------------------------------------------------
def _to_number(zval):
    if hasattr(zval, "is_int_value") and zval.is_int_value():
        return zval.as_long()
    s = zval.as_decimal(12).rstrip("?")
    return float(Decimal(s))


# ---------------------------------------------------------------
# 核心計算 (pure)
# 返回 (tax, final_params, diff)
# ---------------------------------------------------------------
def _calculate_internal(
    *,
    OperatingRevenueTotal: int = 0,
    SalesReturn: int = 0,
    SalesAllowance: int = 0,
    OperatingCost: int = 0,
    OperatingExpensesLosses: int = 0,
    NonOperatingRevenueTotal: int = 0,
    NonOperatingLossExpenses: int = 0,
    Prev10LossDeduction: int = 0,
    TaxIncentiveExempt: int = 0,
    ExemptSecuritiesIncome: int = 0,
    ExemptLandIncome: int = 0,
    Article4_4HouseLandGain: int = 0,
    is_full_year: bool = True,
    m_partial: int = 12,
    free_vars: Optional[List[str]] = None,
    constraints: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[int, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Pure Z3 核心：在給定 free_vars + constraints 下最小化營所稅。
    constraints 的語意與其他稅種一致，直接交由 constraint_utils.apply_linear_constraints 處理。
    """
    free_vars = set(free_vars or [])
    cons = constraints or {}
    opt = Optimize()

    # --- 宣告 Z3 變數 ---
    def R(n: str) -> Real:
        return Real(f"{n}_z")

    ovt_z   = R("OperatingRevenueTotal")
    sret_z  = R("SalesReturn")
    sall_z  = R("SalesAllowance")
    ocost_z = R("OperatingCost")
    oexp_z  = R("OperatingExpensesLosses")
    nrevt_z = R("NonOperatingRevenueTotal")
    nrevl_z = R("NonOperatingLossExpenses")
    prev_z  = R("Prev10LossDeduction")
    exem_z  = R("TaxIncentiveExempt")
    sec_z   = R("ExemptSecuritiesIncome")
    land_z  = R("ExemptLandIncome")
    art44_z = R("Article4_4HouseLandGain")

    # Map: 名稱 → (z3var, 原始值, builtin 檢查 list)
    param_map: Dict[str, Tuple[Any, int, List[Any]]] = {
        "OperatingRevenueTotal":   (ovt_z,   OperatingRevenueTotal,      [lambda v: v >= 0]),
        "SalesReturn":             (sret_z,  SalesReturn,               [lambda v: v >= 0]),
        "SalesAllowance":          (sall_z,  SalesAllowance,            [lambda v: v >= 0]),
        "OperatingCost":           (ocost_z, OperatingCost,             [lambda v: v >= 0]),
        "OperatingExpensesLosses": (oexp_z,  OperatingExpensesLosses,   [lambda v: v >= 0]),
        "NonOperatingRevenueTotal":(nrevt_z, NonOperatingRevenueTotal,  [lambda v: v >= 0]),
        "NonOperatingLossExpenses":(nrevl_z, NonOperatingLossExpenses,  [lambda v: v >= 0]),
        "Prev10LossDeduction":     (prev_z,  Prev10LossDeduction,       [lambda v: v >= 0]),
        "TaxIncentiveExempt":      (exem_z,  TaxIncentiveExempt,        [lambda v: v >= 0]),
        "ExemptSecuritiesIncome":  (sec_z,   ExemptSecuritiesIncome,    [lambda v: v >= 0]),
        "ExemptLandIncome":        (land_z,  ExemptLandIncome,          [lambda v: v >= 0]),
        "Article4_4HouseLandGain": (art44_z, Article4_4HouseLandGain,   [lambda v: v >= 0]),
    }

    # --- 綁定 fixed/free 及 builtin v>=0 檢查 ---
    for name, (zv, orig_val, checks) in param_map.items():
        if name not in free_vars:
            opt.add(zv == orig_val)
        for chk in checks:
            opt.add(chk(zv))

    # --- 使用共用 constraint_utils.apply_linear_constraints 套用所有約束 ---
    # 只需要名稱 → (z3var,) 的 mapping
    params_for_constraints = {name: (zv,) for name, (zv, _orig, _checks) in param_map.items()}
    apply_linear_constraints(opt, params_for_constraints, cons, debug=False)

    # --- 計算流程 ---
    ovn_z = Real("OperatingRevenueNet")
    ogp_z = Real("OperatingGrossProfit")
    onp_z = Real("OperatingNetProfit")
    yinc_z= Real("YearlyIncome")
    p_z   = Real("P")
    tax_z = Real("TaxExpression")

    # 營收與成本
    opt.add(ovn_z == ovt_z - sret_z - sall_z)
    opt.add(ogp_z == ovn_z - ocost_z)
    opt.add(onp_z == ogp_z - oexp_z)
    opt.add(yinc_z== onp_z + nrevt_z - nrevl_z)
    opt.add(p_z   == yinc_z - prev_z - exem_z - sec_z - land_z - art44_z)

    # 級距
    FULL1, FULL2 = 120_000, 200_000
    if is_full_year:
        expr = If(
            p_z <= FULL1, 0,
            If(p_z <= FULL2, (p_z - FULL1) * 0.5, p_z * 0.20),
        )
    else:
        padj_z = Real("P_adj")
        opt.add(padj_z == p_z * 12 / m_partial)
        expr = If(
            padj_z <= FULL1, 0,
            If(padj_z <= FULL2,
               (padj_z - FULL1) * 0.5 * (m_partial / 12),
               padj_z * 0.20 * (m_partial / 12)),
        )

    opt.add(tax_z == expr)

    # --- 最小化目標 & 求解 ---
    opt.minimize(tax_z)
    if opt.check() != sat:
        raise UnsatError("constraint set unsat")

    m = opt.model()
    final_tax_f = _to_number(m[tax_z])
    final_tax = int(round(final_tax_f))

    # --- 彙整 final_params & diff（unpack 3 elements） ---
    final_params: Dict[str, Dict[str, Any]] = {}
    diff: Dict[str, Dict[str, Any]] = {}
    for name, (zv, orig_val, _checks) in param_map.items():
        val = _to_number(m[zv])
        final_params[name] = {
            "value": val,
            "type": "free" if name in free_vars else "fixed",
        }
        if val != orig_val:
            diff[name] = {
                "original": orig_val,
                "optimized": val,
                "difference": val - orig_val,
            }

    return final_tax, final_params, diff


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------
def calculate_business_income_tax(
    *,
    OperatingRevenueTotal: int = 0,
    SalesReturn: int = 0,
    SalesAllowance: int = 0,
    OperatingCost: int = 0,
    OperatingExpensesLosses: int = 0,
    NonOperatingRevenueTotal: int = 0,
    NonOperatingLossExpenses: int = 0,
    Prev10LossDeduction: int = 0,
    TaxIncentiveExempt: int = 0,
    ExemptSecuritiesIncome: int = 0,
    ExemptLandIncome: int = 0,
    Article4_4HouseLandGain: int = 0,
    is_full_year: bool = True,
    m_partial: int = 12,
    free_vars: Optional[List[str]] = None,
    constraints: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    回傳格式與所得稅一致：
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
    started_time = time.perf_counter()

    # 組成完整參數表（顯式、型別清楚，便於上層 Caller 作型別感知補值）
    args: Dict[str, Any] = {
        "OperatingRevenueTotal": OperatingRevenueTotal,
        "SalesReturn": SalesReturn,
        "SalesAllowance": SalesAllowance,
        "OperatingCost": OperatingCost,
        "OperatingExpensesLosses": OperatingExpensesLosses,
        "NonOperatingRevenueTotal": NonOperatingRevenueTotal,
        "NonOperatingLossExpenses": NonOperatingLossExpenses,
        "Prev10LossDeduction": Prev10LossDeduction,
        "TaxIncentiveExempt": TaxIncentiveExempt,
        "ExemptSecuritiesIncome": ExemptSecuritiesIncome,
        "ExemptLandIncome": ExemptLandIncome,
        "Article4_4HouseLandGain": Article4_4HouseLandGain,
        "is_full_year": is_full_year,
        "m_partial": m_partial,
        "free_vars": free_vars or [],
    }
    cons = constraints or {}

    # ① baseline（完全固定、不套用任何 constraint）
    base_args = {k: v for k, v in args.items() if k not in ("free_vars", "constraints")}
    baseline_tax, _, _ = _calculate_internal(constraints={}, **base_args)

    # ② baseline + constraints 可行性檢查
    try:
        _calculate_internal(constraints=cons, **base_args)
        baseline_status = "sat"
        baseline_with_constraints = baseline_tax
    except UnsatError:
        baseline_status = "unsat"
        baseline_with_constraints = None

    # ③ manual_free 分支
    mode = "baseline"
    optimized = baseline_tax
    status = baseline_status
    final_params: Dict[str, Any] = {}
    diff: Dict[str, Any] = {}

    if free_vars:
        mode = "manual_free"
        try:
            optimized, final_params, diff = _calculate_internal(
                free_vars=free_vars,
                constraints=cons,
                **base_args,
            )
            status = "sat"
        except UnsatError:
            optimized = None
            status = "unsat"

    print(f"Calculation time business income tax: {time.perf_counter() - started_time:.3f} seconds")
    return {
        "mode": mode,
        "input_params": base_args,
        "baseline": baseline_tax,
        "baseline_status": baseline_status,
        "baseline_with_constraints": baseline_with_constraints,
        "optimized": optimized,
        "status": status,
        "diff": diff,
        "final_params": final_params,
        "constraints": cons,
    }