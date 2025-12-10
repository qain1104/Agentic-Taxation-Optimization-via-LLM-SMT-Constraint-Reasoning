# -*- coding: utf-8 -*-
"""
foreigner_income_tax.py ── 外僑綜所稅計算 + Z3 最適化（util 版）

變更重點：
1) 支援 legacy key `variable_constraints`，並與 `constraints` 合併。
2) 對 0/1 類變數加上上界（disability_count、long_term_care_count）。
3) 最佳化後以「最終參數」凍結重算，覆蓋 optimized，避免前端顯示殘值。
4) free_vars 名稱檢查。
5) 📌 使用共用的 apply_linear_constraints(util) 處理所有 constraints。
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional, Set
import calendar, time
from datetime import date

from z3 import (
    Optimize, Int, RealVal, ToReal, ToInt, If, And, sat
)

from tax_calculators.constraint_utils import apply_linear_constraints

# ─── 共用錯誤型別 ────────────────────────────────────────────────
class UnsatError(Exception):
    """Raised when constraint set is UNSAT but we want to report gracefully."""
    pass


# ─── 稅法常數（與本國人一致）──────────────────────────────────────
DEFAULTS: Dict[str, int] = {
    "basic_living_exp_per_person": 210_000,
    "savings_investment_deduction_limit": 270_000,
    "disability_deduction_per_person": 218_000,
    "education_deduction_per_student": 25_000,
    "long_term_care_deduction_per_person": 120_000,
    "rent_deduction_limit": 180_000,
    "personal_exemption_under70": 97_000,
    "personal_exemption_over70": 145_500,
    "standard_deduction_single": 131_000,
    "standard_deduction_married": 262_000,
    "salary_special_deduction_max": 218_000,
    "bracket1_upper": 590_000,
    "bracket2_upper": 1_330_000,
    "bracket3_upper": 2_660_000,
    "bracket4_upper": 4_980_000,
    "bracket1_rate": 0.05,
    "bracket2_rate": 0.12,
    "bracket3_rate": 0.20,
    "bracket4_rate": 0.30,
    "bracket5_rate": 0.40,
    "bracket1_sub": 0,
    "bracket2_sub": 41_300,
    "bracket3_sub": 147_700,
    "bracket4_sub": 413_700,
    "bracket5_sub": 911_700,
}

# ───────────────────────────────────────────────────────────────
# 1. 核心函式 _calculate_foreigner_tax_internal
# ───────────────────────────────────────────────────────────────
def _calculate_foreigner_tax_internal(
    *,
    days_of_stay: int = 365,
    is_departure: bool = False,
    is_married: bool = False,
    salary_self: int = 0,
    salary_spouse: int = 0,
    salary_dep: int = 0,
    interest_income: int = 0,
    interest_spouse: int = 0,
    interest_dep: int = 0,
    other_income: int = 0,
    other_income_spouse: int = 0,
    other_income_dep: int = 0,
    cnt_under_70: int = 0,
    cnt_over_70: int = 0,
    use_itemized: bool = False,
    itemized_deduction: int = 0,
    property_loss_deduction: int = 0,
    disability_count: int = 0,
    education_count: int = 0,
    education_fee: int = 0,
    preschool_count: int = 0,
    long_term_care_count: int = 0,
    rent_deduction: int = 0,
    free_vars: Optional[List[str]] = None,
    constraints: Optional[Dict[str, Dict[str, Any]]] = None,
    **override,
) -> Tuple[int, int, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:

    C = {**DEFAULTS, **override}
    free_vars = free_vars or []
    constraints = constraints or {}

    opt = Optimize()

    # ---------- 1. 宣告 Z3 參數 ----------
    def I(n: str): return Int(f"{n}_z")

    z = {
        # 收入 / 扣除類
        "salary_self": I("salary_self"), "salary_spouse": I("salary_spouse"), "salary_dep": I("salary_dep"),
        "interest_income": I("interest_income"), "interest_spouse": I("interest_spouse"), "interest_dep": I("interest_dep"),
        "other_income": I("other_income"), "other_income_spouse": I("other_income_spouse"), "other_income_dep": I("other_income_dep"),
        "itemized_deduction": I("itemized_deduction"), "property_loss_deduction": I("property_loss_deduction"),
        "rent_deduction": I("rent_deduction"),
        # 人數 / 次數
        "cnt_under_70": I("cnt_under_70"), "cnt_over_70": I("cnt_over_70"),
        "disability_count": I("disability_count"), "education_count": I("education_count"),
        "education_fee": I("education_fee"), "preschool_count": I("preschool_count"),
        "long_term_care_count": I("long_term_care_count"),
        "days_of_stay": I("days_of_stay"),
    }

    # ---------- 2. 固定 / 自由 ----------
    PARAMS: Dict[str, Tuple[Any, int, List[Any]]] = {}

    def _add_param(k: str, v: int, *extra):
        PARAMS[k] = (z[k], v, [lambda t: t >= 0, *extra])

    _add = _add_param
    _add("salary_self", salary_self)
    _add("salary_spouse", salary_spouse)
    _add("salary_dep", salary_dep)

    _add("interest_income", interest_income)
    _add("interest_spouse", interest_spouse)
    _add("interest_dep", interest_dep)

    _add("other_income", other_income)
    _add("other_income_spouse", other_income_spouse)
    _add("other_income_dep", other_income_dep)

    _add("itemized_deduction", itemized_deduction)
    _add("property_loss_deduction", property_loss_deduction)
    _add("rent_deduction", rent_deduction, lambda v: v <= 10_000_000)

    _add("cnt_under_70", cnt_under_70)
    _add("cnt_over_70",  cnt_over_70)

    # ★ 防呆：0/1 上界
    _add("disability_count", disability_count, lambda v: And(v >= 0, v <= 1))
    _add("education_count", education_count)
    _add("education_fee", education_fee)
    _add("preschool_count", preschool_count)
    _add("long_term_care_count", long_term_care_count, lambda v: And(v >= 0, v <= 1))

    # days_of_stay：0 ~ 去年天數（含閏年），避免亂飛
    last_year = date.today().year - 1
    days_in_last_year = 366 if calendar.isleap(last_year) else 365
    _add("days_of_stay", days_of_stay, lambda v: And(v >= 0, v <= days_in_last_year))

    # 寫入「固定 / free var」的內建 constraint
    for k, (zv, val, checks) in PARAMS.items():
        if k in free_vars:
            for c in checks:
                opt.add(c(zv))
        else:
            opt.add(zv == val)
            for c in checks:
                opt.add(c(zv))

    # ---------- 3. 使用共用 util 套用線性 constraints ----------
    # 支援：
    #   - days_of_stay + ... <= ...
    #   - itemized_deduction > rent_deduction + education_fee
    #   - property_loss_deduction <= other_income * 0.3
    #   等一般線性表達式
    apply_linear_constraints(opt, PARAMS, constraints, debug=False)

    # ---------- 4. 比例 adj(val) = floor(val * ratio) ----------
    if not is_departure:
        ratio_r = RealVal(1)
    else:
        last_year = date.today().year - 1
        days_in_last_year = 366 if calendar.isleap(last_year) else 365
        ratio_r = ToReal(z["days_of_stay"]) / RealVal(days_in_last_year)

    def adj(val: int):
        return ToInt(RealVal(val) * ratio_r)

    # ---------- 5. 稅額模型 ----------
    SAL_MAX = C["salary_special_deduction_max"]

    self_sp = Int("self_sp")
    sp_sp   = Int("sp_sp")
    dep_sp  = Int("dep_sp")
    opt.add(self_sp == If(z["salary_self"]   >= SAL_MAX, SAL_MAX, z["salary_self"]))
    opt.add(sp_sp   == If(z["salary_spouse"] >= SAL_MAX, SAL_MAX, z["salary_spouse"]))
    opt.add(dep_sp  == If(z["salary_dep"]    >= SAL_MAX, SAL_MAX, z["salary_dep"]))

    self_after = Int("self_after")
    sp_after   = Int("sp_after")
    dep_after  = Int("dep_after")
    opt.add(self_after == z["salary_self"]   - self_sp)
    opt.add(sp_after   == z["salary_spouse"] - sp_sp)
    opt.add(dep_after  == z["salary_dep"]    - dep_sp)
    for v in (self_after, sp_after, dep_after):
        opt.add(v >= 0)

    total_income = Int("total_income")
    opt.add(total_income == (
        self_after + sp_after + dep_after +
        z["interest_income"] + z["interest_spouse"] + z["interest_dep"] +
        z["other_income"] + z["other_income_spouse"] + z["other_income_dep"]
    ))

    total_ex = Int("total_ex")
    opt.add(total_ex == (
        z["cnt_under_70"] * adj(C["personal_exemption_under70"]) +
        z["cnt_over_70"]  * adj(C["personal_exemption_over70"])
    ))

    std_single  = adj(C["standard_deduction_single"])
    std_married = adj(C["standard_deduction_married"])
    std_ded_expr = std_married if is_married else std_single
    chosen_ded = Int("chosen_ded")
    opt.add(chosen_ded == If(use_itemized, z["itemized_deduction"], std_ded_expr))

    interest_sum = z["interest_income"] + z["interest_spouse"] + z["interest_dep"]
    sav_inv = Int("sav_inv")
    opt.add(
        sav_inv ==
        If(
            interest_sum <= C["savings_investment_deduction_limit"],
            interest_sum,
            C["savings_investment_deduction_limit"],
        )
    )

    disability_ded = z["disability_count"] * C["disability_deduction_per_person"]

    edu_ded = Int("edu_ded")
    opt.add(
        edu_ded ==
        If(
            z["education_fee"] <= 0,
            0,
            If(
                z["education_fee"] >= z["education_count"] * C["education_deduction_per_student"],
                z["education_count"] * C["education_deduction_per_student"],
                z["education_fee"],
            ),
        )
    )

    preschool_ded = Int("preschool_ded")
    opt.add(
        preschool_ded ==
        If(
            z["preschool_count"] <= 0,
            0,
            If(
                z["preschool_count"] == 1,
                150_000,
                150_000 + (z["preschool_count"] - 1) * 225_000,
            ),
        )
    )

    long_term = z["long_term_care_count"] * C["long_term_care_deduction_per_person"]

    rent_lim = Int("rent_lim")
    opt.add(
        rent_lim == If(
            z["rent_deduction"] >= C["rent_deduction_limit"],
            C["rent_deduction_limit"],
            z["rent_deduction"],
        )
    )

    total_people = z["cnt_under_70"] + z["cnt_over_70"]
    basic_need   = total_people * adj(C["basic_living_exp_per_person"])

    base_ded = (
        total_ex + chosen_ded + sav_inv + disability_ded +
        edu_ded + preschool_ded + long_term + rent_lim
    )

    basic_diff = Int("basic_diff")
    opt.add(basic_diff == If(basic_need > base_ded, basic_need - base_ded, 0))

    total_ded = base_ded + z["property_loss_deduction"] + basic_diff
    net_inc   = Int("net_inc")
    opt.add(net_inc == total_income - total_ded)

    net_pos = Int("net_pos")
    opt.add(net_pos == If(net_inc < 0, 0, net_inc))

    x = ToReal(net_pos)
    Cb = C
    tax_r = If(x <= Cb["bracket1_upper"], x * Cb["bracket1_rate"] - Cb["bracket1_sub"],
        If(x <= Cb["bracket2_upper"], x * Cb["bracket2_rate"] - Cb["bracket2_sub"],
        If(x <= Cb["bracket3_upper"], x * Cb["bracket3_rate"] - Cb["bracket3_sub"],
        If(x <= Cb["bracket4_upper"], x * Cb["bracket4_rate"] - Cb["bracket4_sub"],
            x * Cb["bracket5_rate"] - Cb["bracket5_sub"]))))

    safe_r = If(tax_r < 0, 0, tax_r)
    final_tax_z = Int("final_tax_z")
    opt.add(final_tax_z == ToInt(safe_r))

    # ---------- 6. 最小化 ----------
    opt.minimize(final_tax_z)

    # ---------- 7. 求解 ----------
    if opt.check() != sat:
        raise UnsatError("constraint set unsat")

    mdl = opt.model()
    tax_val = mdl[final_tax_z].as_long()

    # ---------- 8. 產出結果 ----------
    final_params: Dict[str, Dict[str, Any]] = {}
    diff: Dict[str, Dict[str, Any]] = {}

    for k, (zv, orig, _) in PARAMS.items():
        v = mdl[zv].as_long()
        final_params[k] = {
            "value": v,
            "type": "free" if k in free_vars else "fixed"
        }
        if v != orig:
            diff[k] = {
                "original": orig,
                "optimized": v,
                "difference": v - orig,
            }

    return tax_val, mdl[net_pos].as_long(), final_params, diff


# ───────────────────────────────────────────────────────────────
# 2. Public API（修正版）
# ───────────────────────────────────────────────────────────────
def calculate_foreigner_income_tax(
    *,
    days_of_stay: int = 365,
    is_departure: bool = False,
    is_married: bool = False,
    salary_self: int = 0,
    salary_spouse: int = 0,
    salary_dep: int = 0,
    interest_income: int = 0,
    interest_spouse: int = 0,
    interest_dep: int = 0,
    other_income: int = 0,
    other_income_spouse: int = 0,
    other_income_dep: int = 0,
    cnt_under_70: int = 0,
    cnt_over_70: int = 0,
    use_itemized: bool = False,
    itemized_deduction: int = 0,
    property_loss_deduction: int = 0,
    disability_count: int = 0,
    education_count: int = 0,
    education_fee: int = 0,
    preschool_count: int = 0,
    long_term_care_count: int = 0,
    rent_deduction: int = 0,
    free_vars: Optional[List[str]] = None,
    constraints: Optional[Dict[str, Dict[str, Any]]] = None,
    **override,
):
    start_time = time.perf_counter()

    # ---- 0) 參數整備 ----
    kwargs = {
        "days_of_stay": days_of_stay,
        "is_departure": is_departure,
        "is_married": is_married,
        "salary_self": salary_self,
        "salary_spouse": salary_spouse,
        "salary_dep": salary_dep,
        "interest_income": interest_income,
        "interest_spouse": interest_spouse,
        "interest_dep": interest_dep,
        "other_income": other_income,
        "other_income_spouse": other_income_spouse,
        "other_income_dep": other_income_dep,
        "cnt_under_70": cnt_under_70,
        "cnt_over_70": cnt_over_70,
        "use_itemized": use_itemized,
        "itemized_deduction": itemized_deduction,
        "property_loss_deduction": property_loss_deduction,
        "disability_count": disability_count,
        "education_count": education_count,
        "education_fee": education_fee,
        "preschool_count": preschool_count,
        "long_term_care_count": long_term_care_count,
        "rent_deduction": rent_deduction,
        "free_vars": free_vars,
        **override,
    }
    kwargs = dict(kwargs)
    free_vars = kwargs.get("free_vars") or []

    # ---- 0.1) 合併 legacy constraints ----
    vc = kwargs.pop("variable_constraints", None)
    if constraints is None:
        constraints = vc or {}
    else:
        constraints = {**(vc or {}), **constraints}

    # ---- 0.2) free_vars 名稱檢查 ----
    _KNOWN_PARAM_NAMES: Set[str] = {
        "days_of_stay",
        "salary_self", "salary_spouse", "salary_dep",
        "interest_income", "interest_spouse", "interest_dep",
        "other_income", "other_income_spouse", "other_income_dep",
        "itemized_deduction", "property_loss_deduction", "rent_deduction",
        "cnt_under_70", "cnt_over_70",
        "disability_count", "education_count", "education_fee",
        "preschool_count", "long_term_care_count",
    }
    for v in free_vars:
        if v not in _KNOWN_PARAM_NAMES:
            raise ValueError(f"Unknown free var: {v}")

    # ---- 自動設定 是否提前離境 ----
    _last_year = date.today().year - 1
    _days_in_last_year = 366 if calendar.isleap(_last_year) else 365

    # 只有當 days_of_stay 不是 free var（即使用者是明確傳固定值）時才自動判斷
    if "days_of_stay" not in free_vars:
        if int(kwargs["days_of_stay"]) != _days_in_last_year:
            kwargs["is_departure"] = True

    # ---- 1) baseline（不帶 free_vars / constraints） ----
    base_tax, _, _, _ = _calculate_foreigner_tax_internal(
        **{k: v for k, v in kwargs.items() if k != "free_vars"},
        constraints={}
    )

    # ---- 2) baseline + constraints 可行性 ----
    try:
        _calculate_foreigner_tax_internal(
            **{k: v for k, v in kwargs.items() if k != "free_vars"},
            constraints=constraints
        )
        baseline_status = "sat"
        baseline_with   = base_tax
    except UnsatError:
        baseline_status = "unsat"
        baseline_with   = None

    mode = "baseline"
    status = baseline_status
    opt_tax = base_tax
    params_out: Dict[str, Any] = {}
    diff_out: Dict[str, Any] = {}
    opt_tax_solver_value: Optional[int] = None  # 除錯用

    # ---- 3) manual_free：有 free_vars 就做最佳化 ----
    if free_vars:
        mode = "manual_free"
        try:
            # 第一次：帶 constraints + free_vars 求最佳解
            opt_tax_solver_value, _, params_out, diff_out = _calculate_foreigner_tax_internal(
                constraints=constraints, **kwargs
            )
            status = "sat"

            # 第二次：用「最終參數」凍結重算，覆蓋 optimized
            frozen_kwargs: Dict[str, Any] = {k: v["value"] for k, v in params_out.items()}
            frozen_kwargs["is_departure"] = kwargs.get("is_departure", False)
            frozen_kwargs["is_married"] = kwargs.get("is_married", False)
            frozen_kwargs["use_itemized"] = kwargs.get("use_itemized", False)

            tax_recalc, _, _, _ = _calculate_foreigner_tax_internal(
                constraints={}, **frozen_kwargs
            )
            opt_tax = tax_recalc

        except UnsatError:
            opt_tax = None
            status = "unsat"

    print("Calculation time foreign:", time.perf_counter() - start_time)
    return {
        "mode": mode,
        "input_params": kwargs,
        "baseline": base_tax,
        "baseline_status": baseline_status,
        "baseline_with_constraints": baseline_with,
        "optimized": opt_tax,
        "optimized_solver": opt_tax_solver_value,
        "status": status,
        "diff": diff_out,
        "combo": [],
        "final_params": params_out,
        "constraints": constraints,
    }
