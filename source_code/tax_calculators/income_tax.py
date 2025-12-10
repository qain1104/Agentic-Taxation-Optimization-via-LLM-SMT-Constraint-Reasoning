# -*- coding: utf-8 -*-
from __future__ import annotations

import re, time, ast
from typing import Dict, List, Tuple, Any, Optional, Set

from z3 import (
    Optimize,
    Int,
    Real,
    RealVal,
    If,
    ToReal,
    ToInt,
    sat,
)
from tax_calculators.constraint_utils import apply_linear_constraints

class UnsatError(Exception):
    """Raised when the constraint set is UNSAT but we want to report gracefully."""
    pass


# ───────── 乘除 RHS pattern（目前只在舊版用到，這版 linear parser 不再依賴它，但保留不影響） ──────────
_RHS_MUL_DIV = re.compile(
    r"^\s*(\w+)\s*([*/])\s*(-?\d+(?:\.\d+)?)\s*$"  # var1 * 2.5  或 var1 / 3
)

# --------------------------------------------------------------------------------------
# Helper: default constants (pulled out so tests can patch easily)
# --------------------------------------------------------------------------------------

DEFAULTS = {
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
    # brackets
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


# --------------------------------------------------------------------------------------
# Core pure function
# --------------------------------------------------------------------------------------

def _calculate_tax_internal(
        *,
        # user data (partial list – full list same as original)
        is_married: bool = False,
        salary_self: int = 0,
        salary_spouse: int = 0,
        salary_dep: int = 0,
        interest_income: int = 0,
        stock_dividend: int = 0,
        house_transaction_gain: int = 0,
        other_income: int = 0,
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
        # new
        constraints: Optional[Dict[str, Dict[str, Any]]] = None,
        # keyword override for constants
        **overrides,
) -> Tuple[int, int, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    
    constants = {**DEFAULTS, **overrides}
    free_vars = free_vars or []
    constraints = constraints or {}

    opt = Optimize()

    # ----------------------------------------------------------------------------------
    # 1. declare z3 vars
    # ----------------------------------------------------------------------------------
    salary_self_z = Int("salary_self_z")
    salary_spouse_z = Int("salary_spouse_z")
    salary_dep_z = Int("salary_dep_z")
    interest_z = Int("interest_z")
    stock_div_z = Int("stock_div_z")
    house_gain_z = Int("house_gain_z")
    other_income_z = Int("other_income_z")
    itemized_ded_z = Int("itemized_ded_z")
    prop_loss_ded_z = Int("prop_loss_ded_z")
    rent_ded_z = Int("rent_ded_z")

    total_income_z = Int("total_income_z")

    self_salary_special_deduction_z = Int("self_salary_special_deduction_z")
    spouse_salary_special_deduction_z = Int("spouse_salary_special_deduction_z")
    dep_salary_special_ded_z = Int("dep_salary_special_ded_z")

    self_salary_after_ded_z = Int("self_salary_after_ded_z")
    spouse_salary_after_ded_z = Int("spouse_salary_after_ded_z")
    dep_salary_after_ded_z = Int("dep_salary_after_ded_z")

    total_exemption_z = Int("total_exemption_z")
    standard_deduction_z = Int("standard_deduction_z")
    chosen_deduction_z = Int("chosen_deduction_z")

    savings_investment_deduction_z = Int("savings_investment_deduction_z")
    disability_deduction_z = Int("disability_deduction_z")
    education_deduction_z = Int("education_deduction_z")

    preschool_count_z = Int("preschool_count_z")
    preschool_deduction_z = Int("preschool_deduction_z")

    long_term_care_deduction_z = Int("long_term_care_deduction_z")
    rent_deduction_z_lim = Int("rent_deduction_z_lim")

    total_people_z = Int("total_people_z")
    basic_living_exp_total_z = Int("basic_living_exp_total_z")
    base_deductions_var_z = Int("base_deductions_var_z")
    basic_living_exp_diff_z = Int("basic_living_exp_diff_z")
    total_deduction_z = Int("total_deduction_z")
    net_taxable_income_z = Int("net_taxable_income_z")
    net_taxable_nonneg_z = Int("net_taxable_nonneg_z")

    rate_calc_r = Real("rate_calc_r")
    final_tax_z = Int("final_tax_z")

    # ----------------------------------------------------------------------------------
    # 2. bind fields to z3 vars & built-in constraints
    # ----------------------------------------------------------------------------------
    params: Dict[str, Tuple[Any, int, List[Any]]] = {
        "salary_self": (salary_self_z, salary_self, [lambda v: v >= 0]),
        "salary_spouse": (salary_spouse_z, salary_spouse, [lambda v: v >= 0]),
        "salary_dep": (salary_dep_z, salary_dep, [lambda v: v >= 0]),
        "interest_income": (interest_z, interest_income, [lambda v: v >= 0]),
        "stock_dividend": (stock_div_z, stock_dividend, [lambda v: v >= 0]),
        "house_transaction_gain": (house_gain_z, house_transaction_gain, [lambda v: v >= 0]),
        "other_income": (other_income_z, other_income, [lambda v: v >= 0]),
        "itemized_deduction": (itemized_ded_z, itemized_deduction, [lambda v: v >= 0]),
        "property_loss_deduction": (prop_loss_ded_z, property_loss_deduction, [lambda v: v >= 0]),
        "rent_deduction": (rent_ded_z, rent_deduction, [lambda v: v >= 0, lambda v: v <= 10_000_000]),
        "cnt_under_70": (Int("cnt_under_70_z"), cnt_under_70, [lambda v: v >= 0]),
        "cnt_over_70": (Int("cnt_over_70_z"), cnt_over_70, [lambda v: v >= 0]),
        "disability_count": (Int("disability_count_z"), disability_count, [lambda v: v >= 0]),
        "education_count": (Int("education_count_z"), education_count, [lambda v: v >= 0]),
        "education_fee": (Int("education_fee_z"), education_fee, [lambda v: v >= 0]),
        "preschool_count": (Int("preschool_count_z_param"), preschool_count, [lambda v: v >= 0]),
        "long_term_care_count": (Int("long_term_care_count_z"), long_term_care_count, [lambda v: v >= 0]),
    }

    # apply initial/fixed vs free
    for param_name, (z3_var, value, builtins) in params.items():
        if param_name in free_vars:  # optimise this variable
            for cons in builtins:
                opt.add(cons(z3_var))
        else:  # fixed to user value
            opt.add(z3_var == value)
            for cons in builtins:
                opt.add(cons(z3_var))

    apply_linear_constraints(opt, params, constraints, debug=False)

    # ----------------------------------------------------------------------------------
    # 3. remaining model equations (use Z3 vars from params when variable can be free)
    # ----------------------------------------------------------------------------------
    c = constants  # alias

    # ------ 常用 Z3 參照（避免誤用 Python 參數值） ------
    cnt_under_70_zv = params["cnt_under_70"][0]
    cnt_over_70_zv  = params["cnt_over_70"][0]
    education_fee_zv    = params["education_fee"][0]
    education_count_zv  = params["education_count"][0]
    preschool_count_param_zv  = params["preschool_count"][0]
    long_term_care_count_zv = params["long_term_care_count"][0]
    rent_ded_input_zv = params["rent_deduction"][0]

    # 薪資特別扣除
    opt.add(self_salary_special_deduction_z == If(salary_self_z >= c["salary_special_deduction_max"],
                                                  c["salary_special_deduction_max"], salary_self_z))
    opt.add(spouse_salary_special_deduction_z == If(salary_spouse_z >= c["salary_special_deduction_max"],
                                                    c["salary_special_deduction_max"], salary_spouse_z))
    opt.add(dep_salary_special_ded_z == If(salary_dep_z >= c["salary_special_deduction_max"],
                                           c["salary_special_deduction_max"], salary_dep_z))

    opt.add(self_salary_after_ded_z == salary_self_z - self_salary_special_deduction_z)
    opt.add(spouse_salary_after_ded_z == salary_spouse_z - spouse_salary_special_deduction_z)
    opt.add(dep_salary_after_ded_z == salary_dep_z - dep_salary_special_ded_z)

    opt.add(self_salary_after_ded_z >= 0)
    opt.add(spouse_salary_after_ded_z >= 0)
    opt.add(dep_salary_after_ded_z >= 0)

    opt.add(total_income_z == (
            self_salary_after_ded_z
            + spouse_salary_after_ded_z
            + dep_salary_after_ded_z
            + interest_z
            + stock_div_z
            + house_gain_z
            + other_income_z
    ))

    opt.add(total_exemption_z == (
            cnt_under_70_zv * c["personal_exemption_under70"]
            + cnt_over_70_zv * c["personal_exemption_over70"]
    ))
    opt.add(standard_deduction_z == (c["standard_deduction_married"] if is_married else c["standard_deduction_single"]))
    opt.add(chosen_deduction_z == If(use_itemized, itemized_ded_z, standard_deduction_z))

    interest_plus_div_z = Int("interest_plus_div_z")
    opt.add(interest_plus_div_z == interest_z + stock_div_z)
    opt.add(
        savings_investment_deduction_z
        == If(
            interest_plus_div_z <= c["savings_investment_deduction_limit"],
            interest_plus_div_z,
            c["savings_investment_deduction_limit"],
        )
    )
    opt.add(disability_deduction_z == params["disability_count"][0] * c["disability_deduction_per_person"])

    # 教育學費扣除（Z3）
    opt.add(
        education_deduction_z
        == If(
            education_fee_zv <= 0,
            0,
            If(
                education_fee_zv >= education_count_zv * c["education_deduction_per_student"],
                education_count_zv * c["education_deduction_per_student"],
                education_fee_zv,
            ),
        )
    )

    # 幼兒學前扣除（Z3）
    opt.add(preschool_count_z == preschool_count_param_zv)
    opt.add(
        preschool_deduction_z
        == If(
            preschool_count_z <= 0,
            0,
            If(
                preschool_count_z == 1,
                150_000,
                150_000 + (preschool_count_z - 1) * 225_000,
            ),
        )
    )

    # 長照扣除（Z3）
    opt.add(long_term_care_deduction_z == long_term_care_count_zv * c["long_term_care_deduction_per_person"])

    # 房租扣除（Z3）
    opt.add(
        rent_deduction_z_lim
        == If(rent_ded_input_zv >= c["rent_deduction_limit"], c["rent_deduction_limit"], rent_ded_input_zv)
    )

    opt.add(total_people_z == cnt_under_70_zv + cnt_over_70_zv)
    opt.add(basic_living_exp_total_z == total_people_z * c["basic_living_exp_per_person"])

    opt.add(
        base_deductions_var_z
        == (
                total_exemption_z
                + chosen_deduction_z
                + savings_investment_deduction_z
                + disability_deduction_z
                + education_deduction_z
                + preschool_deduction_z
                + long_term_care_deduction_z
                + rent_deduction_z_lim
        )
    )
    opt.add(
        basic_living_exp_diff_z
        == If(basic_living_exp_total_z > base_deductions_var_z, basic_living_exp_total_z - base_deductions_var_z, 0)
    )
    opt.add(total_deduction_z == base_deductions_var_z + prop_loss_ded_z + basic_living_exp_diff_z)
    opt.add(net_taxable_income_z == total_income_z - total_deduction_z)
    opt.add(net_taxable_nonneg_z == If(net_taxable_income_z < 0, 0, net_taxable_income_z))

    x = ToReal(net_taxable_nonneg_z)
    opt.add(
        rate_calc_r
        == If(
            x <= c["bracket1_upper"],
            (x * c["bracket1_rate"]) - c["bracket1_sub"],
            If(
                x <= c["bracket2_upper"],
                (x * c["bracket2_rate"]) - c["bracket2_sub"],
                If(
                    x <= c["bracket3_upper"],
                    (x * c["bracket3_rate"]) - c["bracket3_sub"],
                    If(
                        x <= c["bracket4_upper"],
                        (x * c["bracket4_rate"]) - c["bracket4_sub"],
                        (x * c["bracket5_rate"]) - c["bracket5_sub"],
                    ),
                ),
            ),
        )
    )
    safe_tax_r = If(rate_calc_r < 0, 0, rate_calc_r)
    opt.add(final_tax_z == ToInt(safe_tax_r))

    # ----------------------------------------------------------------------------------
    # 4. solve optimisation: minimise final_tax_z
    # ----------------------------------------------------------------------------------
    opt.minimize(final_tax_z)

    check_res = opt.check()
    if check_res != sat:
        raise UnsatError("constraint set unsat")
    
    model = opt.model()
    final_tax = model[final_tax_z].as_long()

    # Build result dictionaries
    final_params: Dict[str, Dict[str, Any]] = {}
    differences: Dict[str, Dict[str, Any]] = {}

    for param_name, (z3_var, orig_value, _) in params.items():
        val = model[z3_var].as_long()
        final_params[param_name] = {
            "value": val,
            "type": "free" if param_name in free_vars else "fixed",
        }
        if val != orig_value:
            differences[param_name] = {
                "original": orig_value,
                "optimized": val,
                "difference": val - orig_value,
            }

    net_taxable_income = model[net_taxable_income_z].as_long()
    return final_tax, net_taxable_income, final_params, differences


def _run_with_high_income_rule(*, constraints: Optional[Dict[str, Dict[str, Any]]] = None, **kwargs):
    """呼叫 _calculate_tax_internal，若淨課稅所得 ≥ 1,330,000 則強制長照 / 房租扣除為 0 後重算。"""
    constraints = constraints or {}
    free_vars = kwargs.get("free_vars") or []

    tax1, net1, fp1, diff1 = _calculate_tax_internal(constraints=constraints, **kwargs)
    threshold = DEFAULTS["bracket2_upper"]

    if net1 is not None and net1 >= threshold:
        # 建立第二次計算的 kwargs
        kwargs2 = dict(kwargs)
        kwargs2["long_term_care_count"] = 0
        kwargs2["rent_deduction"] = 0
        # 若這兩個欄位原本在 free_vars，移除之，避免被優化成非零
        new_free = [v for v in free_vars if v not in ("long_term_care_count", "rent_deduction")]
        kwargs2["free_vars"] = new_free if new_free else None

        tax2, net2, fp2, diff2 = _calculate_tax_internal(constraints=constraints, **kwargs2)
        return tax2, net2, fp2, diff2

    return tax1, net1, fp1, diff1


"""
calculate_comprehensive_income_tax (updated)
-------------------------------------------
* 內建高所得規則：若淨課稅所得 ≥ 1,330,000，強制 long_term_care_count / rent_deduction = 0 再重算。
* 行為等同 tax_solver_113_final。
* 其他介面與回傳格式維持不變。
"""
from typing import Dict, Any, Optional

def calculate_comprehensive_income_tax(
    *,
    constraints: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs,
):
    """
    公開 API：

    回傳結構：
        {
            mode:             "manual_free" | "baseline",
            input_params:     kwargs,
            baseline:         <固定參數稅額>,
            baseline_status:  "sat" | "unsat",
            optimized:        <最佳化稅額>  # 已用最終參數凍結重算後的值
            optimized_solver: <求解器第一次回傳的最佳化稅額>  # 便於除錯
            status:           "sat" | "unsat",
            diff:             {...},
            final_params:     {變數: {value, type}},
            constraints:      <最終採用的 constraints>,
        }
    """
    start_time = time.perf_counter()

    # ---- 0) 入參處理與兼容 legacy key ------------------------------------------
    kwargs = dict(kwargs)  # 本地複製避免副作用
    free_vars = kwargs.get("free_vars") or []

    # 允許呼叫方用 legacy 名稱 `variable_constraints`
    vc = kwargs.pop("variable_constraints", None)
    if constraints is None:
        constraints = vc or {}
    else:
        # 以 constraints 為主，merge legacy 進來
        constraints = {**(vc or {}), **constraints}

    # ---- 0.1) free_vars 名稱防呆（避免打錯字） -----------------------------------
    _KNOWN_PARAM_NAMES: Set[str] = {
        "salary_self", "salary_spouse", "salary_dep",
        "interest_income", "stock_dividend", "house_transaction_gain",
        "other_income", "itemized_deduction", "property_loss_deduction",
        "rent_deduction",
        "cnt_under_70", "cnt_over_70",
        "disability_count", "education_count", "education_fee",
        "preschool_count", "long_term_care_count",
    }
    for v in free_vars:
        if v not in _KNOWN_PARAM_NAMES:
            raise ValueError(f"Unknown free var: {v}")

    # ---- 1) baseline：套用高所得規則（不含 free_vars、不含 constraints） ----------
    base_kwargs = {k: v for k, v in kwargs.items() if k != "free_vars"}
    baseline_tax, _, _, _ = _run_with_high_income_rule(
        constraints={},  # baseline 不帶 constraints
        **base_kwargs,
    )

    # ---- 2) baseline_status：固定值 + constraints 可行性檢查 ----------------------
    try:
        _calculate_tax_internal(constraints=constraints, **base_kwargs)
        baseline_status = "sat"
    except UnsatError:
        baseline_status = "unsat"

    # 預設（沒有 free_vars 時就回 baseline）
    mode = "baseline"
    opt_tax: Optional[int] = baseline_tax
    params_out: Dict[str, Dict[str, Any]] = {}
    diff_out: Dict[str, Dict[str, Any]] = {}
    status = baseline_status
    opt_tax_solver_value: Optional[int] = None  # 除錯用

    # ---- 3) 若有 free_vars → manual_free 最佳化 -------------------------------
    if free_vars:
        mode = "manual_free"
        try:
            # 3.1）第一次：帶 free_vars + constraints 做最佳化
            opt_tax_solver_value, _, params_out, diff_out = _run_with_high_income_rule(
                constraints=constraints,
                **kwargs,
            )
            status = "sat"

            # 3.2）用「最佳化出的最終參數」凍結重算一次（不帶 free_vars、不帶 constraints）
            frozen_kwargs: Dict[str, Any] = {k: v["value"] for k, v in params_out.items()}
            if "is_married" in kwargs:
                frozen_kwargs["is_married"] = kwargs["is_married"]
            if "use_itemized" in kwargs:
                frozen_kwargs["use_itemized"] = kwargs["use_itemized"]

            tax_recalc, _, _, _ = _calculate_tax_internal(
                constraints={},  # 凍結重算不帶任何 constraints
                **frozen_kwargs
            )
            opt_tax = tax_recalc

        except UnsatError:
            opt_tax = None
            status = "unsat"

    # ---- 4) 組裝回傳 ----------------------------------------------------------
    elapsed = time.perf_counter() - start_time
    print(elapsed)

    return {
        "mode": mode,
        "input_params": kwargs,
        "baseline": baseline_tax,
        "baseline_status": baseline_status,
        "optimized": opt_tax,
        "optimized_solver": opt_tax_solver_value,
        "status": status,
        "diff": diff_out,
        "final_params": params_out,
        "constraints": constraints,
    }
