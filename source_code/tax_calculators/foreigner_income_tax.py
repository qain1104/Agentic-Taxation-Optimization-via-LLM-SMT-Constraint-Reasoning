# -*- coding: utf-8 -*-
"""foreigner_income_tax.py ── 外僑（居住者）綜合所得稅試算 + Z3 最適化

你提供的 notebook 會把本檔的計算結果，拿去跟 eTax Portal 的
「外僑綜合所得稅試算」頁面做比對。

本版修正兩個造成差異的重點：

1) **排富條款（房屋租金支出特別扣除額、長期照顧特別扣除額）**
   - 自 113 年度（2024）起，若「減除房租＋長照後」綜所稅適用稅率
     達 20%（含）以上，則**不得**列報「房屋租金支出特別扣除額」與
     「長期照顧特別扣除額」。
   - eTax 試算器會套用該條件；原本版本未套用，因此本機稅額偏低。

   參考（財政部稅務入口網 Q&A）：
   - 房屋租金支出特別扣除額：1210（排富條款包含 20% 稅率與 750 萬基本所得額）

2) **不再自動覆寫 is_departure**
   - eTax 網頁是由使用者用「是否提前離境」單選決定是否按居留天數比例。
   - 原本版本會依 days_of_stay 與去年天數（閏年 366）自動把 is_departure
     改成 True，導致在閏年資料下與網頁不一致。

其餘保留原本功能：
- 支援 legacy key `variable_constraints`
- 對 0/1 類變數加上上界（disability_count、long_term_care_count）
- 最佳化後以「最終參數」凍結重算，避免前端顯示殘值
- free_vars 名稱檢查
- 使用共用 apply_linear_constraints(util)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional, Set
import calendar
import time
from datetime import date

from z3 import Optimize, Int, RealVal, ToReal, ToInt, If, And, Or, sat

from tax_calculators.constraint_utils import apply_linear_constraints


class UnsatError(Exception):
    """Raised when constraint set is UNSAT but we want to report gracefully."""


# ─── 稅法常數（以 113 年度試算常用值為主；eTax 目前對應此組） ─────────────
DEFAULTS: Dict[str, Any] = {
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
    # progressive brackets
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
    # 排富條款：所得基本稅額條例之「基本所得額扣除額」（113年度：750萬）
    # 本試算器未完整實作基本所得額計算；此參數僅保留介面。
    "amt_basic_income_deduction": 7_500_000,
}


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
    """Internal solver-based calculator.

    Returns:
        tax (int), net_pos (int), final_params (dict), diff (dict)
    """

    C = {**DEFAULTS, **override}
    free_vars = free_vars or []
    constraints = constraints or {}

    opt = Optimize()

    # ---------- 1) Declare Z3 parameters ----------
    def I(n: str):
        return Int(f"{n}_z")

    z = {
        # income
        "salary_self": I("salary_self"),
        "salary_spouse": I("salary_spouse"),
        "salary_dep": I("salary_dep"),
        "interest_income": I("interest_income"),
        "interest_spouse": I("interest_spouse"),
        "interest_dep": I("interest_dep"),
        "other_income": I("other_income"),
        "other_income_spouse": I("other_income_spouse"),
        "other_income_dep": I("other_income_dep"),
        # deductions
        "itemized_deduction": I("itemized_deduction"),
        "property_loss_deduction": I("property_loss_deduction"),
        "rent_deduction": I("rent_deduction"),
        # counts
        "cnt_under_70": I("cnt_under_70"),
        "cnt_over_70": I("cnt_over_70"),
        "disability_count": I("disability_count"),
        "education_count": I("education_count"),
        "education_fee": I("education_fee"),
        "preschool_count": I("preschool_count"),
        "long_term_care_count": I("long_term_care_count"),
        "days_of_stay": I("days_of_stay"),
    }

    # ---------- 2) Fix/free & basic domains ----------
    PARAMS: Dict[str, Tuple[Any, int, List[Any]]] = {}

    def _add_param(k: str, v: int, *extra_checks):
        PARAMS[k] = (z[k], v, [lambda t: t >= 0, *extra_checks])

    _add_param("salary_self", salary_self)
    _add_param("salary_spouse", salary_spouse)
    _add_param("salary_dep", salary_dep)

    _add_param("interest_income", interest_income)
    _add_param("interest_spouse", interest_spouse)
    _add_param("interest_dep", interest_dep)

    _add_param("other_income", other_income)
    _add_param("other_income_spouse", other_income_spouse)
    _add_param("other_income_dep", other_income_dep)

    _add_param("itemized_deduction", itemized_deduction)
    _add_param("property_loss_deduction", property_loss_deduction)
    _add_param("rent_deduction", rent_deduction, lambda v: v <= 10_000_000)

    _add_param("cnt_under_70", cnt_under_70)
    _add_param("cnt_over_70", cnt_over_70)

    # 0/1 guardrails
    _add_param("disability_count", disability_count, lambda v: And(v >= 0, v <= 1))
    _add_param("long_term_care_count", long_term_care_count, lambda v: And(v >= 0, v <= 1))

    _add_param("education_count", education_count)
    _add_param("education_fee", education_fee)
    _add_param("preschool_count", preschool_count)

    # days_of_stay range by last year's actual days (leap aware)
    last_year = date.today().year - 1
    days_in_last_year = 366 if calendar.isleap(last_year) else 365
    _add_param("days_of_stay", days_of_stay, lambda v: And(v >= 0, v <= days_in_last_year))

    for k, (zv, val, checks) in PARAMS.items():
        if k in free_vars:
            for ck in checks:
                opt.add(ck(zv))
        else:
            opt.add(zv == val)
            for ck in checks:
                opt.add(ck(zv))

    # ---------- 2.4) Spouse guardrail ----------
    # 若未結婚（is_married=False），強制所有配偶相關欄位 = 0
    # 避免 solver 在總所得等式裡「憑空長出配偶收入」(tie-break / multiple optima issue)
    if not is_married:
        opt.add(z["salary_spouse"] == 0)
        opt.add(z["interest_spouse"] == 0)
        opt.add(z["other_income_spouse"] == 0)

    # ---------- 2.5) Dependent guardrail ----------
    # 若未提供扶養人數（cnt_under_70 + cnt_over_70 = 0），禁止 solver「自己長出扶養親屬」
    # 並強制所有扶養親屬相關欄位 = 0（避免 dep income/interest/other 影響稅額）
    if int(cnt_under_70 or 0) + int(cnt_over_70 or 0) == 0:
        # cnt 的 Z3 變數
        opt.add(z["cnt_under_70"] == 0)
        opt.add(z["cnt_over_70"] == 0)

        # 扶養親屬相關欄位（你可自行增減）
        dep_related_vars = [
            z["salary_dep"],
            z["interest_dep"],
            z["other_income_dep"],
        ]
        for vv in dep_related_vars:
            opt.add(vv == 0)

    # ---------- 3) External constraints (optional) ----------
    apply_linear_constraints(opt, PARAMS, constraints, debug=False)

    # ---------- 4) Ratio adjustment for departure ----------
    if not is_departure:
        ratio_r = RealVal(1)
    else:
        # same last_year/days_in_last_year as above
        ratio_r = ToReal(z["days_of_stay"]) / RealVal(days_in_last_year)

    def adj(val: int):
        # eTax 的做法是以整數（元）計算；比例後採 ToInt (truncate)
        return ToInt(RealVal(val) * ratio_r)

    # ---------- 5) Build tax model ----------
    SAL_MAX = int(C["salary_special_deduction_max"])

    self_sp = Int("self_sp")
    sp_sp = Int("sp_sp")
    dep_sp = Int("dep_sp")
    opt.add(self_sp == If(z["salary_self"] >= SAL_MAX, SAL_MAX, z["salary_self"]))
    opt.add(sp_sp == If(z["salary_spouse"] >= SAL_MAX, SAL_MAX, z["salary_spouse"]))
    opt.add(dep_sp == If(z["salary_dep"] >= SAL_MAX, SAL_MAX, z["salary_dep"]))

    self_after = Int("self_after")
    sp_after = Int("sp_after")
    dep_after = Int("dep_after")
    opt.add(self_after == z["salary_self"] - self_sp)
    opt.add(sp_after == z["salary_spouse"] - sp_sp)
    opt.add(dep_after == z["salary_dep"] - dep_sp)
    opt.add(self_after >= 0)
    opt.add(sp_after >= 0)
    opt.add(dep_after >= 0)

    total_income = Int("total_income")
    opt.add(
        total_income
        == (
            self_after
            + sp_after
            + dep_after
            + z["interest_income"]
            + z["interest_spouse"]
            + z["interest_dep"]
            + z["other_income"]
            + z["other_income_spouse"]
            + z["other_income_dep"]
        )
    )

    total_ex = Int("total_ex")
    opt.add(
        total_ex
        == (
            z["cnt_under_70"] * adj(int(C["personal_exemption_under70"]))
            + z["cnt_over_70"] * adj(int(C["personal_exemption_over70"]))
        )
    )

    std_single = adj(int(C["standard_deduction_single"]))
    std_married = adj(int(C["standard_deduction_married"]))
    std_ded_expr = std_married if is_married else std_single

    chosen_ded = Int("chosen_ded")
    opt.add(
        chosen_ded
        == If(
            use_itemized,
            If(z["itemized_deduction"] >= std_ded_expr, z["itemized_deduction"], std_ded_expr),
            std_ded_expr,
        )
    )

    # savings & investment
    interest_sum = z["interest_income"] + z["interest_spouse"] + z["interest_dep"]
    sav_inv = Int("sav_inv")
    opt.add(
        sav_inv
        == If(
            interest_sum <= int(C["savings_investment_deduction_limit"]),
            interest_sum,
            int(C["savings_investment_deduction_limit"]),
        )
    )

    disability_ded = z["disability_count"] * int(C["disability_deduction_per_person"])

    # tuition
    edu_ded = Int("edu_ded")
    opt.add(
        edu_ded
        == If(
            z["education_fee"] <= 0,
            0,
            If(
                z["education_fee"]
                >= z["education_count"] * int(C["education_deduction_per_student"]),
                z["education_count"] * int(C["education_deduction_per_student"]),
                z["education_fee"],
            ),
        )
    )

    # pre-school (113 年度起：第 1 名 150k，第 2 名起每名 225k)
    preschool_ded = Int("preschool_ded")
    opt.add(
        preschool_ded
        == If(
            z["preschool_count"] <= 0,
            0,
            If(
                z["preschool_count"] == 1,
                150_000,
                150_000 + (z["preschool_count"] - 1) * 225_000,
            ),
        )
    )

    # long-term care
    long_term_raw = z["long_term_care_count"] * int(C["long_term_care_deduction_per_person"])

    # rent for housing (cap)
    rent_lim_raw = Int("rent_lim_raw")
    opt.add(
        rent_lim_raw
        == If(
            z["rent_deduction"] >= int(C["rent_deduction_limit"]),
            int(C["rent_deduction_limit"]),
            z["rent_deduction"],
        )
    )

    total_people = z["cnt_under_70"] + z["cnt_over_70"]
    basic_need = total_people * adj(int(C["basic_living_exp_per_person"]))

    # ---- Scenario WITH (claim long-term care + rent) ----
    base_ded_with = (
        total_ex
        + chosen_ded
        + sav_inv
        + disability_ded
        + edu_ded
        + preschool_ded
        + long_term_raw
        + rent_lim_raw
    )

    basic_diff_with = Int("basic_diff_with")
    opt.add(basic_diff_with == If(basic_need > base_ded_with, basic_need - base_ded_with, 0))

    total_ded_with = base_ded_with + z["property_loss_deduction"] + basic_diff_with

    net_inc_with = Int("net_inc_with")
    opt.add(net_inc_with == total_income - total_ded_with)

    net_pos_with = Int("net_pos_with")
    opt.add(net_pos_with == If(net_inc_with < 0, 0, net_inc_with))

    # ---- Scenario NO (do NOT claim long-term care + rent) ----
    base_ded_no = (
        total_ex
        + chosen_ded
        + sav_inv
        + disability_ded
        + edu_ded
        + preschool_ded
    )

    basic_diff_no = Int("basic_diff_no")
    opt.add(basic_diff_no == If(basic_need > base_ded_no, basic_need - base_ded_no, 0))

    total_ded_no = base_ded_no + z["property_loss_deduction"] + basic_diff_no

    net_inc_no = Int("net_inc_no")
    opt.add(net_inc_no == total_income - total_ded_no)

    net_pos_no = Int("net_pos_no")
    opt.add(net_pos_no == If(net_inc_no < 0, 0, net_inc_no))

    # ---- Progressive tax function (resident) ----
    def progressive_tax_int(net_pos_int, prefix: str):
        x = ToReal(net_pos_int)
        tax_r = If(
            x <= int(C["bracket1_upper"]),
            x * C["bracket1_rate"] - int(C["bracket1_sub"]),
            If(
                x <= int(C["bracket2_upper"]),
                x * C["bracket2_rate"] - int(C["bracket2_sub"]),
                If(
                    x <= int(C["bracket3_upper"]),
                    x * C["bracket3_rate"] - int(C["bracket3_sub"]),
                    If(
                        x <= int(C["bracket4_upper"]),
                        x * C["bracket4_rate"] - int(C["bracket4_sub"]),
                        x * C["bracket5_rate"] - int(C["bracket5_sub"]),
                    ),
                ),
            ),
        )

        safe_r = If(tax_r < 0, 0, tax_r)
        tax_int = Int(f"{prefix}_tax")
        opt.add(tax_int == ToInt(safe_r))
        return tax_int

    tax_with = progressive_tax_int(net_pos_with, "with")
    tax_no = progressive_tax_int(net_pos_no, "no")

    # ---- 排富條款：扣除（房租＋長照）後，適用稅率 ≥ 20% -> 不得列報 ----
    # 依級距判斷：落在第 3 級距（20%）以上，即 net_pos_with > bracket2_upper
    disallow_lt_rent = net_pos_with > int(C["bracket2_upper"])

    net_pos = Int("net_pos")
    opt.add(net_pos == If(disallow_lt_rent, net_pos_no, net_pos_with))

    final_tax_z = Int("final_tax_z")
    opt.add(final_tax_z == If(disallow_lt_rent, tax_no, tax_with))

    # ---------- 6) Minimize ----------
    opt.minimize(final_tax_z)

    # ---------- 7) Solve ----------
    if opt.check() != sat:
        raise UnsatError("constraint set unsat")

    mdl = opt.model()
    tax_val = mdl[final_tax_z].as_long()

    # ---------- 8) Output params & diff ----------
    final_params: Dict[str, Dict[str, Any]] = {}
    diff: Dict[str, Dict[str, Any]] = {}

    for k, (zv, orig, _) in PARAMS.items():
        v = mdl[zv].as_long()
        final_params[k] = {"value": v, "type": "free" if k in free_vars else "fixed"}
        if v != orig:
            diff[k] = {"original": orig, "optimized": v, "difference": v - orig}

    return tax_val, mdl[net_pos].as_long(), final_params, diff


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
    """Public API.

    與 eTax 試算器比對時：
    - 會尊重你傳入的 is_departure（不再自動推導覆寫）
    - 會套用房租/長照排富條款（適用稅率 >= 20% 則不得扣除）
    """

    start_time = time.perf_counter()

    kwargs: Dict[str, Any] = {
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

    free_vars = kwargs.get("free_vars") or []

    # ---- 0.1) merge legacy constraints ----
    vc = kwargs.pop("variable_constraints", None)
    if constraints is None:
        constraints = vc or {}
    else:
        constraints = {**(vc or {}), **constraints}

    # ---- 0.2) free_vars name check ----
    _KNOWN_PARAM_NAMES: Set[str] = {
        "days_of_stay",
        "salary_self",
        "salary_spouse",
        "salary_dep",
        "interest_income",
        "interest_spouse",
        "interest_dep",
        "other_income",
        "other_income_spouse",
        "other_income_dep",
        "itemized_deduction",
        "property_loss_deduction",
        "rent_deduction",
        "cnt_under_70",
        "cnt_over_70",
        "disability_count",
        "education_count",
        "education_fee",
        "preschool_count",
        "long_term_care_count",
    }
    for v in free_vars:
        if v not in _KNOWN_PARAM_NAMES:
            raise ValueError(f"Unknown free var: {v}")

    # ---- 1) baseline (no free_vars / no constraints) ----
    base_tax, _, _, _ = _calculate_foreigner_tax_internal(
        **{k: v for k, v in kwargs.items() if k != "free_vars"},
        constraints={},
    )

    # ---- 2) baseline feasibility with constraints ----
    try:
        _calculate_foreigner_tax_internal(
            **{k: v for k, v in kwargs.items() if k != "free_vars"},
            constraints=constraints,
        )
        baseline_status = "sat"
        baseline_with = base_tax
    except UnsatError:
        baseline_status = "unsat"
        baseline_with = None

    mode = "baseline"
    status = baseline_status
    opt_tax: Optional[int] = base_tax
    params_out: Dict[str, Any] = {}
    diff_out: Dict[str, Any] = {}
    opt_tax_solver_value: Optional[int] = None

    # ---- 3) manual_free (optimize when free_vars provided) ----
    if free_vars:
        mode = "manual_free"
        try:
            opt_tax_solver_value, _, params_out, diff_out = _calculate_foreigner_tax_internal(
                constraints=constraints,
                **kwargs,
            )
            status = "sat"

            # Freeze final parameters and re-evaluate without constraints
            frozen_kwargs: Dict[str, Any] = {k: v["value"] for k, v in params_out.items()}
            frozen_kwargs["is_departure"] = kwargs.get("is_departure", False)
            frozen_kwargs["is_married"] = kwargs.get("is_married", False)
            frozen_kwargs["use_itemized"] = kwargs.get("use_itemized", False)

            tax_recalc, _, _, _ = _calculate_foreigner_tax_internal(constraints={}, **frozen_kwargs)
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
