from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from z3 import (
    Bool,
    If,
    Int,
    Optimize,
    Real,
    Solver,
    ToInt,
    ToReal,
    is_false,
    is_true,
    sat,
    set_param,
    unsat,
)

from z3.z3util import get_vars

# 114年度的計算規則參數
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

# 官網有的變數
KNOWN_PARAM_NAMES = {
    "salary_self", "salary_spouse", "salary_dep",
    "interest_income", "stock_dividend", "house_transaction_gain",
    "other_income", "itemized_deduction", "property_loss_deduction",
    "rent_deduction", "cnt_under_70", "cnt_over_70",
    "disability_count", "education_count", "education_fee",
    "preschool_count", "long_term_care_count",
}

# 內建測試案例：對齊 20-case batch 裡的「綜所」兩個案例。
# case 0：已婚夫妻薪資總額 300 萬，在夫妻間分配以最小化綜所稅。
# case 1：單身年薪 120 萬，扶養一位年長父親，身障與長照扣除擇一。
BUILTIN_PAYLOADS = [
    {
        "case_id": "income_case_0_married_salary_split",
        "description": "已婚夫妻今年薪資總額固定為 300 萬，可在本人與配偶間自由分配，目標最小化綜合所得稅。",
        "payload": {
            "is_married": True,
            "salary_self": 3_000_000,
            "salary_spouse": 0,
            "salary_dep": 0,
            "interest_income": 0,
            "stock_dividend": 0,
            "house_transaction_gain": 0,
            "other_income": 0,
            "cnt_under_70": 0,
            "cnt_over_70": 0,
            "preschool_count": 0,
            "long_term_care_count": 0,
            "education_count": 0,
            "disability_count": 0,
            "education_fee": 0,
            "itemized_deduction": 0,
            "property_loss_deduction": 0,
            "rent_deduction": 0,
            "free_vars": ["salary_self", "salary_spouse"],
            "constraints": {
                "salary_self + salary_spouse": {"==": 3_000_000.0}
            },
        },
    },
    {
        "case_id": "income_case_1_disability_vs_long_term_care",
        "description": "單身納稅人年薪 120 萬，扶養一位年長父親；身心障礙扣除與長照扣除只能擇一，目標最小化綜合所得稅。",
        "payload": {
            "is_married": False,
            "salary_self": 1_200_000,
            "salary_spouse": 0,
            "salary_dep": 0,
            "interest_income": 0,
            "stock_dividend": 0,
            "house_transaction_gain": 0,
            "other_income": 0,
            "cnt_under_70": 0,
            "cnt_over_70": 1,
            "preschool_count": 0,
            "long_term_care_count": 0,
            "education_count": 0,
            "disability_count": 0,
            "education_fee": 0,
            "itemized_deduction": 0,
            "property_loss_deduction": 0,
            "rent_deduction": 0,
            "free_vars": ["disability_count", "long_term_care_count"],
            "constraints": {
                "disability_count + long_term_care_count": {"==": 1},
                "disability_count": {">=": 0, "<=": 1},
                "long_term_care_count": {">=": 0, "<=": 1},
            },
        },
    },
]

# Backward-compatible default used when no --case is specified.
BUILTIN_PAYLOAD = BUILTIN_PAYLOADS[0]["payload"]

# 追蹤constraint的資料結構，方便後續分析和報告生成。
@dataclass
class TrackedConstraint:
    name: str
    group: str
    releasable: bool
    expr: Any
    expr_str: str
    note: str = ""
    active: bool = True

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "releasable": self.releasable,
            "expr": self.expr_str,
            "note": self.note,
            "active": self.active,
        }

# 把 constraint name 清成安全字串。
# 例如：
# salary_self + salary_spouse == 3000000
# 會被清成類似：
# salary_self_salary_spouse_3000000
def sanitize_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.:-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:180] or "constraint"

# 從 Z3 model 裡取出整數值
def z3_int_value(model, var, default: int = 0) -> int:
    v = model.eval(var, model_completion=True)
    try:
        return int(v.as_long())
    except Exception:
        try:
            return int(str(v))
        except Exception:
            return default

# 把傳進來的 constraint 字串轉成 Z3 expression。
class LinearExpressionParser:
    """Small AST parser for linear arithmetic over known Z3 Int variables."""

    def __init__(self, env: Dict[str, Any]):
        self.env = env

    # salary_self + salary_spouse 會變成：BinOp(Name("salary_self"), Add(), Name("salary_spouse"))
    def parse(self, text: str) -> Any:
        node = ast.parse(str(text), mode="eval").body
        return self._expr(node)


    def _expr(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            if node.id not in self.env:
                raise ValueError(f"unknown variable in expression: {node.id}")
            return self.env[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, float) and abs(node.value - int(node.value)) < 1e-9:
                return int(node.value)
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._expr(node.operand)
        if isinstance(node, ast.BinOp):
            lhs = self._expr(node.left)
            rhs = self._expr(node.right)
            if isinstance(node.op, ast.Add):
                return lhs + rhs
            if isinstance(node.op, ast.Sub):
                return lhs - rhs
            if isinstance(node.op, ast.Mult):
                return lhs * rhs
            if isinstance(node.op, ast.Div):
                return lhs / rhs
        raise ValueError(f"unsupported expression node: {ast.dump(node)}")

# 為了支援兩種 payload 格式用的
def normalize_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an app payload or direct kwargs into calculator kwargs."""
    if "user_params" in raw and isinstance(raw.get("user_params"), dict):
        out = dict(raw["user_params"])
        # Some app payloads put constraints/free_vars at top level.
        if raw.get("constraints") and "constraints" not in out:
            out["constraints"] = raw["constraints"]
        if raw.get("free_vars") and "free_vars" not in out:
            out["free_vars"] = raw["free_vars"]
        return out
    return dict(raw)

# 主體 SMT case 類別，負責建構 Z3 Optimize 問題、追蹤 constraint、以及求解。
class IncomeTaxSMTCase:
    def __init__(
        self,
        payload: Dict[str, Any],
        *,
        objective: str = "best",
        released: Optional[Set[str]] = None,
        constants: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.payload = normalize_payload(payload)
        self.objective = objective
        self.released = set(released or set())
        self.constants = {**DEFAULTS, **(constants or {})}
        self.opt = Optimize()
        self.tracked: List[TrackedConstraint] = []
        self._name_counts: Dict[str, int] = {}
        self.params: Dict[str, Tuple[Any, int, List[Tuple[str, Any]]]] = {}
        self.vars: Dict[str, Any] = {}
        self.final_tax_z = None
        self.net_taxable_income_z = None
        self._built = False

    def _unique_name(self, base: str) -> str:
        base = sanitize_name(base)
        n = self._name_counts.get(base, 0)
        self._name_counts[base] = n + 1
        return base if n == 0 else f"{base}__{n}"

    def add(self, expr: Any, *, name: str, group: str = "tax_law", releasable: bool = False, note: str = "") -> None:
        nm = self._unique_name(name)
        active = nm not in self.released and name not in self.released
        rec = TrackedConstraint(
            name=nm,
            group=group,
            releasable=bool(releasable),
            expr=expr,
            expr_str=str(expr),
            note=note,
            active=active,
        )
        self.tracked.append(rec)
        if active:
            self.opt.add(expr)

    def _param_value(self, name: str, default: int = 0) -> int:
        v = self.payload.get(name, default)
        if v is None:
            return default
        if isinstance(v, bool):
            return int(v)
        try:
            return int(float(v))
        except Exception:
            return default

    def build(self) -> "IncomeTaxSMTCase":
        if self._built:
            return self
        self._built = True
        p = self.payload
        c = self.constants
        is_married = bool(p.get("is_married", False))
        use_itemized = bool(p.get("use_itemized", False))
        free_vars = set(p.get("free_vars") or [])
        constraints = p.get("constraints") or {}

        # Parameter variables.
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
        cnt_under_70_z = Int("cnt_under_70_z")
        cnt_over_70_z = Int("cnt_over_70_z")
        disability_count_z = Int("disability_count_z")
        education_count_z = Int("education_count_z")
        education_fee_z = Int("education_fee_z")
        preschool_count_param_z = Int("preschool_count_z_param")
        long_term_care_count_z = Int("long_term_care_count_z")

        self.vars.update({
            "salary_self": salary_self_z,
            "salary_spouse": salary_spouse_z,
            "salary_dep": salary_dep_z,
            "interest_income": interest_z,
            "stock_dividend": stock_div_z,
            "house_transaction_gain": house_gain_z,
            "other_income": other_income_z,
            "itemized_deduction": itemized_ded_z,
            "property_loss_deduction": prop_loss_ded_z,
            "rent_deduction": rent_ded_z,
            "cnt_under_70": cnt_under_70_z,
            "cnt_over_70": cnt_over_70_z,
            "disability_count": disability_count_z,
            "education_count": education_count_z,
            "education_fee": education_fee_z,
            "preschool_count": preschool_count_param_z,
            "long_term_care_count": long_term_care_count_z,
        })

        # The third item contains domain constraints as (label_suffix, expr_lambda).
        self.params = {
            "salary_self": (salary_self_z, self._param_value("salary_self"), [("nonnegative", lambda v: v >= 0)]),
            "salary_spouse": (salary_spouse_z, self._param_value("salary_spouse"), [("nonnegative", lambda v: v >= 0)]),
            "salary_dep": (salary_dep_z, self._param_value("salary_dep"), [("nonnegative", lambda v: v >= 0)]),
            "interest_income": (interest_z, self._param_value("interest_income"), [("nonnegative", lambda v: v >= 0)]),
            "stock_dividend": (stock_div_z, self._param_value("stock_dividend"), [("nonnegative", lambda v: v >= 0)]),
            "house_transaction_gain": (house_gain_z, self._param_value("house_transaction_gain"), [("nonnegative", lambda v: v >= 0)]),
            "other_income": (other_income_z, self._param_value("other_income"), [("nonnegative", lambda v: v >= 0)]),
            "itemized_deduction": (itemized_ded_z, self._param_value("itemized_deduction"), [("nonnegative", lambda v: v >= 0)]),
            "property_loss_deduction": (prop_loss_ded_z, self._param_value("property_loss_deduction"), [("nonnegative", lambda v: v >= 0)]),
            "rent_deduction": (rent_ded_z, self._param_value("rent_deduction"), [("nonnegative", lambda v: v >= 0), ("upper_10000000", lambda v: v <= 10_000_000)]),
            "cnt_under_70": (cnt_under_70_z, self._param_value("cnt_under_70"), [("nonnegative", lambda v: v >= 0)]),
            "cnt_over_70": (cnt_over_70_z, self._param_value("cnt_over_70"), [("nonnegative", lambda v: v >= 0)]),
            "disability_count": (disability_count_z, self._param_value("disability_count"), [("nonnegative", lambda v: v >= 0)]),
            "education_count": (education_count_z, self._param_value("education_count"), [("nonnegative", lambda v: v >= 0)]),
            "education_fee": (education_fee_z, self._param_value("education_fee"), [("nonnegative", lambda v: v >= 0)]),
            "preschool_count": (preschool_count_param_z, self._param_value("preschool_count"), [("nonnegative", lambda v: v >= 0)]),
            "long_term_care_count": (long_term_care_count_z, self._param_value("long_term_care_count"), [("nonnegative", lambda v: v >= 0)]),
        }

        # Fixed values vs free variables.
        for name, (zv, value, builtins) in self.params.items():
            if name not in free_vars:
                self.add(zv == value, name=f"fixed.{name}", group="fixed_input", releasable=True)
            for suffix, fn in builtins:
                self.add(fn(zv), name=f"domain.{name}.{suffix}", group="domain", releasable=False)

        if not is_married:
            self.add(salary_spouse_z == 0, name="branch.unmarried.spouse_salary_zero", group="branch_rule", releasable=False)

        if self._param_value("cnt_under_70") + self._param_value("cnt_over_70") == 0:
            self.add(cnt_under_70_z == 0, name="branch.no_dependents.cnt_under_70_zero", group="branch_rule", releasable=False)
            self.add(cnt_over_70_z == 0, name="branch.no_dependents.cnt_over_70_zero", group="branch_rule", releasable=False)
            self.add(salary_dep_z == 0, name="branch.no_dependents.salary_dep_zero", group="branch_rule", releasable=False)

        self._add_user_constraints(constraints)

        # Derived variables.
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
        total_income_no_div_z = Int("total_income_no_div_z")
        net_taxable_income_no_div_z = Int("net_taxable_income_no_div_z")
        net_taxable_nonneg_no_div_z = Int("net_taxable_nonneg_no_div_z")
        rate_calc_r = Real("rate_calc_r")
        rate_calc_no_div_r = Real("rate_calc_no_div_r")
        progressive_tax_with_div_z = Int("progressive_tax_with_div_z")
        progressive_tax_no_div_z = Int("progressive_tax_no_div_z")
        dividend_credit_raw_z = Int("dividend_credit_raw_z")
        dividend_credit_z = Int("dividend_credit_z")
        combined_net_tax_z = Int("combined_net_tax_z")
        combined_tax_due_z = Int("combined_tax_due_z")
        combined_refund_z = Int("combined_refund_z")
        dividend_tax_z = Int("dividend_tax_z")
        separate_net_tax_z = Int("separate_net_tax_z")
        separate_tax_due_z = Int("separate_tax_due_z")
        separate_refund_z = Int("separate_refund_z")
        final_tax_z = Int("final_tax_z")

        self.final_tax_z = final_tax_z
        self.net_taxable_income_z = net_taxable_income_z
        self.vars.update({
            "total_income_z": total_income_z,
            "total_income_no_div_z": total_income_no_div_z,
            "total_exemption_z": total_exemption_z,
            "total_deduction_z": total_deduction_z,
            "net_taxable_income_z": net_taxable_income_z,
            "net_taxable_nonneg_z": net_taxable_nonneg_z,
            "net_taxable_income_no_div_z": net_taxable_income_no_div_z,
            "net_taxable_nonneg_no_div_z": net_taxable_nonneg_no_div_z,
            "progressive_tax_with_div_z": progressive_tax_with_div_z,
            "progressive_tax_no_div_z": progressive_tax_no_div_z,
            "dividend_credit_raw_z": dividend_credit_raw_z,
            "dividend_credit_z": dividend_credit_z,
            "combined_net_tax_z": combined_net_tax_z,
            "dividend_tax_z": dividend_tax_z,
            "separate_net_tax_z": separate_net_tax_z,
            "final_tax_z": final_tax_z,
        })

        self.add(
            self_salary_special_deduction_z == If(salary_self_z >= c["salary_special_deduction_max"], c["salary_special_deduction_max"], salary_self_z),
            name="law.salary_special.self_cap",
        )
        self.add(
            spouse_salary_special_deduction_z == If(salary_spouse_z >= c["salary_special_deduction_max"], c["salary_special_deduction_max"], salary_spouse_z),
            name="law.salary_special.spouse_cap",
        )
        self.add(
            dep_salary_special_ded_z == If(salary_dep_z >= c["salary_special_deduction_max"], c["salary_special_deduction_max"], salary_dep_z),
            name="law.salary_special.dep_cap",
        )
        self.add(self_salary_after_ded_z == salary_self_z - self_salary_special_deduction_z, name="law.salary_after_ded.self")
        self.add(spouse_salary_after_ded_z == salary_spouse_z - spouse_salary_special_deduction_z, name="law.salary_after_ded.spouse")
        self.add(dep_salary_after_ded_z == salary_dep_z - dep_salary_special_ded_z, name="law.salary_after_ded.dep")
        self.add(self_salary_after_ded_z >= 0, name="law.salary_after_ded.self_nonnegative")
        self.add(spouse_salary_after_ded_z >= 0, name="law.salary_after_ded.spouse_nonnegative")
        self.add(dep_salary_after_ded_z >= 0, name="law.salary_after_ded.dep_nonnegative")

        self.add(total_income_z == (
            self_salary_after_ded_z + spouse_salary_after_ded_z + dep_salary_after_ded_z
            + interest_z + stock_div_z + house_gain_z + other_income_z
        ), name="law.total_income")
        self.add(total_income_no_div_z == (
            self_salary_after_ded_z + spouse_salary_after_ded_z + dep_salary_after_ded_z
            + interest_z + house_gain_z + other_income_z
        ), name="law.total_income_no_dividend")
        self.add(total_exemption_z == (
            cnt_under_70_z * c["personal_exemption_under70"]
            + cnt_over_70_z * c["personal_exemption_over70"]
        ), name="law.exemption")
        self.add(standard_deduction_z == (c["standard_deduction_married"] if is_married else c["standard_deduction_single"]), name="law.standard_deduction")
        self.add(chosen_deduction_z == If(use_itemized, itemized_ded_z, standard_deduction_z), name="law.chosen_deduction")

        interest_plus_div_z = Int("interest_plus_div_z")
        self.add(interest_plus_div_z == interest_z, name="law.savings.interest_only")
        self.add(
            savings_investment_deduction_z == If(
                interest_plus_div_z <= c["savings_investment_deduction_limit"],
                interest_plus_div_z,
                c["savings_investment_deduction_limit"],
            ),
            name="law.savings.cap",
        )
        self.add(disability_deduction_z == disability_count_z * c["disability_deduction_per_person"], name="law.disability_deduction")
        self.add(
            education_deduction_z == If(
                education_fee_z <= 0,
                0,
                If(
                    education_fee_z >= education_count_z * c["education_deduction_per_student"],
                    education_count_z * c["education_deduction_per_student"],
                    education_fee_z,
                ),
            ),
            name="law.education_deduction",
        )
        self.add(preschool_count_z == preschool_count_param_z, name="law.preschool_count_alias")
        self.add(
            preschool_deduction_z == If(
                preschool_count_z <= 0,
                0,
                If(preschool_count_z == 1, 150_000, 150_000 + (preschool_count_z - 1) * 225_000),
            ),
            name="law.preschool_deduction",
        )
        self.add(long_term_care_deduction_z == long_term_care_count_z * c["long_term_care_deduction_per_person"], name="law.long_term_care_deduction")
        self.add(rent_deduction_z_lim == If(rent_ded_z >= c["rent_deduction_limit"], c["rent_deduction_limit"], rent_ded_z), name="law.rent_deduction_cap")
        self.add(total_people_z == cnt_under_70_z + cnt_over_70_z, name="law.total_people")
        self.add(basic_living_exp_total_z == total_people_z * c["basic_living_exp_per_person"], name="law.basic_living.total")
        self.add(
            base_deductions_var_z == (
                total_exemption_z + chosen_deduction_z + savings_investment_deduction_z
                + disability_deduction_z + education_deduction_z + preschool_deduction_z
                + long_term_care_deduction_z + rent_deduction_z_lim
            ),
            name="law.base_deductions",
        )
        self.add(
            basic_living_exp_diff_z == If(
                basic_living_exp_total_z > base_deductions_var_z,
                basic_living_exp_total_z - base_deductions_var_z,
                0,
            ),
            name="law.basic_living.diff",
        )
        self.add(total_deduction_z == base_deductions_var_z + prop_loss_ded_z + basic_living_exp_diff_z, name="law.total_deduction")
        self.add(net_taxable_income_z == total_income_z - total_deduction_z, name="law.net_taxable_income")
        self.add(net_taxable_nonneg_z == If(net_taxable_income_z < 0, 0, net_taxable_income_z), name="law.net_taxable_nonnegative")
        self.add(net_taxable_income_no_div_z == total_income_no_div_z - total_deduction_z, name="law.net_taxable_income_no_dividend")
        self.add(net_taxable_nonneg_no_div_z == If(net_taxable_income_no_div_z < 0, 0, net_taxable_income_no_div_z), name="law.net_taxable_no_dividend_nonnegative")

        x = ToReal(net_taxable_nonneg_z)
        self.add(
            rate_calc_r == If(
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
            ),
            name="law.progressive.rate_with_dividend",
        )
        safe_tax_r = If(rate_calc_r < 0, 0, rate_calc_r)
        self.add(progressive_tax_with_div_z == ToInt(safe_tax_r), name="law.progressive.tax_with_dividend")

        x2 = ToReal(net_taxable_nonneg_no_div_z)
        self.add(
            rate_calc_no_div_r == If(
                x2 <= c["bracket1_upper"],
                (x2 * c["bracket1_rate"]) - c["bracket1_sub"],
                If(
                    x2 <= c["bracket2_upper"],
                    (x2 * c["bracket2_rate"]) - c["bracket2_sub"],
                    If(
                        x2 <= c["bracket3_upper"],
                        (x2 * c["bracket3_rate"]) - c["bracket3_sub"],
                        If(
                            x2 <= c["bracket4_upper"],
                            (x2 * c["bracket4_rate"]) - c["bracket4_sub"],
                            (x2 * c["bracket5_rate"]) - c["bracket5_sub"],
                        ),
                    ),
                ),
            ),
            name="law.progressive.rate_no_dividend",
        )
        safe_tax_no_div_r = If(rate_calc_no_div_r < 0, 0, rate_calc_no_div_r)
        self.add(progressive_tax_no_div_z == ToInt(safe_tax_no_div_r), name="law.progressive.tax_no_dividend")

        self.add(dividend_credit_raw_z == (stock_div_z * 85) / 1000, name="law.dividend.credit_raw")
        self.add(dividend_credit_z == If(dividend_credit_raw_z > 80_000, 80_000, dividend_credit_raw_z), name="law.dividend.credit_cap")
        self.add(combined_net_tax_z == progressive_tax_with_div_z - dividend_credit_z, name="law.combined.net_tax")
        self.add(combined_tax_due_z == If(combined_net_tax_z < 0, 0, combined_net_tax_z), name="law.combined.tax_due")
        self.add(combined_refund_z == If(combined_net_tax_z < 0, -combined_net_tax_z, 0), name="law.combined.refund")
        self.add(dividend_tax_z == (stock_div_z * 28) / 100, name="law.separate.dividend_tax_28")
        self.add(separate_net_tax_z == progressive_tax_no_div_z + dividend_tax_z, name="law.separate.net_tax")
        self.add(separate_tax_due_z == separate_net_tax_z, name="law.separate.tax_due")
        self.add(separate_refund_z == 0, name="law.separate.refund")

        best_net_tax_z = If(combined_net_tax_z <= separate_net_tax_z, combined_net_tax_z, separate_net_tax_z)
        if self.objective == "combined":
            self.add(final_tax_z == combined_net_tax_z, name="objective_link.final_tax_combined", group="objective_link")
        elif self.objective == "separate":
            self.add(final_tax_z == separate_net_tax_z, name="objective_link.final_tax_separate", group="objective_link")
        else:
            self.add(final_tax_z == best_net_tax_z, name="objective_link.final_tax_best", group="objective_link")

        self.opt.minimize(final_tax_z)
        return self

    def _add_user_constraints(self, constraints: Dict[str, Any]) -> None:
        if not constraints:
            return
        parser = LinearExpressionParser(self.vars)
        op_map = {
            "==": lambda a, b: a == b,
            "=": lambda a, b: a == b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
        }
        for lhs, ops in constraints.items():
            if not isinstance(ops, dict):
                continue
            lhs_expr = parser.parse(lhs)
            for op, rhs in ops.items():
                if op not in op_map:
                    raise ValueError(f"unsupported op: {op}")
                rhs_values = rhs if isinstance(rhs, (list, tuple)) else [rhs]
                for rhs_one in rhs_values:
                    if isinstance(rhs_one, str):
                        rhs_expr = parser.parse(rhs_one)
                    else:
                        if isinstance(rhs_one, float) and abs(rhs_one - int(rhs_one)) < 1e-9:
                            rhs_expr = int(rhs_one)
                        else:
                            rhs_expr = rhs_one
                    expr = op_map[op](lhs_expr, rhs_expr)
                    pretty_rhs = str(int(rhs_one)) if isinstance(rhs_one, (int, float)) and abs(float(rhs_one) - int(rhs_one)) < 1e-9 else str(rhs_one)
                    if "salary_self" in lhs and "other_income" in lhs and op in ("==", "=") and pretty_rhs == "3000000":
                        nm = "user_constraint.gross_income_sum_eq_3000000"
                    else:
                        nm = f"user_constraint.{sanitize_name(lhs)}.{sanitize_name(op)}.{sanitize_name(pretty_rhs)}"
                    self.add(expr, name=nm, group="user_constraint", releasable=True, note=f"{lhs} {op} {rhs_one}")

    def solve(self) -> Dict[str, Any]:
        self.build()
        res = self.opt.check()
        if res != sat:
            return {"status": str(res), "released": sorted(self.released)}
        model = self.opt.model()
        fields = {}
        for name, (zv, original, _) in self.params.items():
            fields[name] = z3_int_value(model, zv)
        derived_keys = [
            "total_income_z",
            "total_income_no_div_z",
            "total_exemption_z",
            "total_deduction_z",
            "net_taxable_income_z",
            "net_taxable_nonneg_z",
            "net_taxable_income_no_div_z",
            "net_taxable_nonneg_no_div_z",
            "progressive_tax_with_div_z",
            "progressive_tax_no_div_z",
            "dividend_credit_raw_z",
            "dividend_credit_z",
            "combined_net_tax_z",
            "dividend_tax_z",
            "separate_net_tax_z",
            "final_tax_z",
        ]
        derived = {k: z3_int_value(model, self.vars[k]) for k in derived_keys if k in self.vars}
        return {
            "status": "sat",
            "objective": self.objective,
            "optimum": z3_int_value(model, self.final_tax_z),
            "net_taxable_income": z3_int_value(model, self.net_taxable_income_z),
            "fields": fields,
            "derived": derived,
            "released": sorted(self.released),
        }
    # step1: 抽unsat core，看看哪些 constraint 是導致無法達成更優解的原因。
    def prove_no_strictly_better(self, optimum: int, *, minimize_core: bool = True) -> Dict[str, Any]:
        self.build()
        if minimize_core:
            try:
                set_param("smt.core.minimize", True)
            except Exception:
                pass
        s = Solver()
        s.set(unsat_core=True)
        name_to_rec: Dict[str, TrackedConstraint] = {}
        for rec in self.tracked:
            if not rec.active:
                continue
            lit_name = rec.name
            lit = Bool(lit_name)
            s.assert_and_track(rec.expr, lit)
            name_to_rec[lit_name] = rec
        goal_name = f"GOAL.final_tax_le_{int(optimum) - 1}"
        s.assert_and_track(self.final_tax_z <= int(optimum) - 1, Bool(goal_name))
        res = s.check()
        if res != unsat:
            return {
                "status": str(res),
                "expected": "unsat",
                "goal": f"final_tax_z <= {int(optimum) - 1}",
                "core": [],
                "core_size": 0,
                "tracked_assertion_count": len(name_to_rec),
            }
        core_names = [c.decl().name() for c in s.unsat_core()]
        core_json: List[Dict[str, Any]] = []
        summary: Dict[str, int] = {}
        for name in core_names:
            if name == goal_name:
                item = {
                    "name": name,
                    "group": "GOAL",
                    "releasable": False,
                    "expr": f"final_tax_z <= {int(optimum) - 1}",
                    "note": "strictly better than optimum",
                }
            else:
                rec = name_to_rec[name]
                item = rec.to_json()
            summary[item["group"]] = summary.get(item["group"], 0) + 1
            core_json.append(item)
        return {
            "status": "unsat",
            "goal": f"final_tax_z <= {int(optimum) - 1}",
            "core_size": len(core_names),
            "tracked_assertion_count": len(name_to_rec),
            "core_summary": summary,
            "core_names": core_names,
            "core": core_json,
            "releasable_core": [x["name"] for x in core_json if x.get("releasable")],
            "non_core_releasable": [
                rec.name for rec in self.tracked
                if rec.active and rec.releasable and rec.name not in set(core_names)
            ],
        }


from itertools import combinations

RELEASE_POLICY_OVERRIDES: Dict[str, Dict[str, Any]] = {}



RELEASE_FAMILIES: Dict[str, List[str]] = {
    # These are useful for pairwise combination tests. For example,
    # education_fee alone has no effect when education_count is fixed at 0,
    # and education_count alone has no effect when education_fee is fixed at 0.
    "education": ["fixed.education_fee", "fixed.education_count"],
    "rent": ["fixed.rent_deduction"],
    "dependents": ["fixed.cnt_under_70", "fixed.cnt_over_70"],
    "care": ["fixed.long_term_care_count", "fixed.disability_count"],
    "preschool": ["fixed.preschool_count"],
    "loss": ["fixed.property_loss_deduction"],
}


def default_release_tests() -> List[Tuple[str, List[str]]]:
    """Manual fallback release tests.

    The new default path is automatic generation from the unsat core.
    This function is kept as a stable fallback and as documentation of the
    original hand-picked tests.
    """
    return [
        ("release gross income equality", ["user_constraint.gross_income_sum_eq_3000000"]),
        ("release property_loss_deduction fixed value", ["fixed.property_loss_deduction"]),
        ("release rent_deduction fixed value", ["fixed.rent_deduction"]),
        ("release cnt_under_70 fixed value", ["fixed.cnt_under_70"]),
        ("release cnt_over_70 fixed value", ["fixed.cnt_over_70"]),
        ("release disability_count fixed value", ["fixed.disability_count"]),
        ("release education_fee fixed value", ["fixed.education_fee"]),
        ("release preschool_count fixed value", ["fixed.preschool_count"]),
        ("release long_term_care_count fixed value", ["fixed.long_term_care_count"]),
        ("release itemized_deduction fixed value", ["fixed.itemized_deduction"]),
        ("release education_count fixed value", ["fixed.education_count"]),
        ("release education_fee and education_count together", ["fixed.education_fee", "fixed.education_count"]),
    ]

# step2 把其中可以放寬的constraints自動變成single release test
def infer_release_tests_from_probe(
    probe: Dict[str, Any],
    *,
    include_non_core: bool = True,
) -> List[Tuple[str, List[str]]]:
    """Generate single-constraint release tests from the unsat-core probe.

    probe["releasable_core"] contains releasable constraints that appear in
    the unsat core. These are direct bottleneck candidates.

    probe["non_core_releasable"] contains releasable constraints that did not
    appear in this particular core. These are useful proof-reduction candidates
    or no-effect checks.
    """
    tests: List[Tuple[str, List[str]]] = []
    seen: Set[str] = set()

    def add_test(name: str, source: str) -> None:
        if not name or name in seen:
            return
        seen.add(name)
        tests.append((f"auto release {name} ({source})", [name]))

    for name in probe.get("releasable_core") or []:
        add_test(str(name), "core")

    if include_non_core:
        for name in probe.get("non_core_releasable") or []:
            add_test(str(name), "non_core")

    return tests


def infer_family_combo_tests(
    available_names: Sequence[str],
    *,
    max_combos: int = 20,
) -> List[Tuple[str, List[str]]]:
    """Generate domain-aware pairwise/family release tests.

    Instead of trying all combinations, only combine variables that belong to
    the same family. This avoids a combinatorial explosion and gives more
    interpretable tests.
    """
    available = set(available_names)
    out: List[Tuple[str, List[str]]] = []
    seen: Set[Tuple[str, ...]] = set()

    for family, names in RELEASE_FAMILIES.items():
        present = [n for n in names if n in available]
        if len(present) < 2:
            continue

        # Pairwise tests within the same family.
        for combo in combinations(sorted(present), 2):
            key = tuple(combo)
            if key in seen:
                continue
            seen.add(key)
            out.append((f"auto family combo {family}: {' + '.join(combo)}", list(combo)))
            if len(out) >= max_combos:
                return out

        # Full family test, if there are more than two fields.
        if len(present) > 2:
            key = tuple(sorted(present))
            if key not in seen:
                seen.add(key)
                out.append((f"auto family combo {family}: all", list(key)))
                if len(out) >= max_combos:
                    return out

    return out


def z3_var_names(expr: Any) -> Set[str]:
    """Return Z3 variable names used inside an expression."""
    try:
        return {v.decl().name() for v in get_vars(expr)}
    except Exception:
        return set()

# step3 COI由build_coi_index()和infer_coi_combo_tests()兩個function組成
# build_coi_index()會先從tracked constraints裡面建立constraint-to-variable和variable-to-constraint的index，還有固定變數到constraint的index
# infer_coi_combo_tests()則是從seed constraint開始，找到相關的constraint和變數，最後產生release combo test。
def build_coi_index(
    tracked: List[TrackedConstraint],
    *,
    active_only: bool = True,
) -> Dict[str, Any]:
    """Build constraint/variable indexes used by COI expansion.

    Returns:
        constraint_to_vars:
            constraint name -> set of Z3 variable names used in its expression.
        var_to_constraints:
            Z3 variable name -> constraints that mention the variable.
        fixed_var_to_constraint:
            Z3 variable name -> fixed.* constraint that fixes that variable.
        constraint_by_name:
            constraint name -> TrackedConstraint.
    """

    constraint_to_vars: Dict[str, Set[str]] = {}
    var_to_constraints: Dict[str, Set[str]] = {}
    fixed_var_to_constraint: Dict[str, str] = {}
    constraint_by_name: Dict[str, TrackedConstraint] = {}

    for rec in tracked:
        if active_only and not rec.active:
            continue

        constraint_by_name[rec.name] = rec
        vars_in_expr = z3_var_names(rec.expr)
        constraint_to_vars[rec.name] = vars_in_expr

        for v in vars_in_expr:
            var_to_constraints.setdefault(v, set()).add(rec.name)

        if rec.group == "fixed_input" and rec.releasable and len(vars_in_expr) == 1:
            only_var = next(iter(vars_in_expr))
            fixed_var_to_constraint[only_var] = rec.name

    return {
        "constraint_to_vars": constraint_to_vars,
        "var_to_constraints": var_to_constraints,
        "fixed_var_to_constraint": fixed_var_to_constraint,
        "constraint_by_name": constraint_by_name,
    }


def infer_coi_combo_tests(
    tracked: List[TrackedConstraint],
    seed_release_names: Sequence[str],
    *,
    max_depth: int = 1,
    max_combo_size: int = 4,
    max_tests: int = 30,
    include_groups: Optional[Set[str]] = None,
) -> List[Tuple[str, List[str]]]:
    """Generate release-combination tests using COI expansion.

    A seed is usually a fixed.* constraint from releasable_core or
    non_core_releasable. For each seed, this function:
      1. Finds the Z3 variable fixed by the seed.
      2. Finds tax/model constraints that mention that variable.
      3. Finds other variables in those constraints.
      4. Maps those variables back to fixed.* constraints if possible.
      5. Emits a combo test containing the seed and those related fixed.* items.

    max_depth=1 intentionally means "same formula layer only". Increasing the
    depth can quickly include many unrelated deduction/tax constraints.
    """

    if include_groups is None:
        include_groups = {"tax_law", "objective_link", "branch_rule"}

    idx = build_coi_index(tracked, active_only=True)
    constraint_to_vars: Dict[str, Set[str]] = idx["constraint_to_vars"]
    var_to_constraints: Dict[str, Set[str]] = idx["var_to_constraints"]
    fixed_var_to_constraint: Dict[str, str] = idx["fixed_var_to_constraint"]
    constraint_by_name: Dict[str, TrackedConstraint] = idx["constraint_by_name"]

    seed_set = {str(x) for x in seed_release_names}
    tests: List[Tuple[str, List[str]]] = []
    seen: Set[Tuple[str, ...]] = set()

    for seed in sorted(seed_set):
        seed_rec = constraint_by_name.get(seed)
        if seed_rec is None:
            continue

        if seed_rec.group != "fixed_input":
            continue

        frontier_vars = set(constraint_to_vars.get(seed, set()))
        visited_vars = set(frontier_vars)
        visited_constraints: Set[str] = set()
        related_fixed: Set[str] = {seed}

        for _depth in range(max_depth):
            next_frontier_vars: Set[str] = set()

            for var_name in frontier_vars:
                for cname in var_to_constraints.get(var_name, set()):
                    if cname in visited_constraints:
                        continue
                    visited_constraints.add(cname)

                    crec = constraint_by_name.get(cname)
                    if crec is None:
                        continue

                    # Only expand through formula-like constraints. This avoids
                    # user constraints and fixed constraints making COI too wide.
                    if crec.group not in include_groups:
                        continue

                    vars_in_c = constraint_to_vars.get(cname, set())

                    # If variables in this formula correspond to fixed inputs,
                    # include them in the combo.
                    for v2 in vars_in_c:
                        fixed_name = fixed_var_to_constraint.get(v2)
                        if fixed_name is not None:
                            related_fixed.add(fixed_name)

                    # Continue expansion only if max_depth > 1.
                    for v2 in vars_in_c:
                        if v2 not in visited_vars:
                            visited_vars.add(v2)
                            next_frontier_vars.add(v2)

            frontier_vars = next_frontier_vars

        if len(related_fixed) <= 1:
            continue

        combo = tuple(sorted(related_fixed))
        if len(combo) > max_combo_size:
            combo = combo[:max_combo_size]

        if combo in seen:
            continue
        seen.add(combo)

        label = f"auto COI combo from {seed}: {' + '.join(combo)}"
        tests.append((label, list(combo)))

        if len(tests) >= max_tests:
            break

    return tests

def infer_pairwise_combo_tests(
    single_test_rows: List[Dict[str, Any]],
    *,
    max_combos: int = 30,
) -> List[Tuple[str, List[str]]]:
    """Generate pairwise release tests from single-release results.

    Heuristic:
    - Only combine fixed.* constraints.
    - Prefer constraints whose single release has no effect.
    - Exclude user_constraint.* because releasing those changes problem meaning.
    - Also include known family combos such as education_fee + education_count.
    """
    zero_effect_names: List[str] = []
    all_fixed_names: List[str] = []

    for row in single_test_rows:
        released = row.get("released") or []
        if len(released) != 1:
            continue
        name = str(released[0])
        if not name.startswith("fixed."):
            continue
        all_fixed_names.append(name)
        if row.get("delta_vs_base") == 0:
            zero_effect_names.append(name)

    tests: List[Tuple[str, List[str]]] = []
    seen: Set[Tuple[str, ...]] = set()

    # First, use family-aware combos from all available fixed candidates.
    for label, names in infer_family_combo_tests(all_fixed_names, max_combos=max_combos):
        key = tuple(sorted(names))
        if key not in seen:
            seen.add(key)
            tests.append((label, names))
            if len(tests) >= max_combos:
                return tests

    # Then, try pairwise combos among zero-effect fixed releases.
    for a, b in combinations(sorted(set(zero_effect_names)), 2):
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        tests.append((f"auto no-effect combo release {a} + {b}", [a, b]))
        if len(tests) >= max_combos:
            break

    return tests



# ---------------------------------------------------------------------------
# Automatic release-policy inference
# ---------------------------------------------------------------------------
# The functions in this block replace the old hand-written RELEASE_POLICY table.
# They infer how to interpret a released constraint by inspecting:
#   - its name, e.g. fixed.cnt_under_70 or fixed.rent_deduction,
#   - the Z3 variable fixed by that constraint,
#   - domain constraints such as v >= 0 or v <= K,
#   - law-level cap formulas such as law.rent_deduction_cap, and
#   - the release-test result itself.


def fixed_field_name(release_name: str) -> Optional[str]:
    """Return the portal field name for a fixed.* constraint."""
    if not release_name.startswith("fixed."):
        return None
    return release_name[len("fixed."):]


def infer_field_type(field: Optional[str]) -> str:
    """Infer a coarse field type from a portal-aligned field name."""
    if not field:
        return "unknown"

    f = field.lower()

    # Counts are factual quantities and should not be freely optimized.
    if f.startswith("cnt_") or f.endswith("_count") or "count" in f:
        return "count_fact"

    # Deductions/fees/losses are monetary claims; they often need receipts or
    # factual evidence. They may be safe only when a law-level cap is present.
    if any(tok in f for tok in ["deduction", "fee", "loss", "rent"]):
        return "monetary_claim"

    # Income-like variables are monetary quantities, but in this lab many are
    # free decision variables rather than fixed release candidates.
    if any(tok in f for tok in ["salary", "income", "dividend", "gain"]):
        return "monetary_income"

    return "unknown_fixed_input"


def _parse_simple_upper_bound(expr_str: str, var_name: str) -> Optional[int]:
    """Best-effort parser for simple upper-bound domain expressions.

    This intentionally uses string matching only for diagnostics. It does not
    affect solver soundness.
    """
    # Examples from Z3 string output are usually:
    #   rent_ded_z <= 10000000
    #   10000000 >= rent_ded_z
    patterns = [
        rf"\b{re.escape(var_name)}\b\s*<=\s*(-?\d+)",
        rf"(-?\d+)\s*>=\s*\b{re.escape(var_name)}\b",
    ]
    for pat in patterns:
        m = re.search(pat, expr_str)
        if m:
            try:
                # The variable can be on either side depending on pattern.
                nums = [g for g in m.groups() if g is not None]
                return int(nums[-1])
            except Exception:
                return None
    return None


def build_release_inference_context(tracked: List[TrackedConstraint]) -> Dict[str, Any]:
    """Build metadata used to classify release-test results automatically.

    For each fixed.* constraint, we record:
      - the underlying Z3 variable name,
      - simple domain upper bounds involving that variable,
      - tax-law formulas mentioning that variable,
      - whether any law-level formula looks like a cap/floor/limit rule.
    """
    idx = build_coi_index(tracked, active_only=True)
    constraint_to_vars: Dict[str, Set[str]] = idx["constraint_to_vars"]
    var_to_constraints: Dict[str, Set[str]] = idx["var_to_constraints"]
    constraint_by_name: Dict[str, TrackedConstraint] = idx["constraint_by_name"]

    fixed_info: Dict[str, Dict[str, Any]] = {}

    for cname, rec in constraint_by_name.items():
        if rec.group != "fixed_input" or not rec.releasable:
            continue

        vars_in_fixed = list(constraint_to_vars.get(cname, set()))
        z3_var = vars_in_fixed[0] if len(vars_in_fixed) == 1 else None
        field = fixed_field_name(cname)
        field_type = infer_field_type(field)

        domain_upper_bounds: List[int] = []
        law_constraints: List[str] = []
        cap_constraints: List[str] = []

        if z3_var is not None:
            for used_by in var_to_constraints.get(z3_var, set()):
                used_rec = constraint_by_name.get(used_by)
                if used_rec is None:
                    continue

                if used_rec.group == "domain":
                    ub = _parse_simple_upper_bound(used_rec.expr_str, z3_var)
                    if ub is not None:
                        domain_upper_bounds.append(ub)

                if used_rec.group in {"tax_law", "branch_rule", "objective_link"}:
                    law_constraints.append(used_rec.name)
                    lname = used_rec.name.lower()
                    estr = used_rec.expr_str.lower()
                    # This is a heuristic diagnostic: a cap-like law formula is
                    # named or written with cap/limit/min-style If conditions.
                    if (
                        "cap" in lname
                        or "limit" in lname
                        or "upper" in lname
                        or ("if(" in estr and (">=" in estr or "<=" in estr))
                    ):
                        cap_constraints.append(used_rec.name)

        fixed_info[cname] = {
            "field": field,
            "field_type": field_type,
            "z3_var": z3_var,
            "domain_upper_bounds": sorted(set(domain_upper_bounds)),
            "law_constraints": sorted(set(law_constraints)),
            "cap_constraints": sorted(set(cap_constraints)),
            "has_domain_upper_bound": bool(domain_upper_bounds),
            "has_law_cap": bool(cap_constraints),
        }

    return {
        "fixed_info": fixed_info,
        "coi_index": idx,
    }

# step4: 看release後稅額有沒有下降、結果合不合理、有沒有上界，最後分成useful, unsafe, no effect, semantic change
def infer_release_policy(
    release_name: str,
    *,
    inference_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Infer how to interpret one released constraint.

    This function replaces the old hand-written RELEASE_POLICY lookup.  It is
    intentionally conservative: if a field is factual, count-like, or an
    unbounded claim, it is marked unsafe unless the model structure shows a
    strong law-level cap.
    """
    # Optional escape hatch for exceptional legal rules.
    if release_name in RELEASE_POLICY_OVERRIDES:
        out = dict(RELEASE_POLICY_OVERRIDES[release_name])
        out["source"] = "override"
        return out

    if release_name.startswith("user_constraint."):
        return {
            "kind": "semantic_defining_constraint",
            "safe": False,
            "source": "name_rule",
            "reason": "This is a user-level constraint; releasing it may change the problem meaning.",
        }

    if not release_name.startswith("fixed."):
        return {
            "kind": "non_fixed_or_unknown",
            "safe": False,
            "source": "name_rule",
            "reason": "Only fixed input constraints are considered safe release candidates by default.",
        }

    ctx = inference_context or {}
    info = (ctx.get("fixed_info") or {}).get(release_name, {})
    field = info.get("field") or fixed_field_name(release_name)
    field_type = info.get("field_type") or infer_field_type(field)
    has_law_cap = bool(info.get("has_law_cap"))
    has_domain_upper = bool(info.get("has_domain_upper_bound"))
    cap_constraints = info.get("cap_constraints") or []
    domain_upper_bounds = info.get("domain_upper_bounds") or []

    if field_type == "count_fact":
        return {
            "kind": "fact_field_needs_bound",
            "safe": False,
            "source": "variable_type_inference",
            "field_type": field_type,
            "reason": "Count fields are factual quantities and cannot be freely optimized without user-provided bounds.",
        }

    if has_law_cap:
        return {
            "kind": "bounded_by_law_formula",
            "safe": True,
            "source": "law_cap_inference",
            "field_type": field_type,
            "cap_constraints": cap_constraints,
            "domain_upper_bounds": domain_upper_bounds,
            "reason": "The released variable appears in a law-level cap/limit formula, so the release is structurally bounded.",
        }

    if field_type == "monetary_claim":
        return {
            "kind": "claim_amount_needs_bound",
            "safe": False,
            "source": "variable_type_inference",
            "field_type": field_type,
            "domain_upper_bounds": domain_upper_bounds,
            "reason": "This monetary claim needs an actual amount, receipt, or explicit upper bound before it can be safely released.",
        }

    if has_domain_upper:
        # Domain upper bounds alone may be too loose, but still useful metadata.
        return {
            "kind": "domain_bounded_but_needs_review",
            "safe": False,
            "source": "domain_bound_inference",
            "field_type": field_type,
            "domain_upper_bounds": domain_upper_bounds,
            "reason": "The variable has a domain upper bound, but no law-level cap was detected; review before recommending.",
        }

    return {
        "kind": "unknown_needs_review",
        "safe": False,
        "source": "fallback",
        "field_type": field_type,
        "reason": "No reliable law-level bound was inferred for this release candidate.",
    }


def infer_combined_release_policy(
    released: Sequence[str],
    *,
    inference_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Infer policy for a single or combination release."""
    policies = [infer_release_policy(x, inference_context=inference_context) for x in released]

    if any(p.get("kind") == "semantic_defining_constraint" for p in policies):
        return {
            "kind": "semantic_change",
            "safe": False,
            "source": "combined_policy",
            "component_policies": policies,
            "reason": "At least one released constraint changes the semantic definition of the case.",
        }

    if policies and all(p.get("safe") is True for p in policies):
        return {
            "kind": "bounded_release",
            "safe": True,
            "source": "combined_policy",
            "component_policies": policies,
            "reason": "All released constraints are inferred to be structurally bounded.",
        }

    return {
        "kind": "needs_domain_bound",
        "safe": False,
        "source": "combined_policy",
        "component_policies": policies,
        "reason": "At least one released constraint is factual, unbounded, or requires user-provided bounds.",
    }

def classify_release_result(
    row: Dict[str, Any],
    base_optimum: int,
    *,
    inference_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify one release-test result using automatic policy inference."""
    released = [str(x) for x in (row.get("released") or [])]
    status = row.get("status")
    new_optimum = row.get("new_optimum")
    delta = row.get("delta_vs_base")
    net_taxable = row.get("net_taxable_income")

    inferred_policy = infer_combined_release_policy(
        released,
        inference_context=inference_context,
    )

    if status != "sat" or not isinstance(new_optimum, int):
        return {
            "category": "infeasible_or_unknown",
            "is_recommendable": False,
            "policy": inferred_policy,
            "reason": "Release test did not return a SAT optimum.",
        }

    if inferred_policy.get("kind") == "semantic_change":
        return {
            "category": "semantic_change",
            "is_recommendable": False,
            "policy": inferred_policy,
            "reason": inferred_policy.get("reason") or "This release changes the case definition.",
        }

    if delta == 0:
        return {
            "category": "no_effect",
            "is_recommendable": False,
            "policy": inferred_policy,
            "reason": "Releasing this constraint does not improve the objective in this case.",
        }

    if isinstance(delta, int) and delta > 0:
        return {
            "category": "worse",
            "is_recommendable": False,
            "policy": inferred_policy,
            "reason": "Releasing this constraint worsens the objective.",
        }

    if isinstance(delta, int) and delta < 0:
        # Strong signal that the solver exploited an unbounded/factual field.
        # This is independent of the inferred policy and acts as a safety brake.
        looks_extreme = (
            new_optimum <= -80_000
            or (isinstance(net_taxable, int) and net_taxable <= 0)
        )

        if inferred_policy.get("safe") is True and not looks_extreme:
            return {
                "category": "useful_bounded_release",
                "is_recommendable": True,
                "tax_saving": -delta,
                "policy": inferred_policy,
                "reason": "Objective improves and automatic inference found a structural law/domain bound.",
            }

        return {
            "category": "unsafe_needs_domain_bound",
            "is_recommendable": False,
            "tax_saving": -delta,
            "policy": inferred_policy,
            "reason": "Objective improves, but automatic inference did not establish a safe bound or the result looks extreme.",
        }

    return {
        "category": "unknown",
        "is_recommendable": False,
        "policy": inferred_policy,
        "reason": "Could not classify this release result.",
    }


def _execute_release_tests(
    payload: Dict[str, Any],
    *,
    objective: str,
    optimum: int,
    release_tests: List[Tuple[str, List[str]]],
    inference_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run a list of release tests and classify their results."""
    tests_out: List[Dict[str, Any]] = []

    for label, names in release_tests:
        c = IncomeTaxSMTCase(payload, objective=objective, released=set(names))
        sol = c.solve()

        row = {
            "test": label,
            "released": names,
            "status": sol.get("status"),
            "new_optimum": sol.get("optimum"),
            "delta_vs_base": (sol.get("optimum") - optimum) if isinstance(sol.get("optimum"), int) else None,
            "net_taxable_income": sol.get("net_taxable_income"),
            "selected_fields": sol.get("fields"),
            "derived": sol.get("derived"),
        }
        row["classification"] = classify_release_result(
            row,
            optimum,
            inference_context=inference_context,
        )
        tests_out.append(row)

    return tests_out

    # 主流程：
    # 抽unsat core -> 
    # 把可以放寬的constraint變成single release test -> 
    # COI 找某個變數在哪些公式裡被用到，再找同一個公式裡的其他相關變數，所以能自動抓出education_fee和education_count這種組合release ->
    # 執行release test並且分類結果
def run_analysis(
    payload: Dict[str, Any],
    *,
    objective: str = "best",
    release_tests: Optional[List[Tuple[str, List[str]]]] = None,
    auto_release: bool = True,
    auto_combinations: bool = True,
    include_non_core: bool = True,
) -> Dict[str, Any]:
    base_case = IncomeTaxSMTCase(payload, objective=objective)
    base = base_case.solve()
    if base.get("status") != "sat":
        return {"base": base, "error": "base case is not sat"}

    optimum = int(base["optimum"])
    # step1
    probe = base_case.prove_no_strictly_better(optimum)

    release_inference_context = build_release_inference_context(base_case.tracked)


    if release_tests is not None:
        tests_to_run = release_tests
        release_source = "manual"
    # 目前走這裡
    elif auto_release:
        tests_to_run = infer_release_tests_from_probe(probe, include_non_core=include_non_core)
        release_source = "auto_from_unsat_core"
    else:
        tests_to_run = default_release_tests()
        release_source = "default_hardcoded"

    tests_out = _execute_release_tests(
        payload,
        objective=objective,
        optimum=optimum,
        release_tests=tests_to_run,
        inference_context=release_inference_context,
    )

    combo_tests: List[Tuple[str, List[str]]] = []
    combo_out: List[Dict[str, Any]] = []
    coi_combo_count = 0
    heuristic_combo_count = 0

    # step2
    if auto_combinations:
        single_release_names: List[str] = []
        for _label, _names in tests_to_run:
            for _n in _names:
                single_release_names.append(str(_n))

        coi_combo_tests = infer_coi_combo_tests(
            base_case.tracked,
            single_release_names,
            max_depth=1,
            max_combo_size=4,
            max_tests=30,
        )
        heuristic_combo_tests = infer_pairwise_combo_tests(tests_out)

        seen_combo_keys: Set[Tuple[str, ...]] = set()
        for label, names in coi_combo_tests:
            key = tuple(sorted(names))
            if key in seen_combo_keys:
                continue
            seen_combo_keys.add(key)
            combo_tests.append((label, names))
            coi_combo_count += 1

        for label, names in heuristic_combo_tests:
            key = tuple(sorted(names))
            if key in seen_combo_keys:
                continue
            seen_combo_keys.add(key)
            combo_tests.append((label, names))
            heuristic_combo_count += 1

        if combo_tests:
            combo_out = _execute_release_tests(
                payload,
                objective=objective,
                optimum=optimum,
                release_tests=combo_tests,
                inference_context=release_inference_context,
            )
            tests_out.extend(combo_out)

    recommended: List[Dict[str, Any]] = []
    unsafe: List[Dict[str, Any]] = []
    no_effect: List[Dict[str, Any]] = []
    semantic_change: List[Dict[str, Any]] = []
    worse: List[Dict[str, Any]] = []
    unknown: List[Dict[str, Any]] = []

    for row in tests_out:
        cls = row.get("classification") or {}
        cat = cls.get("category")
        if cat == "useful_bounded_release":
            recommended.append(row)
        elif cat == "unsafe_needs_domain_bound":
            unsafe.append(row)
        elif cat == "no_effect":
            no_effect.append(row)
        elif cat == "semantic_change":
            semantic_change.append(row)
        elif cat == "worse":
            worse.append(row)
        else:
            unknown.append(row)

    return {
        "base": base,
        "unsat_core_probe": probe,
        "release_generation": {
            "source": release_source,
            "single_test_count": len(tests_to_run),
            "combo_test_count": len(combo_tests),
            "coi_combo_count": coi_combo_count,
            "heuristic_combo_count": heuristic_combo_count,
            "combo_strategy": "coi_plus_heuristic",
            "include_non_core": include_non_core,
            "release_policy": "automatic_domain_bound_and_variable_type_inference",
        },
        "release_tests": tests_out,
        "release_summary": {
            "recommended": recommended,
            "unsafe_needs_domain_bound": unsafe,
            "no_effect": no_effect,
            "semantic_change": semantic_change,
            "worse": worse,
            "unknown": unknown,
        },
    }


def money(v: Any) -> str:
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.0f}"
    return str(v)


def _classification_label(row: Dict[str, Any]) -> str:
    cls = row.get("classification") or {}
    cat = cls.get("category") or ""
    return str(cat)


def render_markdown(report: Dict[str, Any]) -> str:
    base = report.get("base", {})
    probe = report.get("unsat_core_probe", {})
    release_generation = report.get("release_generation", {})
    lines = []
    lines.append("# Income Tax SMT Unsat Core Lab")
    lines.append("")
    lines.append("## Base solve")
    lines.append(f"- status: `{base.get('status')}`")
    lines.append(f"- objective: `{base.get('objective')}`")
    lines.append(f"- optimum final_tax_z: **{money(base.get('optimum'))}**")
    lines.append(f"- net_taxable_income_z: **{money(base.get('net_taxable_income'))}**")
    lines.append("")
    lines.append("## Strictly-better UNSAT probe")
    lines.append(f"- added goal: `{probe.get('goal')}`")
    lines.append(f"- status: `{probe.get('status')}`")
    lines.append(f"- tracked assertions: `{probe.get('tracked_assertion_count')}`")
    lines.append(f"- core size: `{probe.get('core_size')}`")
    lines.append("")
    lines.append("### Core summary")
    lines.append("| group | count |")
    lines.append("|---|---:|")
    for group, count in sorted((probe.get("core_summary") or {}).items()):
        lines.append(f"| `{group}` | {count} |")
    lines.append("")
    lines.append("### Releasable constraints in this core")
    rel = probe.get("releasable_core") or []
    if rel:
        for name in rel:
            lines.append(f"- `{name}`")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("### Releasable constraints not in this core")
    non = probe.get("non_core_releasable") or []
    if non:
        for name in non:
            lines.append(f"- `{name}`")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Release generation")
    lines.append(f"- source: `{release_generation.get('source')}`")
    lines.append(f"- single release tests: `{release_generation.get('single_test_count')}`")
    lines.append(f"- combination tests: `{release_generation.get('combo_test_count')}`")
    lines.append(f"- COI combination tests: `{release_generation.get('coi_combo_count')}`")
    lines.append(f"- heuristic combination tests: `{release_generation.get('heuristic_combo_count')}`")
    lines.append(f"- combo strategy: `{release_generation.get('combo_strategy')}`")
    lines.append(f"- release policy: `{release_generation.get('release_policy')}`")
    lines.append(f"- include non-core releasable: `{release_generation.get('include_non_core')}`")
    lines.append("")

    lines.append("## Release tests")
    lines.append("| test | released | status | new optimum | delta vs base | net taxable | classification |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for row in report.get("release_tests") or []:
        names = ", ".join(f"`{n}`" for n in row.get("released") or [])
        lines.append(
            f"| {row.get('test')} | {names} | `{row.get('status')}` | "
            f"{money(row.get('new_optimum'))} | {money(row.get('delta_vs_base'))} | "
            f"{money(row.get('net_taxable_income'))} | `{_classification_label(row)}` |"
        )

    lines.append("")
    lines.append("## Release interpretation")
    summary = report.get("release_summary") or {}

    def _section(title: str, key: str, empty: str) -> None:
        rows = summary.get(key) or []
        lines.append("")
        lines.append(f"### {title}")
        if not rows:
            lines.append(f"- {empty}")
            return
        for row in rows:
            cls = row.get("classification") or {}
            released = ", ".join(f"`{x}`" for x in row.get("released") or [])
            reason = cls.get("reason") or ""
            if cls.get("tax_saving") is not None:
                lines.append(
                    f"- {released}: tax saving **{money(cls.get('tax_saving'))}**, "
                    f"new optimum **{money(row.get('new_optimum'))}**. {reason}"
                )
            else:
                lines.append(
                    f"- {released}: new optimum **{money(row.get('new_optimum'))}**, "
                    f"delta **{money(row.get('delta_vs_base'))}**. {reason}"
                )

    _section("Recommended bounded release candidates", "recommended", "none")
    _section("Unsafe release candidates requiring bounds", "unsafe_needs_domain_bound", "none")
    _section("No-effect release tests", "no_effect", "none")
    _section("Semantic-changing releases", "semantic_change", "none")
    _section("Worse release tests", "worse", "none")
    _section("Unknown / infeasible release tests", "unknown", "none")

    lines.append("")
    lines.append("## Full core names")
    for name in probe.get("core_names") or []:
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append("> Reminder: Z3 unsat cores are not unique; use group/name patterns rather than exact ordering.")
    return "\n".join(lines)


def get_builtin_case(case_index: int) -> Dict[str, Any]:
    """Return one built-in case entry by index."""
    if case_index < 0 or case_index >= len(BUILTIN_PAYLOADS):
        raise ValueError(f"Unknown --case {case_index}; available cases: 0..{len(BUILTIN_PAYLOADS) - 1}")
    return BUILTIN_PAYLOADS[case_index]


def load_payload(path: Optional[str], *, case_index: int = 0) -> Dict[str, Any]:
    if not path:
        return dict(get_builtin_case(case_index)["payload"])
    with open(path, "r", encoding="utf-8") as f:
        return normalize_payload(json.load(f))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run income-tax unsat-core and release-candidate tests.")
    ap.add_argument("--payload", help="JSON file containing either user_params or direct calculator kwargs.")
    ap.add_argument("--case", type=int, default=0, help="Built-in case index to run when --payload is not provided.")
    ap.add_argument("--objective", default="best", choices=["best", "combined", "separate"])
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--md-out", default=None)
    ap.add_argument("--print-core", action="store_true", help="Print full core entries to stdout.")
    ap.add_argument("--manual-release-tests", action="store_true", help="Use the old hard-coded release-test list.")
    ap.add_argument("--no-auto-combinations", action="store_true", help="Disable inferred pairwise/family release tests.")
    ap.add_argument("--core-only", action="store_true", help="Do not include non-core releasable constraints in auto release tests.")
    args = ap.parse_args(argv)

    payload = load_payload(args.payload, case_index=args.case)
    case_meta = None if args.payload else get_builtin_case(args.case)
    report = run_analysis(
        payload,
        objective=args.objective,
        release_tests=default_release_tests() if args.manual_release_tests else None,
        auto_release=not args.manual_release_tests,
        auto_combinations=not args.no_auto_combinations,
        include_non_core=not args.core_only,
    )

    default_tag = "custom" if args.payload else f"case{args.case}"
    json_path = Path(args.json_out or f"income_tax_unsat_report_{default_tag}.json")
    md_path = Path(args.md_out or f"income_tax_unsat_report_{default_tag}.md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    base = report.get("base", {})
    probe = report.get("unsat_core_probe", {})
    if case_meta:
        print("=== Built-in case ===")
        print(f"case: {args.case}")
        print(f"case_id: {case_meta.get('case_id')}")
        print(f"description: {case_meta.get('description')}")
        print("")
    print("=== Base solve ===")
    print(f"status: {base.get('status')}")
    print(f"objective: {base.get('objective')}")
    print(f"optimum: {money(base.get('optimum'))}")
    print(f"net_taxable_income: {money(base.get('net_taxable_income'))}")
    print("")
    print("=== UNSAT probe ===")
    print(f"goal: {probe.get('goal')}")
    print(f"status: {probe.get('status')}")
    print(f"tracked_assertions: {probe.get('tracked_assertion_count')}")
    print(f"core_size: {probe.get('core_size')}")
    print(f"core_summary: {json.dumps(probe.get('core_summary'), ensure_ascii=False)}")
    print("")
    print("=== Release generation ===")
    gen = report.get("release_generation", {})
    print(f"source: {gen.get('source')}")
    print(f"single_tests: {gen.get('single_test_count')}")
    print(f"combo_tests: {gen.get('combo_test_count')}")
    print(f"coi_combo_tests: {gen.get('coi_combo_count')}")
    print(f"heuristic_combo_tests: {gen.get('heuristic_combo_count')}")
    print(f"combo_strategy: {gen.get('combo_strategy')}")
    print(f"release_policy: {gen.get('release_policy')}")
    print("")
    print("=== Release tests ===")
    for row in report.get("release_tests", []):
        cls = row.get("classification") or {}
        print(
            f"- {row['test']}: status={row['status']}, "
            f"new_optimum={money(row.get('new_optimum'))}, "
            f"delta={money(row.get('delta_vs_base'))}, "
            f"class={cls.get('category')}"
        )
    print("")
    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")

    if args.print_core:
        print("")
        print("=== Full core ===")
        for item in probe.get("core", []):
            print(json.dumps(item, ensure_ascii=False, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
