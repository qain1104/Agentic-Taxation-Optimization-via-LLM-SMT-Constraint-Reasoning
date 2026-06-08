# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import calendar
import json
from datetime import date
from typing import Any, Dict, Optional, Sequence

from z3 import If, Int, Optimize, Real, RealVal, ToInt, ToReal

try:
    from .tracked_tax_core import BaseTrackedTaxCase, normalize_payload, run_tracked_analysis, write_report
except ImportError:
    from tracked_tax_core import BaseTrackedTaxCase, normalize_payload, run_tracked_analysis, write_report


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
    "bracket1_upper": 590_000,
    "bracket2_upper": 1_330_000,
    "bracket3_upper": 2_660_000,
    "bracket4_upper": 4_980_000,
    "bracket1_rate": "0.05",
    "bracket2_rate": "0.12",
    "bracket3_rate": "0.20",
    "bracket4_rate": "0.30",
    "bracket5_rate": "0.40",
    "bracket1_sub": 0,
    "bracket2_sub": 41_300,
    "bracket3_sub": 147_700,
    "bracket4_sub": 413_700,
    "bracket5_sub": 911_700,
}


BUILTIN_PAYLOADS = [
    {
        "case_id": "foreigner_case_0_departure_investment_split",
        "description": "單身外籍員工年薪 250 萬，另有投資收益 50 萬，可分配為利息所得與其他所得；去年居留 200 天後離境，目標最小化外僑綜所稅。",
        "payload": {
            "days_of_stay": 200,
            "is_departure": True,
            "is_married": False,
            "salary_self": 2_500_000,
            "salary_spouse": 0,
            "salary_dep": 0,
            "interest_income": 0,
            "interest_spouse": 0,
            "interest_dep": 0,
            "other_income": 0,
            "other_income_spouse": 0,
            "other_income_dep": 0,
            "cnt_under_70": 0,
            "cnt_over_70": 0,
            "use_itemized": False,
            "itemized_deduction": 0,
            "property_loss_deduction": 0,
            "disability_count": 0,
            "education_count": 0,
            "education_fee": 0,
            "preschool_count": 0,
            "long_term_care_count": 0,
            "rent_deduction": 0,
            "free_vars": ["interest_income", "other_income"],
            "constraints": {
                "interest_income + other_income": {"=": 500_000},
            },
        },
    },
    {
        "case_id": "foreigner_case_1_short_stay_rent_ltc",
        "description": "單身外籍人士年薪 150 萬，居留天數 90-180 天可調，租金扣除與長照扣除可調，目標最小化外僑綜所稅。",
        "payload": {
            "days_of_stay": 90,
            "is_departure": True,
            "is_married": False,
            "salary_self": 1_500_000,
            "salary_spouse": 0,
            "salary_dep": 0,
            "interest_income": 0,
            "interest_spouse": 0,
            "interest_dep": 0,
            "other_income": 0,
            "other_income_spouse": 0,
            "other_income_dep": 0,
            "cnt_under_70": 0,
            "cnt_over_70": 0,
            "use_itemized": False,
            "itemized_deduction": 0,
            "property_loss_deduction": 0,
            "disability_count": 0,
            "education_count": 0,
            "education_fee": 0,
            "preschool_count": 0,
            "long_term_care_count": 0,
            "rent_deduction": 240_000,
            "free_vars": ["days_of_stay", "rent_deduction", "long_term_care_count"],
            "constraints": {
                "days_of_stay": {">=": 90, "<=": 180},
                "rent_deduction": {"<=": 240_000},
                "long_term_care_count": {"<=": 1},
            },
        },
    },
]
BUILTIN_PAYLOAD = BUILTIN_PAYLOADS[0]["payload"]


class ForeignerIncomeTaxSMTCase(BaseTrackedTaxCase):
    """Tracked SMT model for foreigner comprehensive income tax.

    This mirrors the original foreigner_income_tax.py model but routes every
    constraint through BaseTrackedTaxCase.add(), so the model can produce
    UNSAT cores, release tests, COI combinations, and automatic release safety
    classifications.
    """

    objective_sense = "min"
    objective_label = "final_tax"

    def build(self):
        if self._built:
            return self
        self._built = True
        self.opt = Optimize()
        p = self.payload
        C = {**DEFAULTS, **{k: v for k, v in p.items() if k in DEFAULTS}}

        is_departure = bool(p.get("is_departure", False))
        is_married = bool(p.get("is_married", False))
        use_itemized = bool(p.get("use_itemized", False))

        param_names = [
            "salary_self", "salary_spouse", "salary_dep",
            "interest_income", "interest_spouse", "interest_dep",
            "other_income", "other_income_spouse", "other_income_dep",
            "itemized_deduction", "property_loss_deduction", "rent_deduction",
            "cnt_under_70", "cnt_over_70", "disability_count", "education_count",
            "education_fee", "preschool_count", "long_term_care_count", "days_of_stay",
        ]

        last_year = date.today().year - 1
        days_in_last_year = 366 if calendar.isleap(last_year) else 365

        for name in param_names:
            z = Real(f"{name}_z")
            checks = [("nonnegative", lambda v: v >= 0)]
            if name == "rent_deduction":
                checks.append(("upper_10000000", lambda v: v <= 10_000_000))
            if name == "days_of_stay":
                checks.append((f"upper_{days_in_last_year}", lambda v, d=days_in_last_year: v <= d))
            self._add_param(name, z, self._num(name), checks)
        self._bind_params()

        V = lambda n: self.params[n][0]
        R = lambda x: RealVal(str(x))

        # Branch guardrails from the original solver.
        if not is_married:
            self.add(V("salary_spouse") == 0, name="branch.unmarried.salary_spouse_zero", group="branch_rule")
            self.add(V("interest_spouse") == 0, name="branch.unmarried.interest_spouse_zero", group="branch_rule")
            self.add(V("other_income_spouse") == 0, name="branch.unmarried.other_income_spouse_zero", group="branch_rule")

        if int(self._num("cnt_under_70")) + int(self._num("cnt_over_70")) == 0:
            self.add(V("cnt_under_70") == 0, name="branch.no_dependents.cnt_under_70_zero", group="branch_rule")
            self.add(V("cnt_over_70") == 0, name="branch.no_dependents.cnt_over_70_zero", group="branch_rule")
            self.add(V("salary_dep") == 0, name="branch.no_dependents.salary_dep_zero", group="branch_rule")
            self.add(V("interest_dep") == 0, name="branch.no_dependents.interest_dep_zero", group="branch_rule")
            self.add(V("other_income_dep") == 0, name="branch.no_dependents.other_income_dep_zero", group="branch_rule")

        # User constraints are added after base variables/branch guards and before tax formulas.
        self._add_user_constraints()

        ratio_r = R(1) if not is_departure else V("days_of_stay") / R(days_in_last_year)
        self.vars["departure_ratio"] = ratio_r

        def adj_int(amount: int):
            return ToReal(ToInt(R(amount) * ratio_r))

        # Salary special deductions.
        self_sp = Real("self_sp")
        spouse_sp = Real("spouse_sp")
        dep_sp = Real("dep_sp")
        self_after = Real("self_after")
        spouse_after = Real("spouse_after")
        dep_after = Real("dep_after")
        total_income = Real("total_income")

        self.vars.update({
            "self_sp": self_sp, "spouse_sp": spouse_sp, "dep_sp": dep_sp,
            "self_after": self_after, "spouse_after": spouse_after, "dep_after": dep_after,
            "total_income": total_income,
        })

        SAL_MAX = int(C["salary_special_deduction_max"])
        self.add(self_sp == If(V("salary_self") >= SAL_MAX, R(SAL_MAX), V("salary_self")), name="law.salary_special.self_cap")
        self.add(spouse_sp == If(V("salary_spouse") >= SAL_MAX, R(SAL_MAX), V("salary_spouse")), name="law.salary_special.spouse_cap")
        self.add(dep_sp == If(V("salary_dep") >= SAL_MAX, R(SAL_MAX), V("salary_dep")), name="law.salary_special.dep_cap")
        self.add(self_after == V("salary_self") - self_sp, name="law.salary_after.self")
        self.add(spouse_after == V("salary_spouse") - spouse_sp, name="law.salary_after.spouse")
        self.add(dep_after == V("salary_dep") - dep_sp, name="law.salary_after.dep")
        self.add(self_after >= 0, name="law.salary_after.self_nonnegative")
        self.add(spouse_after >= 0, name="law.salary_after.spouse_nonnegative")
        self.add(dep_after >= 0, name="law.salary_after.dep_nonnegative")

        self.add(
            total_income == (
                self_after + spouse_after + dep_after
                + V("interest_income") + V("interest_spouse") + V("interest_dep")
                + V("other_income") + V("other_income_spouse") + V("other_income_dep")
            ),
            name="law.total_income",
        )

        total_ex = Real("total_ex")
        chosen_ded = Real("chosen_ded")
        interest_sum = Real("interest_sum")
        sav_inv = Real("sav_inv")
        edu_ded = Real("edu_ded")
        preschool_ded = Real("preschool_ded")
        rent_lim_raw = Real("rent_lim_raw")
        basic_need = Real("basic_need")
        basic_diff_with = Real("basic_diff_with")
        basic_diff_no = Real("basic_diff_no")
        net_inc_with = Real("net_inc_with")
        net_inc_no = Real("net_inc_no")
        net_pos_with = Real("net_pos_with")
        net_pos_no = Real("net_pos_no")
        net_pos = Real("net_pos")
        final_tax = Int("final_tax")

        self.vars.update({
            "total_ex": total_ex,
            "chosen_ded": chosen_ded,
            "interest_sum": interest_sum,
            "sav_inv": sav_inv,
            "edu_ded": edu_ded,
            "preschool_ded": preschool_ded,
            "rent_lim_raw": rent_lim_raw,
            "basic_need": basic_need,
            "basic_diff_with": basic_diff_with,
            "basic_diff_no": basic_diff_no,
            "net_inc_with": net_inc_with,
            "net_inc_no": net_inc_no,
            "net_pos_with": net_pos_with,
            "net_pos_no": net_pos_no,
            "net_pos": net_pos,
            "final_tax": final_tax,
        })
        self.final_tax_z = final_tax
        self.net_taxable_z = net_pos
        self.objective_z = final_tax

        self.add(
            total_ex == V("cnt_under_70") * adj_int(int(C["personal_exemption_under70"]))
            + V("cnt_over_70") * adj_int(int(C["personal_exemption_over70"])),
            name="law.exemption.prorated",
        )

        std_expr = adj_int(int(C["standard_deduction_married"] if is_married else C["standard_deduction_single"]))
        self.add(
            chosen_ded == If(use_itemized, If(V("itemized_deduction") >= std_expr, V("itemized_deduction"), std_expr), std_expr),
            name="law.chosen_deduction",
        )

        self.add(interest_sum == V("interest_income") + V("interest_spouse") + V("interest_dep"), name="law.interest_sum")
        self.add(
            sav_inv == If(interest_sum <= int(C["savings_investment_deduction_limit"]), interest_sum, R(int(C["savings_investment_deduction_limit"]))),
            name="law.savings_investment.cap",
        )

        disability_ded = V("disability_count") * R(int(C["disability_deduction_per_person"]))
        long_term_raw = V("long_term_care_count") * R(int(C["long_term_care_deduction_per_person"]))
        self.vars["disability_ded"] = disability_ded
        self.vars["long_term_raw"] = long_term_raw

        self.add(
            edu_ded == If(
                V("education_fee") <= 0,
                R(0),
                If(
                    V("education_fee") >= V("education_count") * R(int(C["education_deduction_per_student"])),
                    V("education_count") * R(int(C["education_deduction_per_student"])),
                    V("education_fee"),
                ),
            ),
            name="law.education_deduction",
        )

        self.add(
            preschool_ded == If(
                V("preschool_count") <= 0,
                R(0),
                If(V("preschool_count") == 1, R(150_000), R(150_000) + (V("preschool_count") - 1) * R(225_000)),
            ),
            name="law.preschool_deduction",
        )

        self.add(
            rent_lim_raw == If(V("rent_deduction") >= int(C["rent_deduction_limit"]), R(int(C["rent_deduction_limit"])), V("rent_deduction")),
            name="law.rent_deduction_cap",
        )

        total_people = V("cnt_under_70") + V("cnt_over_70")
        self.vars["total_people"] = total_people
        self.add(basic_need == total_people * adj_int(int(C["basic_living_exp_per_person"])), name="law.basic_living.need")

        base_ded_with = Real("base_ded_with")
        base_ded_no = Real("base_ded_no")
        total_ded_with = Real("total_ded_with")
        total_ded_no = Real("total_ded_no")
        self.vars.update({
            "base_ded_with": base_ded_with,
            "base_ded_no": base_ded_no,
            "total_ded_with": total_ded_with,
            "total_ded_no": total_ded_no,
        })

        self.add(
            base_ded_with == total_ex + chosen_ded + sav_inv + disability_ded + edu_ded + preschool_ded + long_term_raw + rent_lim_raw,
            name="law.base_deduction.with_ltc_rent",
        )
        self.add(basic_diff_with == If(basic_need > base_ded_with, basic_need - base_ded_with, R(0)), name="law.basic_living.diff_with")
        self.add(total_ded_with == base_ded_with + V("property_loss_deduction") + basic_diff_with, name="law.total_deduction.with_ltc_rent")
        self.add(net_inc_with == total_income - total_ded_with, name="law.net_income.with_ltc_rent")
        self.add(net_pos_with == If(net_inc_with < 0, R(0), net_inc_with), name="law.net_positive.with_ltc_rent")

        self.add(
            base_ded_no == total_ex + chosen_ded + sav_inv + disability_ded + edu_ded + preschool_ded,
            name="law.base_deduction.no_ltc_rent",
        )
        self.add(basic_diff_no == If(basic_need > base_ded_no, basic_need - base_ded_no, R(0)), name="law.basic_living.diff_no")
        self.add(total_ded_no == base_ded_no + V("property_loss_deduction") + basic_diff_no, name="law.total_deduction.no_ltc_rent")
        self.add(net_inc_no == total_income - total_ded_no, name="law.net_income.no_ltc_rent")
        self.add(net_pos_no == If(net_inc_no < 0, R(0), net_inc_no), name="law.net_positive.no_ltc_rent")

        def progressive_tax_real(net_pos_expr, prefix: str):
            tax_r = Real(f"{prefix}_tax_real")
            self.vars[f"{prefix}_tax_real"] = tax_r
            x = net_pos_expr
            expr = If(
                x <= int(C["bracket1_upper"]),
                x * R(C["bracket1_rate"]) - R(int(C["bracket1_sub"])),
                If(
                    x <= int(C["bracket2_upper"]),
                    x * R(C["bracket2_rate"]) - R(int(C["bracket2_sub"])),
                    If(
                        x <= int(C["bracket3_upper"]),
                        x * R(C["bracket3_rate"]) - R(int(C["bracket3_sub"])),
                        If(
                            x <= int(C["bracket4_upper"]),
                            x * R(C["bracket4_rate"]) - R(int(C["bracket4_sub"])),
                            x * R(C["bracket5_rate"]) - R(int(C["bracket5_sub"])),
                        ),
                    ),
                ),
            )
            self.add(tax_r == If(expr < 0, R(0), expr), name=f"law.progressive.{prefix}")
            return tax_r

        tax_with_r = progressive_tax_real(net_pos_with, "with_ltc_rent")
        tax_no_r = progressive_tax_real(net_pos_no, "no_ltc_rent")
        disallow_lt_rent = net_pos_with > int(C["bracket2_upper"])
        self.vars["disallow_ltc_rent_flag_as_netpos_condition"] = net_pos_with
        self.add(net_pos == If(disallow_lt_rent, net_pos_no, net_pos_with), name="law.high_income_rule.net_pos_choice")
        self.add(final_tax == ToInt(If(disallow_lt_rent, tax_no_r, tax_with_r)), name="law.high_income_rule.final_tax_choice")

        self._optimize()
        return self


def calculate_foreigner_income_tax_tracked(**payload):
    return ForeignerIncomeTaxSMTCase(payload).solve()


def load_payload(path: Optional[str], case_index: int) -> Dict[str, Any]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return normalize_payload(json.load(f))
    return dict(BUILTIN_PAYLOADS[case_index]["payload"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Tracked foreigner income tax UNSAT-core/release lab")
    ap.add_argument("--case", type=int, default=0, choices=range(len(BUILTIN_PAYLOADS)))
    ap.add_argument("--payload")
    ap.add_argument("--json-out", default="foreigner_income_tax_unsat_report.json")
    ap.add_argument("--md-out", default="foreigner_income_tax_unsat_report.md")
    ap.add_argument("--print-core", action="store_true")
    ap.add_argument("--no-auto-combinations", action="store_true")
    ap.add_argument("--core-only", action="store_true")
    ap.add_argument("--no-release-tests", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args(argv)
    if getattr(args, "probe_only", False):
        args.no_release_tests = True
        args.no_auto_combinations = True
        args.core_only = True
    payload = load_payload(args.payload, args.case)
    report = run_tracked_analysis(
        ForeignerIncomeTaxSMTCase,
        payload,
        auto_release=not getattr(args, "no_release_tests", False),
        auto_combinations=not getattr(args, "no_auto_combinations", False) and not getattr(args, "no_release_tests", False),
        include_non_core=not getattr(args, "core_only", False),
    )
    write_report(report, json_out=args.json_out, md_out=args.md_out, title="Foreigner Income Tax Tracked SMT Lab")

    base = report.get("base", {})
    probe = report.get("unsat_core_probe", {})
    gen = report.get("release_generation", {})
    print("=== Base solve ===")
    print(f"status: {base.get('status')}")
    print(f"objective: {base.get('objective_label')} / {base.get('objective_sense')}")
    print(f"optimum: {base.get('optimum')}")
    print(f"tax: {base.get('tax')}")
    print(f"net_taxable: {base.get('net_taxable')}")
    print("")
    print("=== UNSAT probe ===")
    print(f"goal: {probe.get('goal')}")
    print(f"status: {probe.get('status')}")
    print(f"tracked_assertions: {probe.get('tracked_assertion_count')}")
    print(f"core_size: {probe.get('core_size')}")
    print("")
    print("=== Release generation ===")
    print(f"source: {gen.get('source')}")
    print(f"single_tests: {gen.get('single_test_count')}")
    print(f"coi_combo_tests: {gen.get('coi_combo_count')}")
    print("")
    print("=== Release tests ===")
    for row in report.get("release_tests", []):
        cls = row.get("classification") or {}
        print(f"- {row['test']}: status={row['status']}, new_optimum={row.get('new_optimum')}, delta={row.get('delta_vs_base')}, class={cls.get('category')}")
    print(f"wrote: {args.json_out}")
    print(f"wrote: {args.md_out}")
    if args.print_core:
        for item in probe.get("core", []):
            print(json.dumps(item, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
