# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Sequence

from z3 import If, Int, Optimize, Real, RealVal, ToInt

try:
    from .tracked_tax_core import BaseTrackedTaxCase, normalize_payload, run_tracked_analysis, write_report
except ImportError:
    from tracked_tax_core import BaseTrackedTaxCase, normalize_payload, run_tracked_analysis, write_report


BUILTIN_PAYLOADS = [
    {
        "case_id": "business_income_case_0_manufacturing_min_tax",
        "description": "製造業營業收入 5,000 萬；銷貨折讓、營業成本、營業費用可調，目標最小化營所稅。",
        "payload": {
            "OperatingRevenueTotal": 50_000_000,
            "SalesReturn": 0,
            "SalesAllowance": 0,
            "OperatingCost": 0,
            "OperatingExpensesLosses": 0,
            "NonOperatingRevenueTotal": 0,
            "NonOperatingLossExpenses": 0,
            "Prev10LossDeduction": 0,
            "TaxIncentiveExempt": 0,
            "ExemptSecuritiesIncome": 0,
            "ExemptLandIncome": 0,
            "Article4_4HouseLandGain": 0,
            "is_full_year": True,
            "m_partial": 12,
            "free_vars": ["SalesAllowance", "OperatingCost", "OperatingExpensesLosses"],
            "constraints": {
                "SalesAllowance": {"<=": 8_000_000},
                "OperatingCost": {"<=": 22_000_000},
                "OperatingExpensesLosses": {"<=": 15_000_000},
                "SalesAllowance + OperatingCost": {"<=": 30_000_000},
            },
        },
    },
    {
        "case_id": "business_income_case_1_startup_partial_year_credits",
        "description": "新創公司營運 8 個月；前十年虧損扣除與投資抵減可調，目標最小化營所稅。",
        "payload": {
            "OperatingRevenueTotal": 24_000_000,
            "SalesReturn": 0,
            "SalesAllowance": 0,
            "OperatingCost": 10_000_000,
            "OperatingExpensesLosses": 6_000_000,
            "NonOperatingRevenueTotal": 500_000,
            "NonOperatingLossExpenses": 200_000,
            "Prev10LossDeduction": 0,
            "TaxIncentiveExempt": 0,
            "ExemptSecuritiesIncome": 0,
            "ExemptLandIncome": 0,
            "Article4_4HouseLandGain": 0,
            "is_full_year": False,
            "m_partial": 8,
            "free_vars": ["Prev10LossDeduction", "TaxIncentiveExempt"],
            "constraints": {
                "Prev10LossDeduction": {"<=": 3_000_000},
                "TaxIncentiveExempt": {"<=": 2_000_000},
                "Prev10LossDeduction + TaxIncentiveExempt": {"<=": "P * 0.6"},
            },
        },
    },
]
BUILTIN_PAYLOAD = BUILTIN_PAYLOADS[0]["payload"]


class BusinessIncomeTaxSMTCase(BaseTrackedTaxCase):
    objective_sense = "min"
    objective_label = "final_tax"

    def build(self):
        if self._built:
            return self
        self._built = True
        self.opt = Optimize()
        p = self.payload
        free_vars = set(p.get("free_vars") or [])
        is_full_year = bool(p.get("is_full_year", True))
        m_partial = int(float(p.get("m_partial", 12) or 12))

        names = [
            "OperatingRevenueTotal", "SalesReturn", "SalesAllowance", "OperatingCost",
            "OperatingExpensesLosses", "NonOperatingRevenueTotal", "NonOperatingLossExpenses",
            "Prev10LossDeduction", "TaxIncentiveExempt", "ExemptSecuritiesIncome",
            "ExemptLandIncome", "Article4_4HouseLandGain",
        ]
        for name in names:
            z = Real(f"{name}_z")
            self._add_param(name, z, self._num(name), [("nonnegative", lambda v: v >= 0)])
        self._bind_params()

        V = lambda n: self.params[n][0]
        OperatingRevenueNet = Real("OperatingRevenueNet")
        OperatingGrossProfit = Real("OperatingGrossProfit")
        OperatingNetProfit = Real("OperatingNetProfit")
        YearlyIncome = Real("YearlyIncome")
        P = Real("P")
        tax_real = Real("tax_real")
        final_tax = Int("final_tax")

        self.vars.update({
            "OperatingRevenueNet": OperatingRevenueNet,
            "OperatingGrossProfit": OperatingGrossProfit,
            "OperatingNetProfit": OperatingNetProfit,
            "YearlyIncome": YearlyIncome,
            "P": P,
            "tax_real": tax_real,
            "final_tax": final_tax,
        })
        self.final_tax_z = final_tax
        self.net_taxable_z = P
        self.objective_z = final_tax

        self.add(OperatingRevenueNet == V("OperatingRevenueTotal") - V("SalesReturn") - V("SalesAllowance"), name="law.operating_revenue_net")
        self.add(OperatingGrossProfit == OperatingRevenueNet - V("OperatingCost"), name="law.operating_gross_profit")
        self.add(OperatingNetProfit == OperatingGrossProfit - V("OperatingExpensesLosses"), name="law.operating_net_profit")
        self.add(YearlyIncome == OperatingNetProfit + V("NonOperatingRevenueTotal") - V("NonOperatingLossExpenses"), name="law.yearly_income")
        self.add(P == YearlyIncome - V("Prev10LossDeduction") - V("TaxIncentiveExempt") - V("ExemptSecuritiesIncome") - V("ExemptLandIncome") - V("Article4_4HouseLandGain"), name="law.taxable_income_P")

        FULL1, FULL2 = 120_000, 200_000
        if is_full_year:
            expr = If(P <= FULL1, 0, If(P <= FULL2, (P - FULL1) * RealVal("0.5"), P * RealVal("0.20")))
            self.add(tax_real == expr, name="law.full_year_tax_brackets")
        else:
            P_adj = Real("P_adj")
            self.vars["P_adj"] = P_adj
            self.add(P_adj == P * 12 / m_partial, name="law.partial_year_adjusted_income")
            expr = If(P_adj <= FULL1, 0, If(P_adj <= FULL2, (P_adj - FULL1) * RealVal("0.5") * RealVal(str(m_partial)) / 12, P_adj * RealVal("0.20") * RealVal(str(m_partial)) / 12))
            self.add(tax_real == expr, name="law.partial_year_tax_brackets")
        self.add(final_tax == ToInt(If(tax_real < 0, 0, tax_real)), name="law.final_tax_floor_nonnegative")
        self._add_user_constraints()
        self._optimize()
        return self


def calculate_business_income_tax_tracked(**payload):
    return BusinessIncomeTaxSMTCase(payload).solve()


def load_payload(path: Optional[str], case_index: int) -> Dict[str, Any]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return normalize_payload(json.load(f))
    return dict(BUILTIN_PAYLOADS[case_index]["payload"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Tracked business income tax UNSAT-core/release lab")
    ap.add_argument("--case", type=int, default=0, choices=range(len(BUILTIN_PAYLOADS)))
    ap.add_argument("--payload")
    ap.add_argument("--json-out", default="business_income_tax_unsat_report.json")
    ap.add_argument("--md-out", default="business_income_tax_unsat_report.md")
    ap.add_argument("--print-core", action="store_true")
    ap.add_argument("--no-auto-combinations", action="store_true")
    ap.add_argument("--core-only", action="store_true")
    ap.add_argument("--no-release-tests", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args(argv)
    if args.probe_only:
        args.no_release_tests = True
        args.no_auto_combinations = True
        args.core_only = True
    if getattr(args, "probe_only", False):
        args.no_release_tests = True
        args.no_auto_combinations = True
        args.core_only = True
    payload = load_payload(args.payload, args.case)
    report = run_tracked_analysis(
        BusinessIncomeTaxSMTCase,
        payload,
        auto_release=not getattr(args, "no_release_tests", False),
        auto_combinations=not getattr(args, "no_auto_combinations", False) and not getattr(args, "no_release_tests", False),
        include_non_core=not getattr(args, "core_only", False),
    )
    write_report(report, json_out=args.json_out, md_out=args.md_out, title="Business Income Tax Tracked SMT Lab")
    base = report.get("base", {})
    probe = report.get("unsat_core_probe", {})
    print("=== Base solve ===")
    print(f"status: {base.get('status')}")
    print(f"objective: {base.get('objective_label')} / {base.get('objective_sense')}")
    print(f"optimum: {base.get('optimum')}")
    print("=== UNSAT probe ===")
    print(f"goal: {probe.get('goal')}")
    print(f"status: {probe.get('status')}")
    print(f"core_size: {probe.get('core_size')}")
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
