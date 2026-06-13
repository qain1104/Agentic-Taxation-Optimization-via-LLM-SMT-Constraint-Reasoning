# -*- coding: utf-8 -*-
"""
estate_tax_tracked.py

Tracked / UNSAT-core-ready version of estate_tax.py.
It preserves the original estate-tax formula, but every constraint is added
through self.add(...), so it supports:
- optimality proof: final_tax_z <= optimum - 1
- unsat core extraction
- automatic release tests
- COI combination release tests
- automatic domain-bound / variable-type inference
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from z3 import And, If, Int, Optimize, Real, RealVal, ToInt, sat

from tracked_tax_core import BaseTrackedTaxCase, normalize_payload, run_tracked_analysis, write_report


class UnsatError(Exception):
    pass


def _norm_period(x) -> int:
    s = str(x).strip().translate(str.maketrans("０１２３４５", "012345"))
    cn = {"一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5}
    n = int(s) if s.isdigit() else cn.get(s)
    if n not in (1, 2, 3, 4, 5):
        raise ValueError(f"death_period must be 1..5 (got {x!r})")
    return n


BUILTIN_PAYLOADS = [
    {
        "case_id": "estate_case_01_farmland_and_debt_deduction",
        "description": "113 年過世，新版級距；農地扣除與未償債務可調，最小化遺產稅。",
        "payload": {
            "death_period": 4,
            "is_military_police": False,
            "land_value": 40_000_000,
            "building_value": 20_000_000,
            "house_value": 10_000_000,
            "deposit_bonds_value": 0,
            "stock_invest_value": 0,
            "cash_gold_jewelry_value": 5_000_000,
            "gift_in_2yrs_value": 0,
            "spouse_count": 1,
            "lineal_descendant_count": 2,
            "father_mother_count": 0,
            "disability_count": 0,
            "dependent_count": 0,
            "farmland_val": 0,
            "inheritance_6to9_val": 0,
            "unpaid_tax_fines_val": 0,
            "unpaid_debts_val": 0,
            "will_management_fee": 0,
            "public_facility_retention_val": 0,
            "spouse_surplus_right_val": 0,
            "gift_tax_offset": 0,
            "foreign_tax_offset": 0,
            "free_vars": ["farmland_val", "unpaid_debts_val"],
            "constraints": {
                "farmland_val": {"<=": 15_000_000},
                "farmland_val - land_value": {"<=": 0},
                "unpaid_debts_val": {"<=": 3_000_000},
            },
        },
    },
    {
        "case_id": "estate_case_02_gift_in_two_years_and_offset",
        "description": "103 年過世，舊制 10%；過世前兩年贈與與贈與稅扣抵可調。",
        "payload": {
            "death_period": 1,
            "is_military_police": False,
            "land_value": 0,
            "building_value": 0,
            "house_value": 0,
            "deposit_bonds_value": 0,
            "stock_invest_value": 120_000_000,
            "cash_gold_jewelry_value": 0,
            "gift_in_2yrs_value": 0,
            "spouse_count": 0,
            "lineal_descendant_count": 0,
            "father_mother_count": 0,
            "disability_count": 0,
            "dependent_count": 0,
            "farmland_val": 0,
            "inheritance_6to9_val": 0,
            "unpaid_tax_fines_val": 0,
            "unpaid_debts_val": 0,
            "will_management_fee": 0,
            "public_facility_retention_val": 0,
            "spouse_surplus_right_val": 0,
            "gift_tax_offset": 0,
            "foreign_tax_offset": 0,
            "free_vars": ["gift_in_2yrs_value", "gift_tax_offset"],
            "constraints": {
                "gift_in_2yrs_value": {">=": 0, "<=": 12_000_000},
                "gift_tax_offset": {">=": 0, "<=": 2_400_000},
                "gift_tax_offset - gift_in_2yrs_value * 0.2": {"<=": 0},
            },
        },
    },
]
BUILTIN_PAYLOAD = BUILTIN_PAYLOADS[0]["payload"]


class EstateTaxSMTCase(BaseTrackedTaxCase):
    PARAM_NAMES = [
        "land_value", "building_value", "house_value", "deposit_bonds_value",
        "stock_invest_value", "cash_gold_jewelry_value", "gift_in_2yrs_value",
        "spouse_count", "lineal_descendant_count", "father_mother_count",
        "disability_count", "dependent_count", "farmland_val", "inheritance_6to9_val",
        "unpaid_tax_fines_val", "unpaid_debts_val", "will_management_fee",
        "public_facility_retention_val", "spouse_surplus_right_val", "gift_tax_offset",
        "foreign_tax_offset",
    ]

    def build(self) -> "EstateTaxSMTCase":
        if self._built:
            return self
        self._built = True
        self.opt = Optimize()
        p = self.payload
        death_period = _norm_period(p.get("death_period", 1))
        is_military_police = bool(p.get("is_military_police", False))

        if death_period in (1, 2):
            base_ex_amt, funeral_amt = 12_000_000, 1_230_000
        else:
            base_ex_amt = 13_330_000
            funeral_amt = 1_230_000 if death_period in (1, 2, 3) else 1_380_000
        if is_military_police:
            base_ex_amt *= 2

        for name in self.PARAM_NAMES:
            z = Real(name)
            self._add_param(name, z, self._num(name, 0.0), [("nonnegative", lambda v: v >= 0)])

        self._bind_params()
        self._add_user_constraints()
        V = lambda n: self.params[n][0]

        asset_sum = Real("asset_sum")
        base_ex_z = Real("base_ex_z")
        funeral_z = Real("funeral_z")
        ded_spouse = Real("ded_spouse")
        ded_lineal = Real("ded_lineal")
        ded_fm = Real("ded_fm")
        ded_dis = Real("ded_dis")
        ded_dep = Real("ded_dep")
        ded_other = Real("ded_other")
        total_ded = Real("total_ded")
        tax_inherit = Real("tax_inherit")
        bracket_r = Real("bracket_r")
        bracket_d = Real("bracket_d")
        offsets = Real("offsets")
        est_tax = Real("est_tax")
        final_tax_int = Int("final_tax_z")
        est_tax_scaled = Int("est_tax_scaled")
        net_tax_int = Int("net_tax_int")

        self.vars.update({
            "asset_sum": asset_sum, "base_ex_z": base_ex_z, "funeral_z": funeral_z,
            "ded_spouse": ded_spouse, "ded_lineal": ded_lineal, "ded_fm": ded_fm,
            "ded_dis": ded_dis, "ded_dep": ded_dep, "ded_other": ded_other,
            "total_ded": total_ded, "tax_inherit": tax_inherit, "bracket_r": bracket_r,
            "bracket_d": bracket_d, "offsets": offsets, "est_tax": est_tax,
            "final_tax_z": final_tax_int, "est_tax_scaled": est_tax_scaled, "net_tax_int": net_tax_int,
        })
        self.final_tax_z = final_tax_int
        self.net_taxable_z = net_tax_int
        self.objective_z = final_tax_int
        self.objective_sense = "min"
        self.objective_label = "final_tax"

        self.add(asset_sum == sum(V(n) for n in ["land_value", "building_value", "house_value", "deposit_bonds_value", "stock_invest_value", "cash_gold_jewelry_value", "gift_in_2yrs_value"]), name="law.asset_sum")
        self.add(base_ex_z == RealVal(base_ex_amt), name="law.base_exemption")
        self.add(funeral_z == RealVal(funeral_amt), name="law.funeral_deduction")
        self.add(ded_spouse == V("spouse_count") * (5_530_000 if death_period > 3 else 4_930_000), name="law.deduction.spouse")
        self.add(ded_lineal == V("lineal_descendant_count") * (560_000 if death_period > 3 else 500_000), name="law.deduction.lineal_descendant")
        self.add(ded_fm == V("father_mother_count") * (1_380_000 if death_period > 3 else 1_230_000), name="law.deduction.father_mother")
        self.add(ded_dis == V("disability_count") * (6_930_000 if death_period > 3 else 6_180_000), name="law.deduction.disability")
        self.add(ded_dep == V("dependent_count") * (560_000 if death_period > 3 else 500_000), name="law.deduction.dependent")
        self.add(ded_other == V("farmland_val") + V("inheritance_6to9_val") + V("unpaid_tax_fines_val") + V("unpaid_debts_val") + funeral_z + V("will_management_fee") + V("public_facility_retention_val") + V("spouse_surplus_right_val"), name="law.deduction.other")
        self.add(total_ded == ded_spouse + ded_lineal + ded_fm + ded_dis + ded_dep + ded_other, name="law.total_deduction")
        self.add(tax_inherit == asset_sum - base_ex_z - total_ded, name="law.net_taxable_estate")

        if death_period == 1:
            rate_expr = And(bracket_r == 0.10, bracket_d == 0)
        elif death_period > 4:
            rate_expr = If(tax_inherit <= 56_210_000, And(bracket_r == 0.10, bracket_d == 0), If(tax_inherit <= 112_420_000, And(bracket_r == 0.15, bracket_d == 2_810_500), And(bracket_r == 0.20, bracket_d == 8_431_500)))
        else:
            rate_expr = If(tax_inherit <= 50_000_000, And(bracket_r == 0.10, bracket_d == 0), If(tax_inherit <= 100_000_000, And(bracket_r == 0.15, bracket_d == 2_500_000), And(bracket_r == 0.20, bracket_d == 7_500_000)))
        self.add(rate_expr, name="law.bracket.rate_and_progressive_difference")
        self.add(offsets == V("gift_tax_offset") + V("foreign_tax_offset"), name="law.offsets")
        self.add(est_tax == If(tax_inherit <= 0, 0, If(tax_inherit * bracket_r - bracket_d - offsets < 0, 0, tax_inherit * bracket_r - bracket_d - offsets)), name="law.estate_tax_real")
        self.add(final_tax_int == ToInt(est_tax), name="objective_link.final_tax_floor", group="objective_link")
        self.add(est_tax_scaled == ToInt(est_tax * 1_000), name="objective_link.tax_scaled")
        self.add(net_tax_int == ToInt(tax_inherit), name="law.net_taxable_int")
        self.opt.minimize(final_tax_int)
        self.opt.minimize(est_tax_scaled)
        return self


def calculate_estate_tax(**kwargs) -> Dict[str, Any]:
    started = time.perf_counter()
    payload = normalize_payload(kwargs)
    base_payload = dict(payload)
    base_payload["free_vars"] = []
    base_payload["constraints"] = {}
    baseline_case = EstateTaxSMTCase(base_payload)
    baseline = baseline_case.solve().get("optimum")
    fixed_payload = dict(payload); fixed_payload["free_vars"] = []
    try:
        baseline_with = EstateTaxSMTCase(fixed_payload).solve()
        baseline_status = baseline_with.get("status")
        baseline_with_constraints = baseline_with.get("optimum") if baseline_status == "sat" else None
    except Exception:
        baseline_status = "unsat"; baseline_with_constraints = None
    mode = "manual_free" if payload.get("free_vars") else "baseline"
    sol = EstateTaxSMTCase(payload).solve() if payload.get("free_vars") else EstateTaxSMTCase(fixed_payload).solve()
    status = sol.get("status")
    optimized = sol.get("optimum") if status == "sat" else None
    fields = sol.get("fields") or {}
    diff = {k: {"original": base_payload.get(k, 0), "optimized": v, "difference": v - float(base_payload.get(k, 0) or 0)} for k, v in fields.items() if k in set(payload.get("free_vars") or []) and v != float(base_payload.get(k, 0) or 0)}
    return {"mode": mode, "input_params": payload, "baseline": baseline, "baseline_status": baseline_status, "baseline_with_constraints": baseline_with_constraints, "optimized": optimized, "status": status, "diff": diff, "final_params": {k: {"value": v, "type": "free" if k in set(payload.get("free_vars") or []) else "fixed"} for k, v in fields.items()}, "constraints": payload.get("constraints") or {}, "elapsed_sec": time.perf_counter() - started}


def run_analysis(payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    return run_tracked_analysis(EstateTaxSMTCase, payload, **kwargs)


def load_payload(path: Optional[str], case_index: int = 0) -> Dict[str, Any]:
    if not path:
        return dict(BUILTIN_PAYLOADS[case_index]["payload"])
    with open(path, "r", encoding="utf-8") as f:
        return normalize_payload(json.load(f))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run estate-tax tracked unsat-core and release analysis.")
    ap.add_argument("--payload")
    ap.add_argument("--case", type=int, default=0, choices=[0, 1])
    ap.add_argument("--json-out", default="estate_tax_unsat_report.json")
    ap.add_argument("--md-out", default="estate_tax_unsat_report.md")
    ap.add_argument("--print-core", action="store_true")
    ap.add_argument("--no-auto-combinations", action="store_true")
    ap.add_argument("--core-only", action="store_true")
    ap.add_argument(
        "--release-scope",
        default="default_only",
        choices=["default_only", "fixed_only", "all"],
        help=(
            "Which constraints may be released. default_only releases only "
            "zero-valued fixed default assumptions; fixed_only releases all fixed.* "
            "constraints; all preserves the previous broad behavior."
        ),
    )
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
    report = run_analysis(
        payload,
        auto_release=not getattr(args, "no_release_tests", False),
        auto_combinations=not getattr(args, "no_auto_combinations", False) and not getattr(args, "no_release_tests", False),
        include_non_core=not getattr(args, "core_only", False),
        release_scope=args.release_scope,
    )
    write_report(report, json_out=args.json_out, md_out=args.md_out, title="Estate Tax SMT Unsat Core Lab")
    base = report.get("base", {}); probe = report.get("unsat_core_probe", {}); gen = report.get("release_generation", {})
    print("=== Base solve ===")
    print(f"status: {base.get('status')}")
    print(f"optimum: {base.get('optimum')}")
    print(f"net_taxable: {base.get('net_taxable')}")
    print("\n=== UNSAT probe ===")
    print(f"goal: {probe.get('goal')}")
    print(f"status: {probe.get('status')}")
    print(f"tracked_assertions: {probe.get('tracked_assertion_count')}")
    print(f"core_size: {probe.get('core_size')}")
    print(f"core_summary: {json.dumps(probe.get('core_summary'), ensure_ascii=False)}")
    print("\n=== Release generation ===")
    print(json.dumps(gen, ensure_ascii=False, indent=2))
    print("\n=== Release tests ===")
    for row in report.get("release_tests", []):
        cls = row.get("classification") or {}
        print(f"- {row['test']}: status={row['status']}, new_optimum={row.get('new_optimum')}, delta={row.get('delta_vs_base')}, class={cls.get('category')}")
    print(f"\nwrote: {args.json_out}\nwrote: {args.md_out}")
    if args.print_core:
        print("\n=== Full core ===")
        for item in probe.get("core", []):
            print(json.dumps(item, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
