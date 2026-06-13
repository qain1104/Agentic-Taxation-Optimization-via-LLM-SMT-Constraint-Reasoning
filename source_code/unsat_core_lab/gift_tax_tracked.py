# -*- coding: utf-8 -*-
"""
gift_tax_tracked.py

Tracked / UNSAT-core-ready version of gift_tax.py.
It preserves the original gift-tax formula, but every constraint is added
through self.add(...), so it supports optimality proof, unsat core extraction,
release tests, COI combo release, and automatic policy inference.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional, Sequence

from z3 import And, If, Int, Optimize, Or, Real, ToInt

from tracked_tax_core import BaseTrackedTaxCase, normalize_payload, run_tracked_analysis, write_report


class UnsatError(Exception):
    pass


BUILTIN_PAYLOADS = [
    {
        "case_id": "gift_case_01_real_estate_allocation",
        "description": "114 年新版制度，不動產贈與；土地、房屋、其他財產可依比例與總額限制調整。",
        "payload": {
            "period_choice": 4,
            "land_value": 25_000_000,
            "ground_value": 5_000_000,
            "house_value": 8_000_000,
            "others_value": 2_000_000,
            "not_included_land": 0,
            "not_included_house": 0,
            "not_included_others": 0,
            "remaining_exemption_98": 0,
            "previous_gift_sum_in_this_year": 0,
            "land_increment_tax": 0,
            "deed_tax": 0,
            "other_gift_burdens": 0,
            "previous_gift_tax_or_credit": 0,
            "new_old_system_adjustment": 0,
            "free_vars": ["land_value", "house_value", "others_value"],
            "constraints": {
                "land_value": {">=": 20_000_000, "<=": 30_000_000, "==": "house_value * 3"},
                "house_value": {">=": 5_000_000, "<=": 10_000_000},
                "land_value + house_value": {"<=": 33_000_000},
                "others_value": {"==": "land_value * 0.1"},
            },
        },
    },
    {
        "case_id": "gift_case_02_cash_gift_range_minimize",
        "description": "113 年制度，現金贈與 others_value 為唯一自由變數，範圍 0 至 200 萬，最小化贈與稅。",
        "payload": {
            "period_choice": 3,
            "land_value": 3_000_000,
            "ground_value": 0,
            "house_value": 0,
            "others_value": 0,
            "not_included_land": 0,
            "not_included_house": 0,
            "not_included_others": 0,
            "remaining_exemption_98": 0,
            "previous_gift_sum_in_this_year": 0,
            "land_increment_tax": 0,
            "deed_tax": 0,
            "other_gift_burdens": 0,
            "previous_gift_tax_or_credit": 0,
            "new_old_system_adjustment": 0,
            "free_vars": ["others_value"],
            "constraints": {"others_value": {">=": 0, "<=": 2_000_000}},
        },
    },
]
BUILTIN_PAYLOAD = BUILTIN_PAYLOADS[0]["payload"]


class GiftTaxSMTCase(BaseTrackedTaxCase):
    PARAM_NAMES = [
        "land_value", "ground_value", "house_value", "others_value",
        "not_included_land", "not_included_house", "not_included_others",
        "remaining_exemption_98", "previous_gift_sum_in_this_year",
        "land_increment_tax", "deed_tax", "other_gift_burdens",
        "previous_gift_tax_or_credit", "new_old_system_adjustment",
    ]

    def build(self) -> "GiftTaxSMTCase":
        if self._built:
            return self
        self._built = True
        self.opt = Optimize()
        p = self.payload
        period_choice = int(float(p.get("period_choice", 1)))

        for name in self.PARAM_NAMES:
            z = Real(name)
            checks = [] if name == "new_old_system_adjustment" else [("nonnegative", lambda v: v >= 0)]
            self._add_param(name, z, self._num(name, 0.0), checks)

        self._bind_params()
        self._add_user_constraints()
        V = lambda n: self.params[n][0]

        T0 = Real("this_total")
        TA = Real("annual_total")
        D = Real("deductions")
        N = Real("net_tax")
        RATE = Real("applied_rate")
        PD = Real("prog_diff")
        TAX_R = Real("tax_real")
        TAX_I = Int("final_tax_z")
        net_tax_int = Int("net_tax_int")

        self.vars.update({"this_total": T0, "annual_total": TA, "deductions": D, "net_tax": N, "applied_rate": RATE, "prog_diff": PD, "tax_real": TAX_R, "final_tax_z": TAX_I, "net_tax_int": net_tax_int})
        self.final_tax_z = TAX_I
        self.net_taxable_z = net_tax_int
        self.objective_z = TAX_I
        self.objective_sense = "min"
        self.objective_label = "final_tax"

        self.add(T0 == V("land_value") + V("ground_value") + V("house_value") + V("others_value"), name="law.this_gift_total")
        self.add(TA == T0 + V("previous_gift_sum_in_this_year"), name="law.annual_gift_total")
        self.add(D == V("land_increment_tax") + V("deed_tax") + V("other_gift_burdens"), name="law.deductions")

        if period_choice == 1:
            self.add(N == T0 - V("remaining_exemption_98") - D, name="law.net_taxable.period1")
            self.add(RATE == 0.10, name="law.period1.rate")
            self.add(PD == 0, name="law.period1.progressive_difference")
            self.add(TAX_R == N * RATE - V("previous_gift_tax_or_credit"), name="law.tax_real.period1")
        elif period_choice == 2:
            c1 = N <= 25_000_000
            c2 = And(N > 25_000_000, N <= 50_000_000)
            self.add(N == TA - 2_200_000 - D, name="law.net_taxable.period2")
            self.add(Or(c1, c2, N > 50_000_000), name="law.bracket.cover.period2")
            self.add(If(c1, And(RATE == 0.10, PD == 0), If(c2, And(RATE == 0.15, PD == 1_250_000), And(RATE == 0.20, PD == 3_750_000))), name="law.bracket.period2")
            self.add(TAX_R == N * RATE - PD - V("previous_gift_tax_or_credit") - V("new_old_system_adjustment"), name="law.tax_real.period2")
        elif period_choice == 3:
            c1 = N <= 25_000_000
            c2 = And(N > 25_000_000, N <= 50_000_000)
            self.add(N == TA - 2_440_000 - D, name="law.net_taxable.period3")
            self.add(Or(c1, c2, N > 50_000_000), name="law.bracket.cover.period3")
            self.add(If(c1, And(RATE == 0.10, PD == 0), If(c2, And(RATE == 0.15, PD == 1_250_000), And(RATE == 0.20, PD == 3_750_000))), name="law.bracket.period3")
            self.add(TAX_R == N * RATE - PD - V("previous_gift_tax_or_credit"), name="law.tax_real.period3")
        elif period_choice == 4:
            c1 = N <= 28_110_000
            c2 = And(N > 28_110_000, N <= 56_210_000)
            self.add(N == TA - 2_440_000 - D, name="law.net_taxable.period4")
            self.add(Or(c1, c2, N > 56_210_000), name="law.bracket.cover.period4")
            self.add(If(c1, And(RATE == 0.10, PD == 0), If(c2, And(RATE == 0.15, PD == 1_405_500), And(RATE == 0.20, PD == 4_216_000))), name="law.bracket.period4")
            self.add(TAX_R == N * RATE - PD - V("previous_gift_tax_or_credit"), name="law.tax_real.period4")
        else:
            raise ValueError("Unsupported period_choice")

        self.add(TAX_I == ToInt(TAX_R), name="objective_link.final_tax_floor", group="objective_link")
        self.add(TAX_I >= 0, name="law.final_tax_nonnegative")
        self.add(net_tax_int == ToInt(N), name="law.net_taxable_int")
        self.opt.minimize(TAX_I)
        return self


def calculate_gift_tax(**kwargs) -> Dict[str, Any]:
    started = time.perf_counter()
    payload = normalize_payload(kwargs)
    base_payload = dict(payload); base_payload["free_vars"] = []; base_payload["constraints"] = {}
    baseline = GiftTaxSMTCase(base_payload).solve().get("optimum")
    fixed_payload = dict(payload); fixed_payload["free_vars"] = []
    fixed_sol = GiftTaxSMTCase(fixed_payload).solve()
    baseline_status = fixed_sol.get("status")
    baseline_with_constraints = fixed_sol.get("optimum") if baseline_status == "sat" else None
    mode = "manual_free" if payload.get("free_vars") else "baseline"
    sol = GiftTaxSMTCase(payload).solve() if payload.get("free_vars") else fixed_sol
    status = sol.get("status")
    optimized = sol.get("optimum") if status == "sat" else None
    fields = sol.get("fields") or {}
    diff = {k: {"original": base_payload.get(k, 0), "optimized": v, "difference": v - float(base_payload.get(k, 0) or 0)} for k, v in fields.items() if k in set(payload.get("free_vars") or []) and v != float(base_payload.get(k, 0) or 0)}
    return {"mode": mode, "input_params": payload, "baseline": baseline, "baseline_status": baseline_status, "baseline_with_constraints": baseline_with_constraints, "optimized": optimized, "status": status, "diff": diff, "final_params": {k: {"value": v, "type": "free" if k in set(payload.get("free_vars") or []) else "fixed"} for k, v in fields.items()}, "constraints": payload.get("constraints") or {}, "elapsed_sec": time.perf_counter() - started}


def run_analysis(payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    return run_tracked_analysis(GiftTaxSMTCase, payload, **kwargs)


def load_payload(path: Optional[str], case_index: int = 0) -> Dict[str, Any]:
    if not path:
        return dict(BUILTIN_PAYLOADS[case_index]["payload"])
    with open(path, "r", encoding="utf-8") as f:
        return normalize_payload(json.load(f))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run gift-tax tracked unsat-core and release analysis.")
    ap.add_argument("--payload")
    ap.add_argument("--case", type=int, default=0, choices=[0, 1])
    ap.add_argument("--json-out", default="gift_tax_unsat_report.json")
    ap.add_argument("--md-out", default="gift_tax_unsat_report.md")
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
    write_report(report, json_out=args.json_out, md_out=args.md_out, title="Gift Tax SMT Unsat Core Lab")
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
