# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Sequence

from z3 import Int, Optimize, Real, RealVal, Sum, ToInt, ToReal

try:
    from .tracked_tax_core import BaseTrackedTaxCase, normalize_payload, run_tracked_analysis, write_report
except ImportError:
    from tracked_tax_core import BaseTrackedTaxCase, normalize_payload, run_tracked_analysis, write_report

ITEMS = {
    "cigarettes_new": {"cn": "紙菸-新制", "type": "tobacco", "base": 1590, "div": 1000, "need_abv": False},
    "cigar_new": {"cn": "雪茄-新制", "type": "tobacco", "base": 1590, "div": 1, "need_abv": False},
    "beer": {"cn": "啤酒", "type": "alcohol", "rate": ("const", 26), "need_abv": False},
    "cooking_wine": {"cn": "料理酒", "type": "alcohol", "rate": ("const", 9), "need_abv": False},
    "cooking_wine_old": {"cn": "料理酒-舊", "type": "alcohol", "rate": ("const", 22), "need_abv": False},
}

BUILTIN_PAYLOADS = [
    {
        "case_id": "tobacco_alcohol_case_0_minimize_tax",
        "description": "菸類合計 3000，酒類合計 500，紙菸為雪茄兩倍，目標最小化菸酒稅。",
        "payload": {
            "mode": "minimize",
            "cigarettes_new.quantity": 120,
            "beer.quantity": 300,
            "cigar_new.quantity": 500,
            "cooking_wine.quantity": 250,
            "free_vars": ["cigarettes_new.quantity", "beer.quantity", "cigar_new.quantity", "cooking_wine.quantity"],
            "constraints": {
                "beer.quantity + cooking_wine.quantity": {"=": 500},
                "cigarettes_new.quantity + cigar_new.quantity": {"=": 3000},
                "cigarettes_new.quantity": {"=": "cigar_new.quantity * 2"},
            },
        },
    },
    {
        "case_id": "tobacco_alcohol_case_1_maximize_qty_under_budget",
        "description": "菸類至少 100、酒類至少 500，紙菸為雪茄三倍，稅額上限 1,000,000 下最大化數量。",
        "payload": {
            "mode": "maximize_qty",
            "budget_tax": 1_000_000,
            "cigarettes_new.quantity": 120,
            "beer.quantity": 300,
            "cigar_new.quantity": 500,
            "cooking_wine_old.quantity": 250,
            "free_vars": ["cigarettes_new.quantity", "beer.quantity", "cigar_new.quantity", "cooking_wine_old.quantity"],
            "constraints": {
                "cigarettes_new.quantity + cigar_new.quantity": {">=": 100},
                "beer.quantity + cooking_wine_old.quantity": {">=": 500},
                "cigarettes_new.quantity": {"=": "cigar_new.quantity * 3"},
            },
        },
    },
]
BUILTIN_PAYLOAD = BUILTIN_PAYLOADS[0]["payload"]


class TobaccoAlcoholTaxSMTCase(BaseTrackedTaxCase):
    def build(self):
        if self._built:
            return self
        self._built = True
        self.opt = Optimize()
        p = self.payload
        mode = p.get("mode", "minimize")
        self.objective_sense = "max" if mode == "maximize_qty" else "min"
        self.objective_label = "total_qty" if mode == "maximize_qty" else "total_tax"

        tax_terms = []
        qty_terms = []
        for slug, spec in ITEMS.items():
            qname = f"{slug}.quantity"
            q_int = Int(f"{slug}_quantity_int")
            q = ToReal(q_int)
            self._add_param(qname, q, self._num(qname), [("nonnegative", lambda v: v >= 0)], releasable=(qname in self.explicit_keys))
            qty_terms.append(q)
            if spec["type"] == "tobacco":
                expr = q * RealVal(str(spec["base"])) / RealVal(str(spec["div"]))
            else:
                kind, rate = spec["rate"]
                expr = q * RealVal(str(rate))
            tax_terms.append(expr)
            row_tax = Int(f"{slug}_tax")
            self.vars[f"{slug}.tax"] = row_tax
            self.add(row_tax == ToInt(expr), name=f"law.{slug}.tax")

        self._bind_params()
        total_tax = Int("total_tax")
        total_qty = Real("total_qty")
        self.vars["total_tax"] = total_tax
        self.vars["total_qty"] = total_qty
        self.final_tax_z = total_tax
        self.net_taxable_z = total_tax
        self.add(total_tax == Sum([self.vars[f"{slug}.tax"] for slug in ITEMS]), name="law.total_tax")
        self.add(total_qty == Sum(qty_terms), name="law.total_qty")
        if mode == "maximize_qty":
            budget = int(float(p.get("budget_tax", p.get("target_tax", 0))))
            self.add(total_tax <= budget, name=f"budget.total_tax_le_{budget}", group="user_constraint", releasable=True)
            self.objective_z = total_qty
        else:
            self.objective_z = total_tax
        self._add_user_constraints()
        self._optimize()
        return self


def calculate_tobacco_alcohol_tax_tracked(**payload):
    return TobaccoAlcoholTaxSMTCase(payload).solve()


def load_payload(path: Optional[str], case_index: int) -> Dict[str, Any]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return normalize_payload(json.load(f))
    return dict(BUILTIN_PAYLOADS[case_index]["payload"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Tracked tobacco/alcohol tax UNSAT-core/release lab")
    ap.add_argument("--case", type=int, default=0, choices=range(len(BUILTIN_PAYLOADS)))
    ap.add_argument("--payload")
    ap.add_argument("--json-out", default="tobacco_alcohol_tax_unsat_report.json")
    ap.add_argument("--md-out", default="tobacco_alcohol_tax_unsat_report.md")
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
        TobaccoAlcoholTaxSMTCase,
        payload,
        auto_release=not getattr(args, "no_release_tests", False),
        auto_combinations=not getattr(args, "no_auto_combinations", False) and not getattr(args, "no_release_tests", False),
        include_non_core=not getattr(args, "core_only", False),
    )
    write_report(report, json_out=args.json_out, md_out=args.md_out, title="Tobacco Alcohol Tax Tracked SMT Lab")
    base = report.get("base", {})
    probe = report.get("unsat_core_probe", {})
    print("=== Base solve ===")
    print(f"status: {base.get('status')}")
    print(f"objective: {base.get('objective_label')} / {base.get('objective_sense')}")
    print(f"optimum: {base.get('optimum')}")
    print(f"tax: {base.get('tax')}")
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
