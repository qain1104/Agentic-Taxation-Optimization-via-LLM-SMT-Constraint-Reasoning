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

RATE = RealVal("0.10")
CATEGORIES = ["car", "yacht", "aircraft", "coral_ivory", "furniture"]

BUILTIN_PAYLOADS = [
    {
        "case_id": "special_goods_case_0_supercar_furniture_min_tax",
        "description": "超跑與高級傢俱，數量可調且總數量至少 3，目標最小化特種貨物稅。",
        "case_class": "SpecialGoodsTaxSMTCase",
        "payload": {
            "kind": "goods",
            "car.price": 3_000_000,
            "car.quantity": 1,
            "furniture.price": 500_000,
            "furniture.quantity": 1,
            "free_vars": ["car.quantity", "furniture.quantity"],
            "constraints": {
                "car.quantity": {">=": 1, "<=": 2},
                "furniture.quantity": {">=": 1, "<=": 5},
                "car.quantity + furniture.quantity": {">=": 3},
            },
        },
    },
    {
        "case_id": "special_services_case_1_membership_fee_min_tax",
        "description": "高級俱樂部入會權利金原價 950,000，售價不得低於 800,000，目標最小化特種勞務稅。",
        "case_class": "SpecialServicesTaxSMTCase",
        "payload": {
            "kind": "services",
            "sales_price": 950_000,
            "free_vars": ["sales_price"],
            "constraints": {"sales_price": {">=": 800_000}},
        },
    },
]
BUILTIN_PAYLOAD = BUILTIN_PAYLOADS[0]["payload"]


class SpecialGoodsTaxSMTCase(BaseTrackedTaxCase):
    objective_sense = "min"
    objective_label = "total_tax"

    def build(self):
        if self._built:
            return self
        self._built = True
        self.opt = Optimize()
        tax_terms = []
        qty_terms = []
        for cat in CATEGORIES:
            pname = f"{cat}.price"
            qname = f"{cat}.quantity"
            p_int = Int(f"{cat}_price_int")
            q_int = Int(f"{cat}_quantity_int")
            price = ToReal(p_int)
            qty = ToReal(q_int)
            self._add_param(pname, price, self._num(pname), [("nonnegative", lambda v: v >= 0)], releasable=(pname in self.explicit_keys))
            self._add_param(qname, qty, self._num(qname), [("nonnegative", lambda v: v >= 0)], releasable=(qname in self.explicit_keys))
            row_tax = Int(f"{cat}_tax")
            self.vars[f"{cat}.tax"] = row_tax
            self.add(row_tax == ToInt(price * qty * RATE), name=f"law.{cat}.tax")
            tax_terms.append(row_tax)
            qty_terms.append(qty)
        self._bind_params()
        total_tax = Int("total_tax")
        total_qty = Real("total_qty")
        self.vars["total_tax"] = total_tax
        self.vars["total_qty"] = total_qty
        self.add(total_tax == Sum(tax_terms), name="law.total_tax")
        self.add(total_qty == Sum(qty_terms), name="law.total_qty")
        self.final_tax_z = total_tax
        self.net_taxable_z = total_qty
        self.objective_z = total_tax
        self._add_user_constraints()
        self._optimize()
        return self


class SpecialServicesTaxSMTCase(BaseTrackedTaxCase):
    objective_sense = "min"
    objective_label = "service_tax"

    def build(self):
        if self._built:
            return self
        self._built = True
        self.opt = Optimize()
        sp = Real("sales_price")
        self._add_param("sales_price", sp, self._num("sales_price"), [("nonnegative", lambda v: v >= 0)], releasable=("sales_price" in self.explicit_keys))
        self._bind_params()
        tax = Int("service_tax")
        self.vars["service_tax"] = tax
        self.add(tax == ToInt(sp * RATE), name="law.service_tax_10_percent")
        self.final_tax_z = tax
        self.net_taxable_z = sp
        self.objective_z = tax
        self._add_user_constraints()
        self._optimize()
        return self


def _case_cls_for_payload(payload: Dict[str, Any]):
    return SpecialServicesTaxSMTCase if payload.get("kind") == "services" or "sales_price" in payload else SpecialGoodsTaxSMTCase


def calculate_special_tax_tracked(**payload):
    return _case_cls_for_payload(payload)(payload).solve()


def load_payload(path: Optional[str], case_index: int) -> Dict[str, Any]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return normalize_payload(json.load(f))
    return dict(BUILTIN_PAYLOADS[case_index]["payload"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Tracked special goods/services tax UNSAT-core/release lab")
    ap.add_argument("--case", type=int, default=0, choices=range(len(BUILTIN_PAYLOADS)))
    ap.add_argument("--payload")
    ap.add_argument("--json-out", default="special_tax_unsat_report.json")
    ap.add_argument("--md-out", default="special_tax_unsat_report.md")
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
    case_cls = _case_cls_for_payload(payload)
    report = run_tracked_analysis(
        case_cls,
        payload,
        auto_release=not getattr(args, "no_release_tests", False),
        auto_combinations=not getattr(args, "no_auto_combinations", False) and not getattr(args, "no_release_tests", False),
        include_non_core=not getattr(args, "core_only", False),
        release_scope=args.release_scope,
    )
    write_report(report, json_out=args.json_out, md_out=args.md_out, title="Special Goods and Services Tax Tracked SMT Lab")
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
