# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, Optional, Sequence

from z3 import Int, Optimize, Real, RealVal, Sum, ToInt

try:
    from .tracked_tax_core import BaseTrackedTaxCase, normalize_payload, run_tracked_analysis, write_report
except ImportError:
    from tracked_tax_core import BaseTrackedTaxCase, normalize_payload, run_tracked_analysis, write_report

VAT_RATE = RealVal("0.05")
NVAT_RATE = {"cat1": 0.01, "cat2": 0.01, "cat3": 0.02, "cat4": 0.05, "cat5": 0.05, "cat6": 0.15, "cat7": 0.25, "cat8": 0.001}

BUILTIN_PAYLOADS = [
    {
        "case_id": "vat_case_0_minimize_row_inputs",
        "description": "加值型營業稅三列資料；row1/row2 進項稅額可調，目標最小化營業稅。",
        "case_class": "VATTaxSMTCase",
        "payload": {
            "row0.output_tax_val": 3_000_000,
            "row0.input_tax_val": 1_100_000,
            "row1.output_tax_val": 1_200_000,
            "row1.input_tax_val": 600_000,
            "row2.output_tax_val": 800_000,
            "row2.input_tax_val": 300_000,
            "free_vars": ["row1.input_tax_val", "row2.input_tax_val"],
            "constraints": {
                "row1.input_tax_val": {"<=": 800_000},
                "row2.input_tax_val": {"<=": 500_000},
                "row1.input_tax_val + row2.input_tax_val": {"<=": 1_100_000},
            },
        },
    },
    {
        "case_id": "nvat_case_1_maximize_sales_under_budget",
        "description": "非加值型營業稅 cat4/cat6/cat8 可調，在稅額上限 70,000 下最大化總銷售額。",
        "case_class": "NVATTaxSMTCase",
        "payload": {
            "mode": "maximize_sales",
            "budget_tax": 70_000,
            "cat4": 900_000,
            "cat6": 100_000,
            "cat8": 0,
            "free_vars": ["cat4", "cat6", "cat8"],
            "constraints": {
                "cat6": {"<=": "cat4 * 0.5"},
                "cat4 + cat6 + cat8": {"<=": 2_000_000},
            },
        },
    },
]
BUILTIN_PAYLOAD = BUILTIN_PAYLOADS[0]["payload"]


class VATTaxSMTCase(BaseTrackedTaxCase):
    objective_sense = "min"
    objective_label = "total_vat"

    def build(self):
        if self._built:
            return self
        self._built = True
        self.opt = Optimize()
        tax_terms = []
        for i in range(3):
            out_name = f"row{i}.output_tax_val"
            in_name = f"row{i}.input_tax_val"
            out_z = Real(f"row{i}_output_tax_val")
            in_z = Real(f"row{i}_input_tax_val")
            self._add_param(out_name, out_z, self._num(out_name), [("nonnegative", lambda v: v >= 0)], releasable=(out_name in self.explicit_keys))
            self._add_param(in_name, in_z, self._num(in_name), [("nonnegative", lambda v: v >= 0)], releasable=(in_name in self.explicit_keys))
            row_tax = Real(f"row{i}_vat")
            self.vars[f"row{i}.vat"] = row_tax
            self.add(row_tax == (out_z - in_z) * VAT_RATE, name=f"law.row{i}.vat")
            tax_terms.append(row_tax)
        self._bind_params()
        total_vat_real = Real("total_vat_real")
        total_vat = Int("total_vat")
        self.vars["total_vat_real"] = total_vat_real
        self.vars["total_vat"] = total_vat
        self.add(total_vat_real == Sum(tax_terms), name="law.total_vat_real")
        self.add(total_vat == ToInt(total_vat_real), name="law.total_vat_floor")
        self.final_tax_z = total_vat
        self.net_taxable_z = total_vat
        self.objective_z = total_vat
        self._add_user_constraints()
        self._optimize()
        return self


class NVATTaxSMTCase(BaseTrackedTaxCase):
    def build(self):
        if self._built:
            return self
        self._built = True
        self.opt = Optimize()
        p = self.payload
        mode = p.get("mode", "minimize")
        self.objective_sense = "max" if mode == "maximize_sales" else "min"
        self.objective_label = "total_sales" if mode == "maximize_sales" else "total_tax"
        tax_terms = []
        sales_terms = []
        for cat, rate in NVAT_RATE.items():
            z = Real(cat)
            self._add_param(cat, z, self._num(cat), [("nonnegative", lambda v: v >= 0)], releasable=(cat in self.explicit_keys))
            row_tax = Real(f"{cat}_tax_real")
            self.vars[f"{cat}.tax_real"] = row_tax
            self.add(row_tax == z * RealVal(str(rate)), name=f"law.{cat}.tax")
            tax_terms.append(row_tax)
            if mode == "maximize_sales":
                # Maximize sales only over fields that define the user scenario.
                # Default zero categories not mentioned by the payload remain
                # absent form defaults, even if their fixed-zero assumption is
                # experimentally released.
                if _payload_mentions_field(p, cat):
                    sales_terms.append(z)
            else:
                sales_terms.append(z)
        self._bind_params()
        total_tax_real = Real("total_tax_real")
        total_tax = Int("total_tax")
        total_sales = Real("total_sales")
        self.vars["total_tax_real"] = total_tax_real
        self.vars["total_tax"] = total_tax
        self.vars["total_sales"] = total_sales
        self.add(total_tax_real == Sum(tax_terms), name="law.total_tax_real")
        self.add(total_tax == ToInt(total_tax_real), name="law.total_tax_floor")
        self.add(total_sales == (Sum(sales_terms) if sales_terms else RealVal("0")), name="law.total_sales")
        self.final_tax_z = total_tax
        self.net_taxable_z = total_sales
        if mode == "maximize_sales":
            budget = int(float(p.get("budget_tax", p.get("target_tax", 0))))
            self.add(total_tax <= budget, name=f"budget.total_tax_le_{budget}", group="user_constraint", releasable=True)
            self.objective_z = total_sales
        else:
            self.objective_z = total_tax
        self._add_user_constraints()
        self._optimize()
        return self



def _payload_mentions_field(payload: Dict[str, Any], field: str) -> bool:
    """Return True when a field belongs to the actual user scenario."""
    if field in set(payload.get("free_vars") or []):
        return True
    if field in payload:
        try:
            return abs(float(payload.get(field) or 0)) > 1e-9
        except Exception:
            return True

    constraints = payload.get("constraints") or {}
    if isinstance(constraints, dict):
        token = re.escape(field)
        pattern = re.compile(rf"(?<![A-Za-z0-9_.]){token}(?![A-Za-z0-9_.])")
        for lhs, rules in constraints.items():
            if pattern.search(str(lhs)):
                return True
            if isinstance(rules, dict):
                for rhs in rules.values():
                    values = rhs if isinstance(rhs, (list, tuple)) else [rhs]
                    for one in values:
                        if isinstance(one, str) and pattern.search(one):
                            return True
    return False


def _case_cls_for_payload(payload: Dict[str, Any]):
    # Row-style VAT payloads have row0.* variables. Otherwise use nVAT.
    if any(str(k).startswith("row0.") for k in payload.keys()):
        return VATTaxSMTCase
    return NVATTaxSMTCase


def calculate_sale_tax_tracked(**payload):
    return _case_cls_for_payload(payload)(payload).solve()


def load_payload(path: Optional[str], case_index: int) -> Dict[str, Any]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return normalize_payload(json.load(f))
    return dict(BUILTIN_PAYLOADS[case_index]["payload"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Tracked sale tax UNSAT-core/release lab")
    ap.add_argument("--case", type=int, default=0, choices=range(len(BUILTIN_PAYLOADS)))
    ap.add_argument("--payload")
    ap.add_argument("--json-out", default="sale_tax_unsat_report.json")
    ap.add_argument("--md-out", default="sale_tax_unsat_report.md")
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
    write_report(report, json_out=args.json_out, md_out=args.md_out, title="Sale Tax Tracked SMT Lab")
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
