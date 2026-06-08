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

SEC_ITEMS = {
    "stock": {"fields": ["tp"], "rate": ("tp", 0.003), "cn": "上市公司股票"},
    "bond": {"fields": ["tp"], "rate": ("tp", 0.001), "cn": "公司債"},
}
FUT_ITEMS = {
    "stock_index": {"fields": ["ca"], "rate": ("ca", 0.00002), "qty": "ca", "cn": "臺股指數期貨"},
    "option": {"fields": ["pa", "ca"], "rate": ("pa", 0.001), "qty": "pa", "cn": "選擇權"},
    "gold": {"fields": ["ca"], "rate": ("ca", 0.0000025), "qty": "ca", "cn": "黃金期貨"},
}

_DEFAULT_FIELD_BY_SLUG = {"stock": "tp", "bond": "tp", "stock_index": "ca", "gold": "ca", "option": "pa"}
_VAR_UNDER = re.compile(r"\b([a-z_]+)_(tp|ep|sc|ca|pa)\b", flags=re.I)
_VAR_SPACE = re.compile(r"\b([a-z_]+)\s+(tp|ep|sc|ca|pa)\b", flags=re.I)
_BARE_SLUG = re.compile(r"\b(" + "|".join(map(re.escape, _DEFAULT_FIELD_BY_SLUG.keys())) + r")\b(?!\s*\.(?:tp|ep|sc|ca|pa))", flags=re.I)

def _canon_var_token(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = _VAR_UNDER.sub(r"\1.\2", s)
    t = _VAR_SPACE.sub(r"\1.\2", t)
    def repl(m):
        slug = m.group(1).lower(); fld = _DEFAULT_FIELD_BY_SLUG.get(slug)
        return f"{slug}.{fld}" if fld else m.group(0)
    return _BARE_SLUG.sub(repl, t)


def _normalize_payload_vars(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (payload or {}).items():
        if k in {"free_vars", "constraints"}:
            continue
        out[_canon_var_token(k)] = v
    out["free_vars"] = [_canon_var_token(x).lower() for x in (payload.get("free_vars") or [])]
    cons = {}
    for k, rule in (payload.get("constraints") or {}).items():
        nk = _canon_var_token(k)
        nrule = {}
        for op, rhs in (rule or {}).items():
            if isinstance(rhs, str):
                nrule[op] = _canon_var_token(rhs)
            else:
                nrule[op] = rhs
        cons[nk] = nrule
    out["constraints"] = cons
    if "which" in payload:
        out["which"] = payload["which"]
    return out


BUILTIN_PAYLOADS = [
    {
        "case_id": "securities_case_0_stock_bond_min_tax",
        "description": "股票交易金額為公司債兩倍，兩者合計 600 萬，目標最小化證券交易稅。",
        "payload": {
            "which": "securities",
            "stock.tp": 0,
            "bond.tp": 0,
            "free_vars": ["stock.tp", "bond.tp"],
            "constraints": {
                "stock.tp": {"=": "bond.tp * 2"},
                "stock.tp + bond.tp": {"=": 6_000_000},
            },
        },
    },
    {
        "case_id": "futures_case_1_index_option_gold_min_tax",
        "description": "股價期貨為黃金期貨兩倍，黃金與選擇權權利金有上下限，目標最小化期貨交易稅。",
        "payload": {
            "which": "futures",
            "stock_index.ca": 0,
            "option.pa": 0,
            "option.ca": 0,
            "gold.ca": 0,
            "free_vars": ["stock_index.ca", "option.pa", "gold.ca"],
            "constraints": {
                "stock_index.ca": {">=": 100_000_000, "<=": 150_000_000, "=": "gold.ca * 2"},
                "gold.ca": {">=": 50_000_000, "<=": 75_000_000},
                "stock_index.ca + gold.ca": {"<=": 220_000_000},
                "option.pa": {">=": 5_000, "<=": 20_000},
            },
        },
    },
]
BUILTIN_PAYLOAD = BUILTIN_PAYLOADS[0]["payload"]


class SecuritiesFuturesTaxSMTCase(BaseTrackedTaxCase):
    objective_sense = "min"
    objective_label = "total_tax"

    def __init__(self, payload: Dict[str, Any], *, released=None) -> None:
        super().__init__(_normalize_payload_vars(payload), released=released)

    def build(self):
        if self._built:
            return self
        self._built = True
        self.opt = Optimize()
        which = self.payload.get("which", "securities")
        spec = SEC_ITEMS if which == "securities" else FUT_ITEMS
        tax_terms = []
        qty_terms = []
        for slug, item in spec.items():
            var_objs = {}
            for fld in item["fields"]:
                key = f"{slug}.{fld}"
                z = Real(f"{slug}_{fld}")
                self._add_param(key, z, self._num(key), [("nonnegative", lambda v: v >= 0)], releasable=(key in self.explicit_keys))
                var_objs[fld] = z
            mode, rate = item["rate"]
            if mode == "tp":
                expr = var_objs["tp"] * RealVal(str(rate))
                qty_terms.append(var_objs["tp"])
            elif mode == "ca":
                expr = var_objs["ca"] * RealVal(str(rate))
                qty_terms.append(var_objs[item.get("qty", "ca")])
            elif mode == "pa":
                expr = var_objs["pa"] * RealVal(str(rate))
                qty_terms.append(var_objs[item.get("qty", "pa")])
            else:
                expr = RealVal("0")
            row_tax = Int(f"{slug}_tax")
            self.vars[f"{slug}.tax"] = row_tax
            self.add(row_tax == ToInt(expr), name=f"law.{slug}.tax")
            tax_terms.append(row_tax)
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


def calculate_securities_futures_tax_tracked(**payload):
    return SecuritiesFuturesTaxSMTCase(payload).solve()


def load_payload(path: Optional[str], case_index: int) -> Dict[str, Any]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return normalize_payload(json.load(f))
    return dict(BUILTIN_PAYLOADS[case_index]["payload"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Tracked securities/futures transaction tax UNSAT-core/release lab")
    ap.add_argument("--case", type=int, default=0, choices=range(len(BUILTIN_PAYLOADS)))
    ap.add_argument("--payload")
    ap.add_argument("--json-out", default="securities_futures_tax_unsat_report.json")
    ap.add_argument("--md-out", default="securities_futures_tax_unsat_report.md")
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
        SecuritiesFuturesTaxSMTCase,
        payload,
        auto_release=not getattr(args, "no_release_tests", False),
        auto_combinations=not getattr(args, "no_auto_combinations", False) and not getattr(args, "no_release_tests", False),
        include_non_core=not getattr(args, "core_only", False),
    )
    write_report(report, json_out=args.json_out, md_out=args.md_out, title="Securities Futures Transaction Tax Tracked SMT Lab")
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
