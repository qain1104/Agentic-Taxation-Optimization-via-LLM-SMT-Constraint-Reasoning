# -*- coding: utf-8 -*-
"""
tracked_tax_core.py

Shared utilities for tracked SMT tax solvers:
- named/tracked constraints
- simple linear expression parser, including variable names with dots
- UNSAT-core optimality probe for both minimization and maximization
- automatic release-test generation
- COI-based combination release
- automatic domain-bound / variable-type inference
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from z3 import Bool, RealVal, Solver, is_int_value, sat, set_param, unsat
from z3.z3util import get_vars


def sanitize_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.:-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:180] or "constraint"


def z3_number_value(model, var, default: int = 0):
    v = model.eval(var, model_completion=True)
    try:
        if is_int_value(v):
            return int(v.as_long())
    except Exception:
        pass
    try:
        s = v.as_decimal(20).rstrip("?")
        fv = float(s)
        return int(round(fv)) if abs(fv - round(fv)) < 1e-9 else fv
    except Exception:
        try:
            s = str(v)
            if "/" in s:
                a, b = s.split("/", 1)
                fv = float(a) / float(b)
            else:
                fv = float(s)
            return int(round(fv)) if abs(fv - round(fv)) < 1e-9 else fv
        except Exception:
            return default


def money(v: Any) -> str:
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.4f}"
    return str(v)


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


class LinearExpressionParser:
    """Small AST parser for arithmetic over known Z3 variables.

    It supports variable names that are not valid Python identifiers, such as
    ``car.price`` or ``row1.input_tax_val``, by rewriting them to safe aliases
    before parsing the expression with Python AST.
    """

    def __init__(self, env: Dict[str, Any]):
        self.env = env

    def _sanitize_expr_and_varmap(self, expr_str: str) -> Tuple[str, Dict[str, str]]:
        s = str(expr_str)
        var_map: Dict[str, str] = {}
        # Longer names first to avoid replacing ``cat1`` inside ``cat10``.
        for idx, name in enumerate(sorted(self.env.keys(), key=len, reverse=True)):
            safe = f"v{idx}"
            pattern = rf"(?<![A-Za-z0-9_.]){re.escape(name)}(?![A-Za-z0-9_.])"
            s_new, n = re.subn(pattern, safe, s)
            if n > 0:
                var_map[safe] = name
                s = s_new
        return s, var_map

    def parse(self, text: Any) -> Any:
        if isinstance(text, (int, float)):
            return RealVal(str(float(text)))
        sanitized, var_map = self._sanitize_expr_and_varmap(str(text))
        node = ast.parse(sanitized, mode="eval").body
        return self._expr(node, var_map, str(text))

    def _expr(self, node: ast.AST, var_map: Dict[str, str], original: str) -> Any:
        if isinstance(node, ast.Name):
            if node.id not in var_map:
                raise ValueError(f"unknown variable '{node.id}' in expression {original!r}")
            orig = var_map[node.id]
            return self.env[orig]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return RealVal(str(float(node.value)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._expr(node.operand, var_map, original)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return self._expr(node.operand, var_map, original)
        if isinstance(node, ast.BinOp):
            lhs = self._expr(node.left, var_map, original)
            rhs = self._expr(node.right, var_map, original)
            if isinstance(node.op, ast.Add):
                return lhs + rhs
            if isinstance(node.op, ast.Sub):
                return lhs - rhs
            if isinstance(node.op, ast.Mult):
                return lhs * rhs
            if isinstance(node.op, ast.Div):
                return lhs / rhs
        raise ValueError(f"unsupported expression node {ast.dump(node)} in {original!r}")


def normalize_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "payload" in raw and isinstance(raw.get("payload"), dict):
        return dict(raw["payload"])
    if "user_params" in raw and isinstance(raw.get("user_params"), dict):
        out = dict(raw["user_params"])
        if raw.get("constraints") and "constraints" not in out:
            out["constraints"] = raw["constraints"]
        if raw.get("free_vars") and "free_vars" not in out:
            out["free_vars"] = raw["free_vars"]
        return out
    return dict(raw or {})


class BaseTrackedTaxCase:
    """Base class. Subclasses must implement build()."""

    objective_sense: str = "min"  # "min" or "max"
    objective_label: str = "objective"

    def __init__(self, payload: Dict[str, Any], *, released: Optional[Set[str]] = None) -> None:
        self.payload = normalize_payload(payload)
        self.released = set(released or set())
        self.tracked: List[TrackedConstraint] = []
        self._name_counts: Dict[str, int] = {}
        self.vars: Dict[str, Any] = {}
        self.params: Dict[str, Tuple[Any, float, List[Tuple[str, Any]], bool]] = {}
        self.final_tax_z = None
        self.net_taxable_z = None
        self.objective_z = None
        self.opt = None
        self._built = False
        self.explicit_keys: Set[str] = set(self.payload.keys()) - {"free_vars", "constraints", "mode", "target_tax", "budget_tax", "case_id", "description", "which", "kind"}

    def _unique_name(self, base: str) -> str:
        base = sanitize_name(base)
        n = self._name_counts.get(base, 0)
        self._name_counts[base] = n + 1
        return base if n == 0 else f"{base}__{n}"

    def add(self, expr: Any, *, name: str, group: str = "tax_law", releasable: bool = False, note: str = "") -> None:
        nm = self._unique_name(name)
        active = nm not in self.released and name not in self.released
        rec = TrackedConstraint(nm, group, bool(releasable), expr, str(expr), note, active)
        self.tracked.append(rec)
        if active:
            self.opt.add(expr)

    def _num(self, key: str, default: float = 0.0) -> float:
        v = self.payload.get(key, default)
        if v is None:
            return default
        if isinstance(v, bool):
            return float(int(v))
        try:
            return float(v)
        except Exception:
            return default

    def _add_param(self, name: str, zvar: Any, value: float, checks: Optional[List[Tuple[str, Any]]] = None, *, releasable: Optional[bool] = None) -> None:
        if releasable is None:
            releasable = name in self.explicit_keys
        self.params[name] = (zvar, value, checks or [], bool(releasable))
        self.vars[name] = zvar

    def _bind_params(self) -> None:
        free_vars = set(self.payload.get("free_vars") or [])
        for name, (zv, value, checks, rel) in self.params.items():
            if name not in free_vars:
                self.add(zv == RealVal(str(float(value))), name=f"fixed.{name}", group="fixed_input", releasable=rel)
            for suffix, fn in checks:
                self.add(fn(zv), name=f"domain.{name}.{suffix}", group="domain", releasable=False)

    def _add_user_constraints(self) -> None:
        constraints = self.payload.get("constraints") or {}
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
                    rhs_expr = parser.parse(rhs_one) if isinstance(rhs_one, str) else RealVal(str(float(rhs_one)))
                    expr = op_map[op](lhs_expr, rhs_expr)
                    nm = f"user_constraint.{sanitize_name(lhs)}.{sanitize_name(op)}.{sanitize_name(rhs_one)}"
                    self.add(expr, name=nm, group="user_constraint", releasable=True, note=f"{lhs} {op} {rhs_one}")

    def build(self):
        raise NotImplementedError

    def _optimize(self) -> None:
        if self.objective_z is None:
            self.objective_z = self.final_tax_z
        if self.objective_sense == "max":
            self.opt.maximize(self.objective_z)
        else:
            self.opt.minimize(self.objective_z)

    def solve(self) -> Dict[str, Any]:
        self.build()
        # Defensive fallback: some older *_tracked.py modules directly call
        # self.opt.minimize(...) / maximize(...) instead of BaseTrackedTaxCase._optimize().
        # In that case objective_z may be None even though final_tax_z is set.
        if self.objective_z is None:
            self.objective_z = self.final_tax_z
        if self.objective_z is None:
            return {
                "status": "error",
                "error": "objective_z is not set and final_tax_z is None",
                "released": sorted(self.released),
                "objective_sense": self.objective_sense,
            }
        res = self.opt.check()
        if res != sat:
            return {"status": str(res), "released": sorted(self.released), "objective_sense": self.objective_sense}
        model = self.opt.model()
        fields = {}
        for name, (zv, _original, _checks, _rel) in self.params.items():
            fields[name] = z3_number_value(model, zv)
        derived = {}
        for k, v in self.vars.items():
            if k not in fields:
                try:
                    derived[k] = z3_number_value(model, v)
                except Exception:
                    pass
        return {
            "status": "sat",
            "objective_sense": self.objective_sense,
            "objective_label": self.objective_label,
            "optimum": int(z3_number_value(model, self.objective_z)),
            "tax": int(z3_number_value(model, self.final_tax_z)) if self.final_tax_z is not None else None,
            "net_taxable": int(z3_number_value(model, self.net_taxable_z)) if self.net_taxable_z is not None else None,
            "fields": fields,
            "derived": derived,
            "released": sorted(self.released),
        }

    def prove_no_strictly_better(self, optimum: int, *, minimize_core: bool = True) -> Dict[str, Any]:
        self.build()
        # Defensive fallback for older modules that set final_tax_z but not objective_z.
        if self.objective_z is None:
            self.objective_z = self.final_tax_z
        if self.objective_z is None:
            return {
                "status": "error",
                "expected": "unsat",
                "goal": None,
                "core": [],
                "core_size": 0,
                "tracked_assertion_count": 0,
                "error": "objective_z is not set and final_tax_z is None",
            }
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
            lit = Bool(rec.name)
            s.assert_and_track(rec.expr, lit)
            name_to_rec[rec.name] = rec
        if self.objective_sense == "max":
            goal_expr = self.objective_z >= int(optimum) + 1
            goal_text = f"{self.objective_label} >= {int(optimum) + 1}"
            goal_name = f"GOAL.{sanitize_name(self.objective_label)}_ge_{int(optimum)+1}"
        else:
            goal_expr = self.objective_z <= int(optimum) - 1
            goal_text = f"{self.objective_label} <= {int(optimum) - 1}"
            goal_name = f"GOAL.{sanitize_name(self.objective_label)}_le_{int(optimum)-1}"
        s.assert_and_track(goal_expr, Bool(goal_name))
        res = s.check()
        if res != unsat:
            return {"status": str(res), "expected": "unsat", "goal": goal_text, "core": [], "core_size": 0, "tracked_assertion_count": len(name_to_rec)}
        core_names = [c.decl().name() for c in s.unsat_core()]
        core_json: List[Dict[str, Any]] = []
        summary: Dict[str, int] = {}
        for name in core_names:
            if name == goal_name:
                item = {"name": name, "group": "GOAL", "releasable": False, "expr": goal_text, "note": "strictly better than optimum"}
            else:
                item = name_to_rec[name].to_json()
            summary[item["group"]] = summary.get(item["group"], 0) + 1
            core_json.append(item)
        return {
            "status": "unsat",
            "goal": goal_text,
            "core_size": len(core_names),
            "tracked_assertion_count": len(name_to_rec),
            "core_summary": summary,
            "core_names": core_names,
            "core": core_json,
            "releasable_core": [x["name"] for x in core_json if x.get("releasable")],
            "non_core_releasable": [rec.name for rec in self.tracked if rec.active and rec.releasable and rec.name not in set(core_names)],
        }


def z3_var_names(expr: Any) -> Set[str]:
    try:
        return {v.decl().name() for v in get_vars(expr)}
    except Exception:
        return set()


def build_coi_index(tracked: List[TrackedConstraint], *, active_only: bool = True) -> Dict[str, Any]:
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
        if rec.group == "fixed_input" and len(vars_in_expr) == 1:
            fixed_var_to_constraint[next(iter(vars_in_expr))] = rec.name
    return {"constraint_to_vars": constraint_to_vars, "var_to_constraints": var_to_constraints, "fixed_var_to_constraint": fixed_var_to_constraint, "constraint_by_name": constraint_by_name}


def infer_release_tests_from_probe(probe: Dict[str, Any], *, include_non_core: bool = True) -> List[Tuple[str, List[str]]]:
    tests: List[Tuple[str, List[str]]] = []
    seen: Set[str] = set()
    def add_test(name: str, source: str) -> None:
        if name and name not in seen:
            seen.add(name)
            tests.append((f"auto release {name} ({source})", [name]))
    for name in probe.get("releasable_core") or []:
        add_test(str(name), "core")
    if include_non_core:
        for name in probe.get("non_core_releasable") or []:
            add_test(str(name), "non_core")
    return tests


def infer_coi_combo_tests(tracked: List[TrackedConstraint], seed_release_names: Sequence[str], *, max_depth: int = 1, max_combo_size: int = 4, max_tests: int = 30, include_groups: Optional[Set[str]] = None) -> List[Tuple[str, List[str]]]:
    if include_groups is None:
        include_groups = {"tax_law", "objective_link", "branch_rule"}
    idx = build_coi_index(tracked, active_only=True)
    constraint_to_vars = idx["constraint_to_vars"]
    var_to_constraints = idx["var_to_constraints"]
    fixed_var_to_constraint = idx["fixed_var_to_constraint"]
    constraint_by_name = idx["constraint_by_name"]
    tests: List[Tuple[str, List[str]]] = []
    seen: Set[Tuple[str, ...]] = set()
    for seed in sorted({str(x) for x in seed_release_names}):
        seed_rec = constraint_by_name.get(seed)
        if seed_rec is None or seed_rec.group != "fixed_input":
            continue
        frontier_vars = set(constraint_to_vars.get(seed, set()))
        visited_vars = set(frontier_vars)
        visited_constraints: Set[str] = set()
        related_fixed: Set[str] = {seed}
        for _ in range(max_depth):
            next_frontier: Set[str] = set()
            for var_name in frontier_vars:
                for cname in var_to_constraints.get(var_name, set()):
                    if cname in visited_constraints:
                        continue
                    visited_constraints.add(cname)
                    crec = constraint_by_name.get(cname)
                    if crec is None or crec.group not in include_groups:
                        continue
                    vars_in_c = constraint_to_vars.get(cname, set())
                    for v2 in vars_in_c:
                        fixed_name = fixed_var_to_constraint.get(v2)
                        if fixed_name:
                            related_fixed.add(fixed_name)
                    for v2 in vars_in_c:
                        if v2 not in visited_vars:
                            visited_vars.add(v2)
                            next_frontier.add(v2)
            frontier_vars = next_frontier
        if len(related_fixed) <= 1:
            continue
        combo = tuple(sorted(related_fixed))[:max_combo_size]
        if combo in seen:
            continue
        seen.add(combo)
        tests.append((f"auto COI combo from {seed}: {' + '.join(combo)}", list(combo)))
        if len(tests) >= max_tests:
            break
    return tests


def fixed_field_name(release_name: str) -> Optional[str]:
    return release_name[len("fixed."):] if release_name.startswith("fixed.") else None


META_PAYLOAD_KEYS = {
    "free_vars", "constraints", "mode", "target_tax", "budget_tax",
    "case_id", "description", "which", "kind", "objective",
}


def field_appears_in_user_constraints(payload: Dict[str, Any], field: str) -> bool:
    """Return True if a field is mentioned in any user constraint expression."""
    constraints = payload.get("constraints") or {}
    if not isinstance(constraints, dict):
        return False

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


def is_default_value_field(payload: Dict[str, Any], field: str) -> bool:
    """Return True when a fixed field behaves like a portal default value.

    In the built-in cases, many portal fields are included in the payload with
    value 0 because the official forms default blank numeric fields to zero.
    These zero-valued fixed fields are valid default-assumption candidates.
    By contrast, free variables, fields used in user constraints, and nonzero
    factual values define the case and are not released under default_only.
    """
    if not field:
        return False
    if field in set(payload.get("free_vars") or []):
        return False
    if field_appears_in_user_constraints(payload, field):
        return False
    if field not in payload:
        return True
    if field in META_PAYLOAD_KEYS:
        return False

    value = payload.get(field)
    if value is None:
        return True
    if isinstance(value, bool):
        return value is False
    try:
        return abs(float(value)) < 1e-9
    except Exception:
        return False


def is_allowed_release_name(name: str, payload: Dict[str, Any], *, release_scope: str) -> bool:
    """Filter release candidates according to RQ4 release scope.

    default_only:
        Only fixed.<field> default-value assumptions are releasable.
        User constraints, budget constraints, free variables, and explicit facts
        define the case and are excluded.

    fixed_only:
        All fixed.<field> constraints are releasable, but user_constraint.* and
        budget.* are excluded.

    all:
        Preserve the previous broad behavior.
    """
    name = str(name)

    if release_scope == "all":
        return True

    if not name.startswith("fixed."):
        return False

    if release_scope == "fixed_only":
        return True

    if release_scope == "default_only":
        field = fixed_field_name(name)
        return bool(field and is_default_value_field(payload, field))

    raise ValueError(f"unknown release_scope: {release_scope}")


def scoped_release_tests_from_probe(
    probe: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    include_non_core: bool = True,
    release_scope: str = "default_only",
    tracked: Optional[List[TrackedConstraint]] = None,
) -> List[Tuple[str, List[str]]]:
    """Generate release tests after applying release-scope filtering."""
    if release_scope == "all":
        return infer_release_tests_from_probe(probe, include_non_core=include_non_core)

    tests: List[Tuple[str, List[str]]] = []
    seen: Set[str] = set()

    def add_test(name: str, source: str) -> None:
        if not name or name in seen:
            return
        if not is_allowed_release_name(name, payload, release_scope=release_scope):
            return
        seen.add(name)
        tests.append((f"auto release {name} ({source})", [name]))

    # Use the full core, not only probe['releasable_core']. Older modules often
    # mark explicit user facts as releasable and default assumptions as not
    # releasable. For default_only RQ4, the scope filter is the source of truth.
    for item in probe.get("core") or []:
        if isinstance(item, dict):
            add_test(str(item.get("name")), "core")

    if include_non_core and tracked is not None:
        core_names = set(probe.get("core_names") or [])
        for rec in tracked:
            if rec.active and rec.name not in core_names:
                add_test(rec.name, "non_core")

    return tests


def scope_combo_tests(
    combo_tests: List[Tuple[str, List[str]]],
    payload: Dict[str, Any],
    *,
    release_scope: str,
) -> List[Tuple[str, List[str]]]:
    """Remove disallowed names from COI combos and drop degenerate combos."""
    if release_scope == "all":
        return combo_tests

    out: List[Tuple[str, List[str]]] = []
    seen: Set[Tuple[str, ...]] = set()
    for label, names in combo_tests:
        scoped = [
            n for n in names
            if is_allowed_release_name(n, payload, release_scope=release_scope)
        ]
        scoped = sorted(set(scoped))
        if len(scoped) < 2:
            continue
        key = tuple(scoped)
        if key in seen:
            continue
        seen.add(key)
        out.append((label, scoped))
    return out


def infer_field_type(field: Optional[str]) -> str:
    if not field:
        return "unknown"
    f = field.lower()
    if f.endswith("_count") or "count" in f or f.startswith("cnt_"):
        return "count_fact"
    if any(tok in f for tok in ["quantity", "qty", "sales", "amount", "base", "price", "tp", "ca", "pa", "input", "output"]):
        return "business_amount_or_quantity"
    if any(tok in f for tok in ["deduction", "fee", "loss", "debt", "fines", "burden", "offset", "credit", "exemption"]):
        return "monetary_claim"
    if any(tok in f for tok in ["value", "val", "land", "house", "stock", "cash", "gift", "cost", "revenue", "allowance", "expenses"]):
        return "monetary_base_or_asset"
    return "unknown_fixed_input"


def _parse_simple_upper_bound(expr_str: str, var_name: str) -> Optional[int]:
    patterns = [rf"\b{re.escape(var_name)}\b\s*<=\s*(-?\d+)", rf"(-?\d+)\s*>=\s*\b{re.escape(var_name)}\b"]
    for pat in patterns:
        m = re.search(pat, expr_str)
        if m:
            nums = [g for g in m.groups() if g is not None]
            try:
                return int(nums[-1])
            except Exception:
                return None
    return None


def build_release_inference_context(tracked: List[TrackedConstraint]) -> Dict[str, Any]:
    idx = build_coi_index(tracked, active_only=True)
    constraint_to_vars = idx["constraint_to_vars"]
    var_to_constraints = idx["var_to_constraints"]
    constraint_by_name = idx["constraint_by_name"]
    fixed_info: Dict[str, Dict[str, Any]] = {}
    for cname, rec in constraint_by_name.items():
        if rec.group != "fixed_input":
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
                    lname = used_rec.name.lower(); estr = used_rec.expr_str.lower()
                    if "cap" in lname or "limit" in lname or "upper" in lname or ("if(" in estr and (">=" in estr or "<=" in estr)):
                        cap_constraints.append(used_rec.name)
        fixed_info[cname] = {"field": field, "field_type": field_type, "z3_var": z3_var, "domain_upper_bounds": sorted(set(domain_upper_bounds)), "law_constraints": sorted(set(law_constraints)), "cap_constraints": sorted(set(cap_constraints)), "has_domain_upper_bound": bool(domain_upper_bounds), "has_law_cap": bool(cap_constraints)}
    return {"fixed_info": fixed_info, "coi_index": idx}


def infer_release_policy(release_name: str, *, inference_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if release_name.startswith("user_constraint."):
        return {"kind": "semantic_defining_constraint", "safe": False, "source": "name_rule", "reason": "This user-level constraint may define the case."}
    if not release_name.startswith("fixed."):
        return {"kind": "non_fixed_or_unknown", "safe": False, "source": "name_rule", "reason": "Only fixed input constraints are considered by default."}
    info = ((inference_context or {}).get("fixed_info") or {}).get(release_name, {})
    field = info.get("field") or fixed_field_name(release_name)
    field_type = info.get("field_type") or infer_field_type(field)
    if field_type == "count_fact":
        return {"kind": "fact_field_needs_bound", "safe": False, "source": "variable_type_inference", "field_type": field_type, "reason": "Count fields are factual and need user-provided bounds."}
    if info.get("has_law_cap"):
        return {"kind": "bounded_by_law_formula", "safe": True, "source": "law_cap_inference", "field_type": field_type, "cap_constraints": info.get("cap_constraints"), "reason": "A law-level cap/limit formula bounds this release."}
    if field_type in {"monetary_claim", "monetary_base_or_asset", "business_amount_or_quantity"}:
        return {"kind": "amount_needs_bound", "safe": False, "source": "variable_type_inference", "field_type": field_type, "reason": "The amount/quantity requires factual evidence or explicit bounds."}
    if info.get("has_domain_upper_bound"):
        return {"kind": "domain_bounded_but_needs_review", "safe": False, "source": "domain_bound_inference", "field_type": field_type, "reason": "A domain bound exists, but no law cap was detected."}
    return {"kind": "unknown_needs_review", "safe": False, "source": "fallback", "field_type": field_type, "reason": "No reliable safe bound inferred."}


def infer_combined_release_policy(released: Sequence[str], *, inference_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    policies = [infer_release_policy(x, inference_context=inference_context) for x in released]
    if any(p.get("kind") == "semantic_defining_constraint" for p in policies):
        return {"kind": "semantic_change", "safe": False, "source": "combined_policy", "component_policies": policies, "reason": "At least one release changes the case definition."}
    if policies and all(p.get("safe") is True for p in policies):
        return {"kind": "bounded_release", "safe": True, "source": "combined_policy", "component_policies": policies, "reason": "All releases are structurally bounded."}
    return {"kind": "needs_domain_bound", "safe": False, "source": "combined_policy", "component_policies": policies, "reason": "At least one release is factual, unbounded, or needs review."}


def classify_release_result(row: Dict[str, Any], base_optimum: int, *, objective_sense: str, inference_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    released = [str(x) for x in (row.get("released") or [])]
    status = row.get("status")
    new_optimum = row.get("new_optimum")
    delta = row.get("delta_vs_base")
    policy = infer_combined_release_policy(released, inference_context=inference_context)
    if status != "sat" or not isinstance(new_optimum, int):
        return {"category": "infeasible_or_unknown", "is_recommendable": False, "policy": policy, "reason": "Release test did not return SAT."}
    if policy.get("kind") == "semantic_change":
        return {"category": "semantic_change", "is_recommendable": False, "policy": policy, "reason": policy.get("reason")}
    if delta == 0:
        return {"category": "no_effect", "is_recommendable": False, "policy": policy, "reason": "No objective improvement."}
    improved = (delta < 0) if objective_sense == "min" else (delta > 0)
    if not improved:
        return {
            "category": "diagnostic_anomaly",
            "is_recommendable": False,
            "policy": policy,
            "reason": (
                "A pure release relaxation should not worsen the optimum. "
                "This indicates a diagnostic issue such as objective encoding, "
                "unbounded default variables, integer/real rounding, or solver optimization behavior."
            ),
        }
    impact = (-delta) if objective_sense == "min" else delta
    looks_extreme = False
    if objective_sense == "min" and new_optimum < 0:
        looks_extreme = True
    if policy.get("safe") is True and not looks_extreme:
        return {"category": "useful_bounded_release", "is_recommendable": True, "objective_improvement": impact, "policy": policy, "reason": "Objective improves and a structural bound was inferred."}
    return {"category": "unsafe_needs_domain_bound", "is_recommendable": False, "objective_improvement": impact, "policy": policy, "reason": "Objective improves but safe bounds were not established or result looks extreme."}


def execute_release_tests(case_cls, payload: Dict[str, Any], *, optimum: int, objective_sense: str, release_tests: List[Tuple[str, List[str]]], inference_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, names in release_tests:
        c = case_cls(payload, released=set(names))
        sol = c.solve()
        row = {"test": label, "released": names, "status": sol.get("status"), "new_optimum": sol.get("optimum"), "delta_vs_base": sol.get("optimum") - optimum if isinstance(sol.get("optimum"), int) else None, "tax": sol.get("tax"), "net_taxable": sol.get("net_taxable"), "selected_fields": sol.get("fields"), "derived": sol.get("derived")}
        row["classification"] = classify_release_result(row, optimum, objective_sense=objective_sense, inference_context=inference_context)
        rows.append(row)
    return rows


def run_tracked_analysis(case_cls, payload: Dict[str, Any], *, release_tests: Optional[List[Tuple[str, List[str]]]] = None, auto_release: bool = True, auto_combinations: bool = True, include_non_core: bool = True, release_scope: str = "default_only") -> Dict[str, Any]:
    base_case = case_cls(payload)
    base = base_case.solve()
    if base.get("status") != "sat":
        return {"base": base, "error": "base case is not sat"}
    optimum = int(base["optimum"])
    objective_sense = base.get("objective_sense", "min")
    probe = base_case.prove_no_strictly_better(optimum)
    inference_context = build_release_inference_context(base_case.tracked)
    if release_tests is not None:
        tests_to_run = release_tests; source = "manual"
    elif auto_release:
        tests_to_run = scoped_release_tests_from_probe(
            probe,
            payload,
            include_non_core=include_non_core,
            release_scope=release_scope,
            tracked=base_case.tracked,
        ); source = f"auto_from_unsat_core_{release_scope}"
    else:
        tests_to_run = []; source = "none"
    rows = execute_release_tests(case_cls, payload, optimum=optimum, objective_sense=objective_sense, release_tests=tests_to_run, inference_context=inference_context)
    combo_tests: List[Tuple[str, List[str]]] = []
    coi_count = 0
    if auto_combinations:
        seed_names = [n for _, names in tests_to_run for n in names]
        coi_combo_tests = infer_coi_combo_tests(base_case.tracked, seed_names, max_depth=1)
        coi_combo_tests = scope_combo_tests(coi_combo_tests, payload, release_scope=release_scope)
        seen: Set[Tuple[str, ...]] = set()
        for label, names in coi_combo_tests:
            key = tuple(sorted(names))
            if key not in seen:
                seen.add(key); combo_tests.append((label, names)); coi_count += 1
        if combo_tests:
            rows.extend(execute_release_tests(case_cls, payload, optimum=optimum, objective_sense=objective_sense, release_tests=combo_tests, inference_context=inference_context))
    summary = {"recommended": [], "unsafe_needs_domain_bound": [], "no_effect": [], "semantic_change": [], "diagnostic_anomaly": [], "worse": [], "unknown": []}
    for row in rows:
        cat = (row.get("classification") or {}).get("category")
        if cat == "useful_bounded_release": summary["recommended"].append(row)
        elif cat == "unsafe_needs_domain_bound": summary["unsafe_needs_domain_bound"].append(row)
        elif cat == "no_effect": summary["no_effect"].append(row)
        elif cat == "semantic_change": summary["semantic_change"].append(row)
        elif cat == "diagnostic_anomaly": summary["diagnostic_anomaly"].append(row)
        elif cat == "worse": summary["worse"].append(row)
        else: summary["unknown"].append(row)
    return {"base": base, "unsat_core_probe": probe, "release_generation": {"source": source, "single_test_count": len(tests_to_run), "combo_test_count": len(combo_tests), "coi_combo_count": coi_count, "combo_strategy": "coi_only", "include_non_core": include_non_core, "release_policy": "automatic_domain_bound_and_variable_type_inference", "release_scope": release_scope}, "release_tests": rows, "release_summary": summary}


def render_markdown(report: Dict[str, Any], *, title: str = "Tracked Tax SMT Unsat Core Lab") -> str:
    base = report.get("base", {})
    probe = report.get("unsat_core_probe", {})
    gen = report.get("release_generation", {})
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Base solve")
    lines.append(f"- status: `{base.get('status')}`")
    lines.append(f"- objective: `{base.get('objective_label')}` / `{base.get('objective_sense')}`")
    lines.append(f"- optimum: **{money(base.get('optimum'))}**")
    lines.append(f"- tax: **{money(base.get('tax'))}**")
    lines.append(f"- net_taxable: **{money(base.get('net_taxable'))}**")
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
    lines.append("### Original releasable constraints in this core before release-scope filtering")
    for name in probe.get("releasable_core") or ["none"]:
        lines.append(f"- `{name}`" if name != "none" else "- none")
    lines.append("")
    lines.append("## Release generation")
    lines.append(f"- source: `{gen.get('source')}`")
    lines.append(f"- single release tests: `{gen.get('single_test_count')}`")
    lines.append(f"- COI combination tests: `{gen.get('coi_combo_count')}`")
    lines.append(f"- release policy: `{gen.get('release_policy')}`")
    lines.append(f"- release scope: `{gen.get('release_scope')}`")
    lines.append("")
    lines.append("## Release tests")
    lines.append("| test | released | status | new optimum | delta vs base | tax | classification |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for row in report.get("release_tests") or []:
        names = ", ".join(f"`{n}`" for n in row.get("released") or [])
        cat = ((row.get("classification") or {}).get("category") or "")
        lines.append(f"| {row.get('test')} | {names} | `{row.get('status')}` | {money(row.get('new_optimum'))} | {money(row.get('delta_vs_base'))} | {money(row.get('tax'))} | `{cat}` |")
    lines.append("")
    lines.append("## Full core names")
    for name in probe.get("core_names") or []:
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append("> Reminder: Z3 unsat cores are not unique; use group/name patterns rather than exact ordering.")
    return "\n".join(lines)


def write_report(report: Dict[str, Any], *, json_out: str, md_out: str, title: str) -> None:
    Path(json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    Path(md_out).write_text(render_markdown(report, title=title), encoding="utf-8")


# ---------------------------------------------------------------------------
# Generic CLI helper
# ---------------------------------------------------------------------------

def load_cli_payload(payload_path: Optional[str], builtin_payloads: Sequence[Dict[str, Any]], case_index: int) -> Dict[str, Any]:
    """Load a payload for a tracked tax script.

    Supports two common built-in formats:
    1. {"case_id": ..., "description": ..., "payload": {...}}
    2. direct payload dict
    """
    if payload_path:
        with open(payload_path, "r", encoding="utf-8") as f:
            return normalize_payload(json.load(f))
    if not builtin_payloads:
        return {}
    item = builtin_payloads[case_index]
    if isinstance(item, dict) and isinstance(item.get("payload"), dict):
        return dict(item["payload"])
    return normalize_payload(item)


def cli_main(
    case_cls,
    builtin_payloads: Sequence[Dict[str, Any]],
    *,
    title: str = "Tracked Tax SMT Lab",
    default_json: str = "tracked_tax_unsat_report.json",
    default_md: str = "tracked_tax_unsat_report.md",
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Shared command-line entry point for *_tracked.py modules.

    This keeps older modules such as cargo_tax_tracked.py working while also
    exposing the same speed/debug flags across modules that choose to use it.
    Modules with their own main() can still import and use BaseTrackedTaxCase,
    normalize_payload, run_tracked_analysis, and write_report directly.
    """
    max_case = max(0, len(builtin_payloads) - 1)
    ap = argparse.ArgumentParser(description=title)
    ap.add_argument("--case", type=int, default=0, choices=range(len(builtin_payloads)) if builtin_payloads else [0])
    ap.add_argument("--payload", help="JSON file containing either a direct payload, {payload: ...}, or {user_params: ...}.")
    ap.add_argument("--json-out", default=default_json)
    ap.add_argument("--md-out", default=default_md)
    ap.add_argument("--print-core", action="store_true", help="Print full unsat-core entries to stdout.")
    ap.add_argument("--no-auto-combinations", action="store_true", help="Disable COI combination release tests.")
    ap.add_argument("--core-only", action="store_true", help="Only test releasable constraints that appear in the returned core.")
    ap.add_argument(
        "--release-scope",
        default="default_only",
        choices=["default_only", "fixed_only", "all"],
        help=(
            "Which constraints may be released. default_only releases only "
            "fixed default-value assumptions; fixed_only releases all fixed.* "
            "constraints; all preserves the previous broad behavior."
        ),
    )
    ap.add_argument("--no-release-tests", action="store_true", help="Do not run release tests; only solve base and extract the optimality core.")
    ap.add_argument("--probe-only", action="store_true", help="Alias for --no-release-tests --no-auto-combinations --core-only.")
    args = ap.parse_args(argv)

    payload = load_cli_payload(args.payload, builtin_payloads, args.case)
    if args.probe_only:
        args.no_release_tests = True
        args.no_auto_combinations = True
        args.core_only = True

    report = run_tracked_analysis(
        case_cls,
        payload,
        auto_release=not args.no_release_tests,
        auto_combinations=not args.no_auto_combinations and not args.no_release_tests,
        include_non_core=not args.core_only,
        release_scope=args.release_scope,
    )
    write_report(report, json_out=args.json_out, md_out=args.md_out, title=title)

    case_meta = None
    if not args.payload and builtin_payloads:
        case_meta = builtin_payloads[args.case]
    if isinstance(case_meta, dict) and (case_meta.get("case_id") or case_meta.get("description")):
        print("=== Built-in case ===")
        print(f"case: {args.case}")
        if case_meta.get("case_id"):
            print(f"case_id: {case_meta.get('case_id')}")
        if case_meta.get("description"):
            print(f"description: {case_meta.get('description')}")
        print("")

    base = report.get("base", {})
    probe = report.get("unsat_core_probe", {})
    gen = report.get("release_generation", {})
    print("=== Base solve ===")
    print(f"status: {base.get('status')}")
    print(f"objective: {base.get('objective_label')} / {base.get('objective_sense')}")
    print(f"optimum: {money(base.get('optimum'))}")
    if base.get("tax") is not None:
        print(f"tax: {money(base.get('tax'))}")
    if base.get("net_taxable") is not None:
        print(f"net_taxable: {money(base.get('net_taxable'))}")
    print("")
    print("=== UNSAT probe ===")
    print(f"goal: {probe.get('goal')}")
    print(f"status: {probe.get('status')}")
    print(f"tracked_assertions: {probe.get('tracked_assertion_count')}")
    print(f"core_size: {probe.get('core_size')}")
    if probe.get("core_summary") is not None:
        print(f"core_summary: {json.dumps(probe.get('core_summary'), ensure_ascii=False)}")
    print("")
    print("=== Release generation ===")
    print(f"source: {gen.get('source')}")
    print(f"single_tests: {gen.get('single_test_count')}")
    print(f"combo_tests: {gen.get('combo_test_count')}")
    print(f"coi_combo_tests: {gen.get('coi_combo_count')}")
    print(f"include_non_core: {gen.get('include_non_core')}")
    print(f"release_policy: {gen.get('release_policy')}")
    print(f"release_scope: {gen.get('release_scope')}")
    print("")
    print("=== Release tests ===")
    rows = report.get("release_tests") or []
    if not rows:
        print("- skipped")
    for row in rows:
        cls = row.get("classification") or {}
        print(
            f"- {row.get('test')}: status={row.get('status')}, "
            f"new_optimum={money(row.get('new_optimum'))}, "
            f"delta={money(row.get('delta_vs_base'))}, "
            f"class={cls.get('category')}"
        )
    print("")
    print(f"wrote: {args.json_out}")
    print(f"wrote: {args.md_out}")

    if args.print_core:
        print("")
        print("=== Full core ===")
        for item in probe.get("core", []):
            print(json.dumps(item, ensure_ascii=False, default=str))
    return 0


__all__ = [
    "TrackedConstraint",
    "LinearExpressionParser",
    "BaseTrackedTaxCase",
    "sanitize_name",
    "normalize_payload",
    "money",
    "z3_number_value",
    "z3_var_names",
    "build_coi_index",
    "infer_release_tests_from_probe",
    "scoped_release_tests_from_probe",
    "is_allowed_release_name",
    "is_default_value_field",
    "infer_coi_combo_tests",
    "build_release_inference_context",
    "infer_release_policy",
    "infer_combined_release_policy",
    "classify_release_result",
    "execute_release_tests",
    "run_tracked_analysis",
    "render_markdown",
    "write_report",
    "load_cli_payload",
    "cli_main",
]
