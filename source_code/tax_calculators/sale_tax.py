# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from typing import Dict, List, Set, Tuple, Any, Optional

from z3 import Optimize, Real, RealVal, Sum, sat
from .constraint_utils import apply_linear_constraints

# ─────────────────────────────────────────────────────────────
# 非加值型稅率（照你列的 8 類）
# ─────────────────────────────────────────────────────────────
NVAT_RATE: Dict[int, float] = {
    1: 0.01, 2: 0.01, 3: 0.02, 4: 0.05,
    5: 0.05, 6: 0.15, 7: 0.25, 8: 0.001,
}

# ─────────────────────────────────────────────────────────────
# 內部小工具
# ─────────────────────────────────────────────────────────────

def _to_number(v):
    """將 z3 數值安全轉成 float。"""
    try:
        return float(str(v))
    except Exception:
        return float(v.as_decimal(20).replace("?", ""))


# ====================================================================
# 1) 加值型 VAT：只吃兩個輸入（銷項稅額、進項稅額），最小化淨稅額
# ====================================================================

def minimize_vat_tax(
    *,
    output_tax_amt: int,   # 銷項課稅基礎（未乘稅）
    input_tax_amt: int,    # 進項課稅基礎（未乘稅）
    free_vars: Optional[List[str]] = None,
    constraints: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    目標：minimize total_vat
    定義：
      out5  = output_tax_amt * 0.05
      in5   = input_tax_amt  * 0.05
      total_vat = out5 - in5

    可在 constraints 裡使用變數名：
      output_tax_amt, input_tax_amt, out5, in5, total_vat
    約束解析交由 constraint_utils.apply_linear_constraints 處理。
    """
    from z3 import Real  # 局部引入，避免上面 import 太多

    RATE_5 = RealVal(0.05)

    started = time.perf_counter()
    free_vars = set(free_vars or [])
    cons = constraints or {}

    # ---- baseline（固定輸入，不帶 constraints）----
    baseline_out5 = int(output_tax_amt * 0.05)
    baseline_in5  = int(input_tax_amt  * 0.05)
    baseline = baseline_out5 - baseline_in5

    # ---- baseline + constraints 可行性 ----
    opt0 = Optimize()
    out0 = Real("output_base_0")
    in0  = Real("input_base_0")
    out5_0 = Real("out5_0")
    in5_0  = Real("in5_0")
    tot0   = Real("total_vat_0")

    opt0.add(out0 == RealVal(output_tax_amt))
    opt0.add(in0  == RealVal(input_tax_amt))
    opt0.add(out5_0 == out0 * RATE_5)
    opt0.add(in5_0  == in0  * RATE_5)
    opt0.add(tot0   == out5_0 - in5_0)

    var_map0 = {
        "output_tax_amt": out0,
        "input_tax_amt":  in0,
        "out5":           out5_0,
        "in5":            in5_0,
        "total_vat":      tot0,
    }
    params_for_constraints0 = {name: (zv,) for name, zv in var_map0.items()}
    apply_linear_constraints(opt0, params_for_constraints0, cons, debug=False)

    if opt0.check() == sat:
        base_stat = "sat"
        base_wc = int(_to_number(opt0.model().eval(tot0)))
    else:
        base_stat = "unsat"
        base_wc = None

    # ---- 無 free_vars → 回 baseline（補回 final_params）----
    if not free_vars:
        final_params = {
            "output_tax_amt": {"value": int(output_tax_amt), "type": "fixed"},
            "input_tax_amt":  {"value": int(input_tax_amt),  "type": "fixed"},
            "out5":           {"value": int(baseline_out5),  "type": "derived"},
            "in5":            {"value": int(baseline_in5),   "type": "derived"},
            "total_vat":      {"value": int(baseline),       "type": "derived"},
        }
        return {
            "mode": "minimize",
            "baseline": baseline,
            "out5%": baseline_out5,
            "in5%": baseline_in5,
            "baseline_status": base_stat,
            "baseline_with_constraints": base_wc,
            "free_vars": [],
            "constraints": cons,
            "final_params": final_params,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    # ---- 有 free_vars → 最小化 total_vat ----
    opt = Optimize()
    out = Real("output_tax_amt")
    inn = Real("input_tax_amt")
    out5 = Real("out5")
    in5 = Real("in5")
    total = Real("total_vat")

    # 綁定可變/固定（允許非負）
    if "output_tax_amt" in free_vars:
        opt.add(out >= 0)
    else:
        opt.add(out == RealVal(output_tax_amt))

    if "input_tax_amt" in free_vars:
        opt.add(inn >= 0)
    else:
        opt.add(inn == RealVal(input_tax_amt))

    # 5% 計算
    opt.add(out5 == out * RATE_5)
    opt.add(in5  == inn * RATE_5)
    opt.add(total == out5 - in5)

    var_map = {
        "output_tax_amt": out,
        "input_tax_amt":  inn,
        "out5":           out5,
        "in5":            in5,
        "total_vat":      total,
    }
    params_for_constraints = {name: (zv,) for name, zv in var_map.items()}
    apply_linear_constraints(opt, params_for_constraints, cons, debug=False)

    opt.minimize(total)

    if opt.check() != sat:
        return {
            "mode": "minimize",
            "baseline": baseline,
            "out5%": baseline_out5,
            "in5%": baseline_in5,
            "baseline_status": base_stat,
            "baseline_with_constraints": base_wc,
            "free_vars": list(free_vars),
            "constraints": cons,
            "status": "unsat",
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    m = opt.model()
    optimized = int(_to_number(m.eval(total)))

    # —— 參數調整（diff）——
    param_diff: Dict[str, Dict[str, int]] = {}
    for k, zv in (("output_tax_amt", out), ("input_tax_amt", inn)):
        if k in free_vars:
            orig = output_tax_amt if k == "output_tax_amt" else input_tax_amt
            optv = int(_to_number(m.eval(zv)))
            if optv != orig:
                param_diff[k] = {
                    "original": int(orig),
                    "optimized": int(optv),
                    "difference": int(optv - orig),
                }

    # —— 最佳解輸入值（final_params）——
    out_val   = int(_to_number(m.eval(out)))
    in_val    = int(_to_number(m.eval(inn)))
    out5_val  = int(_to_number(m.eval(out5)))
    in5_val   = int(_to_number(m.eval(in5)))
    total_val = int(_to_number(m.eval(total)))

    final_params = {
        "output_tax_amt": {"value": out_val,   "type": ("free" if "output_tax_amt" in free_vars else "fixed")},
        "input_tax_amt":  {"value": in_val,    "type": ("free" if "input_tax_amt"  in free_vars else "fixed")},
        "out5":           {"value": out5_val,  "type": "derived"},
        "in5":            {"value": in5_val,   "type": "derived"},
        "total_vat":      {"value": total_val, "type": "derived"},
    }

    return {
        "mode": "minimize",
        "baseline": baseline,
        "out5%": baseline_out5,
        "in5%": baseline_in5,
        "baseline_status": base_stat,
        "baseline_with_constraints": base_wc,
        "optimized": optimized,
        "param_diff": param_diff,   # 向下相容
        "diff": param_diff,         # 標準鍵名，供報告使用
        "final_params": final_params,
        "free_vars": list(free_vars),
        "constraints": cons,
        "status": "sat",
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


# ====================================================================
# 2) 非加值型 nVAT：8 個變數全部展開（預設 0）
#    - minimize：最小化總稅額
#    - maximize：在稅額上限內最大化總銷售額
# 變數命名（兩組別名都支援）：cat1..cat8 / 對應語義別名
# ====================================================================

_NVAT_KEYS = [
    ("cat1", 1, "small_business"),
    ("cat2", 2, "reinsurance"),
    ("cat3", 3, "finance_core"),
    ("cat4", 4, "bank_insurance_core"),
    ("cat5", 5, "finance_noncore"),
    ("cat6", 6, "nightclub"),
    ("cat7", 7, "geisha"),
    ("cat8", 8, "agri_wholesale"),
]

def _normalize_nvat_inputs(kwargs: Dict[str, Any]) -> Dict[str, int]:
    """把缺省值補成 0，同時支援 catN 與語義別名。"""
    out: Dict[str, int] = {}
    alias_to_cat = {
        "small_business": "cat1",
        "reinsurance": "cat2",
        "finance_core": "cat3",
        "bank_insurance_core": "cat4",
        "finance_noncore": "cat5",
        "nightclub": "cat6",
        "geisha": "cat7",
        "agri_wholesale": "cat8",
    }
    # 先把語義別名轉成 catN 放進一份乾淨 dict
    canon_kwargs: Dict[str, Any] = {}
    for k, v in (kwargs or {}).items():
        if k in alias_to_cat:
            canon_kwargs[alias_to_cat[k]] = v
        else:
            canon_kwargs[k] = v

    for cat_key, _, alias in _NVAT_KEYS:
        v = canon_kwargs.get(cat_key, canon_kwargs.get(alias, 0)) or 0
        out[cat_key] = int(v)
    return out

def _build_nvat_model(nvars: Dict[str, int], free_vars: Set[str], constraints: Dict):
    opt = Optimize()
    var_map: Dict[str, Real] = {}
    row_terms: List[Tuple[Any, str]] = []

    for cat_key, code, _ in _NVAT_KEYS:
        rv = Real(cat_key)
        var_map[cat_key] = rv
        if cat_key in free_vars:
            opt.add(rv >= 0)
        else:
            opt.add(rv == RealVal(nvars[cat_key]))
        row_terms.append((rv * RealVal(NVAT_RATE[code]), cat_key))

    total = Real("total_tax")
    opt.add(total == Sum([t for t, _ in row_terms]))

    # 使用共用 apply_linear_constraints，讓 constraints 可以用 cat1..cat8 以及 total_tax
    params_for_constraints = {name: (zv,) for name, zv in var_map.items()}
    params_for_constraints["total_tax"] = (total,)
    apply_linear_constraints(opt, params_for_constraints, constraints or {}, debug=False)

    return opt, var_map, row_terms, total

def _nvat_final_params_from_items(
    items: List[Dict[str, int]],
    free_vars: Set[str],
    *,
    add_totals: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, int | str]]:
    """
    將 optimized_items（或 baseline_items 對應構造）轉成 final_params：
      - cat1..cat8：value=銷售額，type=free/fixed
      - total_tax（derived），如果有提供 add_totals，可一併帶上 total_sales
    """
    fp: Dict[str, Dict[str, int | str]] = {}
    for it in items:
        cat = it.get("category")
        sales = int(it.get("sales_val", 0))
        key = f"cat{cat}"
        fp[key] = {"value": sales, "type": ("free" if key in free_vars else "fixed")}
    if add_totals:
        if "total_tax" in add_totals:
            fp["total_tax"] = {"value": int(add_totals["total_tax"]), "type": "derived"}
        if "total_sales" in add_totals:
            fp["total_sales"] = {"value": int(add_totals["total_sales"]), "type": "derived"}
    return fp

def minimize_nvat_tax(
    *,
    # 8 個變數（任一缺省視作 0）
    cat1: int = 0, cat2: int = 0, cat3: int = 0, cat4: int = 0,
    cat5: int = 0, cat6: int = 0, cat7: int = 0, cat8: int = 0,
    free_vars: Optional[List[str]] = None,
    constraints: Optional[Dict[str, Dict[str, Any]]] = None,
    **aliases,  # 允許傳語義別名：small_business=..., nightlife=... 等
):
    free_vars = set(free_vars or [])
    cons = constraints or {}
    nvars = _normalize_nvat_inputs({**locals(), **aliases})

    # baseline
    baseline_items = []
    baseline_sales = 0
    for cat_key, code, _ in _NVAT_KEYS:
        sales = nvars[cat_key]
        tax   = int(sales * NVAT_RATE[code])
        baseline_items.append({"category": code, "sales_val": sales, "tax": tax})
        baseline_sales += sales
    baseline = sum(x["tax"] for x in baseline_items)

    # baseline + constraints 可行性
    opt0, _, _, total0 = _build_nvat_model(nvars, set(), cons)
    if opt0.check() == sat:
        base_stat = "sat"
        base_wc = int(_to_number(opt0.model().eval(total0)))
    else:
        base_stat = "unsat"
        base_wc = None

    if not free_vars:
        # baseline 也回 final_params（type=fixed；附 total_tax / total_sales）
        final_params = _nvat_final_params_from_items(
            baseline_items, free_vars,
            add_totals={"total_tax": baseline, "total_sales": baseline_sales},
        )
        return {
            "mode": "minimize",
            "baseline": baseline,
            "baseline_items": baseline_items,
            "baseline_total_sales": baseline_sales,
            "baseline_status": base_stat,
            "baseline_with_constraints": base_wc,
            "free_vars": [],
            "constraints": cons,
            "final_params": final_params,
        }

    opt, var_map, row_terms, total = _build_nvat_model(nvars, free_vars, cons)
    opt.minimize(total)
    if opt.check() != sat:
        return {
            "mode": "minimize",
            "baseline": baseline,
            "baseline_items": baseline_items,
            "baseline_total_sales": baseline_sales,
            "baseline_status": base_stat,
            "baseline_with_constraints": base_wc,
            "free_vars": list(free_vars),
            "constraints": cons,
            "status": "unsat",
        }

    m = opt.model()
    optimized = int(_to_number(m.eval(total)))
    optimized_items = []
    diff = {}
    param_diff = {}

    for term, cat_key in row_terms:
        code = int(cat_key.replace("cat", ""))
        sales_v = int(_to_number(m.eval(var_map[cat_key])))
        tax_v   = int(_to_number(m.eval(term)))
        optimized_items.append({"category": code, "sales_val": sales_v, "tax": tax_v})

    # diff（以各類稅額比較）
    for bi in baseline_items:
        oi = next((x for x in optimized_items if x["category"] == bi["category"]), None)
        if oi and bi["tax"] != oi["tax"]:
            diff[bi["category"]] = {
                "original": bi["tax"],
                "optimized": oi["tax"],
                "difference": bi["tax"] - oi["tax"],
            }

    # param_diff（以銷售額比較）
    for cat_key in var_map:
        if cat_key in free_vars:
            orig = nvars[cat_key]
            optv = int(_to_number(m.eval(var_map[cat_key])))
            if optv != orig:
                param_diff[cat_key] = {
                    "original": orig,
                    "optimized": optv,
                    "difference": orig - optv,
                }

    final_params = _nvat_final_params_from_items(
        optimized_items, free_vars,
        add_totals={"total_tax": optimized, "total_sales": sum(x["sales_val"] for x in optimized_items)},
    )

    return {
        "mode": "minimize",
        "baseline": baseline,
        "baseline_items": baseline_items,
        "baseline_total_sales": baseline_sales,
        "optimized": optimized,
        "optimized_items": optimized_items,
        "diff": diff,
        "param_diff": param_diff,
        "free_vars": list(free_vars),
        "constraints": cons,
        "final_params": final_params,
        "status": "sat",
    }

def maximize_nvat_under_budget(
    *,
    cat1: int = 0, cat2: int = 0, cat3: int = 0, cat4: int = 0,
    cat5: int = 0, cat6: int = 0, cat7: int = 0, cat8: int = 0,
    free_vars: Optional[List[str]] = None,
    constraints: Optional[Dict[str, Dict[str, Any]]] = None,
    budget_tax: Optional[int] = None,
    **aliases,
):
    if budget_tax is None:
        raise ValueError("maximize_nvat_under_budget 需要 budget_tax")

    free_vars = set(free_vars or [])
    cons = constraints or {}
    nvars = _normalize_nvat_inputs({**locals(), **aliases})

    # baseline
    baseline_items = []
    baseline_sales = 0
    for cat_key, code, _ in _NVAT_KEYS:
        sales = nvars[cat_key]
        tax   = int(sales * NVAT_RATE[code])
        baseline_items.append({"category": code, "sales_val": sales, "tax": tax})
        baseline_sales += sales
    baseline = sum(x["tax"] for x in baseline_items)

    # baseline+constraints 可行性
    opt0, _, _, total0 = _build_nvat_model(nvars, set(), cons)
    if opt0.check() == sat:
        base_stat = "sat"
        base_wc = int(_to_number(opt0.model().eval(total0)))
    else:
        base_stat = "unsat"
        base_wc = None

    if not free_vars:
        final_params = _nvat_final_params_from_items(
            baseline_items, free_vars,
            add_totals={"total_tax": baseline, "total_sales": baseline_sales},
        )
        return {
            "mode": "maximize",
            "baseline": baseline,
            "baseline_items": baseline_items,
            "baseline_total_sales": baseline_sales,
            "baseline_status": base_stat,
            "baseline_with_constraints": base_wc,
            "free_vars": [],
            "constraints": cons,
            "budget_tax": budget_tax,
            "final_params": final_params,
        }

    opt, var_map, row_terms, total = _build_nvat_model(nvars, free_vars, cons)
    opt.add(total <= RealVal(budget_tax))
    total_sales = Sum([var_map[k] for k in var_map]) if var_map else RealVal(0)
    opt.maximize(total_sales)

    if opt.check() != sat:
        return {
            "mode": "maximize",
            "baseline": baseline,
            "baseline_items": baseline_items,
            "baseline_total_sales": baseline_sales,
            "baseline_status": base_stat,
            "baseline_with_constraints": base_wc,
            "free_vars": list(free_vars),
            "constraints": cons,
            "budget_tax": budget_tax,
            "status": "unsat",
        }

    m = opt.model()
    optimized_total_tax = int(_to_number(m.eval(total)))
    optimized_total_sales = int(_to_number(m.eval(total_sales)))

    optimized_items = []
    for term, cat_key in row_terms:
        code = int(cat_key.replace("cat", ""))
        optimized_items.append({
            "category": code,
            "sales_val": int(_to_number(m.eval(var_map[cat_key]))),
            "tax": int(_to_number(m.eval(term))),
        })

    param_diff = {}
    for k, zv in var_map.items():
        if k in free_vars:
            orig = nvars[k]
            optv = int(_to_number(m.eval(zv)))
            if optv != orig:
                param_diff[k] = {
                    "original": orig,
                    "optimized": optv,
                    "difference": orig - optv,
                }

    final_params = _nvat_final_params_from_items(
        optimized_items, free_vars,
        add_totals={"total_tax": optimized_total_tax, "total_sales": optimized_total_sales},
    )

    return {
        "mode": "maximize",
        "baseline": baseline,
        "baseline_items": baseline_items,
        "baseline_total_sales": baseline_sales,
        "baseline_status": base_stat,
        "baseline_with_constraints": base_wc,
        "optimized_total_tax": optimized_total_tax,
        "optimized_total_sales": optimized_total_sales,
        "optimized_items": optimized_items,
        "param_diff": param_diff,
        "free_vars": list(free_vars),
        "constraints": cons,
        "budget_tax": budget_tax,
        "final_params": final_params,
        "status": "sat",
    }