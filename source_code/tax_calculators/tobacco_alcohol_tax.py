# tax_calculators/ta_tax.py
# 菸酒稅（展開式變數 + 中文標籤 + 最小化/最大化）
# - 變數型式：<slug>.quantity；需 ABV 的 4 類再有 <slug>.alcohol_content
# - ABV 需要者：brew_other / distilled / reprocessed_le20 / alcohol_other
# - 菸稅（不含健保捐）：紙菸以 1,590 元/千支；其他以 1,590 元/單位
# - 酒稅：依子類別決定率（部分 * ABV），全部 × 數量
import re, time
from typing import Dict, List, Tuple, Optional

from z3 import Optimize, Real, Int, RealVal, ToReal, Sum, sat
from .util import _to_number
from .constraint_utils import apply_linear_constraints  # 共用線性約束 parser

# ─────────────────────────────────────────
# 品項定義（slug、中文名與稅制）
# ─────────────────────────────────────────
ITEMS = {
    # 菸（新制／舊制）
    "cigarettes_new":     {"cn": "紙菸-新制",     "type": "tobacco", "base": 1590, "div": 1000, "need_abv": False},
    "tobacco_cut_new":    {"cn": "菸絲-新制",     "type": "tobacco", "base": 1590, "div": 1,    "need_abv": False},
    "cigar_new":          {"cn": "雪茄-新制",     "type": "tobacco", "base": 1590, "div": 1,    "need_abv": False},
    "tobacco_other_new":  {"cn": "其他菸品-新制", "type": "tobacco", "base": 1590, "div": 1,    "need_abv": False},

    "cigarettes_old":     {"cn": "紙菸-舊制",     "type": "tobacco", "base": 1590, "div": 1000, "need_abv": False},
    "tobacco_cut_old":    {"cn": "菸絲-舊制",     "type": "tobacco", "base": 1590, "div": 1,    "need_abv": False},
    "cigar_old":          {"cn": "雪茄-舊制",     "type": "tobacco", "base": 1590, "div": 1,    "need_abv": False},
    "tobacco_other_old":  {"cn": "其他菸品-舊制", "type": "tobacco", "base": 1590, "div": 1,    "need_abv": False},

    # 酒（全部有 quantity；其中 4 類需要 ABV）
    "beer":               {"cn": "釀造 → 啤酒",     "type": "alcohol", "rate": ("const", 26),   "need_abv": False},
    "brew_other":         {"cn": "釀造 → 其他",     "type": "alcohol", "rate": ("abv", 7),     "need_abv": True},
    "distilled":          {"cn": "蒸餾酒類",         "type": "alcohol", "rate": ("abv", 2.5),   "need_abv": True},
    "reprocessed_gt20":   {"cn": "再製 >20%",        "type": "alcohol", "rate": ("const", 185), "need_abv": False},
    "reprocessed_le20":   {"cn": "再製 ≤20%",        "type": "alcohol", "rate": ("abv", 7),     "need_abv": True},
    "cooking_wine":       {"cn": "料理酒",           "type": "alcohol", "rate": ("const", 9),   "need_abv": False},
    "cooking_wine_old":   {"cn": "料理酒-舊",       "type": "alcohol", "rate": ("const", 22),  "need_abv": False},
    "alcohol_other":      {"cn": "其他酒類",         "type": "alcohol", "rate": ("abv", 7),     "need_abv": True},
    "ethanol":            {"cn": "酒精",             "type": "alcohol", "rate": ("const", 15),  "need_abv": False},
    "ethanol_old":        {"cn": "酒精-舊",         "type": "alcohol", "rate": ("const", 11),  "need_abv": False},
}

# 中文標籤（與 cargo_tax 同型式）：<slug>.quantity / <slug>.alcohol_content
FIELD_LABELS: Dict[str, str] = {}
for slug, spec in ITEMS.items():
    FIELD_LABELS[f"{slug}.quantity"] = f"{spec['cn']}- 數量"
    if spec["type"] == "alcohol" and spec["need_abv"]:
        FIELD_LABELS[f"{slug}.alcohol_content"] = f"{spec['cn']}- 酒精度(%)"

# 供 report renderer 使用
def _label(key: str) -> str:
    return FIELD_LABELS.get(key, key)

# ─────────────────────────────────────────
# 工具：解析／建立變數與模型
# ─────────────────────────────────────────

def _expand_payload_with_defaults(kwargs: dict) -> dict:
    """把所有展開欄位補成 0 / 0.0（缺值時）。"""
    data = {}
    for slug, spec in ITEMS.items():
        qk = f"{slug}.quantity"
        qv = kwargs.get(qk, 0)
        try:
            qv = int(float(qv))
        except Exception:
            qv = 0
        data[qk] = max(qv, 0)

        if spec["type"] == "alcohol" and spec["need_abv"]:
            ak = f"{slug}.alcohol_content"
            av = kwargs.get(ak, 0.0)
            try:
                av = float(av)
            except Exception:
                av = 0.0
            data[ak] = max(av, 0.0)
    return data


def _row_tax_expr(slug: str, q, abv=None):
    spec = ITEMS[slug]
    if spec["type"] == "tobacco":
        base, div = spec["base"], spec["div"]
        return RealVal(base) * (q / RealVal(div))
    # alcohol
    kind, k = spec["rate"]
    if kind == "const":
        rate = RealVal(k)
    else:  # "abv"
        if abv is None:
            abv = RealVal(0)
        rate = RealVal(k) * abv
    return rate * q


def _build_model_expanded(free_vars, constraints, values):
    """
    建立 Optimize 模型（展開式）：
    - 每個 slug 產生 quantity 變數（Int → ToReal）；需 ABV 的再建 alcohol_content（Real）
    - total_tax = Σ 各品項稅額（向 Z3 表達式）
    - total_qty = Σ 各 quantity
    - constraints 交給 constraint_utils.apply_linear_constraints，支援：
        • 單變數：   a >= 10
        • 線性合成： a + b - 2*c <= 100
        • 比例等式： a = b * 0.3、a = b / 2 等
        • 也可用 total_tax / total_qty 作條件
    """
    opt = Optimize()
    var_map = {}
    item_exprs = []
    qty_vars = []

    for slug, spec in ITEMS.items():
        # 數量：整數（以 ToReal 進稅額公式）
        q_int = Int(f"{slug}_qty")
        q = ToReal(q_int)
        var_map[f"{slug}.quantity"] = q
        if f"{slug}.quantity" in free_vars:
            opt.add(q_int >= 0)
        else:
            opt.add(q_int == int(values.get(f"{slug}.quantity", 0)))

        # ABV（必要者）：實數
        abv = None
        if spec["type"] == "alcohol" and spec["need_abv"]:
            a = Real(f"{slug}_abv")
            var_map[f"{slug}.alcohol_content"] = a
            if f"{slug}.alcohol_content" in free_vars:
                opt.add(a >= 0)
            else:
                opt.add(a == RealVal(float(values.get(f"{slug}.alcohol_content", 0.0))))
            abv = a

        expr = _row_tax_expr(slug, q, abv)
        item_exprs.append(expr)
        qty_vars.append(q)

    total_tax = Real("total_tax")
    opt.add(total_tax == Sum(item_exprs))
    total_qty = Real("total_qty")
    opt.add(total_qty == Sum(qty_vars))

    # ── 套用共用線性約束引擎 ─────────────────────────────
    # constraint_utils.apply_linear_constraints 只需要 name -> (z3_var,) 的 mapping
    params_for_constraints = {name: (z,) for name, z in var_map.items()}
    # 也讓使用者可以直接寫 total_tax / total_qty 當變數
    params_for_constraints["total_tax"] = (total_tax,)
    params_for_constraints["total_qty"] = (total_qty,)

    apply_linear_constraints(opt, params_for_constraints, constraints or {}, debug=False)

    return opt, var_map, total_tax, total_qty


def _final_params_dict(opt_values, free_vars):
    out = {}
    fv = set(free_vars or [])
    for k, v in (opt_values or {}).items():
        out[k] = {
            "value": v,
            "type": "free" if (k in fv) else "fixed",
            "label": _label(k),
        }
    return out


def _collect_model_values(var_map, model):
    return {k: _to_number(model.eval(z)) for k, z in var_map.items()}


def _baseline(values: dict) -> int:
    """用純 Python 公式算 baseline 稅額（向下取整）。"""
    total = 0.0
    for slug, spec in ITEMS.items():
        q = int(values.get(f"{slug}.quantity", 0) or 0)
        if spec["type"] == "tobacco":
            total += spec["base"] * (q / spec["div"])
        else:
            kind, k = spec["rate"]
            if kind == "const":
                rate = k
            else:
                abv = float(values.get(f"{slug}.alcohol_content", 0.0) or 0.0)
                rate = k * abv
            total += rate * q
    return int(total)  # 向下取整一致化

# ─────────────────────────────────────────
# 對外 API（與 cargo_tax 對齊）
# ─────────────────────────────────────────
def minimize_ta_tax(*, free_vars=None, constraints=None, **kwargs):
    """
    最小化總稅額（展開式變數版）。
    kwargs：展開欄位（見 FIELD_LABELS 的 keys）
    constraints：交給 constraint_utils，支援線性與比例條件。
    """
    start = time.perf_counter()
    free_vars = set(free_vars or [])
    constraints = constraints or {}
    values = _expand_payload_with_defaults(kwargs)
    baseline = _baseline(values)

    if not free_vars:
        return {
            "baseline": baseline,
            "optimized": baseline,
            "optimized_items": {},
            "diff": {},
            "param_diff": {},
            "final_params": {},
            "free_vars": [],
            "constraints": constraints,
            "target_tax": None,
            "elapsed_sec": time.perf_counter() - start,
            "input_params": values,
        }

    opt, var_map, total_tax, _ = _build_model_expanded(free_vars, constraints, values)
    opt.minimize(total_tax)
    if opt.check() != sat:
        return {
            "baseline": baseline,
            "no_solution": True,
            "free_vars": list(free_vars),
            "constraints": constraints,
            "target_tax": None,
            "elapsed_sec": time.perf_counter() - start,
            "input_params": values,
        }

    m = opt.model()
    optimized = int(_to_number(m.eval(total_tax)))
    opt_values = _collect_model_values(var_map, m)

    # 參數差異（只列自由變數）
    param_diff = {}
    for k in free_vars:
        if k in opt_values:
            orig = values.get(k, 0)
            newv = opt_values[k]
            if float(orig) != float(newv):
                param_diff[k] = {
                    "original": orig,
                    "optimized": newv,
                    "difference": float(newv) - float(orig),
                }

    return {
        "baseline": baseline,
        "optimized": optimized,
        "optimized_items": opt_values,
        "diff": param_diff,              # 與 cargo_tax 對齊：用 param_diff 當 diff 顯示
        "param_diff": param_diff,
        "final_params": _final_params_dict(opt_values, free_vars),
        "free_vars": list(free_vars),
        "constraints": constraints,
        "target_tax": None,
        "elapsed_sec": time.perf_counter() - start,
        "input_params": values,
    }


def maximize_ta_qty(*, free_vars=None, constraints=None, budget_tax=None, **kwargs):
    """
    在『總稅額 ≤ budget_tax』的條件下，最大化總數量。
    - 回傳欄位與 cargo_tax 的 maximize_cargo_qty 對齊：
      optimized_total_qty / optimized_total_tax / optimized_sales(鏡射)/ optimized(鏡射)
    """
    if budget_tax is None:
        raise ValueError("請提供 budget_tax（稅額上限）")

    start = time.perf_counter()
    free_vars = set(free_vars or [])
    constraints = constraints or {}
    values = _expand_payload_with_defaults(kwargs)
    baseline = _baseline(values)

    if not free_vars:
        return {
            "baseline": baseline,
            "optimized_total_qty": 0,
            "optimized_total_tax": baseline,
            "optimized": baseline,
            "optimized_sales": 0,
            "optimized_items": {},
            "param_diff": {},
            "final_params": {},
            "free_vars": [],
            "constraints": constraints,
            "target_tax": budget_tax,
            "elapsed_sec": time.perf_counter() - start,
            "input_params": values,
        }

    opt, var_map, total_tax, total_qty = _build_model_expanded(free_vars, constraints, values)
    opt.add(total_tax <= RealVal(float(budget_tax)))
    opt.maximize(total_qty)

    if opt.check() != sat:
        return {
            "baseline": baseline,
            "no_solution": True,
            "free_vars": list(free_vars),
            "constraints": constraints,
            "target_tax": budget_tax,
            "elapsed_sec": time.perf_counter() - start,
            "input_params": values,
        }

    m = opt.model()
    optimized_total_qty = int(_to_number(m.eval(total_qty)))
    optimized_total_tax = int(_to_number(m.eval(total_tax)))
    opt_values = _collect_model_values(var_map, m)

    # 參數差異（只列自由變數）
    param_diff = {}
    for k in free_vars:
        if k in opt_values:
            orig = values.get(k, 0)
            newv = opt_values[k]
            if float(orig) != float(newv):
                param_diff[k] = {
                    "original": orig,
                    "optimized": newv,
                    "difference": float(newv) - float(orig),
                }

    return {
        "baseline": baseline,
        "optimized_total_qty": optimized_total_qty,
        "optimized_total_tax": optimized_total_tax,
        "optimized": optimized_total_tax,        # 相容給「最佳稅額」
        "optimized_sales": optimized_total_qty,  # 相容給「最佳數量」
        "optimized_items": opt_values,
        "param_diff": param_diff,
        "final_params": _final_params_dict(opt_values, free_vars),
        "free_vars": list(free_vars),
        "constraints": constraints,
        "target_tax": budget_tax,
        "elapsed_sec": time.perf_counter() - start,
        "input_params": values,
    }
