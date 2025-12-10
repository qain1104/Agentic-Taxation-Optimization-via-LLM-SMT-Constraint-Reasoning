# -*- coding: utf-8 -*-
"""
cargo_tax.py  (展開式 + final_params + 中文標籤)
修復與強化：
- RHS 正規化：支援單元素 list/tuple、中文單位（萬/億/兆）、百分比、含千分位逗號。
- 單變數約束：RHS 支援 var2、var2*k、var2/k、純常數。
- 合計 LHS（a + b + ...）：RHS 支援「純常數、純變數、var2*k、var2/k」。
- total_tax 改為 Int（避免 Real/Int sort mismatch）。
"""

import time
import re
from z3 import Optimize, Real, Int, RealVal, IntVal, ToReal, Sum, sat
from .util import _to_number
from .constraint_utils import apply_linear_constraints  # ★ 共用線性約束 parser

ITEMS = {
    # 水泥 (fixed; 公噸)
    "cement_white":            {"mode": "fixed", "unit_tax": 600,  "cn": "白水泥"},
    "cement_portland_I":       {"mode": "fixed", "unit_tax": 320,  "cn": "卜特蘭I型水泥"},
    "cement_blast_furnace":    {"mode": "fixed", "unit_tax": 196,  "cn": "卜特蘭高爐水泥"},
    "cement_other":            {"mode": "fixed", "unit_tax": 440,  "cn": "代水泥及其他"},

    # 油氣類 (fixed)
    "oil_gasoline":            {"mode": "fixed", "unit_tax": 6830, "cn": "汽油"},
    "oil_diesel":              {"mode": "fixed", "unit_tax": 3990, "cn": "柴油"},
    "oil_kerosene":            {"mode": "fixed", "unit_tax": 4250, "cn": "煤油"},
    "oil_jetfuel":             {"mode": "fixed", "unit_tax": 610,  "cn": "航空燃油"},
    "oil_fueloil":             {"mode": "fixed", "unit_tax": 110,  "cn": "燃料油"},
    "oil_solvent":             {"mode": "fixed", "unit_tax": 720,  "cn": "溶劑油"},
    "lpg":                     {"mode": "fixed", "unit_tax": 690,  "cn": "液化石油氣"},

    # 橡膠輪胎 (ad valorem)
    "tire_bus_truck":          {"mode": "adval", "rate": 0.10,     "cn": "大客/大貨車輪胎"},
    "tire_other":              {"mode": "adval", "rate": 0.15,     "cn": "其他橡膠輪胎"},
    "tire_inner_solid":        {"mode": "adval", "rate": 0.00,     "cn": "內胎/實心輪胎等"},

    # 飲料品 (ad valorem)
    "drink_diluted_juice":     {"mode": "adval", "rate": 0.08,     "cn": "稀釋天然果蔬汁"},
    "drink_other":             {"mode": "adval", "rate": 0.15,     "cn": "其他飲料"},
    "drink_pure_juice":        {"mode": "adval", "rate": 0.00,     "cn": "天然果汁類"},

    # 平板玻璃 (ad valorem)
    "glass_plain":             {"mode": "adval", "rate": 0.10,     "cn": "一般平板玻璃"},
    "glass_conductive_mold":   {"mode": "adval", "rate": 0.00,     "cn": "導電/模具用玻璃"},

    # 電器 (ad valorem)
    "fridge":                  {"mode": "adval", "rate": 0.13,     "cn": "冰箱"},
    "tv":                      {"mode": "adval", "rate": 0.13,     "cn": "彩色電視機"},
    "hvac_central":            {"mode": "adval", "rate": 0.15,     "cn": "中央空調"},
    "hvac_non_central":        {"mode": "adval", "rate": 0.20,     "cn": "非中央空調"},
    "dehumidifier":            {"mode": "adval", "rate": 0.15,     "cn": "除濕機"},
    "vcr":                     {"mode": "adval", "rate": 0.13,     "cn": "錄影機"},
    "recorder":                {"mode": "adval", "rate": 0.10,     "cn": "錄音機"},
    "stereo":                  {"mode": "adval", "rate": 0.10,     "cn": "音響組合"},
    "oven":                    {"mode": "adval", "rate": 0.15,     "cn": "電烤箱"},

    # 車輛 (ad valorem)
    "car_le_2000cc":           {"mode": "adval", "rate": 0.25,     "cn": "小客車 ≤2000cc"},
    "car_gt_2000cc":           {"mode": "adval", "rate": 0.30,     "cn": "小客車 >2000cc"},
    "truck_bus":               {"mode": "adval", "rate": 0.15,     "cn": "貨車/大客車等"},
    "motorcycle":              {"mode": "adval", "rate": 0.17,     "cn": "機車"},
}

# 中文標籤
HUMAN_LABELS = {}
for slug, spec in ITEMS.items():
    HUMAN_LABELS[f"{slug}.quantity"] = f"{spec['cn']}- 數量"
    if spec["mode"] == "adval":
        HUMAN_LABELS[f"{slug}.assessed_price"] = f"{spec['cn']}- 每單位完稅價格"

_ARITH_CMP = {
    '>':  lambda a, b: a >  b,  '>=': lambda a, b: a >= b,
    '<':  lambda a, b: a <  b,  '<=': lambda a, b: a <= b,
    '=':  lambda a, b: a == b,  '==': lambda a, b: a == b,
}

# 乘除 RHS pattern
_RHS_MUL_DIV = re.compile(r"^\s*(\w+)\s*([*/])\s*(-?\d+(?:\.\d+)?)\s*$")

# 中文單位與 RHS 正規化
_CN_UNITS = {"萬": 1e4, "亿": 1e8, "億": 1e8, "兆": 1e12}

def _parse_cn_amount(s: str) -> float | None:
    if not isinstance(s, str):
        return None
    s = s.strip().replace(",", "")
    m = re.match(r"^(-?\d+(?:\.\d+)?)([萬亿億兆]?)$", s)
    if m:
        v = float(m.group(1)); u = m.group(2)
        return v * _CN_UNITS[u] if u else v
    m2 = re.match(r"^(-?\d+(?:\.\d+)?)\s*%$", s)  # 60% → 0.6
    if m2:
        return float(m2.group(1)) / 100.0
    return None

def _normalize_rhs(rhs):
    """回傳 float 或可解析表達式（var、var*k、var/k）；單元素 list/tuple 會攤平。"""
    if isinstance(rhs, (list, tuple)):
        if len(rhs) == 1:
            rhs = rhs[0]
        else:
            raise ValueError(f"RHS list expects 1 element, got {rhs}")
    if isinstance(rhs, (int, float)):
        return float(rhs)
    if isinstance(rhs, str):
        s = rhs.strip()
        if _RHS_MUL_DIV.match(s) or re.fullmatch(r"\w+", s):
            return s
        v = _parse_cn_amount(s)
        if v is not None:
            return v
        try:
            return float(s)
        except ValueError:
            return s
    return rhs

def _is_int_like(x) -> bool:
    try:
        return float(x).is_integer()
    except Exception:
        return False

def _num_val(num: float):
    return IntVal(int(num)) if _is_int_like(num) else RealVal(float(num))

def _expand_payload_with_defaults(kwargs: dict) -> dict:
    payload = {}
    for slug, spec in ITEMS.items():
        qkey = f"{slug}.quantity"
        qv = kwargs.get(qkey, 0)
        try:
            qv = int(float(qv))
        except Exception:
            qv = 0
        payload[qkey] = max(qv, 0)

        if spec["mode"] == "adval":
            pkey = f"{slug}.assessed_price"
            pv = kwargs.get(pkey, 0.0)
            try:
                pv = float(pv)
            except Exception:
                pv = 0.0
            payload[pkey] = max(pv, 0.0)
    return payload

def _build_model_expanded(free_vars, constraints, values):
    opt = Optimize()
    var_map = {}
    item_tax_ints, qty_vars = [], []

    # 1) 建立所有商品的 quantity / price 變數與 Z3 模型
    for slug, spec in ITEMS.items():
        q_int = Int(f"{slug}_qty")
        q = ToReal(q_int)
        qkey = f"{slug}.quantity"
        var_map[qkey] = q

        if qkey in free_vars:
            opt.add(q_int >= 0)
        else:
            opt.add(q_int == int(values.get(qkey, 0)))

        if spec["mode"] == "fixed":
            expr_real = RealVal(spec["unit_tax"]) * q
        else:
            p = Real(f"{slug}_price")
            pkey = f"{slug}.assessed_price"
            var_map[pkey] = p
            if pkey in free_vars:
                opt.add(p >= 0)
            else:
                opt.add(p == RealVal(float(values.get(pkey, 0.0))))
            expr_real = RealVal(spec["rate"]) * p * q

        t_int = Int(f"{slug}_tax_int")
        opt.add(ToReal(t_int) <= expr_real)
        opt.add(expr_real < ToReal(t_int) + 1)
        item_tax_ints.append((slug, t_int))
        qty_vars.append(q)

    # 2) total_tax / total_qty
    total_tax = Int("total_tax")
    opt.add(total_tax == Sum([t for _, t in item_tax_ints]))

    total_qty = Real("total_qty")
    opt.add(total_qty == Sum(qty_vars))

    # 3) 套用共用線性 constraints（constraint_utils）
    #
    #   - params 只會被 apply_linear_constraints 拿來提供 Z3 expr，
    #     第二個元素與第三個元素不會真的用到，所以給 0, [] 即可。
    #
    params_for_constraints = {name: (zexpr, 0, []) for name, zexpr in var_map.items()}
    # 若未來要支援 total_tax / total_qty 也可出現在使用者 constraints：
    params_for_constraints["total_tax"] = (total_tax, 0, [])
    params_for_constraints["total_qty"] = (total_qty, 0, [])

    apply_linear_constraints(opt, params_for_constraints, constraints, debug=False)

    return opt, var_map, item_tax_ints, total_tax, total_qty

def _collect_optimized_items(var_map, model):
    return {k: _to_number(model.eval(z)) for k, z in var_map.items()}

def _compute_param_diff(values, opt_values, free_vars):
    diffs = {}
    for k in free_vars or []:
        if k in opt_values:
            orig = values.get(k, 0)
            newv = opt_values[k]
            if float(orig) != float(newv):
                diffs[k] = {
                    "original": orig,
                    "optimized": newv,
                    "difference": float(newv) - float(orig),
                }
    return diffs

def _final_params_dict(opt_values, free_vars):
    out = {}
    fv = set(free_vars or [])
    for k, v in (opt_values or {}).items():
        out[k] = {
            "value": v,
            "type": "free" if (k in fv) else "fixed",
            "label": HUMAN_LABELS.get(k),
        }
    return out

def _baseline(values):
    total = 0
    for slug, spec in ITEMS.items():
        q = int(values.get(f"{slug}.quantity", 0))
        if spec["mode"] == "fixed":
            total += int(spec["unit_tax"] * q)
        else:
            p = float(values.get(f"{slug}.assessed_price", 0.0))
            total += int(spec["rate"] * p * q)
    return total

# ── 最小化總稅額 ──────────────────────────
def minimize_cargo_tax(*, free_vars=None, constraints=None, budget_tax=None, **kwargs):
    start = time.perf_counter()
    free_vars = set(free_vars or [])
    constraints = constraints or {}
    values = _expand_payload_with_defaults(kwargs)

    baseline = _baseline(values)
    if not free_vars:
        return {
            "baseline": baseline, "optimized": baseline,
            "optimized_items": {}, "diff": {}, "param_diff": {},
            "final_params": {}, "free_vars": [], "constraints": constraints,
            "target_tax": None, "elapsed_sec": time.perf_counter() - start,
        }

    opt, var_map, _items, total_tax, _ = _build_model_expanded(free_vars, constraints, values)
    opt.minimize(total_tax)
    if opt.check() != sat:
        raise ValueError("無可行解，請檢查 free_vars / constraints")

    m = opt.model()
    optimized = int(_to_number(m.eval(total_tax)))
    opt_values = _collect_optimized_items(var_map, m)
    pdiff = _compute_param_diff(values, opt_values, free_vars)

    return {
        "baseline": baseline,
        "optimized": optimized,
        "optimized_items": opt_values,
        "diff": pdiff,
        "param_diff": pdiff,
        "final_params": _final_params_dict(opt_values, free_vars),
        "free_vars": list(free_vars),
        "constraints": constraints,
        "target_tax": None,
        "elapsed_sec": time.perf_counter() - start,
    }

# ── 診斷不可行原因 ─────────────────────────
def _diagnose_infeasible_reason(*, free_vars, constraints, values, budget_tax):
    opt_base, var_map, _items, total_tax, total_qty = _build_model_expanded(free_vars, constraints, values)
    if opt_base.check() != sat:
        hints = []
        for key, rule in (constraints or {}).items():
            if not isinstance(rule, dict):
                continue
            parts = [p.strip() for p in re.split(r"\+", str(key)) if p.strip()]
            all_fixed = True
            lhs_val = 0.0
            can_eval = True
            for p in parts:
                if " " in p and "." not in p:
                    sp = p.split()
                    if len(sp) == 2 and sp[1] in ("quantity", "assessed_price"):
                        p = f"{sp[0]}.{sp[1]}"
                if p in free_vars:
                    all_fixed = False
                    can_eval = False
                    break
                if p.endswith(".quantity") or p.endswith(".assessed_price"):
                    v = values.get(p, 0.0)
                    try:
                        lhs_val += float(v)
                    except Exception:
                        can_eval = False
                        break
                else:
                    can_eval = False
                    break
            if not can_eval:
                continue
            if all_fixed:
                for op, rhs in (rule or {}).items():
                    try:
                        rn = _normalize_rhs(rhs if not isinstance(rhs, (list, tuple)) else rhs[0])
                        rhs_num = float(rn) if isinstance(rn, (int, float)) else None
                    except Exception:
                        rhs_num = None
                    if rhs_num is None:
                        continue
                    violated = False
                    if op == ">=":
                        violated = not (lhs_val >= rhs_num)
                    elif op == ">":
                        violated = not (lhs_val > rhs_num)
                    elif op == "<=":
                        violated = not (lhs_val <= rhs_num)
                    elif op == "<":
                        violated = not (lhs_val < rhs_num)
                    elif op in ("=", "=="):
                        violated = not (lhs_val == rhs_num)
                    if violated:
                        hints.append(f"{key} {op} {rhs_num} 被固定參數鎖死（目前左側={lhs_val}）")
        reason = "約束條件與固定參數衝突，模型本身不可行。"
        if hints:
            reason += " 可能的衝突條件：\n- " + "\n- ".join(hints)
        return reason, {"type": "constraints_conflict"}

    opt_min, _, _, total_tax_min, _ = _build_model_expanded(free_vars, constraints, values)
    opt_min.minimize(total_tax_min)
    if opt_min.check() != sat:
        return "診斷時發生非預期狀況（base SAT, minimize UNSAT）。", {"type": "unexpected"}

    m_min = opt_min.model()
    min_tax = int(_to_number(m_min.eval(total_tax_min)))

    try:
        bgt = float(budget_tax)
    except Exception:
        bgt = None

    if bgt is not None and min_tax > bgt:
        gap = int(min_tax - bgt)
        reason = (
            "就算把所有可調變數壓到最低，稅額下限仍超過設定上限。\n"
            f"- 稅額下限（在現有約束下）：{min_tax}\n"
            f"- 目前稅額上限：{int(bgt)}\n"
            f"- 超出：{gap}"
        )
        return reason, {"type": "budget_too_low", "min_tax": min_tax, "budget_tax": int(bgt), "gap": gap}

    return "參數與上限理論上可行，但最大全量求解失敗。請降低求解複雜度或放寬部分離散條件後再試。", {
        "type": "maximize_failed",
        "min_tax": min_tax,
        "budget_tax": int(bgt) if bgt is not None else None,
    }

def maximize_cargo_qty(*, free_vars=None, constraints=None, budget_tax=None, **kwargs):
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
            "optimized_items": {}, "param_diff": {},
            "final_params": {}, "free_vars": [], "constraints": constraints,
            "target_tax": budget_tax, "elapsed_sec": time.perf_counter() - start,
        }

    opt0, var_map0, _items0, total_tax0, total_qty0 = _build_model_expanded(free_vars, constraints, values)
    if opt0.check() != sat:
        reason, _ = _diagnose_infeasible_reason(
            free_vars=free_vars, constraints=constraints, values=values, budget_tax=budget_tax
        )
        raise ValueError(f"無可行解：{reason}")

    reason_min, detail_min = _diagnose_infeasible_reason(
        free_vars=free_vars, constraints=constraints, values=values, budget_tax=budget_tax
    )
    if detail_min.get("type") == "budget_too_low":
        raise ValueError(f"無可行解：{reason_min}")

    opt, var_map, _items, total_tax, total_qty = _build_model_expanded(free_vars, constraints, values)
    opt.add(total_tax <= IntVal(int(float(budget_tax))))
    opt.maximize(total_qty)

    if opt.check() != sat:
        reason, _ = _diagnose_infeasible_reason(
            free_vars=free_vars, constraints=constraints, values=values, budget_tax=budget_tax
        )
        raise ValueError(f"無可行解：{reason}")

    m = opt.model()
    optimized_total_qty = int(_to_number(m.eval(total_qty)))
    optimized_total_tax = int(_to_number(m.eval(total_tax)))
    opt_values = _collect_optimized_items(var_map, m)
    pdiff = _compute_param_diff(values, opt_values, free_vars)

    return {
        "baseline": baseline,
        "optimized_total_qty": optimized_total_qty,
        "optimized_total_tax": optimized_total_tax,
        "optimized": optimized_total_tax,
        "optimized_sales": optimized_total_qty,
        "optimized_items": opt_values,
        "param_diff": pdiff,
        "final_params": _final_params_dict(opt_values, free_vars),
        "free_vars": list(free_vars),
        "constraints": constraints,
        "target_tax": budget_tax,
        "diff": pdiff,
        "elapsed_sec": time.perf_counter() - start,
    }
