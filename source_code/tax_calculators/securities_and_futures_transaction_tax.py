# -*- coding: utf-8 -*-
"""
securities_and_futures_transaction_tax.py
展開式變數版（無 rows）：
- 未提供的欄位自動視為 0
- 有 target_tax → 在上限內「最大化總交易量」
- 無 target_tax → 「最小化總稅額」
- 支援 constraints（交給 constraint_utils.apply_linear_constraints 處理）：
  • 單變數比較：stock.tp >= 2e8
  • 二元加減 / 線性式：stock.tp + bond.tp >= 2e8
  • 乘除等式：stock.tp = bond.tp * 1.2  或  option.pa = option.ca / 3
  • RHS 可為單值或列表（例如 {">=": [2e8, 3e8]}）
- 支援變數名正規化：
  • "stock_tp" / "stock tp" 皆會自動正規化為 "stock.tp"
"""

import json
import time
import re
from typing import Dict, Tuple, Optional, List

from z3 import Optimize, Real, RealVal, Sum, sat
from z3 import IntNumRef, RatNumRef

from .util import _to_number
from .constraint_utils import apply_linear_constraints

# ========= 變數名正規化：把底線/空白/裸 slug 轉成 dot 形式 =========

# 裸 slug 的預設欄位：證券→tp；期貨→ca；選擇權→pa
_DEFAULT_FIELD_BY_SLUG = {
    # 證券
    "stock": "tp",
    "bond": "tp",
    "warrant": "tp",
    # 履約兩種沒有自然單欄位，故不給預設（避免歧義）
    # "warrant_delivery_stock": None,
    # "warrant_delivery_cash":  None,

    # 期貨
    "stock_index": "ca",
    "interest_rate_30": "ca",
    "interest_rate_10": "ca",
    "gold": "ca",
    "option": "pa",  # 選擇權稅基用權利金
}

_VAR_UNDER = re.compile(r'\b([a-z_]+)_(tp|ep|sc|ca|pa)\b', flags=re.I)
_VAR_SPACE = re.compile(r'\b([a-z_]+)\s+(tp|ep|sc|ca|pa)\b', flags=re.I)
# 裸 slug：需確保右邊**不是**已經跟了 .tp/.ep/... 才補
_BARE_SLUG = re.compile(
    r'\b(' + "|".join(map(re.escape, _DEFAULT_FIELD_BY_SLUG.keys())) + r')\b(?!\s*\.(?:tp|ep|sc|ca|pa))',
    flags=re.I
)

def _canon_var_token(s: str) -> str:
    """將 'stock_tp' / 'stock tp' / 'stock' 轉為 'stock.tp'（option → pa；期貨多數 → ca）。"""
    if not isinstance(s, str):
        return s
    t = _VAR_UNDER.sub(r'\1.\2', s)
    t = _VAR_SPACE.sub(r'\1.\2', t)

    def _repl(m):
        slug = m.group(1).lower()
        fld = _DEFAULT_FIELD_BY_SLUG.get(slug)
        return f"{slug}.{fld}" if fld else m.group(0)

    t = _BARE_SLUG.sub(_repl, t)
    return t

def _canon_expr(expr: str) -> str:
    """把表達式內所有變數正規化，並縮減多餘空白。"""
    if not isinstance(expr, str):
        return expr
    t = _canon_var_token(expr)
    return re.sub(r'\s+', ' ', t).strip()


# ========= 證券/期貨品項與稅率 =========

SEC_ITEMS = {
    # 證券（tp / ep*sc）
    "stock":                 {"cn": "（一）公司股票／股權證書",   "fields": ["tp"],        "rate": ("tp", 0.003)},
    "bond":                  {"cn": "（二）公司債及其他有價證券", "fields": ["tp"],        "rate": ("tp", 0.001)},
    "warrant":               {"cn": "（三）認購（售）權證",       "fields": ["tp"],        "rate": ("tp", 0.001)},
    "warrant_delivery_stock":{"cn": "（四）權證履約—股票移轉",   "fields": ["ep","sc"],   "rate": ("ep*sc", 0.003)},
    "warrant_delivery_cash": {"cn": "（四）權證履約—現金結算",   "fields": ["ep","sc"],   "rate": ("ep*sc", 0.001)},
}

FUT_ITEMS = {
    # 期貨（大多用 ca；選擇權用 pa）
    "stock_index":      {"cn": "（一）股價期貨",              "fields": ["ca"],      "rate": ("ca", 0.00002),       "qty": "ca"},
    "interest_rate_30": {"cn": "（二）利率期貨30天CP",      "fields": ["ca"],      "rate": ("ca", 0.000000125),   "qty": "ca"},
    "interest_rate_10": {"cn": "（二）利率期貨10年期債",    "fields": ["ca"],      "rate": ("ca", 0.00000125),    "qty": "ca"},
    "option":           {"cn": "（三）選擇權及期貨選擇權",  "fields": ["pa","ca"], "rate": ("pa", 0.001),          "qty": "pa"},
    "gold":             {"cn": "（四）黃金期貨",              "fields": ["ca"],      "rate": ("ca", 0.0000025),     "qty": "ca"},
}


# ========= 中文標籤 =========

SEC_FIELD_LABELS: Dict[str, str] = {
    "stock.tp":                   "公司股票／股權證書- 交易金額",
    "bond.tp":                    "公司債及其他有價證券- 交易金額",
    "warrant.tp":                 "認購（售）權證- 交易金額",
    "warrant_delivery_stock.ep":  "權證履約—股票移轉- 履約價格",
    "warrant_delivery_stock.sc":  "權證履約—股票移轉- 股數",
    "warrant_delivery_cash.ep":   "權證履約—現金結算- 履約價格",
    "warrant_delivery_cash.sc":   "權證履約—現金結算- 股數",
}

FUT_FIELD_LABELS: Dict[str, str] = {
    "stock_index.ca":             "股價期貨-契約金額",
    "interest_rate_30.ca":        "利率期貨-30天CP-契約金額",
    "interest_rate_10.ca":        "利率期貨-10年期債-契約金額",
    "option.pa":                  "選擇權及期貨選擇權-權利金金額",
    "option.ca":                  "選擇權及期貨選擇權-契約金額",
    "gold.ca":                    "黃金期貨-契約金額",
}

# 合併（給 ReasoningAgent / report 使用）
FIELD_LABELS: Dict[str, str] = {}
FIELD_LABELS.update(SEC_FIELD_LABELS)
FIELD_LABELS.update(FUT_FIELD_LABELS)


# ========= constraints 解析 & 範圍過濾 =========

# 抽取變數 token 用（stock.tp / option.pa ...）
_VAR_TOKEN = re.compile(r'\b([a-z_]+\.(?:tp|ep|sc|ca|pa))\b', flags=re.I)

def _parse_constraints(constraints_json) -> Dict:
    """
    接受 dict 或 JSON 字串；key 與 RHS 字串中的變數都先正規化（底線/空白 → dot）。
    支援 RHS 為列表（逐條加入）。
    """
    if isinstance(constraints_json, dict):
        obj = constraints_json
    elif not constraints_json:
        obj = {}
    else:
        obj = json.loads(constraints_json)
        if not isinstance(obj, dict):
            raise ValueError("constraints 需為 JSON 物件")

    norm: Dict[str, Dict] = {}
    for k, rule in (obj or {}).items():
        nk = _canon_expr(k)
        nrule = {}
        for op, rhs in (rule or {}).items():
            if isinstance(rhs, list):
                nrule[op] = [_canon_expr(x) if isinstance(x, str) else x for x in rhs]
            else:
                nrule[op] = _canon_expr(rhs) if isinstance(rhs, str) else rhs
        norm[nk] = nrule
    return norm

def _filter_constraints_for_vars(constraints: Dict, allowed_vars: set) -> Dict:
    """
    只保留「只使用 allowed_vars 中變數」的約束；
    若某條約束的 key/RHS 牽涉到其他變數（例如 futures 的變數卻套在 securities），就略過。
    """
    out: Dict[str, Dict] = {}
    for key, rule in (constraints or {}).items():
        used = set()

        # key 裡的變數
        if isinstance(key, str):
            for t in _VAR_TOKEN.findall(key):
                used.add(t.lower())

        # RHS 裡的變數
        for rhs in (rule or {}).values():
            vals = rhs if isinstance(rhs, list) else [rhs]
            for rv in vals:
                if isinstance(rv, str):
                    for t in _VAR_TOKEN.findall(rv):
                        used.add(t.lower())

        # 有牽涉變數且超出 allowed_vars → 略過
        if used and not used.issubset(allowed_vars):
            continue

        out[key] = rule
    return out


# ========= 小工具 =========

def _expand_with_defaults(items_spec: Dict[str, Dict], kwargs: Dict) -> Dict[str, float]:
    """
    把所有展開欄位補成數字（預設 0）；接受 dot/under/space 三種鍵名。
    例如：傳入 stock_tp=2e8 → 會被正規化到 stock.tp。
    """
    canon_kwargs: Dict[str, float] = {}
    for k, v in (kwargs or {}).items():
        canon_kwargs[_canon_var_token(k)] = v

    out: Dict[str, float] = {}
    for slug, spec in items_spec.items():
        for f in spec["fields"]:
            key = f"{slug}.{f}"
            v = canon_kwargs.get(key, 0)
            try:
                out[key] = float(v)
            except Exception:
                out[key] = 0.0
    return out


def _param_diff(
    var_map: Dict[str, Real],
    free_vars: set,
    values: Dict[str, float],
    model
) -> Dict[str, Dict[str, float]]:
    """回傳 { var: {original, optimized, difference} }。"""
    diffs: Dict[str, Dict[str, float]] = {}
    for k, z in var_map.items():
        if k not in free_vars:
            continue
        zval = model.eval(z)
        if not isinstance(zval, (IntNumRef, RatNumRef)):
            continue
        orig = values.get(k, 0.0)
        optv = _to_number(zval)
        if optv != orig:
            diffs[k] = {"original": orig, "optimized": optv, "difference": optv - orig}
    return diffs


# ========= 證券：建模 & baseline =========

def _securities_model(values: Dict[str, float], free_vars: set, constraints: Dict):
    opt = Optimize()
    var_map: Dict[str, Real] = {}
    row_exprs: List[Tuple] = []

    # 宣告變數與稅額式
    for slug, spec in SEC_ITEMS.items():
        tp = ep = sc = None
        if "tp" in spec["fields"]:
            tp = Real(f"{slug}_tp")
            var_map[f"{slug}.tp"] = tp
            opt.add(tp >= 0 if f"{slug}.tp" in free_vars else tp == RealVal(values.get(f"{slug}.tp", 0.0)))
        if "ep" in spec["fields"]:
            ep = Real(f"{slug}_ep")
            var_map[f"{slug}.ep"] = ep
            opt.add(ep >= 0 if f"{slug}.ep" in free_vars else ep == RealVal(values.get(f"{slug}.ep", 0.0)))
        if "sc" in spec["fields"]:
            sc = Real(f"{slug}_sc")
            var_map[f"{slug}.sc"] = sc
            opt.add(sc >= 0 if f"{slug}.sc" in free_vars else sc == RealVal(values.get(f"{slug}.sc", 0.0)))

        mode, r = spec["rate"]
        if mode == "tp":
            expr = tp * RealVal(r)
        elif mode == "ep*sc":
            expr = ep * sc * RealVal(r)
        else:
            expr = RealVal(0)
        row_exprs.append((expr, slug))

    total_tax = Real("total_tax")
    opt.add(total_tax == Sum([e for e, _ in row_exprs]))

    # 使用共用 constraint_utils.apply_linear_constraints
    params_for_constraints = {name: (z,) for name, z in var_map.items()}
    apply_linear_constraints(opt, params_for_constraints, constraints or {}, debug=False)

    return opt, row_exprs, var_map, total_tax


def _securities_baseline(values: Dict[str, float]) -> int:
    total = 0.0
    for slug, spec in SEC_ITEMS.items():
        mode, r = spec["rate"]
        if mode == "tp":
            total += values.get(f"{slug}.tp", 0.0) * r
        elif mode == "ep*sc":
            total += values.get(f"{slug}.ep", 0.0) * values.get(f"{slug}.sc", 0.0) * r
    return int(total)


# ========= 期貨：建模 & baseline =========

def _futures_model(values: Dict[str, float], free_vars: set, constraints: Dict):
    opt = Optimize()
    var_map: Dict[str, Real] = {}
    row_exprs: List[Tuple] = []

    for slug, spec in FUT_ITEMS.items():
        ca = pa = None
        if "ca" in spec["fields"]:
            ca = Real(f"{slug}_ca")
            var_map[f"{slug}.ca"] = ca
            opt.add(ca >= 0 if f"{slug}.ca" in free_vars else ca == RealVal(values.get(f"{slug}.ca", 0.0)))
        if "pa" in spec["fields"]:
            pa = Real(f"{slug}_pa")
            var_map[f"{slug}.pa"] = pa
            opt.add(pa >= 0 if f"{slug}.pa" in free_vars else pa == RealVal(values.get(f"{slug}.pa", 0.0)))

        mode, r = spec["rate"]
        if mode == "pa":
            expr = pa * RealVal(r)
        elif mode == "ca":
            expr = ca * RealVal(r)
        else:
            expr = RealVal(0)
        row_exprs.append((expr, slug))

    total_tax = Real("total_tax")
    opt.add(total_tax == Sum([e for e, _ in row_exprs]))

    # 使用共用 constraint_utils.apply_linear_constraints
    params_for_constraints = {name: (z,) for name, z in var_map.items()}
    apply_linear_constraints(opt, params_for_constraints, constraints or {}, debug=False)

    return opt, row_exprs, var_map, total_tax


def _futures_baseline(values: Dict[str, float]) -> int:
    total = 0.0
    for slug, spec in FUT_ITEMS.items():
        mode, r = spec["rate"]
        if mode == "pa":
            total += values.get(f"{slug}.pa", 0.0) * r
        elif mode == "ca":
            total += values.get(f"{slug}.ca", 0.0) * r
    return int(total)


# ========= 核心優化器（共用） =========

def _optimize_generic(*, which: str, free_vars=None, constraints=None, target_tax=None, **kwargs):
    """
    which: 'securities' 或 'futures'
    - free_vars：可用底線/空白/點，會自動正規化為 dot 形式
    - constraints：dict 或 JSON 字串，支援線性表達式（交給 constraint_utils）
    - target_tax：若提供，會在上限內最大化 total_qty；否則最小化 total_tax
    """
    # 正規化 free_vars（統一成 dot + 小寫）
    free_vars = { _canon_var_token(x).lower() for x in (free_vars or []) }

    # constraints 先做 JSON & 變數正規化，再依 which 過濾只含該類變數的條件
    constraints_parsed = _parse_constraints(constraints or {})

    if which == "securities":
        spec = SEC_ITEMS
        values = _expand_with_defaults(spec, kwargs)
        # 本類別可用的變數集合（小寫）
        allowed_vars = {
            f"{slug}.{f}".lower()
            for slug, sp in SEC_ITEMS.items()
            for f in sp["fields"]
        }
        scoped_constraints = _filter_constraints_for_vars(constraints_parsed, allowed_vars)
        baseline = _securities_baseline(values)
        opt, row_exprs, var_map, total_tax = _securities_model(values, free_vars, scoped_constraints)

        # 交易量：tp；履約兩種使用 ep*sc
        qty_terms = []
        for slug, s in SEC_ITEMS.items():
            if s["rate"][0] == "tp":
                qty_terms.append(var_map[f"{slug}.tp"])
            else:
                qty_terms.append(var_map[f"{slug}.ep"] * var_map[f"{slug}.sc"])
        field_labels = FIELD_LABELS

    else:
        spec = FUT_ITEMS
        values = _expand_with_defaults(spec, kwargs)
        allowed_vars = {
            f"{slug}.{f}".lower()
            for slug, sp in FUT_ITEMS.items()
            for f in sp["fields"]
        }
        scoped_constraints = _filter_constraints_for_vars(constraints_parsed, allowed_vars)
        baseline = _futures_baseline(values)
        opt, row_exprs, var_map, total_tax = _futures_model(values, free_vars, scoped_constraints)

        # 交易量：選擇權用 pa，其餘用 ca
        qty_terms = []
        for slug, s in FUT_ITEMS.items():
            qty_key = s.get("qty", "ca")
            qty_terms.append(var_map[f"{slug}.{qty_key}"])
        field_labels = FIELD_LABELS

    # 未放行 → 回 baseline
    if not free_vars:
        return {
            "baseline": baseline,
            "optimized_total_tax": baseline,
            "optimized_total_qty": 0 if target_tax is not None else None,
            "optimized_items": {},
            "diff": {},
            "param_diff": {},
            "final_params": {},
            "free_vars": [],
            "constraints": scoped_constraints,
            "target_tax": target_tax,
            "field_labels": field_labels,
        }

    # 稅額上限
    if target_tax is not None:
        opt.add(total_tax <= RealVal(float(target_tax)))

    total_qty = Real("total_qty")
    opt.add(total_qty == Sum(qty_terms))

    # 目標
    if target_tax is not None:
        opt.maximize(total_qty)
    else:
        opt.minimize(total_tax)

    # 求解
    if opt.check() != sat:
        return {
            "baseline": baseline,
            "free_vars": list(free_vars),
            "constraints": scoped_constraints,
            "target_tax": target_tax,
            "no_solution": True,
            "field_labels": field_labels,
        }

    m = opt.model()
    optimized_total_tax = int(_to_number(m.eval(total_tax)))
    optimized_total_qty = int(_to_number(m.eval(total_qty))) if target_tax is not None else None

    # 每品項稅額
    optimized_items = {}
    for expr, slug in row_exprs:
        optimized_items[slug] = int(_to_number(m.eval(expr)))

    # 參數差異與 final_params
    pdiff = _param_diff(var_map, free_vars, values, m)
    final_params = {}
    for k, z in var_map.items():
        v = _to_number(m.eval(z))
        t = "free" if (k in free_vars) else "fixed"
        final_params[k] = {"value": v, "type": t}

    return {
        "baseline": baseline,
        "optimized_total_tax": optimized_total_tax,
        "optimized_total_qty": optimized_total_qty,
        "optimized_items": optimized_items,
        "param_diff": pdiff,
        "final_params": final_params,
        "free_vars": list(free_vars),
        "constraints": scoped_constraints,
        "target_tax": target_tax,
        "field_labels": field_labels,
    }


# ========= Public APIs =========

def compute_securities_transaction_tax(
    *,
    free_vars=None,
    constraints=None,
    target_tax=None,
    **kwargs
) -> Dict:
    """
    展開式變數（未提供者自動 0）：
      - stock.tp / bond.tp / warrant.tp
      - warrant_delivery_stock.ep, warrant_delivery_stock.sc
      - warrant_delivery_cash.ep,  warrant_delivery_cash.sc
    """
    t0 = time.perf_counter()
    res = _optimize_generic(
        which="securities",
        free_vars=free_vars,
        constraints=constraints,
        target_tax=target_tax,
        **kwargs
    )
    print(f"[securities] done in {time.perf_counter() - t0:.3f}s")
    return res


def compute_futures_transaction_tax(
    *,
    free_vars=None,
    constraints=None,
    target_tax=None,
    **kwargs
) -> Dict:
    """
    展開式變數（未提供者自動 0）：
      - stock_index.ca / interest_rate_30.ca / interest_rate_10.ca / gold.ca
      - option.pa（稅基）與 option.ca（方便做約束）
    """
    t0 = time.perf_counter()
    res = _optimize_generic(
        which="futures",
        free_vars=free_vars,
        constraints=constraints,
        target_tax=target_tax,
        **kwargs
    )
    print(f"[futures] done in {time.perf_counter() - t0:.3f}s")
    return res