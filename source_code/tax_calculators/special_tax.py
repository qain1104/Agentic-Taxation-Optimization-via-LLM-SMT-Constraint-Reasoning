# tax_calculators/special_tax.py
from uuid import uuid4
from typing import List, Dict, Optional
from z3 import (
    Context, Optimize,
    Real, Int, RealVal, IntVal, ToInt, ToReal, Sum, sat
)
import re, time

# ────────────────────────── 全域設定 ──────────────────────────
RATE = 0.10
CATEGORIES = ["car", "yacht", "aircraft", "coral_ivory", "furniture"]

_INT   = lambda ctx, p: Int(f'{p}_{uuid4().hex}', ctx)
_REAL  = lambda ctx, p: Real(f'{p}_{uuid4().hex}', ctx)
_RVAL  = lambda ctx, v: RealVal(v, ctx)
_IVAL  = lambda ctx, v: IntVal(v, ctx)

# ────────────────────────── 共用小工具 ──────────────────────────
def _to_num(z3v):
    try:
        return float(z3v.as_decimal(20).rstrip('?'))
    except Exception:
        return float(str(z3v))

def _expand_values(values: Optional[Dict]) -> Dict:
    """補齊五大類的 price / quantity，未提供則預設 0。"""
    v = dict(values or {})
    for cat in CATEGORIES:
        v.setdefault(f"{cat}.price", 0)
        v.setdefault(f"{cat}.quantity", 0)
    return v

def _baseline_tax(values: Dict):
    total, items = 0, []
    for i, cat in enumerate(CATEGORIES):
        price = int(values.get(f"{cat}.price", 0))
        qty   = int(values.get(f"{cat}.quantity", 0))
        tax   = int(price * qty * RATE)
        total += tax
        items.append({"row": i, "cat": cat, "price": price, "quantity": qty, "tax": tax})
    return total, items

# ────────────────────── constraints 正規式與解析器 ───────────────
_ARITH_CMP = { '>':lambda a,b:a>b, '>=':lambda a,b:a>=b,
               '<':lambda a,b:a<b, '<=':lambda a,b:a<=b,
               '=':lambda a,b:a==b, '==':lambda a,b:a==b,
               '!=':lambda a,b:a!=b }

# 允許：car.quantity + yacht.quantity、car.price = yacht.price * 2 等
_BIN_EXPR    = re.compile(r'^(?:\w+)\.(?:price|quantity)\s*([+\-])\s*(?:\w+)\.(?:price|quantity)$')
_RHS_MUL_DIV = re.compile(r'^(?:\w+)\.(?:price|quantity)\s*([*/])\s*(\d+(?:\.\d+)?)$')

def _coerce_num(rhs):
    """把 constraints 裡面的 RHS 轉成 float，支援數字或單元素 list/tuple。"""
    if isinstance(rhs, (list, tuple)):
        rhs = rhs[0] if rhs else 0
    return float(rhs)

def _engaged_cats(free_vars, constraints):
    """
    找出被『牽涉』的品項：
    - 出現在 free_vars 的變數（e.g., car.quantity）
    - 出現在 constraints key 的變數（e.g., car.quantity + yacht.quantity >= 10）
    """
    engaged = set()
    for fv in (free_vars or []):
        if isinstance(fv, str) and '.' in fv:
            cat = fv.split('.', 1)[0]
            if cat in CATEGORIES:
                engaged.add(cat)
    for key in (constraints or {}).keys():
        if not isinstance(key, str):
            continue
        key_ns = key.replace(' ', '')
        parts = re.split(r'[+\-]', key_ns)
        for p in parts:
            if '.' in p:
                cat = p.split('.', 1)[0]
                if cat in CATEGORIES:
                    engaged.add(cat)
    return engaged

def apply_constraints(opt, var_map, cons_json, free_vars, ctx):
    cons_json = cons_json or {}
    for key, rule in cons_json.items():
        key_ns = key.replace(' ', '')
        rule = rule or {}

        # A) 單變數大小比較
        if key_ns in var_map:
            v = var_map[key_ns]
            for cmp_op, rhs in rule.items():
                if cmp_op in ('>', '>=', '<', '<='):
                    try:
                        opt.add(_ARITH_CMP[cmp_op](v, _RVAL(ctx, _coerce_num(rhs))))
                    except Exception:
                        # 忽略非數值 RHS 的比較（避免整體失敗）
                        pass
            # 單變數等式（數值 or 表達式）
            if '=' in rule or '==' in rule:
                rhs_val = rule.get('=', rule.get('=='))
                # 若是數值或單元素陣列，直接等於
                try:
                    num = _coerce_num(rhs_val)
                    opt.add(v == _RVAL(ctx, num))
                except Exception:
                    # 否則當作表達式字串處理（例如 yacht.quantity * 2）
                    rhs_txt = str(rhs_val).strip()
                    rhs_ns = rhs_txt.replace(' ', '')
                    m2 = _RHS_MUL_DIV.match(rhs_ns)
                    if m2:
                        parts = re.split(r'([*/])', rhs_ns, maxsplit=1)
                        rhs_var, op_md, k = parts[0], parts[1], parts[2]
                        if rhs_var in var_map:
                            c = _RVAL(ctx, float(k))
                            expr = var_map[rhs_var] * c if op_md == '*' else var_map[rhs_var] / c
                            opt.add(v == expr)
                    else:
                        # 盡量嘗試轉數字
                        try:
                            opt.add(v == _RVAL(ctx, float(rhs_txt)))
                        except Exception:
                            pass
            continue

        # B) 二元加減（例如 car.quantity + yacht.quantity >= 10）
        m = _BIN_EXPR.match(key_ns)
        if m:
            if '+' in key_ns:
                lhs1, lhs2 = key_ns.split('+')
                has_vars = (lhs1 in var_map) and (lhs2 in var_map)
                if not has_vars:
                    continue
                expr = var_map[lhs1] + var_map[lhs2]
            else:
                lhs1, lhs2 = key_ns.split('-')
                has_vars = (lhs1 in var_map) and (lhs2 in var_map)
                if not has_vars:
                    continue
                expr = var_map[lhs1] - var_map[lhs2]
            for cmp_op, rhs in (rule or {}).items():
                if cmp_op in ('>', '>=', '<', '<='):
                    try:
                        opt.add(_ARITH_CMP[cmp_op](expr, _RVAL(ctx, _coerce_num(rhs))))
                    except Exception:
                        pass
            continue

        # C) 其他情況：key 不是已知變數名稱或二元式，且指定等式
        #    支援 a=b*k / a=b/k 或 a=常數
        if ('=' in rule or '==' in rule) and key_ns in var_map:
            v = var_map[key_ns]
            rhs_val = rule.get('=', rule.get('=='))
            try:
                num = _coerce_num(rhs_val)
                opt.add(v == _RVAL(ctx, num))
            except Exception:
                rhs_txt = str(rhs_val).strip()
                rhs_ns = rhs_txt.replace(' ', '')
                m2 = _RHS_MUL_DIV.match(rhs_ns)
                if m2:
                    parts = re.split(r'([*/])', rhs_ns, maxsplit=1)
                    rhs_var, op_md, k = parts[0], parts[1], parts[2]
                    if rhs_var in var_map:
                        c = _RVAL(ctx, float(k))
                        expr = var_map[rhs_var] * c if op_md == '*' else var_map[rhs_var] / c
                        opt.add(v == expr)
                else:
                    try:
                        opt.add(v == _RVAL(ctx, float(rhs_txt)))
                    except Exception:
                        pass

# ────────────────────── 建模（固定五類） ────────────────────────
def _build_model(values, free_vars, constraints, *, with_qty_objective, ctx):
    opt, vmap = Optimize(ctx=ctx), {}
    rate = _RVAL(ctx, RATE)

    # 找出被牽涉的品項（退化解防護會用到）
    engaged = _engaged_cats(free_vars, constraints)

    for cat in CATEGORIES:
        # price：Int→Real
        p_int = _INT(ctx, f'{cat}_price')
        pz    = ToReal(p_int)
        opt.add(p_int >= _IVAL(ctx, 0))
        if f'{cat}.price' not in free_vars:
            opt.add(p_int == _IVAL(ctx, int(values.get(f'{cat}.price', 0))))
        vmap[f'{cat}.price'] = pz

        # quantity：Int→Real
        q_int = _INT(ctx, f'{cat}_qty')
        q     = ToReal(q_int)
        opt.add(q_int >= _IVAL(ctx, 0))
        if f'{cat}.quantity' not in free_vars:
            opt.add(q_int == _IVAL(ctx, int(values.get(f'{cat}.quantity', 0))))
        vmap[f'{cat}.quantity'] = q

        # ⭐ 退化解防護：
        # 若此品項被牽涉（free_vars 或 constraints key 出現），
        # 且價格未提供（=0 且不是自由變數），則禁止其數量非零。
        if (cat in engaged) and (int(values.get(f'{cat}.price', 0)) == 0) and (f'{cat}.price' not in free_vars):
            opt.add(q == _RVAL(ctx, 0))

    total_tax = _REAL(ctx, 'total_tax')
    item_taxes = [ ToInt(vmap[f'{c}.price'] * vmap[f'{c}.quantity'] * rate) for c in CATEGORIES ]
    opt.add(total_tax == Sum(item_taxes))

    apply_constraints(opt, vmap, constraints, free_vars, ctx)

    total_qty = None
    if with_qty_objective:
        total_qty = _REAL(ctx, 'total_qty')
        opt.add(total_qty == Sum([ vmap[f'{c}.quantity'] for c in CATEGORIES ]))

    return opt, vmap, total_tax, total_qty

def _compute_param_diff(vmap, free_vars, values, model):
    diff = {}
    for key in free_vars:
        if key not in vmap:
            continue
        orig = int(values.get(key, 0))
        optv = int(_to_num(model.eval(vmap[key])))
        if optv != orig:
            diff[key] = {'original': orig, 'optimized': optv, 'difference': optv - orig}
    return diff

def _collect_items(vmap, model):
    items = []
    for i, cat in enumerate(CATEGORIES):
        price = int(_to_num(model.eval(vmap[f'{cat}.price'])))
        qty   = int(_to_num(model.eval(vmap[f'{cat}.quantity'])))
        tax   = int(price * qty * RATE)
        items.append({'row': i, 'cat': cat, 'price': price, 'quantity': qty, 'tax': tax})
    return items

# ────────────────────── minimize（特種貨物） ─────────────────────
# 只吃扁平欄位；以 **values 收所有 car.price 等鍵，完全移除 rows 相關邏輯
def minimize_sg_tax(*, free_vars: Optional[List[str]] = None, constraints: Optional[Dict] = None, **values):
    started = time.perf_counter()
    free_vars, constraints = set(free_vars or []), constraints or {}
    values = _expand_values(values)

    # baseline
    baseline_total, baseline_items = _baseline_tax(values)

    # baseline + constraints（不放行）
    if constraints:
        ctx0 = Context()
        opt0, vmap0, total0, _ = _build_model(values, set(), constraints, with_qty_objective=False, ctx=ctx0)
        if opt0.check() == sat:
            m0 = opt0.model()
            baseline_status = 'sat'
            baseline_wc = int(_to_num(m0.eval(total0)))
            baseline_wc_items = _collect_items(vmap0, m0)
        else:
            baseline_status, baseline_wc, baseline_wc_items = 'unsat', None, None
    else:
        baseline_status, baseline_wc, baseline_wc_items = 'na', baseline_total, baseline_items

    # 優化
    ctx = Context()
    opt, vmap, total_tax, _ = _build_model(values, free_vars, constraints, with_qty_objective=False, ctx=ctx)
    opt.minimize(total_tax)
    if opt.check() != sat:
        return {'no_solution': True, 'free_vars': list(free_vars), 'constraints': constraints}

    m = opt.model()
    optimized = int(_to_num(m.eval(total_tax)))
    items = _collect_items(vmap, m)
    pdiff = _compute_param_diff(vmap, free_vars, values, m)

    print(f"minimize_sg_tax 耗時：{time.perf_counter() - started:.4f} 秒")
    return {
        'baseline': baseline_total,
        'baseline_items': baseline_items,
        'baseline_status': baseline_status,
        'baseline_with_constraints': baseline_wc,
        'baseline_with_constraints_items': baseline_wc_items,
        'optimized': optimized,
        'optimized_total_tax': optimized,
        'optimized_items': items,
        'param_diff': pdiff,
        'diff': pdiff,
        'final_params': {
            k: {
                'value': int(_to_num(m.eval(vmap[k]))),
                'type': ('free' if k in free_vars else 'fixed')
            }
            for k in vmap.keys()
        },
        'free_vars': list(free_vars),
        'constraints': constraints
    }

# ────────────────────── maximize（在上限內最大化數量） ───────────
# 同樣只吃扁平欄位；以 **values 收鍵
def maximize_qty_under_budget_sg(*, free_vars: Optional[List[str]] = None,
                                 constraints: Optional[Dict] = None, budget_tax: Optional[int] = None, **values):
    if budget_tax is None:
        raise ValueError('缺少 budget_tax')
    free_vars, constraints = set(free_vars or []), constraints or {}
    values = _expand_values(values)

    baseline_total, baseline_items = _baseline_tax(values)

    # baseline + constraints
    if constraints:
        ctx0 = Context()
        opt0, vmap0, total0, _ = _build_model(values, set(), constraints, with_qty_objective=False, ctx=ctx0)
        if opt0.check() == sat:
            m0 = opt0.model()
            baseline_status = 'sat'
            baseline_wc = int(_to_num(m0.eval(total0)))
            baseline_wc_items = _collect_items(vmap0, m0)
        else:
            baseline_status, baseline_wc, baseline_wc_items = 'unsat', None, None
    else:
        baseline_status, baseline_wc, baseline_wc_items = 'na', baseline_total, baseline_items

    # 優化
    ctx = Context()
    opt, vmap, total_tax, total_qty = _build_model(values, free_vars, constraints, with_qty_objective=True, ctx=ctx)
    opt.add(total_tax <= _RVAL(ctx, float(budget_tax)))
    opt.maximize(total_qty)
    if opt.check() != sat:
        return {'no_solution': True, 'free_vars': list(free_vars), 'constraints': constraints, 'budget_tax': budget_tax}

    m = opt.model()
    opt_tax = int(_to_num(m.eval(total_tax)))
    opt_qty = int(_to_num(m.eval(total_qty)))
    items = _collect_items(vmap, m)
    pdiff = _compute_param_diff(vmap, free_vars, values, m)

    return {
        'baseline': baseline_total,
        'baseline_items': baseline_items,
        'baseline_status': baseline_status,
        'baseline_with_constraints': baseline_wc,
        'baseline_with_constraints_items': baseline_wc_items,
        'optimized_total_tax': opt_tax,
        'optimized_total_qty': opt_qty,
        'optimized_items': items,
        'param_diff': pdiff,
        'diff': pdiff,
        'final_params': {
            k: {
                'value': int(_to_num(m.eval(vmap[k]))),
                'type': ('free' if k in free_vars else 'fixed')
            }
            for k in vmap.keys()
        },
        'free_vars': list(free_vars),
        'constraints': constraints,
        'budget_tax': budget_tax
    }

# ────────────────────── 勞務（最小化稅額） ──────────────────────
def minimize_special_services_tax(*, sales_price: float, free_vars: Optional[List[str]] = None, constraints: Optional[Dict] = None):
    started = time.perf_counter()
    free_vars, constraints = set(free_vars or []), constraints or {}
    ctx  = Context(); opt = Optimize(ctx=ctx)
    sp_i = _INT(ctx, 'sales_price_int')
    sp   = ToReal(sp_i)
    tax  = _INT(ctx, 'tax')

    opt.add(sp_i >= _IVAL(ctx, 0))
    if 'sales_price' not in free_vars:
        opt.add(sp_i == _IVAL(ctx, int(sales_price)))

    if 'sales_price' in (constraints or {}):
        for cmp_op, rhs in constraints['sales_price'].items():
            if cmp_op in _ARITH_CMP:
                try:
                    opt.add(_ARITH_CMP[cmp_op](sp, _RVAL(ctx, _coerce_num(rhs))))
                except Exception:
                    pass

    opt.add(tax == ToInt(sp * _RVAL(ctx, RATE)), tax >= _IVAL(ctx, 0))
    opt.minimize(tax)

    if opt.check() != sat:
        return {'no_solution': True, 'constraints': constraints}

    m = opt.model()
    print(f"minimize_special_services_tax 耗時：{time.perf_counter() - started:.4f} 秒")
    opt_sp = int(_to_num(m.eval(sp)))
    return {
        'baseline': int(sales_price * RATE),
        'optimized': int(m.eval(tax).as_long()),
        'optimized_tax': int(m.eval(tax).as_long()),
        'final_params': {'sales_price': {'value': opt_sp, 'type': ('free' if 'sales_price' in free_vars else 'fixed')}},
        'param_diff': ({'sales_price': {
            'original': int(sales_price), 'optimized': opt_sp, 'difference': opt_sp - int(sales_price)
        }} if 'sales_price' in free_vars else {}),
        'diff': ({'sales_price': {
            'original': int(sales_price), 'optimized': opt_sp, 'difference': opt_sp - int(sales_price)
        }} if 'sales_price' in free_vars else {}),
        'constraints': constraints,
        'free_vars': list(free_vars)
    }
