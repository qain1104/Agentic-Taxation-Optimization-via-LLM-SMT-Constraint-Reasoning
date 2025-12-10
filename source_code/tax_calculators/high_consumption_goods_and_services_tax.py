from uuid import uuid4
from z3 import Context, Optimize, Real, Int, RealVal, IntVal, ToInt, sat
from .constraint_utils import apply_linear_constraints

RATE = 0.10  # 10%

def _fresh_real(ctx: Context, prefix: str):
    return Real(f"{prefix}_{uuid4().hex}", ctx)

def _fresh_int(ctx: Context, prefix: str):
    return Int(f"{prefix}_{uuid4().hex}", ctx)

def _R(ctx: Context, num: int, den: int = 1):
    # 用有理數建立 Real，避免浮點誤差
    return RealVal(num, ctx) / RealVal(den, ctx)

def _RV(ctx: Context, x):
    # 用字串或整數建立 RealVal，減少 float 誤差
    if isinstance(x, (int,)):
        return RealVal(x, ctx)
    try:
        return RealVal(str(float(x)), ctx)
    except Exception:
        return RealVal(0, ctx)

def calculate_special_goods_tax(
    item_type: str,
    base_amount: float,
    *,
    free_vars: list | None = None,
    constraints: dict | None = None,
):
    """
    以 10% 稅率計算高額消費貨物稅；若 'base_amount' 在 free_vars 中，則最小化稅額（→ 會壓到 0）。
    回傳最終稅額（int）。若想更豐富可改回傳 dict。
    """
    free_vars = set(free_vars or [])
    constraints = constraints or {}

    # clamp 避免 UNSAT
    try:
        base_amount = max(0.0, float(base_amount))
    except Exception:
        base_amount = 0.0

    ctx = Context()
    opt = Optimize(ctx=ctx)

    ba_z = _fresh_real(ctx, "ba")
    tax_z = _fresh_int(ctx, "tax")

    zero = _RV(ctx, 0)
    rate = _R(ctx, 1, 10)  # 0.1 as rational

    opt.add(ba_z >= zero)
    if "base_amount" not in free_vars:
        opt.add(ba_z == _RV(ctx, base_amount))

    opt.add(tax_z == ToInt(ba_z * rate), tax_z >= IntVal(0, ctx))
    opt.minimize(tax_z)

    # 套用 constraints
    params_for_constraints = {
        "base_amount": (ba_z,),
        "tax": (tax_z,),
    }
    apply_linear_constraints(opt, params_for_constraints, constraints, debug=False)

    assert opt.check() == sat  # 這個模型在上面 clamp 後理論上永遠 SAT
    return opt.model()[tax_z].as_long()

def calculate_special_services_tax(
    sales_price: float,
    *,
    free_vars: list | None = None,
    constraints: dict | None = None,
):
    """
    以 10% 稅率計算高額消費勞務稅；若 'sales_price' 在 free_vars 中，則最小化稅額（→ 會壓到 0）。
    回傳最終稅額（int）。
    """
    free_vars = set(free_vars or [])
    constraints = constraints or {}

    # clamp 避免 UNSAT
    try:
        sales_price = max(0.0, float(sales_price))
    except Exception:
        sales_price = 0.0

    ctx = Context()
    opt = Optimize(ctx=ctx)

    sp_z  = _fresh_real(ctx, "sp")
    tax_z = _fresh_int(ctx, "tax")

    zero = _RV(ctx, 0)
    rate = _R(ctx, 1, 10)  # 0.1 as rational

    opt.add(sp_z >= zero)
    if "sales_price" not in free_vars:
        opt.add(sp_z == _RV(ctx, sales_price))

    opt.add(tax_z == ToInt(sp_z * rate), tax_z >= IntVal(0, ctx))
    opt.minimize(tax_z)

    # 套用 constraints
    params_for_constraints = {
        "sales_price": (sp_z,),
        "tax": (tax_z,),
    }
    apply_linear_constraints(opt, params_for_constraints, constraints, debug=False)

    assert opt.check() == sat
    return opt.model()[tax_z].as_long()