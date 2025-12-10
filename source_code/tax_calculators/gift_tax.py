# -*- coding: utf-8 -*-
"""
gift_tax_optimizer.py ── 贈與稅最適化模組
功能：
  • 支援 free_vars（指定哪些變數可調整）
  • 支援 variable_constraints（線性表達式比較、比例條件等，委由 constraint_utils）
  • 回傳 dict：mode / input_params / baseline / baseline_status /
                baseline_with_constraints / optimized / status /
                diff / final_params / constraints
"""

from __future__ import annotations
import re, time
from typing import Dict, List, Tuple, Any, Optional

from z3 import Optimize, Real, Int, If, And, Or, ToInt, sat
from .constraint_utils import apply_linear_constraints  # 共用線性約束 parser

# ─── 共用錯誤與工具 ───────────────────────────────────────────────
class UnsatError(Exception):
    """Constraint set is UNSAT but we want to report gracefully."""
    pass

# 支援乘除 RHS pattern（只給 _normalize_rhs 用，預檢等式時解析字串）
_RHS_MUL_DIV = re.compile(r"^\s*(\w+)\s*([*/])\s*(-?\d+(?:\.\d+)?)\s*$")

# ───────── 中文數字單位與 RHS 正規化（只用在 _check_conflict_fixed） ─────────
_CN_UNITS = {"萬": 1e4, "亿": 1e8, "億": 1e8, "兆": 1e12}


def _parse_cn_amount(s: str) -> float | None:
    """解析純數字字串（可含小數）+ 中文單位（萬/億/兆）、百分比、或含千分位逗號。"""
    if not isinstance(s, str):
        return None
    s = s.strip().replace(",", "")
    m = re.match(r"^(-?\d+(?:\.\d+)?)([萬亿億兆]?)$", s)
    if m:
        v = float(m.group(1))
        u = m.group(2)
        return v * _CN_UNITS[u] if u else v
    m2 = re.match(r"^(-?\d+(?:\.\d+)?)\s*%$", s)  # 60% → 0.6
    if m2:
        return float(m2.group(1)) / 100.0
    return None


def _normalize_rhs(rhs):
    """
    規一 RHS 為：
      - int/float：純常數
      - 其他      ：原樣或轉回字串（讓上層判斷是否為純常數）
    僅用在 _check_conflict_fixed（預檢固定值等式），不參與真正約束生成。
    """
    # 攤平單元素 list/tuple
    if isinstance(rhs, (list, tuple)):
        if len(rhs) == 1:
            rhs = rhs[0]
        else:
            raise ValueError(f"RHS list/tuple expects a single element, got {rhs}")

    if isinstance(rhs, (int, float)):
        return rhs

    if isinstance(rhs, str):
        s = rhs.strip()
        # 有 * / 或看起來像變數名，就交給 Z3，不做預檢
        if _RHS_MUL_DIV.match(s):
            return s
        if re.fullmatch(r"\w+", s):
            return s
        v = _parse_cn_amount(s)
        if v is not None:
            return v
        try:
            return float(s)  # 一般數字字串
        except ValueError:
            return s
    return rhs


def _check_conflict_fixed(name: str, fixed: float, cons: Dict[str, Any]):
    """
    只有當 cons["="] 是「純常數」時才做預檢；若 RHS 是變數或含 */ 的運算式，交給 Z3。
    同時支援單元素 list/tuple、中文單位、百分比等。
    """
    if "=" not in cons:
        return
    rhsn = _normalize_rhs(cons["="])
    if isinstance(rhsn, (int, float)):
        if fixed != float(rhsn):
            raise UnsatError(f"{name} fixed value {fixed} != {rhsn}")
    # 其餘型別（純變數名或運算式）不預先檢查，留給 Z3 + apply_linear_constraints 處理


# ───────────────────────────────────────────────────────────────
# 1. 贈與稅核心函式
# ───────────────────────────────────────────────────────────────
def _calculate_gift_tax_internal(
    *,
    period_choice: int,
    land_value: float,
    ground_value: float,
    house_value: float,
    others_value: float,
    not_included_land: float,
    not_included_house: float,
    not_included_others: float,
    remaining_exemption_98: float = 0,
    previous_gift_sum_in_this_year: float = 0,
    land_increment_tax: float = 0,
    deed_tax: float = 0,
    other_gift_burdens: float = 0,
    previous_gift_tax_or_credit: float = 0,
    new_old_system_adjustment: float = 0,
    free_vars: Optional[List[str]] = None,
    constraints: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[int, int, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    計算並最小化贈與稅：
      free_vars       可優化變數列表
      constraints     所有比較式條件（交給 constraint_utils.apply_linear_constraints）
    回傳：(最佳稅額, 課稅贈與淨額, final_params, diff)
    """
    free_vars = set(free_vars or [])
    cons = constraints or {}
    opt = Optimize()

    # 1. 宣告 Z3 變數與初始參數
    def R(name: str):
        return Real(name)

    params: Dict[str, Tuple[Any, float, List[Any]]] = {
        "land_value": (R("land_value"), land_value, [lambda v: v >= 0]),
        "ground_value": (R("ground_value"), ground_value, [lambda v: v >= 0]),
        "house_value": (R("house_value"), house_value, [lambda v: v >= 0]),
        "others_value": (R("others_value"), others_value, [lambda v: v >= 0]),
        "not_included_land": (R("not_included_land"), not_included_land, [lambda v: v >= 0]),
        "not_included_house": (R("not_included_house"), not_included_house, [lambda v: v >= 0]),
        "not_included_others": (R("not_included_others"), not_included_others, [lambda v: v >= 0]),
        "remaining_exemption_98": (R("remaining_exemption_98"), remaining_exemption_98, [lambda v: v >= 0]),
        "previous_gift_sum_in_this_year": (
            R("previous_gift_sum_in_this_year"),
            previous_gift_sum_in_this_year,
            [lambda v: v >= 0],
        ),
        "land_increment_tax": (R("land_increment_tax"), land_increment_tax, [lambda v: v >= 0]),
        "deed_tax": (R("deed_tax"), deed_tax, [lambda v: v >= 0]),
        "other_gift_burdens": (R("other_gift_burdens"), other_gift_burdens, [lambda v: v >= 0]),
        "previous_gift_tax_or_credit": (
            R("previous_gift_tax_or_credit"),
            previous_gift_tax_or_credit,
            [lambda v: v >= 0],
        ),
        "new_old_system_adjustment": (R("new_old_system_adjustment"), new_old_system_adjustment, []),
    }

    # 2. 綁定固定或自由變數
    for name, (zv, orig, checks) in params.items():
        extra = cons.get(name, {})
        if name in free_vars:
            # 自由變數：僅加內建檢查
            for c in checks:
                opt.add(c(zv))
        else:
            # 固定變數：檢查等式衝突，再綁定
            _check_conflict_fixed(name, orig, extra)
            opt.add(zv == orig)
            for c in checks:
                opt.add(c(zv))

    # 3. 套用所有 user 定義的比較式條件（共用 linear expression parser）
    #    constraint_utils.apply_linear_constraints 只需要 name -> (z3_var,) 的 mapping
    params_for_constraints = {name: (zv,) for name, (zv, _, _) in params.items()}
    apply_linear_constraints(opt, params_for_constraints, cons, debug=False)

    # 4. 贈與稅核心公式
    V = lambda n: params[n][0]

    T0 = R("this_total")
    opt.add(T0 == V("land_value") + V("ground_value") + V("house_value") + V("others_value"))

    TA = R("annual_total")
    opt.add(TA == T0 + V("previous_gift_sum_in_this_year"))

    D = R("deductions")
    opt.add(D == V("land_increment_tax") + V("deed_tax") + V("other_gift_burdens"))

    N = R("net_tax")  # 淨課稅額（可能負）
    RATE = R("applied_rate")
    PD = R("prog_diff")
    TAX_R = R("tax_real")
    TAX_I = Int("tax_int")
    # floor 且不可負
    opt.add(TAX_I == ToInt(TAX_R), TAX_I >= 0)

    # 分期計算
    if period_choice == 1:
        opt.add(N == T0 - V("remaining_exemption_98") - D)
        opt.add(RATE == 0.10, PD == 0)
        opt.add(TAX_R == N * RATE - V("previous_gift_tax_or_credit"))

    elif period_choice == 2:
        opt.add(N == TA - 2_200_000 - D)
        c1 = N <= 25_000_000
        c2 = And(N > 25_000_000, N <= 50_000_000)
        opt.add(Or(c1, c2, N > 50_000_000))
        opt.add(
            If(
                c1,
                And(RATE == 0.10, PD == 0),
                If(c2, And(RATE == 0.15, PD == 1_250_000), And(RATE == 0.20, PD == 3_750_000)),
            )
        )
        opt.add(
            TAX_R
            == N * RATE
            - PD
            - V("previous_gift_tax_or_credit")
            - V("new_old_system_adjustment")
        )

    elif period_choice == 3:
        opt.add(N == TA - 2_440_000 - D)
        c1 = N <= 25_000_000
        c2 = And(N > 25_000_000, N <= 50_000_000)
        opt.add(Or(c1, c2, N > 50_000_000))
        opt.add(
            If(
                c1,
                And(RATE == 0.10, PD == 0),
                If(c2, And(RATE == 0.15, PD == 1_250_000), And(RATE == 0.20, PD == 3_750_000)),
            )
        )
        opt.add(TAX_R == N * RATE - PD - V("previous_gift_tax_or_credit"))

    elif period_choice == 4:
        opt.add(N == TA - 2_440_000 - D)
        c1 = N <= 28_110_000
        c2 = And(N > 28_110_000, N <= 56_210_000)
        opt.add(Or(c1, c2, N > 56_210_000))
        opt.add(
            If(
                c1,
                And(RATE == 0.10, PD == 0),
                If(c2, And(RATE == 0.15, PD == 1_405_500), And(RATE == 0.20, PD == 4_216_000)),
            )
        )
        opt.add(TAX_R == N * RATE - PD - V("previous_gift_tax_or_credit"))

    else:
        raise ValueError("Unsupported period_choice")

    # 5. 求解並最小化
    opt.minimize(TAX_I)
    if opt.check() != sat:
        raise UnsatError("constraint set unsat")
    m = opt.model()
    best_tax = m[TAX_I].as_long()
    net_taxable = m.evaluate(ToInt(N)).as_long()

    # 6. 收集所有變數最終值及差異
    final_params: Dict[str, Dict[str, Any]] = {}
    diff: Dict[str, Dict[str, Any]] = {}
    for name, (zv, orig, _) in params.items():
        mv = m.evaluate(zv)
        if mv.is_int_value():
            val = mv.as_long()
        else:
            # 以 12 位小數取得，再轉 float
            s = mv.as_decimal(12).rstrip("?")
            val = float(s)
        final_params[name] = {
            "value": val,
            "type": "free" if name in free_vars else "fixed",
        }
        if val != orig:
            diff[name] = {"original": orig, "optimized": val, "difference": val - orig}

    return best_tax, net_taxable, final_params, diff


# ───────────────────────────────────────────────────────────────
# 2. Public API
# ───────────────────────────────────────────────────────────────
def calculate_gift_tax(
    *,
    period_choice: int = 1,
    land_value: float = 0.0,
    ground_value: float = 0.0,
    house_value: float = 0.0,
    others_value: float = 0.0,
    not_included_land: float = 0.0,
    not_included_house: float = 0.0,
    not_included_others: float = 0.0,
    remaining_exemption_98: float = 0.0,
    previous_gift_sum_in_this_year: float = 0.0,
    land_increment_tax: float = 0.0,
    deed_tax: float = 0.0,
    other_gift_burdens: float = 0.0,
    previous_gift_tax_or_credit: float = 0.0,
    new_old_system_adjustment: float = 0.0,
    free_vars: Optional[List[str] | str] = None,
    constraints: Optional[Dict[str, Dict[str, Any]]] = None,
    # 如上游不小心多塞參數，避免噴 unexpected kw
    **_extra,
) -> Dict[str, Any]:
    """
    公開函式：計算贈與稅並可最佳化
    回傳結構：
      {
        mode,
        input_params,
        baseline,
        baseline_status,
        baseline_with_constraints,
        optimized,
        status,
        diff,
        final_params,
        constraints
      }
    """
    started_time = time.perf_counter()

    # -- 正規化 free_vars：list / CSV -> list（去重、保序）
    def _normalize_free_vars(fv) -> List[str]:
        if not fv:
            return []
        if isinstance(fv, list):
            arr = [str(x).strip() for x in fv if str(x).strip()]
        else:
            arr = [t.strip() for t in str(fv).split(",") if t.strip()]
        seen, out = set(), []
        for v in arr:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    fv_list = _normalize_free_vars(free_vars)
    cons = constraints or {}

    # 僅把 internal 需要的鍵丟進 kwargs（不含 constraints）
    kwargs = dict(
        period_choice=period_choice,
        land_value=land_value,
        ground_value=ground_value,
        house_value=house_value,
        others_value=others_value,
        not_included_land=not_included_land,
        not_included_house=not_included_house,
        not_included_others=not_included_others,
        remaining_exemption_98=remaining_exemption_98,
        previous_gift_sum_in_this_year=previous_gift_sum_in_this_year,
        land_increment_tax=land_increment_tax,
        deed_tax=deed_tax,
        other_gift_burdens=other_gift_burdens,
        previous_gift_tax_or_credit=previous_gift_tax_or_credit,
        new_old_system_adjustment=new_old_system_adjustment,
        free_vars=fv_list,
    )

    # (1) baseline（完全固定，不帶 constraints）
    baseline, _, _, _ = _calculate_gift_tax_internal(
        **{k: v for k, v in kwargs.items() if k != "free_vars"},
        constraints={},
    )

    # (2) baseline + constraints 可行性檢查
    try:
        _calculate_gift_tax_internal(
            **{k: v for k, v in kwargs.items() if k != "free_vars"},
            constraints=cons,
        )
        baseline_status = "sat"
        baseline_with = baseline
    except UnsatError:
        baseline_status = "unsat"
        baseline_with = None

    # (3) 如果有 free_vars，就做 manual_free 最佳化
    mode, optimized, status = "baseline", baseline, baseline_status
    diff_out: Dict[str, Any] = {}
    params_out: Dict[str, Any] = {}
    if kwargs["free_vars"]:
        mode = "manual_free"
        try:
            optimized, _, params_out, diff_out = _calculate_gift_tax_internal(
                **kwargs,
                constraints=cons,
            )
            status = "sat"
        except UnsatError:
            optimized = None
            status = "unsat"

    print(f"Total time for calculate_gift_tax: {time.perf_counter() - started_time:.2f} seconds")
    return {
        "mode": mode,
        "input_params": kwargs,
        "baseline": baseline,
        "baseline_status": baseline_status,
        "baseline_with_constraints": baseline_with,
        "optimized": optimized,
        "status": status,
        "diff": diff_out,
        "final_params": params_out,
        "constraints": cons,
    }
