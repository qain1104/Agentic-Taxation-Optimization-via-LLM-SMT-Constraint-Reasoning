import re
import ast
from typing import Dict, Any, Tuple, Set, Mapping, Optional

from z3 import Optimize, RealVal, ToReal, Real, Int, IntVal, ToReal

ParamsLike = Mapping[str, Tuple[Any, ...]]


def apply_linear_constraints(
    opt: Optimize,
    params: ParamsLike,
    cons_dict: Dict[str, Dict[str, Any]],
    debug: bool = False,
) -> None:
    """
    通用線性約束套用器：

    - 支援：LHS (op) RHS
    - LHS / RHS 都可以是線性運算式：
        * 變數：a
        * 常數：100000, 0.3
        * +, -, *（僅允許「常數 × 線性式」）
        * /（僅允許「線性式 ÷ 常數」，結果視為 Real）
        * 括號：(a + b) * 2
    - 額外支援「比例」形式：
        * (線性式) / (線性式) ⋈ c   （c 為常數）
          會自動轉成： (線性式) ⋈ c * (線性式) 並加上分母 > 0
    - 不允許：一般情況下變數在分母（除非被上面比例特例處理）、變數 × 變數。

    `params` 只會用到第一個元素（z3_var），其他 tuple 內容忽略。
    """

    if not cons_dict:
        return

    # 允許的比較子
    cmp_ops = {">", ">=", "<", "<=", "=", "=="}

    # Int / Real 比較 mapping
    cmp_map_int = {
        ">":  lambda a, b: a >  b,
        ">=": lambda a, b: a >= b,
        "<":  lambda a, b: a <  b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "=":  lambda a, b: a == b,
    }
    cmp_map_real = cmp_map_int  # Real 也用同樣語意

    # ---- 小工具：檢查 Z3 expr 型別 ----
    def _is_real_sort(e) -> bool:
        return hasattr(e, "is_real") and e.is_real()

    def _is_int_sort(e) -> bool:
        return hasattr(e, "is_int") and e.is_int()

    # ---- 中文數字單位與 RHS 正規化 ----
    _CN_UNITS = {"萬": 1e4, "亿": 1e8, "億": 1e8, "兆": 1e12}

    def parse_cn_amount(s: str) -> Optional[float]:
        """
        解析純數值（可含小數）+ 中文單位（萬/億/兆），
        或含千分位逗號、百分比。
        僅在「純數值」情境使用；若有變數/運算符就不處理。
        """
        if not isinstance(s, str):
            return None
        s = s.strip().replace(",", "")
        m = re.match(r"^(-?\d+(?:\.\d+)?)([萬亿億兆]?)$", s)
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            if unit:
                val *= _CN_UNITS[unit]
            return val
        # 百分比："60%" → 0.6
        m2 = re.match(r"^(-?\d+(?:\.\d+)?)\s*%$", s)
        if m2:
            return float(m2.group(1)) / 100.0
        return None

    def _normalize_rhs(rhs: Any) -> Any:
        """
        把 RHS 規整成：
            - int/float：純常數
            - str：一般線性運算式字串，交給 AST parser
            - 其他：原樣丟回（最後會被視為錯誤）
        也支援單元素 list/tuple。
        """
        # 展開單元素 list/tuple
        if isinstance(rhs, (list, tuple)):
            if len(rhs) == 1:
                rhs = rhs[0]
            else:
                raise ValueError(f"RHS list/tuple expects a single element, got {rhs}")

        # 直接是數值
        if isinstance(rhs, (int, float)):
            return rhs

        # 字串：可能是 常數 or 運算式
        if isinstance(rhs, str):
            s = rhs.strip()
            # 先試中文單位/百分比
            val = parse_cn_amount(s)
            if val is not None:
                return val
            # 再試一般純數字
            try:
                return float(s.replace(",", ""))
            except ValueError:
                # 交給線性 parser 處理（例如 "a + b"）
                return s

        # 其他型別
        return rhs

    # --- 把 expr 裡的變數名稱換成安全識別符號 ---
    def _sanitize_expr_and_varmap(expr_str: str) -> Tuple[str, Dict[str, str]]:
        s = str(expr_str)
        var_map: Dict[str, str] = {}

        # 為避免較短名稱蓋到較長名稱，先以長度由長到短排序
        for idx, name in enumerate(sorted(params.keys(), key=len, reverse=True)):
            safe_name = f"v{idx}"
            pattern = rf"(?<![A-Za-z0-9_.]){re.escape(name)}(?![A-Za-z0-9_.])"
            s_new, n = re.subn(pattern, safe_name, s)
            if n > 0:
                var_map[safe_name] = name
                s = s_new

        return s, var_map

    # --- 核心：把一個線性運算式字串 parse 成 (z3_expr, is_real, vars_set) ---
    def _parse_linear_expr(expr_str: str) -> Tuple[Any, bool, Set[str]]:
        """
        回傳：
            expr    : Z3 算術表達式 (Int 或 Real)
            is_real : True 代表含有浮點 / 除法 / Real 變數（需要用 Real 比較）
            vars_set: 運算式中出現的變數名稱集合
        """
        sanitized, var_map = _sanitize_expr_and_varmap(expr_str)
        try:
            node = ast.parse(sanitized, mode="eval").body
        except SyntaxError:
            raise ValueError(f"Invalid linear expression syntax: {expr_str!r}")

        def visit(n: ast.AST) -> Tuple[Any, bool, Set[str]]:
            # 變數
            if isinstance(n, ast.Name):
                safe = n.id
                if safe not in var_map:
                    raise ValueError(
                        f"Unknown variable '{safe}' in expression {expr_str!r}"
                    )
                orig = var_map[safe]
                if orig not in params:
                    raise ValueError(
                        f"Unknown variable '{orig}' in expression {expr_str!r}"
                    )
                z, *_ = params[orig]  # 可能是 Int 或 Real
                # 依 Z3 expr 本身型別決定 is_real
                is_real = _is_real_sort(z)
                return z, is_real, {orig}

            # 常數
            if isinstance(n, ast.Constant):
                v = n.value
                if not isinstance(v, (int, float)):
                    raise ValueError(
                        f"Non-numeric literal {v!r} in expression {expr_str!r}"
                    )
                if isinstance(v, int):
                    return v, False, set()
                else:
                    return RealVal(v), True, set()

            # 單元運算：+x / -x
            if isinstance(n, ast.UnaryOp) and isinstance(
                n.op, (ast.UAdd, ast.USub)
            ):
                expr, is_real, vs = visit(n.operand)
                if isinstance(n.op, ast.USub):
                    expr = -expr
                return expr, is_real, vs

            # 二元運算：+,-,*,/
            if isinstance(n, ast.BinOp) and isinstance(
                n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
            ):
                left_expr, left_real, left_vars = visit(n.left)
                right_expr, right_real, right_vars = visit(n.right)

                # 加減：線性 + 線性 -> 線性
                if isinstance(n.op, (ast.Add, ast.Sub)):
                    is_real = (
                        left_real
                        or right_real
                        or _is_real_sort(left_expr)
                        or _is_real_sort(right_expr)
                    )
                    lhs_z3, rhs_z3 = left_expr, right_expr
                    if is_real:
                        if _is_int_sort(lhs_z3):
                            lhs_z3 = ToReal(lhs_z3)
                        if _is_int_sort(rhs_z3):
                            rhs_z3 = ToReal(rhs_z3)
                    expr = lhs_z3 + rhs_z3 if isinstance(n.op, ast.Add) else lhs_z3 - rhs_z3
                    vars_set = left_vars | right_vars
                    return expr, is_real, vars_set

                # 乘法：只允許「常數 × 線性式」
                if isinstance(n.op, ast.Mult):
                    if left_vars and right_vars:
                        raise ValueError(
                            f"Non-linear term (variable * variable) "
                            f"in expression {expr_str!r}"
                        )

                    # 有變數的一側是 term，另一側是 scalar
                    if left_vars:
                        term_expr, term_real, term_vars = (
                            left_expr,
                            left_real,
                            left_vars,
                        )
                        scalar_expr, scalar_real, scalar_vars = (
                            right_expr,
                            right_real,
                            right_vars,
                        )
                    else:
                        term_expr, term_real, term_vars = (
                            right_expr,
                            right_real,
                            right_vars,
                        )
                        scalar_expr, scalar_real, scalar_vars = (
                            left_expr,
                            left_real,
                            left_vars,
                        )

                    if scalar_vars:
                        raise ValueError(
                            f"Non-linear scalar in expression {expr_str!r}"
                        )

                    is_real = (
                        term_real
                        or scalar_real
                        or _is_real_sort(term_expr)
                        or _is_real_sort(scalar_expr)
                    )
                    term_z3, scalar_z3 = term_expr, scalar_expr
                    if is_real and _is_int_sort(term_z3):
                        term_z3 = ToReal(term_z3)
                    expr = term_z3 * scalar_z3
                    return expr, is_real, term_vars

                # 除法：僅允許「線性式 ÷ 常數」，結果當作 Real
                if isinstance(n.op, ast.Div):
                    if right_vars:
                        # 一般情況的「變數在分母」不在這裡處理
                        raise ValueError(
                            f"Non-linear division (variable in denominator) "
                            f"in expression {expr_str!r}"
                        )
                    term_expr, term_real, term_vars = (
                        left_expr,
                        left_real,
                        left_vars,
                    )
                    scalar_expr, scalar_real, scalar_vars = (
                        right_expr,
                        right_real,
                        right_vars,
                    )
                    if scalar_vars:
                        raise ValueError(
                            f"Non-linear scalar in division in expression {expr_str!r}"
                        )

                    term_z3, scalar_z3 = term_expr, scalar_expr
                    if not _is_real_sort(term_z3):
                        term_z3 = ToReal(term_z3)
                    if not _is_real_sort(scalar_z3):
                        if isinstance(scalar_z3, (int, float)):
                            # 純數字 → 先變成 Z3 Real 常數
                            scalar_z3 = RealVal(float(scalar_z3))
                        else:
                            # 已經是 Z3 Int/Real expr，正常轉型
                            scalar_z3 = ToReal(scalar_z3)
                    expr = term_z3 / scalar_z3
                    return expr, True, term_vars

            # 其他 AST 節點不允許
            raise ValueError(
                f"Unsupported expression construct "
                f"{ast.dump(n)} in {expr_str!r}"
            )

        return visit(node)

    # --- 把 RHS（可能是數字 or 字串）轉為線性式 ---
    def _parse_side(side: Any) -> Tuple[Any, bool, Set[str]]:
        val = _normalize_rhs(side)
        # 純數字：直接當常數
        if isinstance(val, (int, float)):
            if isinstance(val, int):
                return val, False, set()
            else:
                return RealVal(val), True, set()
        # 字串：當線性式 parse
        if isinstance(val, str):
            return _parse_linear_expr(val)
        raise ValueError(f"Unsupported RHS type: {type(side)} ({side!r})")

    # --- 嘗試偵測「比例約束」: (Num / Den) ⋈ c, 其中 c 為常數 ---
    _ratio_pat = re.compile(r"^\s*(.+?)\s*/\s*(.+?)\s*$")

    # --- 主迴圈：對每一條 expr op rhs 產生 Z3 約束 ---
    for expr, rule in (cons_dict or {}).items():
        if not isinstance(rule, dict):
            raise ValueError(f"Invalid constraint rule for {expr}: {rule!r}")

        expr_str = str(expr)

        for cmp_op, rhs in rule.items():
            if cmp_op not in cmp_ops:
                raise ValueError(f"Unsupported comparator '{cmp_op}' in {expr}")

            # 先試「比例特例」： (Num / Den) ⋈ c
            m_ratio = _ratio_pat.match(expr_str)
            if m_ratio:
                num_str, den_str = m_ratio.groups()
                rhsn = _normalize_rhs(rhs)

                # 只在 RHS 是純常數 & 比較子是 >,>=,<,<= 時啟動比例特例
                if isinstance(rhsn, (int, float)) and cmp_op in {">", ">=", "<", "<="}:
                    num_expr, num_is_real, _ = _parse_linear_expr(num_str)
                    den_expr, den_is_real, _ = _parse_linear_expr(den_str)

                    c_val = float(rhsn)
                    c_r = RealVal(c_val)

                    # 轉成 Real 語意
                    num_r = num_expr if _is_real_sort(num_expr) else ToReal(num_expr)
                    den_r = den_expr if _is_real_sort(den_expr) else ToReal(den_expr)

                    # 分母 > 0 避免除以 0，同時也讓不等式方向有意義
                    opt.add(den_r > 0)

                    rhs_expr_ratio = c_r * den_r
                    if cmp_op == ">=":
                        constraint = num_r >= rhs_expr_ratio
                    elif cmp_op == ">":
                        constraint = num_r > rhs_expr_ratio
                    elif cmp_op == "<=":
                        constraint = num_r <= rhs_expr_ratio
                    else:  # "<"
                        constraint = num_r < rhs_expr_ratio

                    if debug:
                        print(
                            f"[Z3 ratio] {expr_str} {cmp_op} {rhsn}  "
                            f"→ {num_str} {cmp_op} {c_val} * ({den_str})"
                        )
                    opt.add(constraint)
                    continue  # 這個 cmp_op 處理完，進下一個

            # 一般線性情境：照原本邏輯處理
            lhs_expr, lhs_real_flag, _ = _parse_linear_expr(expr_str)
            rhs_expr, rhs_real_flag, _ = _parse_side(rhs)

            # 根據旗標 + 實際 sort 決定要不要用 Real 語意
            lhs_is_real_sort = _is_real_sort(lhs_expr)
            rhs_is_real_sort = _is_real_sort(rhs_expr)
            use_real = (
                lhs_real_flag
                or rhs_real_flag
                or lhs_is_real_sort
                or rhs_is_real_sort
            )

            lhs_z3, rhs_z3 = lhs_expr, rhs_expr

            if use_real:
                # 需要 Real 語意：把 Int 轉成 Real，Real 保持不動
                if _is_int_sort(lhs_z3):
                    lhs_z3 = ToReal(lhs_z3)
                if _is_int_sort(rhs_z3):
                    rhs_z3 = ToReal(rhs_z3)
                constraint = cmp_map_real[cmp_op](lhs_z3, rhs_z3)
            else:
                # 純整數語意
                constraint = cmp_map_int[cmp_op](lhs_z3, rhs_z3)

            if debug:
                print(f"[Z3] {expr_str} {cmp_op} {rhs}  →  {lhs_z3} {cmp_op} {rhs_z3}")

            opt.add(constraint)