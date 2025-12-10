# tax_calculators/util.py

from z3 import IntNumRef, RatNumRef

def _to_number(val):
    """
    將 Z3 回傳的 IntNumRef、RatNumRef 或其他 ArithRef 轉成 Python 浮點數或整數。
    """
    # 整數
    if isinstance(val, IntNumRef):
        return val.as_long()
    # 有理數
    if isinstance(val, RatNumRef):
        s = str(val)           # 可能是 "12345/1000" 或 "12.345"
        if '/' in s:
            num, den = s.split('/')
            return float(num) / float(den)
        return float(s)
    # 其餘 ArithRef 也用字串解析
    s = str(val)
    if '/' in s:
        num, den = s.split('/')
        return float(num) / float(den)
    return float(s)
