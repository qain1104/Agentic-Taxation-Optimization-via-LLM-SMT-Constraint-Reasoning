# vat_nl_parser.py
# NL → payload for 加值型營業稅 (VAT)
import re, json
from typing import Dict, Any, List, Optional, Tuple

# 數字 + 單位
_NUM = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*(億|萬)?')
# 銷項 / 進項 捕捉
_OUT = re.compile(r'(?:銷項(?:稅額)?|output|out)\s*[:：]?\s*(\d[\d,]*(?:\.\d+)?)\s*(億|萬)?', re.I)
_INN = re.compile(r'(?:進項(?:稅額)?|input|in)\s*[:：]?\s*(\d[\d,]*(?:\.\d+)?)\s*(億|萬)?', re.I)

# 自由變數語彙
_FREE_FLAG_IN  = re.compile(r'(自由進項|進項自由)', re.I)
_FREE_FLAG_OUT = re.compile(r'(自由銷項|銷項自由)', re.I)
_FREE_VARS_ANY = re.compile(r'row\d+\.(?:input_tax_val|output_tax_val)')

# 夾帶 JSON
_JSON_BLOCK = re.compile(r'\{.*\}', re.S)


def _scale(num: str, unit: Optional[str]) -> float:
    n = float(num.replace(',', ''))
    if unit == '億':
        n *= 1e8
    elif unit == '萬':
        n *= 1e4
    return float(n)


def _split(text: str) -> List[str]:
    parts = re.split(r'[；;、\n]+', text)
    return [p.strip() for p in parts if p.strip()]


def _maybe_constraints(text: str) -> Dict:
    m = _JSON_BLOCK.search(text)
    if not m:
        return {}
    try:
        js = json.loads(m.group(0))
        return js if isinstance(js, dict) else {}
    except Exception:
        return {}


def _extract_numbers(seg: str) -> List[Tuple[float, Optional[str]]]:
    return [(_scale(n, u), u) for n, u in _NUM.findall(seg)]


def _find_pair(seg: str) -> Tuple[Optional[int], Optional[int]]:
    """
    依優先序抓 (out, in)；若沒標籤、但有兩個數字，則第一個視為銷項、第二個視為進項。
    """
    out_v = inn_v = None
    m = _OUT.search(seg)
    if m:
        out_v = int(_scale(m.group(1), m.group(2)))
    m = _INN.search(seg)
    if m:
        inn_v = int(_scale(m.group(1), m.group(2)))

    if out_v is None or inn_v is None:
        nums = [int(float(n.replace(',', ''))) for n, _ in _NUM.findall(seg)]
        if len(nums) >= 2:
            if out_v is None: out_v = nums[0]
            if inn_v is None: inn_v = nums[1]

    return out_v, inn_v


def _free_vars_for_row(seg: str, row_idx: int) -> List[str]:
    flags = []
    if _FREE_FLAG_IN.search(seg):
        flags.append(f"row{row_idx}.input_tax_val")
    if _FREE_FLAG_OUT.search(seg):
        flags.append(f"row{row_idx}.output_tax_val")
    return flags


def _collect_explicit_free_vars(text: str) -> List[str]:
    # 支援「放行 row0.input_tax_val,row1.output_tax_val」之類的明確宣告
    return list(set(_FREE_VARS_ANY.findall(text)))


def nl_to_payload(text: str) -> Dict[str, Any]:
    """
    輸入例：
      - 銷項 120 萬、進項 80 萬；銷項 30萬 進項 5萬（自由進項）
      - out 2,000,000 in 1,200,000；放行 row0.input_tax_val
      - 夾帶 constraints：{ "row1.output_tax_val": {"<=": 500000} }
    輸出：
      { rows:[{output_tax_val, input_tax_val},...], free_vars:[...], constraints:{...} }
    """
    if not text or not text.strip():
        return {"rows": [], "free_vars": [], "constraints": {}}

    rows: List[Dict[str, int]] = []
    free_vars: List[str] = []
    constraints = _maybe_constraints(text)

    row_idx = 0
    for seg in _split(text):
        out_v, inn_v = _find_pair(seg)
        if out_v is None and inn_v is None:
            continue
        rows.append({
            "output_tax_val": int(out_v or 0),
            "input_tax_val": int(inn_v or 0),
        })
        free_vars.extend(_free_vars_for_row(seg, row_idx))
        row_idx += 1

    # 併入明確 free_vars 宣告
    free_vars.extend(_collect_explicit_free_vars(text))
    free_vars = sorted(set(free_vars))

    return {
        "rows": rows,
        "free_vars": free_vars,
        "constraints": constraints,
    }
