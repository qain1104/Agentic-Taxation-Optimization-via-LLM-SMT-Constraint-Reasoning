# special_goods_nl_parser.py
# NL → payload for 特種貨物稅（高額消費貨物）→ values 介面
import re, json
from typing import Dict, Any, List, Optional, Tuple

# ─────────────────────────────────────────────────────────
# 數字與單位（億/萬/千/百）
_NUM = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*(億|萬|千|百)?')

# 目標稅額 / 上限 / 預算
_TARGET = re.compile(
    r'(?:目標|上限|預算|budget)\s*(?:稅額|tax)?\s*[:：]?\s*(\d[\d,]*(?:\.\d+)?)\s*(億|萬|千|百)?',
    re.I
)

# 取一段 JSON（常用於 constraints）
_JSON_BLOCK = re.compile(r'\{.*\}', re.S)

# 支援的 values-自由變數：car.price / car.quantity / …
_FREE_VARS_VALUES = re.compile(r'\b(?:car|yacht|aircraft|coral_ivory|furniture)\.(?:price|quantity)\b', re.I)

# 量詞（推斷數量）
_QTY_UNITS = r'(台|部|輛|艘|架|只|支|顆|件|套)'

# ─────────────────────────────────────────────────────────
# 品項對應（與後端欄位對齊）
#   car           ：（一）小客車：300 萬元以上
#   yacht         ：（二）遊艇：30.48 公尺
#   aircraft      ：（三）飛機／直升機：300 萬元以上
#   coral_ivory   ：（四）珊瑚／象牙等：50 萬元以上
#   furniture     ：（五）家具：50 萬元以上
_ITEM_ALIASES: List[Tuple[str, str]] = [
    # car
    (r'(?:小客車|汽車|轎車|超跑|名車|跑車|自用小客車)', 'car'),
    # yacht
    (r'(?:遊艇|艇)', 'yacht'),
    # aircraft
    (r'(?:飛機|直升機|私人飛機|航機|旋翼機)', 'aircraft'),
    # coral / ivory
    (r'(?:珊瑚|象牙|象牙製品|珊瑚製品|玳瑁|龜殼|稀有材質飾品)', 'coral_ivory'),
    # furniture
    (r'(?:家具|高級家具|傢俱)', 'furniture'),
]

_ALLOWED_CODES = {'car', 'yacht', 'aircraft', 'coral_ivory', 'furniture'}

# ─────────────────────────────────────────────────────────
# 基本工具

def _scale(num: str, unit: Optional[str]) -> float:
    n = float(num.replace(',', ''))
    if unit == '億':
        n *= 1e8
    elif unit == '萬':
        n *= 1e4
    elif unit == '千':
        n *= 1e3
    elif unit == '百':
        n *= 1e2
    return float(n)

def _split(text: str) -> List[str]:
    parts = re.split(r'[；;、，\n]+', text)
    return [p.strip() for p in parts if p.strip()]

def _extract_target(text: str) -> Optional[int]:
    m = _TARGET.search(text)
    if not m:
        return None
    return int(_scale(m.group(1), m.group(2)))

def _maybe_constraints(text: str):
    m = _JSON_BLOCK.search(text)
    if not m:
        return {}
    try:
        js = json.loads(m.group(0))
        return js if isinstance(js, dict) else {}
    except Exception:
        return {}

def _match_item(seg: str) -> Optional[str]:
    # 1) 中文同義詞 → 標準代碼
    for pat, code in _ITEM_ALIASES:
        if re.search(pat, seg):
            return code
    # 2) 允許直接輸入英文字代碼或在 "item/category/種類/品項" 欄位中填寫
    m = re.search(r'(?:item|category|種類|品項)\s*[:：]?\s*([A-Za-z_]+)', seg)
    if m:
        cand = m.group(1).strip().lower()
        if cand in _ALLOWED_CODES:
            return cand
    # 3) 如果整段只有代碼字樣
    tokens = re.findall(r'[A-Za-z_]+', seg)
    for t in tokens:
        if t.lower() in _ALLOWED_CODES:
            return t.lower()
    return None

def _extract_numbers(seg: str) -> List[Tuple[float, Optional[str]]]:
    return [(_scale(n, u), u) for n, u in _NUM.findall(seg)]

def _find_price(seg: str) -> Optional[int]:
    # 明確標示的單價
    m = re.search(r'(?:單價|價格|售價|金額|price)\s*[:：]?\s*(\d[\d,]*(?:\.\d+)?)\s*(億|萬|千|百)?', seg)
    if m:
        return int(_scale(m.group(1), m.group(2)))
    # 沒標示 → 取第一個數字當單價
    nums = _extract_numbers(seg)
    if nums:
        return int(nums[0][0])
    return None

def _find_quantity(seg: str) -> Optional[int]:
    # 明確標示數量
    m = re.search(r'(?:數量|qty|quantity)\s*[:：xX×*]?\s*(\d[\d,]*)', seg)
    if m:
        return int(m.group(1).replace(',', ''))
    # 「x 2」「×2」樣式
    m = re.search(r'[xX×*]\s*(\d[\d,]*)', seg)
    if m:
        return int(m.group(1).replace(',', ''))
    # 「2台/2部/2艘/2架…」
    m = re.search(r'(\d[\d,]*)\s*' + _QTY_UNITS, seg)
    if m:
        return int(m.group(1).replace(',', ''))
    # 若段落中有兩個以上數字，第二個當數量（常見：<單價> <數量>）
    nums = [int(float(n.replace(',', ''))) for n, _ in _NUM.findall(seg)]
    if len(nums) >= 2:
        return int(nums[1])
    return None

def _find_free_vars(text: str) -> List[str]:
    """抓 values 風格的自由變數，例如：car.price, yacht.quantity"""
    vars_found = set(m.group(0).lower() for m in _FREE_VARS_VALUES.finditer(text))
    # 也容忍使用者輸入以逗號/空白分隔的片段中出現合法 token
    tokens = re.findall(r'[A-Za-z_]+\.(?:price|quantity)', text)
    for t in tokens:
        t2 = t.strip().lower()
        if t2 in {f'{c}.{f}' for c in _ALLOWED_CODES for f in ('price', 'quantity')}:
            vars_found.add(t2)
    return sorted(vars_found)

# ─────────────────────────────────────────────────────────
# 主要轉換函式：回傳 values 介面
def nl_to_values_payload(text: str) -> Dict[str, Any]:
    """
    輸入（例）：
      - 小客車 800萬 x 2；遊艇 1.2億 x 1；目標稅額 500萬
      - 放行 car.quantity,yacht.price
      - 夾帶 constraints JSON：{ "car.price": {">=": 5000000} }

    輸出（values 介面）：
      {
        "values": {
          "car.price": 8000000, "car.quantity": 2,
          "yacht.price": 120000000, "yacht.quantity": 1,
          "aircraft.price": 0, "aircraft.quantity": 0,
          "coral_ivory.price": 0, "coral_ivory.quantity": 0,
          "furniture.price": 0, "furniture.quantity": 0
        },
        "free_vars": ["car.quantity", "yacht.price"],
        "constraints": {...},
        "budget_tax": 5000000,
        "inferred_intent": "maximize_qty"
      }
    """
    # 預設 0 的值（未提及也要給 0，方便後端直接吃）
    values: Dict[str, int] = {}
    for code in _ALLOWED_CODES:
        values[f'{code}.price'] = 0
        values[f'{code}.quantity'] = 0

    if not text or not text.strip():
        return {"values": values, "free_vars": [], "constraints": {}}

    constraints = _maybe_constraints(text)
    budget_tax = _extract_target(text)
    free_vars = _find_free_vars(text)

    # 逐段解析品項／單價／數量
    for seg in _split(text):
        code = _match_item(seg)
        if not code:
            continue

        price = _find_price(seg)
        qty   = _find_quantity(seg)

        # 缺漏則用 0；若同一品項被多次提及，數量相加、價格以最後一次非 None 覆蓋
        if qty is not None:
            values[f'{code}.quantity'] = int(values.get(f'{code}.quantity', 0)) + int(qty)
        if price is not None:
            values[f'{code}.price'] = int(price)

    payload: Dict[str, Any] = {
        "values": values,
        "free_vars": free_vars,
        "constraints": constraints,
    }
    if budget_tax is not None:
        payload["budget_tax"] = budget_tax
        payload["inferred_intent"] = "maximize_qty"

    return payload

# 保留舊名字以避免外部尚未改 SPEC 的情況（向下相容）。
# 若外部仍呼叫 nl_to_payload，就轉呼 nl_to_values_payload。
def nl_to_payload(text: str) -> Dict[str, Any]:
    return nl_to_values_payload(text)
