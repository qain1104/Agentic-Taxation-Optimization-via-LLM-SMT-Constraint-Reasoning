# futures_nl_parser.py
# NL → payload for 期貨交易稅
import re, json
from typing import Dict, Any, List, Optional

# ------------------ 基本 regex ------------------
_NUM = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*(億|萬|千|百)?')
TARGET = re.compile(r'(?:目標|上限|預算|budget)\s*(?:稅額|tax)?\s*[:：]?\s*(\d[\d,]*(?:\.\d+)?)\s*(億|萬)?', re.I)
JSON_BLOCK = re.compile(r'\{.*\}', re.S)

# ------------------ 顯示名與別名 ------------------
CANONICAL_NAMES: Dict[str, str] = {
    "stock_index": "股價期貨",
    "interest_rate_30": "利率期貨（30天CP）",
    "interest_rate_10": "利率期貨（10年期）",
    "option": "選擇權",
    "gold": "黃金期貨",
}

# 常見中文別名（可自行擴充）
FUTURES_ITEM_ALIASES: Dict[str, List[str]] = {
    "stock_index": ["股票", "股價期貨", "股指期貨", "股價", "股指"],
    "interest_rate_30": ["利率期貨30", "30天", "30 天", "30天CP", "CP30天", "短天期利率期貨", "CP"],
    "interest_rate_10": ["利率期貨10", "10年", "10 年", "10年期公債", "十年債", "十年期公債", "10年期"],
    "option": ["選擇權", "期貨選擇權", "options"],
    "gold": ["黃金期貨", "黃金"],
}

# ------------------ 工具函式 ------------------
def _scale(v: str, unit: Optional[str]) -> float:
    n = float(v.replace(',', ''))
    if unit == '億':
        n *= 1e8
    elif unit == '萬':
        n *= 1e4
    elif unit == '千':
        n *= 1e3
    elif unit == '百':
        n *= 1e2
    return float(n)

def _extract_target(text: str) -> Optional[float]:
    m = TARGET.search(text)
    if not m:
        return None
    return _scale(m.group(1), m.group(2))

def _maybe_constraints(text: str):
    m = JSON_BLOCK.search(text)
    if not m:
        return {}
    try:
        cons = json.loads(m.group(0))
        return cons if isinstance(cons, dict) else {}
    except Exception:
        return {}

def _split(text: str) -> List[str]:
    parts = re.split(r'[；;、，\n]+', text)
    return [p.strip() for p in parts if p.strip()]

def _amount(seg: str) -> Optional[float]:
    """
    從片段挑『最像金額』的數字：
      1) 先移除時間數（如 30天 / 10年 / 3月 / 15日）
      2) 還有數字時，優先取『有單位（億/萬/千/百）』的最後一個；否則取最後一個數字
    """
    # 去掉明顯非金額的時間用數字
    seg2 = re.sub(r'\d[\d,]*(?:\.\d+)?\s*(?:天|日|年|月)\b', '', seg)
    matches = list(_NUM.finditer(seg2))
    if not matches:
        return None
    with_unit = [m for m in matches if m.group(2)]
    m = with_unit[-1] if with_unit else matches[-1]
    return _scale(m.group(1), m.group(2))

# ------------------ 類別判斷 ------------------
def _is_stock_index(seg: str) -> bool:
    return any(k in seg for k in FUTURES_ITEM_ALIASES["stock_index"])

def _is_interest_30(seg: str) -> bool:
    s = seg.replace(" ", "")
    return (
        any(k.replace(" ", "") in s for k in FUTURES_ITEM_ALIASES["interest_rate_30"]) or
        ("利率期貨" in seg and ("30天" in s or "CP" in s))
    )

def _is_interest_10(seg: str) -> bool:
    s = seg.replace(" ", "")
    return (
        any(k.replace(" ", "") in s for k in FUTURES_ITEM_ALIASES["interest_rate_10"]) or
        ("利率期貨" in seg and ("10年" in s or "十年" in s))
    )

def _is_option(seg: str) -> bool:
    return any(k in seg for k in FUTURES_ITEM_ALIASES["option"])

def _is_gold(seg: str) -> bool:
    return any(k in seg for k in FUTURES_ITEM_ALIASES["gold"])

# ------------------ 主要轉換 ------------------
def nl_to_payload(text: str) -> Dict[str, Any]:
    """
    支援：
      - 「股價期貨 10 億；利率期貨 30天CP 5億；選擇權 2000萬；黃金期貨 3億」
      - 「目標稅額 1000萬」
      - 夾帶一段 JSON constraints {...}
    """
    if not text or not text.strip():
        return {"rows": [], "free_vars": [], "constraints": {}}

    rows: List[Dict[str, Any]] = []
    row_aliases: Dict[str, str] = {}  # 供 ConstraintAgent 以中文別名對應 row{i}
    target_tax = _extract_target(text)
    constraints = _maybe_constraints(text)

    for seg in _split(text):
        amt = _amount(seg)
        if amt is None:
            continue

        # 依判斷建立 row，並且把別名映射到該 row 索引
        def _append_row(tax_item: str, row_obj: Dict[str, Any]):
            idx = len(rows)
            row_obj["tax_item"] = tax_item
            row_obj.setdefault("main_name", CANONICAL_NAMES.get(tax_item, tax_item))
            rows.append(row_obj)
            # 建 row_aliases：canonical + 所有中文別名（去空白版本也收錄）
            for zh in [CANONICAL_NAMES.get(tax_item, tax_item)] + FUTURES_ITEM_ALIASES.get(tax_item, []):
                if not isinstance(zh, str) or not zh.strip():
                    continue
                row_aliases[zh] = f"row{idx}"
                row_aliases[zh.replace(" ", "")] = f"row{idx}"

        if _is_stock_index(seg):
            _append_row("stock_index", {"ca": amt})
            continue

        if _is_interest_30(seg):
            _append_row("interest_rate_30", {"ca": amt})
            continue

        if _is_interest_10(seg):
            _append_row("interest_rate_10", {"ca": amt})
            continue

        if _is_option(seg):
            # option 需要 pa；與前端一致，若未填 pa，pa = ca
            _append_row("option", {"ca": amt, "pa": amt})
            continue

        if _is_gold(seg):
            _append_row("gold", {"ca": amt})
            continue

    payload: Dict[str, Any] = {
        "rows": rows,
        "free_vars": [],                 # NL 預設不放行；交給 UI/ConstraintAgent 決定
        "constraints": constraints,
    }
    if target_tax is not None:
        payload["target_tax"] = target_tax
        # 小提示：有上限多半是「在上限內最大化」
        payload["inferred_intent"] = "maximize_qty"
    if row_aliases:
        payload["row_aliases"] = row_aliases

    return payload
