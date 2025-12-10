# securities_nl_parser.py
# NL → payload for 證券交易稅
import re, json
from typing import Dict, Any, List, Optional

# ----- 中文數字/單位 -----
_NUM = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*(億|萬|千|百)?')
_TARGET = re.compile(r'(?:目標|上限|預算|budget)\s*(?:稅額|tax)?\s*[:：]?\s*(\d[\d,]*(?:\.\d+)?)\s*(億|萬)?', re.I)
JSON_BLOCK = re.compile(r'\{.*\}', re.S)

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
    m = _TARGET.search(text)
    if not m:
        return None
    num, unit = m.group(1), m.group(2)
    return _scale(num, unit)

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

# ----- 類型與關鍵字 -----
# Canonical 稅目英文鍵 → 中文顯示名
CANONICAL_NAMES: Dict[str, str] = {
    "stock": "股票",
    "bond": "債券",
    "warrant": "權證",
    "warrant_delivery_stock": "權證履約（股票移轉）",
    "warrant_delivery_cash": "權證履約（現金結算）",
}

# 稅目 → 可被用來指稱該列的中文別名（你可以自由增刪）
SECURITIES_ITEM_ALIASES: Dict[str, List[str]] = {
    "stock": [
        "股票", "公司股票", "普通股", "現股", "上市股票", "上櫃股票", "股權", "股權證書"
    ],
    "bond": [
        "債券", "公司債", "企業債", "金融債", "可轉債", "有價證券-債"
    ],
    "warrant": [
        "權證", "認購權證", "認售權證", "買權證", "賣權證", "權證交易"
    ],
    "warrant_delivery_stock": [
        "權證履約 股票移轉", "履約 股票移轉", "履約 股票", "股票移轉履約"
    ],
    "warrant_delivery_cash": [
        "權證履約 現金結算", "現金結算", "履約 現金"
    ],
}

def _is_stock(seg: str) -> bool:
    return any(k in seg for k in SECURITIES_ITEM_ALIASES["stock"])

def _is_bond(seg: str) -> bool:
    return any(k in seg for k in SECURITIES_ITEM_ALIASES["bond"])

def _is_warrant(seg: str) -> bool:
    return any(k in seg for k in SECURITIES_ITEM_ALIASES["warrant"]) and '履約' not in seg

def _is_wd_stock(seg: str) -> bool:
    return any(k in seg for k in SECURITIES_ITEM_ALIASES["warrant_delivery_stock"])

def _is_wd_cash(seg: str) -> bool:
    return any(k in seg for k in SECURITIES_ITEM_ALIASES["warrant_delivery_cash"])

def _find_number(seg: str) -> Optional[float]:
    # 避免把「股數」抓成金額：先拿掉股數片段
    seg_ = re.sub(r'(股數|股)\s*\d[\d,]*(?:\.\d+)?', '', seg)
    m = _NUM.search(seg_)
    if not m:
        return None
    return _scale(m.group(1), m.group(2))

def _find_ep(seg: str) -> Optional[float]:
    m = re.search(r'(履約價|履約價格|執行價|執行價格|EP)\s*[:：]?\s*(\d[\d,]*(?:\.\d+)?)', seg)
    if not m:
        return None
    return float(m.group(2).replace(',', ''))

def _find_sc(seg: str) -> Optional[float]:
    m = re.search(r'(股數|股|張)\s*[:：]?\s*(\d[\d,]*(?:\.\d+)?)', seg)
    if not m:
        return None
    return float(m.group(2).replace(',', ''))

def nl_to_payload(text: str) -> Dict[str, Any]:
    """
    支援：
      - 「股票 500 萬；債券 2 億；權證 300 萬」
      - 「權證履約 股票移轉 履約價 50 股數 10000」
      - 「權證履約 現金結算 EP 60 股 8000」
      - 「目標稅額 300 萬」
      - 夾帶一段 JSON constraints {...}
    """
    if not text or not text.strip():
        return {"rows": [], "free_vars": [], "constraints": {}}

    rows: List[Dict[str, Any]] = []
    target_tax = _extract_target(text)
    constraints = _maybe_constraints(text)
    row_aliases: Dict[str, str] = {}  # ← 讓 ConstraintAgent 用

    for seg in _split(text):
        # 權證履約（股票移轉 / 現金結算）：需要 EP 與 SC
        if _is_wd_stock(seg):
            ep, sc = _find_ep(seg), _find_sc(seg)
            if ep is not None and sc is not None:
                idx = len(rows)
                tax_item = "warrant_delivery_stock"
                row = {"tax_item": tax_item, "ep": ep, "sc": sc, "main_name": CANONICAL_NAMES[tax_item]}
                rows.append(row)
                # 建 row_aliases
                for zh in SECURITIES_ITEM_ALIASES[tax_item]:
                    row_aliases[zh] = f"row{idx}"
                    row_aliases[zh.replace(" ", "")] = f"row{idx}"
            continue

        if _is_wd_cash(seg):
            ep, sc = _find_ep(seg), _find_sc(seg)
            if ep is not None and sc is not None:
                idx = len(rows)
                tax_item = "warrant_delivery_cash"
                row = {"tax_item": tax_item, "ep": ep, "sc": sc, "main_name": CANONICAL_NAMES[tax_item]}
                rows.append(row)
                for zh in SECURITIES_ITEM_ALIASES[tax_item]:
                    row_aliases[zh] = f"row{idx}"
                    row_aliases[zh.replace(" ", "")] = f"row{idx}"
            continue

        # 其餘（股票/債券/權證）只要交易金額 tp
        amt = _find_number(seg)
        if amt is None:
            continue

        if _is_stock(seg):
            idx = len(rows)
            tax_item = "stock"
            row = {"tax_item": tax_item, "tp": amt, "main_name": CANONICAL_NAMES[tax_item]}
            rows.append(row)
            for zh in SECURITIES_ITEM_ALIASES[tax_item]:
                row_aliases[zh] = f"row{idx}"
                row_aliases[zh.replace(" ", "")] = f"row{idx}"

        elif _is_bond(seg):
            idx = len(rows)
            tax_item = "bond"
            row = {"tax_item": tax_item, "tp": amt, "main_name": CANONICAL_NAMES[tax_item]}
            rows.append(row)
            for zh in SECURITIES_ITEM_ALIASES[tax_item]:
                row_aliases[zh] = f"row{idx}"
                row_aliases[zh.replace(" ", "")] = f"row{idx}"

        elif _is_warrant(seg):
            idx = len(rows)
            tax_item = "warrant"
            row = {"tax_item": tax_item, "tp": amt, "main_name": CANONICAL_NAMES[tax_item]}
            rows.append(row)
            for zh in SECURITIES_ITEM_ALIASES[tax_item]:
                row_aliases[zh] = f"row{idx}"
                row_aliases[zh.replace(" ", "")] = f"row{idx}"

    payload: Dict[str, Any] = {
        "rows": rows,
        "free_vars": [],                 # NL 預設不放行；交給 UI/第二階段決定
        "constraints": constraints,
    }
    if target_tax is not None:
        payload["target_tax"] = target_tax
    # ★ 關鍵：把列別名帶出去，交給 ConstraintAgent 用
    if row_aliases:
        payload["row_aliases"] = row_aliases
    return payload
