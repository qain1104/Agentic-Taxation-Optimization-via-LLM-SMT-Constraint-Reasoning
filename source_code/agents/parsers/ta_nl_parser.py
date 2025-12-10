# ta_nl_parser.py
# 以自然語言解析菸酒稅輸入，輸出 payload:
# { rows: [{category, subcategory, quantity, (alcohol_content?), main_name, sub_name}],
#   free_vars: [], constraints: {}, target_tax? }
import re
from typing import List, Dict, Any, Optional

# ---- 子類別中文標籤（與前端常數一致）----
TOBACCO_LABELS = {
    1: "紙菸（新制）",
    2: "菸絲（新制）",
    3: "雪茄（新制）",
    4: "其他菸品（新制）",
    5: "紙菸（舊制）",
    6: "菸絲（舊制）",
    7: "雪茄（舊制）",
    8: "其他菸品（舊制）",
}
ALCOHOL_LABELS = {
    1: "釀造 – 啤酒",
    2: "釀造 – 其他",
    3: "蒸餾酒類",
    4: "再製 >20%",
    5: "再製 ≤20%",
    6: "料理酒",
    7: "料理酒(舊)",
    8: "其他酒類",
    9: "酒精",
    10: "酒精(舊)",
}

# 需要 ABV 的酒類子別（保留以供參考）
ALCOHOL_ABV_REQUIRED = {2, 3, 5, 8}

# ---- 關鍵字對應（盡量寬鬆）----
TOBACCO_KEYS = {
    1: ["紙菸", "紙煙", "香菸", "香煙", "新制紙菸", "紙菸（新制）", "紙菸(新制)"],
    2: ["菸絲", "煙絲", "新制菸絲", "菸絲（新制）", "菸絲(新制)"],
    3: ["雪茄", "新制雪茄", "雪茄（新制）", "雪茄(新制)"],
    4: ["其他菸", "其他菸品", "新制其他", "其他菸品（新制）", "其他菸品(新制)"],
    5: ["紙菸（舊制）", "紙菸(舊制)", "舊制紙菸"],
    6: ["菸絲（舊制）", "菸絲(舊制)", "舊制菸絲"],
    7: ["雪茄（舊制）", "雪茄(舊制)", "舊制雪茄"],
    8: ["其他菸品（舊制）", "其他菸(舊制)", "舊制其他菸"],
}
ALCOHOL_KEYS = {
    1: ["啤酒", "釀造→啤酒", "釀造 啤酒"],
    2: ["釀造 其他", "釀造→其他", "發酵酒", "葡萄酒", "米酒(發酵)", "發酵類"],
    3: ["蒸餾", "蒸餾酒", "蒸餾酒類", "烈酒", "威士忌", "白蘭地", "高粱"],
    4: ["再製 >20", "再製>20", "再製大於20", "再製 高度"],
    5: ["再製 ≤20", "再製<=20", "再製 小於等於20", "低度再製"],
    6: ["料理酒"],
    7: ["料理酒(舊)", "舊制料理酒", "舊料理酒"],
    8: ["其他酒", "其他酒類"],
    9: ["酒精", "乙醇"],
    10: ["酒精(舊)", "舊制酒精", "舊酒精"],
}

# ---- 正則 ----
_NUM_WITH_UNIT = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*(億|萬|千|百)?')
_PERCENT = re.compile(r'(\d+(?:\.\d+)?)\s*%')
_ABV = re.compile(r'(?:ABV|酒精度|酒度|濃度)\s*[:：]?\s*(\d+(?:\.\d+)?)\s*%?', re.I)
_TARGET = re.compile(r'(?:目標|上限|預算|budget)\s*(?:稅額|tax|:|：)?\s*(\d[\d,]*(?:\.\d+)?)\s*(億|萬)?', re.I)
_LEADING_INDEX = re.compile(r'^\s*(?:[（(]?\d{1,2}[)）\.、]|[（(]?\d{1,2}\s*\)\s*)\s*')  # 1. / 1、/ (1) / 1) / （1）等

# 常見數量單位（僅作為「鄰近提示」，不會影響數值縮放）
_QTY_UNITS = r"(條|包|支|公升|升|L|毫升|ml|瓶|罐|箱|打|桶|公斤|公克|噸|公噸|件|個)\b"

def _norm(s: str) -> str:
    return (s or "").strip()

def _scale_unit(num_str: str, unit: Optional[str]) -> int:
    n = float(num_str.replace(',', ''))
    if unit == '億':
        n *= 1e8
    elif unit == '萬':
        n *= 1e4
    elif unit == '千':
        n *= 1e3
    elif unit == '百':
        n *= 1e2
    return int(round(n))

def _extract_target_tax(text: str) -> Optional[int]:
    m = _TARGET.search(text or "")
    if not m:
        return None
    num, unit = m.group(1), m.group(2)
    return _scale_unit(num, unit)

def _extract_abv(seg: str) -> Optional[float]:
    s = seg or ""
    # 先找 ABV 關鍵詞
    m = _ABV.search(s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # 句子同時有「再製」與「%」通常是分類描述，不當 ABV
    if "再製" in s:
        return None
    # 退而求其次：純百分比
    m2 = _PERCENT.search(s)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            return None
    return None

def _extract_amount(seg: str) -> Optional[int]:
    """
    從片段抓「數量」。
    修復點：
      1) 忽略句首列表序號（例如 "1. 紙菸..."），避免把 1 當數量
      2) 移除百分比（避免把 45% 或 5% 當數量）
      3) 優先找在『數量/qty/quantity』或『常見數量單位』附近的數字
      4) 否則取片段裡最大的數字（含萬/億縮放）
    """
    if not seg:
        return None

    # 1) 去列表序號
    s = _LEADING_INDEX.sub('', seg)

    # 2) 去百分比
    s_no_pct = _PERCENT.sub('', s)

    # 3a) 先找『數量/qty/quantity』附近的數字（含萬/億）
    m = re.search(r'(?:數量|qty|quantity)\D*' + _NUM_WITH_UNIT.pattern, s_no_pct, re.I)
    if m:
        num, unit = m.group(1), (m.group(2) if m.lastindex and m.lastindex >= 2 else None)
        return _scale_unit(num, unit)

    # 3b) 再找『數字 +（萬|億 可選）+ 常見數量單位』
    m2 = re.search(r'(\d[\d,]*(?:\.\d+)?)\s*(萬|億)?\s*' + _QTY_UNITS, s_no_pct, re.I)
    if m2:
        return _scale_unit(m2.group(1), m2.group(2))

    # 4) 否則取片段中最大的數字（含萬/億）
    best = None
    for mm in _NUM_WITH_UNIT.finditer(s_no_pct):
        val = _scale_unit(mm.group(1), mm.group(2) if mm.lastindex and mm.lastindex >= 2 else None)
        if val is not None and (best is None or val > best):
            best = val
    return best

def _guess_cat(seg: str) -> Optional[str]:
    s = seg or ""
    if any(k in s for k in ["菸", "紙菸", "雪茄", "菸絲", "香菸", "香煙"]):
        return "菸"
    if any(k in s for k in ["酒", "啤酒", "蒸餾", "再製", "料理酒", "酒精"]):
        return "酒"
    return None

def _match_sub_from_keywords(cat: str, seg: str) -> Optional[int]:
    s = seg or ""
    keys = TOBACCO_KEYS if cat == "菸" else ALCOHOL_KEYS
    for sub, words in keys.items():
        for w in words:
            if w and w in s:
                return sub
    # 允許句首「1. / 1、 / (1) / 1) / （1）」等數字索引標記子類別
    m = re.match(r'^\s*[（(]?(\d{1,2})[)）\.、]\s*', s)
    if m:
        idx = int(m.group(1))
        if cat == "菸" and 1 <= idx <= 8:
            return idx
        if cat == "酒" and 1 <= idx <= 10:
            return idx
    return None

def _label(cat: str, sub: int) -> str:
    return (TOBACCO_LABELS if cat == "菸" else ALCOHOL_LABELS).get(sub, f"子類別 {sub}")

def _build_row(cat: str, sub: int, qty: int, abv: Optional[float]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "category": cat,
        "subcategory": sub,
        "quantity": qty,
        "main_name": "菸品" if cat == "菸" else "酒類",
        "sub_name": _label(cat, sub),
    }
    # 只要使用者有提供 ABV 就保留（即使該子類不需要，計算端可自行忽略）
    if cat == "酒" and abv is not None:
        row["alcohol_content"] = float(abv)
    return row

def _split_segments(text: str) -> List[str]:
    # 以常見分隔符切句：； ; 、 ， \n
    parts = re.split(r'[；;、，\n]+', text or "")
    return [p.strip() for p in parts if p and p.strip()]

def nl_to_ta_payload(text: str) -> Dict[str, Any]:
    """
    將自然語言轉為 TA payload。
    支援示例：
      - "紙菸 3萬條；啤酒 500 公升；再製>20% 1000 公升 45%"
      - "蒸餾酒 2,500 L，ABV 40%"
      - "目標稅額 500萬"
    """
    if not text or not text.strip():
        return {"rows": [], "free_vars": [], "constraints": {}}

    target_tax = _extract_target_tax(text)
    segments = _split_segments(text)

    rows: List[Dict[str, Any]] = []
    for seg in segments:
        seg_norm = _norm(seg)
        if not seg_norm:
            continue

        cat = _guess_cat(seg_norm)
        if not cat:
            # 若句子沒明說，嘗試從關鍵字推斷；找不到就略過
            if any(k in seg_norm for k in ("啤酒", "蒸餾", "再製", "料理酒", "酒精")):
                cat = "酒"
            elif any(k in seg_norm for k in ("紙菸", "雪茄", "菸絲", "香菸", "香煙")):
                cat = "菸"
            else:
                continue

        sub = _match_sub_from_keywords(cat, seg_norm)
        if not sub:
            # 類別已知但子別未知，給合理預設：菸→1(紙菸新制)，酒→1(啤酒)
            sub = 1

        abv = _extract_abv(seg_norm)
        qty = _extract_amount(seg_norm)
        if qty is None:
            # 沒抓到數量就跳過此段
            continue

        rows.append(_build_row(cat, sub, qty, abv))

    payload: Dict[str, Any] = {
        "rows": rows,
        "free_vars": [],       # 預設不從 NL 放自由變數；留給 UI/使用者指定
        "constraints": {},     # 預設無文字 constraints；由 UI 的 JSON 對應進來
    }
    if target_tax is not None:
        payload["target_tax"] = target_tax

    return payload

# 與 CallerAgent 的 nl_parser 通用入口相容
def nl_to_payload(text: str) -> Dict[str, Any]:
    return nl_to_ta_payload(text)
