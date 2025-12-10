# agents/cargo_nl_parser.py
# -*- coding: utf-8 -*-
import re
from typing import Dict, Any, List, Optional, Tuple

# ============================================================
#   常數（對齊你的 CARGO_CATS / SUB_MAP）
# ============================================================

# —— 固定課稅（單位課稅） —— #
UNIT_TAX_MAP: Dict[str, Dict[str, float]] = {
    "水泥": {
        "白水泥：600元/公噸": 600,
        "卜特蘭I型水泥：320元/公噸": 320,
        "卜特蘭高爐水泥：196元/公噸": 196,
        "代水泥及其他：440元/公噸": 440,
    },
    "油氣類": {
        "汽油：6,830元/公秉": 6830,
        "柴油：3,990元/公秉": 3990,
        "煤油：4,250元/公秉": 4250,
        "航空燃油：610元/公秉": 610,
        "燃料油：110元/公秉": 110,
        "溶劑油：720元/公秉": 720,
        "液化石油氣：690元/公噸": 690,
    },
}

# —— 從價課稅（需要價格） —— #
RATE_MAP: Dict[str, Dict[str, float]] = {
    "橡膠輪胎": {
        "大客/大貨車 10%": 0.10,
        "其他橡膠輪胎 15%": 0.15,
        "內胎/實心輪胎等 0%": 0.0,
    },
    "飲料品": {
        "稀釋天然果蔬汁 8%": 0.08,
        "其他飲料 15%": 0.15,
        "天然果汁類 0%": 0.0,
    },
    "平板玻璃": {
        "一般平板玻璃 10%": 0.10,
        "導電/模具用玻璃 0%": 0.0,
    },
    "電器": {
        "冰箱 13%": 0.13,
        "彩色電視機 13%": 0.13,
        "中央空調 15%": 0.15,
        "非中央空調 20%": 0.20,
        "除濕機 15%": 0.15,
        "錄影機 13%": 0.13,
        "錄音機 10%": 0.10,
        "音響組合 10%": 0.10,
        "電烤箱 15%": 0.15,
    },
    "車輛": {
        "小客車 ≤2000cc 25%": 0.25,
        "小客車 >2000cc 30%": 0.30,
        "貨車/大客車等 15%": 0.15,
        "機車 17%": 0.17,
    },
}

# —— 常見自然語彙 → 主/子類別快速映射 —— #
# 固定課稅（單位課稅）直接指定 unit_tax
FIXED_TOKENS: Dict[str, Tuple[str, str, float]] = {
    "白水泥": ("水泥", "白水泥：600元/公噸", 600),
    "卜特蘭i": ("水泥", "卜特蘭I型水泥：320元/公噸", 320),
    "卜特蘭高爐": ("水泥", "卜特蘭高爐水泥：196元/公噸", 196),
    "代水泥": ("水泥", "代水泥及其他：440元/公噸", 440),
    "汽油": ("油氣類", "汽油：6,830元/公秉", 6830),
    "柴油": ("油氣類", "柴油：3,990元/公秉", 3990),
    "煤油": ("油氣類", "煤油：4,250元/公秉", 4250),
    "航空燃油": ("油氣類", "航空燃油：610元/公秉", 610),
    "燃料油": ("油氣類", "燃料油：110元/公秉", 110),
    "溶劑油": ("油氣類", "溶劑油：720元/公秉", 720),
    "液化石油氣": ("油氣類", "液化石油氣：690元/公噸", 690),
}

# 從價課稅（需價格）直接指定 rate
ADVAL_TOKENS: Dict[str, Tuple[str, str, float]] = {
    # 橡膠輪胎
    "大客車輪胎": ("橡膠輪胎", "大客/大貨車 10%", 0.10),
    "大貨車輪胎": ("橡膠輪胎", "大客/大貨車 10%", 0.10),
    "其他橡膠輪胎": ("橡膠輪胎", "其他橡膠輪胎 15%", 0.15),
    "內胎": ("橡膠輪胎", "內胎/實心輪胎等 0%", 0.0),
    "實心輪胎": ("橡膠輪胎", "內胎/實心輪胎等 0%", 0.0),

    # 飲料
    "稀釋果汁": ("飲料品", "稀釋天然果蔬汁 8%", 0.08),
    "稀釋天然果蔬汁": ("飲料品", "稀釋天然果蔬汁 8%", 0.08),
    "其他飲料": ("飲料品", "其他飲料 15%", 0.15),
    "天然果汁": ("飲料品", "天然果汁類 0%", 0.0),

    # 平板玻璃
    "平板玻璃": ("平板玻璃", "一般平板玻璃 10%", 0.10),
    "導電玻璃": ("平板玻璃", "導電/模具用玻璃 0%", 0.0),
    "模具用玻璃": ("平板玻璃", "導電/模具用玻璃 0%", 0.0),

    # 電器
    "冰箱": ("電器", "冰箱 13%", 0.13),
    "彩色電視機": ("電器", "彩色電視機 13%", 0.13),
    "中央空調": ("電器", "中央空調 15%", 0.15),
    "非中央空調": ("電器", "非中央空調 20%", 0.20),
    "除濕機": ("電器", "除濕機 15%", 0.15),
    "錄影機": ("電器", "錄影機 13%", 0.13),
    "錄音機": ("電器", "錄音機 10%", 0.10),
    "音響組合": ("電器", "音響組合 10%", 0.10),
    "電烤箱": ("電器", "電烤箱 15%", 0.15),

    # 車輛（若句中提及 cc，會由下方 cc 判斷覆寫）
    "貨車": ("車輛", "貨車/大客車等 15%", 0.15),
    "大客車": ("車輛", "貨車/大客車等 15%", 0.15),
    "機車": ("車輛", "機車 17%", 0.17),
    "小客車": ("車輛", "小客車 ≤2000cc 25%", 0.25),  # 預設 25%，遇 >2000cc 由 cc 規則改為 30%
}

# ============================================================
#   數字與單位處理（阿拉伯數字 / 中文數字、萬/億/兆）
# ============================================================
_UNIT_MULT = {"萬": 1e4, "億": 1e8, "兆": 1e12}

_CN_DIG = {"零":0,"一":1,"二":2,"兩":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}
_CN_UNIT = {"十":10,"百":100,"千":1000}

def _fw_to_hw(s: str) -> str:
    return s.translate(str.maketrans(
        "０１２３４５６７８９．，％",
        "0123456789.,%"
    ))

def _cn_to_float(token: str) -> float:
    """
    簡易中文數字轉 float，支援「四十」「一百二十」「三千五百」「一點二五」。
    （萬/億/兆 的倍率在外層處理）
    """
    s = token.strip()
    if not s:
        return 0.0

    # 小數：「一點二五」
    if "點" in s:
        int_part, frac_part = s.split("點", 1)
        iv = _cn_to_float(int_part) if int_part else 0.0
        fv = 0.0
        base = 0.1
        for ch in frac_part:
            if ch in _CN_DIG:
                d = _CN_DIG[ch]
            elif ch.isdigit():
                d = int(ch)
            else:
                d = 0
            fv += d * base
            base *= 0.1
        return iv + fv

    # 阿拉伯數字（含逗號）
    if re.fullmatch(r"[0-9][0-9,]*(?:\.[0-9]+)?", s):
        return float(s.replace(",", ""))

    # 中文整數
    total = 0
    num = 0
    for ch in s:
        if ch in _CN_DIG:
            num = num * 10 + _CN_DIG[ch]
        elif ch in _CN_UNIT:
            u = _CN_UNIT[ch]
            if num == 0:
                num = 1  # 「十」視為 1×10
            total += num * u
            num = 0
        else:
            # 其他字略過
            pass
    total += num
    return float(total or 0.0)

def _num_token_to_float(token: str) -> float:
    """同時支援阿拉伯數字(含逗號小數)與中文數字（全半形容錯）。"""
    t = _fw_to_hw(token.strip())
    if re.fullmatch(r"[0-9][0-9,]*(?:\.[0-9]+)?", t):
        return float(t.replace(",", ""))
    return _cn_to_float(t)

# ============================================================
#   分隔 & 抽取（數量 / 價格 / cc）
# ============================================================
DELIMS = r"[\n；;、]+"  # 項目分隔：換行、分號、頓號等

NUM_TOKEN = r"([0-9０-９][0-9０-９,，]*(?:\.[0-9０-９]+)?|[零一二三四五六七八九十百千兩點]+)"
UNITS = r"(單位|噸|頓|公噸|公秉|台|部|輛|個|件|瓶|片|條|支|公升|升)"

# 例：「60噸」「三千 公秉」「200 個」「五十 台」
QTY_RE = re.compile(rf"{NUM_TOKEN}\s*{UNITS}")

# 例：「單價 1200」「每台完稅價 15000」「每個價格 一萬五千」「完稅價 3,000」「售價 2.5萬」
PRICE_RE = re.compile(
    rf"(?:NT\$|NTD|\$)?\s*"
    rf"(?:單價|售價|價格|完稅價|完稅價格|每(?:台|部|輛|個|件|瓶|片|條|支|公升|升)(?:完稅價|價格|售價)?|每單位(?:完稅)?價格?)"
    rf"\s*{NUM_TOKEN}\s*(萬|億|兆)?"
)

# 例：車輛 cc
CC_RE = re.compile(r"(\d{3,5})\s*cc", re.I)

# 稅額上限/預算等金額提示（支援中文數字與『萬/億/兆』）
_MONEY_HINT_RE = re.compile(
    rf"(上限|預算|不超過|至多|目標|稅額上限)\s*[:：]?\s*{NUM_TOKEN}\s*(萬|億|兆)?"
)

# 自由變數指標（允許 row0.quantity / row1.assessed_price）
_FREE_VARS_RE = re.compile(r"row\d+\.(?:quantity|assessed_price)")

# ============================================================
#   基本工具
# ============================================================
def _split_items(text: str) -> List[str]:
    parts = re.split(DELIMS, text.strip())
    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"^[•\*\-．·\s]+", "", p)  # 去前導項符號
        cleaned.append(p)
    return cleaned

def _extract_qty(line: str) -> Optional[float]:
    m = QTY_RE.search(line)
    if not m:
        return None
    return _num_token_to_float(m.group(1))

def _extract_price(line: str) -> Optional[float]:
    m = PRICE_RE.search(line)
    if not m:
        return None
    base = _num_token_to_float(m.group(1))
    unit = m.group(2)
    if unit in _UNIT_MULT:
        base *= _UNIT_MULT[unit]
    return base

def _extract_cc(line: str) -> Optional[int]:
    m = CC_RE.search(line)
    if not m:
        return None
    try:
        return int(_fw_to_hw(m.group(1)))
    except Exception:
        return None

def _match_key_by_inclusion(line: str, table: Dict[str, Any]) -> Optional[str]:
    # 在 table.keys() 裡找第一個出現在 line 的 key
    for key in table.keys():
        if key and key in line:
            return key
    return None

def _find_in_nested_map(line: str) -> Optional[Tuple[str, str, float, str]]:
    """
    從 UNIT_TAX_MAP / RATE_MAP 的 key 做『主類別/子類別』直接匹配。
    傳回：(main, sub, value, mode)；mode ∈ {"fixed","adValorem"}
    """
    # 固定課稅
    for main, subs in UNIT_TAX_MAP.items():
        for sub, unit_tax in subs.items():
            if main in line and sub.split("：")[0] in line:
                return (main, sub, float(unit_tax), "fixed")
    # 從價課稅
    for main, subs in RATE_MAP.items():
        for sub, rate in subs.items():
            # 子類別有百分比，切掉數字前面詞再比對
            sub_key = sub.split(" ")[0]
            if main in line and sub_key in line:
                return (main, sub, float(rate), "adValorem")
    return None

def _try_vehicle_rate_by_cc(cc: int) -> Tuple[str, float]:
    """依 cc 推斷小客車稅率子類別與 rate"""
    if cc is None:
        return ("小客車 ≤2000cc 25%", 0.25)
    return ("小客車 ≤2000cc 25%", 0.25) if cc <= 2000 else ("小客車 >2000cc 30%", 0.30)

# ============================================================
#   主流程：自然語句 → canonical rows（與 cargo_tax.py 對齊）
# ============================================================
def nl_to_cargo_payload(text: str) -> Dict[str, Any]:
    """
    將自然語句解析為 cargo 稅 payload：
      {
        rows: [
          # fixed row
          { "mode":"fixed", "unit_tax":<number>, "quantity":<int>, "main_name":<str>, "sub_name":<str> },
          # ad valorem row
          { "mode":"adValorem", "rate":<float>, "assessed_price":<number>, "quantity":<int>, "main_name":<str>, "sub_name":<str> }
        ],
        free_vars: [...],
        constraints: {...},
        inferred_intent: "minimize" | "maximize_qty",
        target_tax?: <int>
      }
    """
    text = _fw_to_hw(text or "")
    rows: List[Dict[str, Any]] = []
    inferred_intent = "minimize"
    target_tax: Optional[int] = None

    # A) 自動偵測『最大化 + 稅額上限』
    m = _MONEY_HINT_RE.search(text)
    if m:
        inferred_intent = "maximize_qty"
        num_token = m.group(2)  # 數值（阿拉伯或中文）
        unit = m.group(3)       # 「萬/億/兆」或 None
        base = _num_token_to_float(num_token)
        mult = _UNIT_MULT.get(unit, 1.0)
        target_tax = int(base * mult)

    free_vars: List[str] = list(dict.fromkeys(_FREE_VARS_RE.findall(text)))

    # B) 逐段解析
    for seg in _split_items(text):
        # 1) 先抓「主/子類別」顯式並列的情況（如：水泥 白水泥 60噸；電器 冰箱 單價15000 x 200台）
        hit = _find_in_nested_map(seg)
        if hit:
            main, sub, val, mode = hit
            qty = int(_extract_qty(seg) or 0)
            if mode == "fixed":
                rows.append({
                    "mode": "fixed",
                    "unit_tax": float(val),
                    "quantity": qty,
                    "main_name": main,
                    "sub_name": sub,
                })
            else:
                price = _extract_price(seg) or 0.0
                rows.append({
                    "mode": "adValorem",
                    "rate": float(val),
                    "assessed_price": float(price),
                    "quantity": qty,
                    "main_name": main,
                    "sub_name": sub,
                })
            continue

        # 2) 固定課稅：快速別名
        tok = _match_key_by_inclusion(seg, FIXED_TOKENS)
        if tok:
            qty = int(_extract_qty(seg) or 0)
            main, sub, unit_tax = FIXED_TOKENS[tok]
            rows.append({
                "mode": "fixed",
                "unit_tax": float(unit_tax),
                "quantity": qty,
                "main_name": main,
                "sub_name": sub,
            })
            continue

        # 3) 從價課稅：快速別名（含車輛 cc 自動判斷）
        tok = _match_key_by_inclusion(seg, ADVAL_TOKENS)
        if tok:
            main, sub, rate = ADVAL_TOKENS[tok]
            # 車輛若提及 cc → 覆寫子類別/稅率
            if main == "車輛":
                cc = _extract_cc(seg)
                if "小客車" in seg:
                    sub, rate = _try_vehicle_rate_by_cc(cc)

            qty = int(_extract_qty(seg) or 0)
            price = _extract_price(seg) or 0.0
            rows.append({
                "mode": "adValorem",
                "rate": float(rate),
                "assessed_price": float(price),
                "quantity": qty,
                "main_name": main,
                "sub_name": sub,
            })
            continue

        # 4) 退而求其次：僅主類別命中 → 以第一個子類別當預設（固定課稅/從價課稅分支）
        main_guess = None
        for main in list(UNIT_TAX_MAP.keys()) + list(RATE_MAP.keys()):
            if main in seg:
                main_guess = main
                break
        if main_guess:
            qty = int(_extract_qty(seg) or 0)
            if main_guess in UNIT_TAX_MAP:
                # 固定課稅：取第一子類別預設
                sub, unit_tax = next(iter(UNIT_TAX_MAP[main_guess].items()))
                rows.append({
                    "mode": "fixed",
                    "unit_tax": float(unit_tax),
                    "quantity": qty,
                    "main_name": main_guess,
                    "sub_name": sub,
                })
            else:
                # 從價課稅：取第一子類別預設 + 價格
                sub, rate = next(iter(RATE_MAP[main_guess].items()))
                price = _extract_price(seg) or 0.0
                rows.append({
                    "mode": "adValorem",
                    "rate": float(rate),
                    "assessed_price": float(price),
                    "quantity": qty,
                    "main_name": main_guess,
                    "sub_name": sub,
                })
            continue

        # 5) 若都沒中，放棄本段（避免產出不完整列）
        # （需要更寬鬆的 NLP 可在此擴充）

    payload: Dict[str, Any] = {
        "rows": rows,
        "free_vars": free_vars,
        "constraints": {},
        "inferred_intent": inferred_intent,
    }
    if target_tax is not None:
        payload["target_tax"] = target_tax
    return payload

# 檔尾別名（框架既有呼叫）
def nl_to_payload(text: str):
    return nl_to_cargo_payload(text)
