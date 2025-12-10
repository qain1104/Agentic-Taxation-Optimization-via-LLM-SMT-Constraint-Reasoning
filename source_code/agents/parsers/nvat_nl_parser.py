# nvat_nl_parser.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import re
import json
from typing import Any, Dict, List

# 類別代碼對應（1~8）與常見別名
CATEGORY_ALIASES: Dict[int, List[str]] = {
    1: ["小規模營業人", "小規模"],
    2: ["再保費收入", "再保費", "再保險保費", "再保"],
    3: ["金融等業專屬本業", "金融專屬本業", "證券業專屬本業", "票券專屬本業", "金融專屬"],
    4: ["銀行/保險本業", "銀行本業", "保險本業", "銀行/保險", "銀行 保險 本業"],
    5: ["金融等非專屬本業", "金融非專屬本業", "非專屬本業", "金融非專屬"],
    6: ["夜總會", "夜店"],
    7: ["酒家等", "酒家", "酒店"],
    8: ["農產品批發市場", "農批", "農批市場", "農產品 批發 市場"],
}

# 關鍵字→代碼
KW2CODE: Dict[str, int] = {}
for code, kws in CATEGORY_ALIASES.items():
    for kw in kws:
        KW2CODE[kw] = code

# 金額樣式
AMOUNT_RE = re.compile(
    r'(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>億|萬|千|百|元|塊|NTD|NT\$|N?T?\$)?',
    re.I
)

# 別名
BUDGET_ALIASES = ("稅額上限", "目標稅額", "預算上限", "上限")
FREE_VARS_ALIASES = ("自由變數", "放行變數")

def _normalize_amount(num_str: str, unit: str) -> int | None:
    try:
        val = float(num_str.replace(",", ""))
    except Exception:
        return None
    mult = 1
    u = (unit or "").strip()
    if u == "億":
        mult = 100_000_000
    elif u == "萬":
        mult = 10_000
    elif u == "千":
        mult = 1_000
    elif u == "百":
        mult = 100
    return int(round(val * mult))

def _find_category(seg: str) -> int | None:
    # 支援 category=6
    m = re.search(r'category\s*=\s*(\d+)', seg, re.I)
    if m:
        c = int(m.group(1))
        return c if 1 <= c <= 8 else None
    # 中文關鍵字
    for kw, code in KW2CODE.items():
        if kw in seg:
            return code
    return None

def _extract_amount(seg: str) -> int | None:
    # 優先找『銷售/營業額/金額』附近數字
    pref = re.search(r'(?:銷售(?:額)?|營業(?:額)?|金額)\D*' + AMOUNT_RE.pattern, seg)
    if pref:
        gd = pref.groupdict()
        return _normalize_amount(gd.get("num", ""), gd.get("unit") or "")
    # 否則取片段最大數字
    best = None
    for m in AMOUNT_RE.finditer(seg):
        gd = m.groupdict()
        val = _normalize_amount(gd.get("num", ""), gd.get("unit") or "")
        if val is not None and (best is None or val > best):
            best = val
    return best

def _extract_budget(text: str) -> int | None:
    # 英文鍵 budget_tax / budget tax / budgettax
    m = re.search(r'budget[_\-\s]?tax\s*=\s*(\d+(?:[.,]\d+)?)', text, re.I)
    if m:
        return _normalize_amount(m.group(1), "")
    m = re.search(r'budget\s*tax\s*[:=]\s*(\d+(?:[.,]\d+)?)', text, re.I)
    if m:
        return _normalize_amount(m.group(1), "")
    m = re.search(r'budgettax\s*[:=]\s*(\d+(?:[.,]\d+)?)', text, re.I)
    if m:
        return _normalize_amount(m.group(1), "")
    # 中文別名
    m = re.search(r'(?:' + "|".join(map(re.escape, BUDGET_ALIASES)) + r')\D*' + AMOUNT_RE.pattern, text)
    if m:
        gd = m.groupdict()
        return _normalize_amount(gd.get("num", ""), gd.get("unit") or "")
    return None

def _extract_free_vars(text: str) -> List[str]:
    """
    自由變數/放行變數：cat1, cat3 或 1,3,6 → ['cat1','cat3','cat6']
    """
    pat = re.compile(r'(?:' + "|".join(map(re.escape, FREE_VARS_ALIASES)) + r')\s*[:：]\s*([a-z0-9,\s、]+)', re.I)
    m = pat.search(text)
    if not m:
        return []
    raw = m.group(1)
    toks = re.split(r'[,\s、]+', raw.strip())
    out: List[str] = []
    for t in toks:
        if not t:
            continue
        t = t.lower()
        if t.startswith("cat"):
            try:
                n = int(t[3:])
            except Exception:
                continue
            if 1 <= n <= 8:
                out.append(f"cat{n}")
        elif t.isdigit():
            n = int(t)
            if 1 <= n <= 8:
                out.append(f"cat{n}")
    # 去重保序
    seen = set(); dedup = []
    for x in out:
        if x not in seen:
            seen.add(x); dedup.append(x)
    return dedup

def _extract_op(text: str) -> str | None:
    t = text.lower()
    if any(k in t for k in ("最大化", "maximize", "最大")):
        return "maximize"
    if any(k in t for k in ("最小化", "minimize", "最小")):
        return "minimize"
    return None

def _maybe_parse_json(text: str) -> Dict[str, Any] | None:
    """支援直接 JSON 指定 cat1..cat8/budget_tax/free_vars/op。"""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None

def _merge_user_params_from_json(d: Dict[str, Any]) -> Dict[str, int]:
    user_params: Dict[str, int] = {}
    # 允許 cat1..cat8；也容忍 category1..category8
    for i in range(1, 9):
        for key in (f"cat{i}", f"category{i}"):
            if key in d:
                try:
                    user_params[f"cat{i}"] = int(d[key])
                except Exception:
                    # 嘗試『800萬』字樣
                    m = AMOUNT_RE.search(str(d[key]))
                    if m:
                        gd = m.groupdict()
                        v = _normalize_amount(gd.get("num", ""), gd.get("unit") or "")
                        if v is not None:
                            user_params[f"cat{i}"] = v
                break
    # budget_tax / budgettax
    for k in ("budget_tax", "budget tax", "budgettax"):
        if k in d:
            try:
                user_params["budget_tax"] = int(d[k])
            except Exception:
                m = AMOUNT_RE.search(str(d[k]))
                if m:
                    gd = m.groupdict()
                    v = _normalize_amount(gd.get("num", ""), gd.get("unit") or "")
                    if v is not None:
                        user_params["budget_tax"] = v
            break
    return user_params

def nl_to_payload(text: str) -> Dict[str, Any]:
    """
    解析自然語言/JSON，直接回傳 SPEC 需要的 tool 格式：
    {
      "tool_name": "nvat_tax",
      "payload": {
        "op": "minimize" | "maximize" | None,
        "user_params": { "cat1":..., "cat2":..., ..., "budget_tax": ... },
        "free_vars": [...],
        "constraints": {}
      }
    }
    """
    text = (text or "").strip()

    # 1) JSON 直貼優先
    jp = _maybe_parse_json(text)
    user_params: Dict[str, int] = {}
    free_vars: List[str] = []
    op: str | None = None

    if jp is not None:
        user_params.update(_merge_user_params_from_json(jp))
        # free_vars
        fv = jp.get("free_vars")
        if isinstance(fv, list):
            # 正常化成 catN
            norm: List[str] = []
            for t in fv:
                s = str(t).lower().strip()
                if s.startswith("cat"):
                    try:
                        n = int(s[3:])
                        if 1 <= n <= 8:
                            norm.append(f"cat{n}")
                    except Exception:
                        continue
                elif s.isdigit():
                    n = int(s)
                    if 1 <= n <= 8:
                        norm.append(f"cat{n}")
            # 去重
            seen = set(); dedup = []
            for x in norm:
                if x not in seen:
                    seen.add(x); dedup.append(x)
            free_vars = dedup
        # op
        op_raw = jp.get("op")
        if isinstance(op_raw, str) and op_raw.lower() in ("minimize", "maximize"):
            op = op_raw.lower()

    # 2) 自然語言：切片找類別+金額
    segments = re.split(r"[；;、\n]+", text)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        cat = _find_category(seg)
        amt = _extract_amount(seg)
        if cat is not None and amt is not None:
            user_params[f"cat{cat}"] = int(amt)

    # 3) budget_tax / op / free_vars 自然語言補強
    bud = _extract_budget(text)
    if bud is not None:
        user_params["budget_tax"] = int(bud)

    if op is None:
        op = _extract_op(text)

    if not free_vars:
        free_vars = _extract_free_vars(text)

    return {
        "tool_name": "nvat_tax",
        "payload": {
            "op": op,  # "minimize" | "maximize" | None
            "user_params": user_params,  # 僅包含有提供的 catN 與（可能有的）budget_tax
            "free_vars": free_vars,
            "constraints": {},  # 文字解析不帶 constraints，交由 ConstraintAgent
        }
    }