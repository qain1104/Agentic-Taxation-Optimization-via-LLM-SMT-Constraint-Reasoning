from __future__ import annotations

# Standard library
import asyncio
import hashlib
import inspect
import json
import time
import logging
import os
import re
import sys
import types
import typing
from dataclasses import dataclass, field
from datetime import datetime
from importlib import import_module
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple, get_args, get_origin

# Third-party
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

def _env_true(name: str) -> bool:
    v = os.getenv(name, "")
    return v.strip().lower() in {"1", "true", "yes", "on"}

LOG_NAME = "tax"
LOG = logging.getLogger(LOG_NAME)
LOG.propagate = False
LOG.setLevel(logging.DEBUG if _env_true("TAX_DEBUG") else logging.INFO)

# Log file path (overridable by TAX_LOG_FILE, default: ./logs/tax_app.log)
_default_log_dir = Path.cwd() / "logs"
log_file_env = os.getenv("TAX_LOG_FILE")

if log_file_env:
    LOG_FILE = Path(log_file_env)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
else:
    _default_log_dir.mkdir(parents=True, exist_ok=True)
    LOG_FILE = _default_log_dir / "tax_app.log"

_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")

# Console handler (stdout)
if not any(isinstance(h, logging.StreamHandler) for h in LOG.handlers):
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(_fmt)
    _sh.setLevel(logging.DEBUG if _env_true("TAX_DEBUG") else logging.INFO)
    LOG.addHandler(_sh)

# Rotating file handler (5 MB x 5)
if not any(isinstance(h, RotatingFileHandler) for h in LOG.handlers):
    _fh = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    _fh.setFormatter(_fmt)
    _fh.setLevel(logging.DEBUG)  # keep everything in file
    LOG.addHandler(_fh)

# child logger example
_log_fin = logging.getLogger(f"{LOG_NAME}.final_report.trigger")

def _append_debug(memory, *parts):
    lines = memory.get("debug_lines", []) or []
    msg = " ".join(str(p) for p in parts)
    lines.append(msg)
    memory.set("debug_lines", lines)
    if _env_true("TAX_DEBUG"):
        LOG.debug(msg)

LOG.info("File logging enabled at: %s", LOG_FILE)

# ---------------------------------------------------------------------------
# Global config & OpenAI setup (dotenv + env var only; no hardcoded secrets)
# ---------------------------------------------------------------------------
try:
    # 若有安裝 python-dotenv，就會自動載入專案根目錄的 .env
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please use `export OPENAI_API_KEY=...` in the shell,"
        "or create a .env file, or pass it in using `--env-file` in docker run."
    )

_async_client = AsyncOpenAI()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


KEYWORD_ROUTER = [
    (
        re.compile(
            r"(Special Goods and Services Tax|Special Goods Tax|High-End Consumption|"
            r"Luxury Cars|Yachts|Private Jet|Luxury Watches|Jewelry|Fur|Cameras)",
            re.I
        ),
        "special_goods_tax"
    ),
    (
        re.compile(
            r"(Special labor service tax|Labor service tax|Special sales tax|High-value consumption)",
            re.I
        ),
        "special_tax"
    ),
    (
        re.compile(
            r"(Non\s*value-added|Non-value-added[\-（(]?\s*|n\s*vat|non[-\s]?vat|Non\s*vat)",
            re.I
        ),
        "nvat_tax"
    ),
    (
        re.compile(
            r"(Value-added|Sales|Input|\bvat\b|value[-\s]?added)",
            re.I
        ),
        "vat_tax"
    ),
    (
        re.compile(
            r"((?<!Special)cargo-tax|White cement|Oil and gas|Rubber tires|Beverages|"
            r"Flat glass|Refrigerators|Vehicles)",
            re.I
        ),
        "cargo_tax_minimize"
    ),
    (
        re.compile(
            r"(Securities transaction tax|Securities trading|Stocks|Corporate bonds|"
            r"Warrants|Performance|Number of shares)",
            re.I
        ),
        "securities_tx_tax"
    ),
    (
        re.compile(
            r"(Futures transaction tax|Futures|Stock futures|Interest rate futures|Gold futures|Options)",
            re.I
        ),
        "futures_tx_tax"
    ),
]

def _route_by_keywords(text: str) -> str | None:
    t = str(text or "")
    # —— Guard：若是「再加條件/補幾個條件/繼續加」類語句，停用關鍵字路由 —— 
    try:
        _raw = t
        _norm = re.sub(r"[\s\u3000\u00A0]+", "", _raw)
        if re.search(r"(?:Continue | Then | Add | Modify).{0,4}(?:add)?constraints|(?:Add more conditions | Add a few more conditions | Continue adding | Keep adding)", _norm, flags=re.I):
            return None
    except Exception:
        pass
    for rx, tool in KEYWORD_ROUTER:
        if rx.search(t):
            return tool
    return None
# ---------------------------------------------------------------------------
# Public pipeline entry for REST API (/run)
# ---------------------------------------------------------------------------

async def run_tax_pipeline(payload: dict) -> dict:
    """
    給 REST API（/run）用的「單輪最佳化」入口。

    輸入 payload 預期格式（對應 /run body）：
        {
            "tool_name": "income_tax",
            "user_params": {...},
            "constraints": {...} | null,
            "free_vars": ["donation", ...] | null,
            "op": "minimize" | "maximize" | null,
            "budget_tax": 80000 | null,
            "raw_query": "原始自然語言說明" | null,
            ...
        }

    回傳格式：
        {
            "status": "ok" | "error",
            "tool_name": "...",
            "kpi": {...},              # baseline / optimized / mode / status
            "final_params": {...} | None,
            "diff": {...} | None,
            "report_md": "## Markdown 報告...",
            "raw_result": {...},       # 原始 calculator 回傳結果
        }
    """
    mem: MemoryStore = MEMORY

    # 每次 /run 重置短期記憶（長期 LTR_MEM 不受影響）
    reset_session(mem)

    tool_name = payload.get("tool_name")
    op = payload.get("op")
    if tool_name:
        mem.set("last_tool", tool_name)
    if op:
        mem.set("op", op)

    exec_agent = ExecuteAgent(memory=mem)
    reasoning_agent = ReasoningAgent(memory=mem)

    # 1) 執行實際稅務工具
    exec_output = await exec_agent.handle(payload)

    # 2) 產出報告（會順便寫入 __latest_report__ 與 reports/last_run）
    reasoning_output = await reasoning_agent.handle(exec_output)

    # 3) 從記憶體抓出 kpi snapshot
    latest = mem.get("__latest_report__") or {}
    kpis = latest.get("kpis") or {}

    # 4) 原始 result（通常包含 final_params / diff 等）
    result = reasoning_output.get("raw_result") or exec_output.get("result") or {}
    final_params = None
    diff = None
    if isinstance(result, dict):
        fp = result.get("final_params")
        if isinstance(fp, dict):
            final_params = fp
        df = result.get("diff")
        if isinstance(df, dict):
            diff = df

    return {
        "status": "ok",
        "tool_name": tool_name,
        "kpi": kpis,
        "final_params": final_params,
        "diff": diff,
        "report_md": reasoning_output.get("text", ""),
        "raw_result": result,
    }

# ---------------------------------------------------------------------------
# Number parsing utilities (共用：百/千/萬/億/兆；支援複合片語如「2億3千萬」)
# ---------------------------------------------------------------------------
def parse_cn_amount(text: str, *, allow_percent: bool = False) -> Optional[float]:
    """
    將含中式單位的數字片語轉為數值。
    支援：
      - 阿拉伯數字：1,234 / 3.5 / 3.5萬 / 1億2,345萬
      - 中文數字：三千五百 / 一點二五億 / 二億三千萬
      - 複合：2億3千萬 → 230,000,000；1億2,345萬 → 123,450,000
    可選：
      - allow_percent=True 時，"60%" → 0.6
    回傳 float 或 None（無法解析時）
    """
    import re
    s = (text or "").strip()
    if not s:
        return None

    # 百/千/萬/億/兆 對應倍率
    UNIT = {"百": 1e2, "千": 1e3, "萬": 1e4, "億": 1e8, "兆": 1e12}

    # 若允許百分比
    if allow_percent:
        m_pct = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*$", s)
        if m_pct:
            return float(m_pct.group(1)) / 100.0

    # 阿拉伯數字 + 單位（可複數段落）
    # 例：1億2,345萬 → [(1, 億), (2345, 萬)]
    parts = re.findall(r"([0-9][0-9,\.]*)\s*([百千萬億兆]?)", s)
    if parts:
        total = 0.0
        matched_any = False
        for num_s, unit in parts:
            if num_s == "":  # 可能是空段
                continue
            matched_any = True
            n = float(num_s.replace(",", ""))
            mul = UNIT.get(unit, 1.0)
            total += n * mul
        if matched_any:
            # 若整段只有單一段且無單位、且原字串其實是純數字（避免把「1 個詞」拆成 1 + 0）
            # 仍然會正確回傳該純數字
            return total if total != 0.0 or re.search(r"[0-9]", s) else None

    # 中文數字：處理到兆 / 億 / 萬 分節（先把「一點二」這種小數也處理）
    CN_NUM = {"零":0,"〇":0,"一":1,"二":2,"兩":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}
    CN_UNIT_SMALL = {"十":10, "百":100, "千":1000}
    SECTION_UNITS = [("兆", 1e12), ("億", 1e8), ("萬", 1e4)]

    def _cn_decimal_to_float(seg: str) -> Optional[float]:
        # 例如：「一點二五」→ 1.25
        if "點" not in seg:
            return None
        left, right = seg.split("點", 1)
        def _int_part(ss: str) -> int:
            # 支援「十、二十、三十五」這種
            if not ss:
                return 0
            total, num = 0, 0
            for ch in ss:
                if ch in CN_UNIT_SMALL:
                    num = 1 if num == 0 else num
                    total += num * CN_UNIT_SMALL[ch]
                    num = 0
                else:
                    num = num*10 + CN_NUM.get(ch, 0)
            return total + num
        L = _int_part(left)
        frac = 0.0
        base = 0.1
        for ch in right:
            if ch in CN_NUM:
                frac += CN_NUM[ch] * base
                base /= 10.0
        return L + frac

    def _cn_int_to_float(seg: str) -> float:
        # 例如：「三千五百」→ 3500；「十二」→ 12；「二十」→ 20
        total, num = 0, 0
        hit = False
        for ch in seg:
            if ch in CN_UNIT_SMALL:
                hit = True
                num = 1 if num == 0 else num
                total += num * CN_UNIT_SMALL[ch]
                num = 0
            else:
                if ch not in CN_NUM:
                    continue
                hit = True
                num = num*10 + CN_NUM[ch]
        return float(total + num) if hit else float("nan")

    # 先按「兆 / 億 / 萬」分節累加；每節內處理「千/百/十」與小數
    def _parse_cn_phrase(phrase: str) -> Optional[float]:
        rest = phrase
        total = 0.0
        for unit_char, mul in SECTION_UNITS:
            if unit_char in rest:
                left, rest = rest.split(unit_char, 1)
                # left 可以是「一點二」或「三千五百」
                val = _cn_decimal_to_float(left)
                if val is None:
                    val = _cn_int_to_float(left)
                    if val != val:  # NaN
                        return None
                total += val * mul
        # 處理最後剩下未帶「萬/億/兆」的尾段
        if "點" in rest:
            tail = _cn_decimal_to_float(rest)
            if tail is None:
                return None
            total += tail
        else:
            v = _cn_int_to_float(rest)
            if v == v:  # not NaN
                total += v
        return total if total != 0.0 else None

    if re.search(r"[零〇一二兩三四五六七八九十百千萬億兆點]", s):
        v = _parse_cn_phrase(s)
        if v is not None:
            return v

    return None

# ---------------------------------------------------------------------------
# —— 規則說明（共用 Markdown）——
# ---------------------------------------------------------------------------
RULES_NOTE_STAGE1 = (
    "#### Rule Explanation (Phase 1: Input)\n"
    "1) Unfilled variable → defaults to 0; Boolean unfilled → defaults to No.\n"
    "2) Variables not in the conditional expression: your input value will be used first; if no value is entered, the default value will be used.\n"
    "3) If a variable is mentioned in the conditional expression later, it is considered to be dominated by that condition and will override the previous input/preset (so that the solver can adjust it).\n"
    "(Note) For SMT/optimization, unfilled = 0 is equivalent to setting a fixed value for the variable unless the conditional statement allows it or rewrites it.\n"
)


def reset_session(memory: MemoryStore):
    """清空本輪對話狀態（長期記憶 LongTermMemory 不受影響）。"""
    for k in [
        "stage", "pending_tool", "pending_missing", "filled_slots", "conversation",
        "op", "last_tool", "last_exec_payload", "last_report_md", "last_result",
        "last_payload", "pending_constraint_payload", "pending_tool_for_constraints",
        "debug_lines",
    ]:
        memory.set(k, None)

# ===== prev-carry helpers =====

def extract_prev_from_result(result: dict) -> dict:
    """
    從工具 result 擷取快照，產出可直接塞回 payload 的 __prev_*。
    """
    if not isinstance(result, dict):
        return {}

    # 取用上一輪「可比較」的稅額：優先 optimized 其次 baseline
    prev_tax = None
    for k in ("optimized", "optimized_total_tax", "total_tax", "tax", "net_tax", "optimized_tax", "baseline", "base_tax"):
        v = result.get(k)
        if isinstance(v, (int, float)):
            prev_tax = float(v)
            break

    prev_final_params = result.get("final_params")
    prev_constraints  = result.get("constraints")

    return {
        "__prev_tax__": prev_tax if isinstance(prev_tax, float) else None,
        "__prev_final_params__": prev_final_params if isinstance(prev_final_params, dict) else None,
        "__prev_constraints__": prev_constraints if isinstance(prev_constraints, dict) else None,
    }


def merge_prev_into_payload(payload: dict | None, prev: dict | None) -> dict:
    """
    注入 __prev_* 到下一輪 payload（不覆蓋已存在且非空的值）。
    """
    payload = dict(payload or {})
    prev = dict(prev or {})

    def _set_if_missing(key: str):
        if key not in payload or payload.get(key) in (None, {}, []):
            if prev.get(key) not in (None, {}, []):
                payload[key] = prev[key]

    for key in ("__prev_tax__", "__prev_final_params__","__prev_constraints__"):
        _set_if_missing(key)

    return payload

# ---------------------------------------------------------------------------
# Simple memory utilities
# ---------------------------------------------------------------------------
class MemoryStore:
    """In-memory key–value store for short-term session state."""

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None):
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value

    def clear(self):
        self._data.clear()


class LongTermMemory:
    """Very light JSON-file persistence for user preferences / common phrases."""

    def __init__(self, path: str = "lt_memory.json"):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)

    def load(self) -> Dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, data: Dict[str, Any]):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def update(self, key: str, value: Any):
        data = self.load()
        data[key] = value
        self.save(data)

LTR_MEM = LongTermMemory()
MEMORY = MemoryStore()

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

from tools_registry import TOOLS  # noqa: F401

# Build global Chroma index once
_docs = [f"{t['name']} ::: {';'.join(t['keywords'])}" for t in TOOLS]
_embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
INDEX = Chroma.from_documents(CharacterTextSplitter(chunk_size=256).create_documents(_docs), _embeddings)

# Utility to find tool meta quickly
TOOL_MAP = {t["name"]: t for t in TOOLS}

# ---------------------------------------------------------------------------
# Field helper (module-level): collect all translatable field tokens for a tool
# ---------------------------------------------------------------------------
def field_set_for_tool(tool_name: str) -> set[str]:
    """
    依據該 tool 的 field_labels / row_fields 動態蒐集「可中文化」的欄位名稱（slug 右側）。
    會自動把像 tax_price_per_unit / unit_price / dutiable_price 等新欄位納入。
    """
    labels = (TOOL_MAP.get(tool_name, {}) or {}).get("field_labels", {}) or {}
    row_fields = set((TOOL_MAP.get(tool_name, {}) or {}).get("row_fields", []) or [])
    right = set()
    for k in labels.keys():
        if isinstance(k, str) and "." in k:
            _, fld = k.split(".", 1)
            right.add(fld.strip())
    right |= row_fields
    # 常見保底鍵，避免空集合
    right |= {"quantity", "assessed_price", "tp", "ep", "sc", "ca", "pa"}
    # 僅保留合法識別子
    return {f for f in right if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", f)}

# ---------------------------------------------------------------------------
# BaseAgent – common async chat helper
# ---------------------------------------------------------------------------

@dataclass
class BaseAgent:
    name: Optional[str] = None
    memory: MemoryStore = field(default_factory=lambda: MEMORY)

    def __post_init__(self):
        if not self.name:
            self.name = self.__class__.__name__

    async def _chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        """LLM wrapper using AsyncOpenAI client."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = await _async_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# CallerAgent
# ---------------------------------------------------------------------------

class CallerAgent(BaseAgent):
    """解析自然語言 → semantic frame → 檢缺 → 工具匹配 / 追問。"""

    # === 常數 & Regex / 區間 ===
    import re as _re
    import datetime as _dt

    _NEXT_PAT = re.compile(r"^(?:next step|next|go|continue|ok)$", re.I)
    _DONE_PAT = re.compile(r"^(?:over|it's done|ok|done|finish)$", re.I)
    _REOPEN_PAT = re.compile(
        r"(?: continue | then | add | modify).{0,4}(?: add)?condition|(?: add more conditions | add a few more conditions | continue adding | add more)",
        re.I
    )
    _DATE_RE = _re.compile(r"(\d{2,3})\s*y?\s*(\d{1,2})?\s*m?\s*(\d{1,2})?")

    _GIFT_BOUNDS = [
        (_dt.date(2009,1,23),  _dt.date(2017,5,11),   1),
        (_dt.date(2017,5,12),  _dt.date(2021,12,31),  2),
        (_dt.date(2022,1,1),   _dt.date(2024,12,31),  3),
        (_dt.date(2025,1,1),   _dt.date(9999,12,31),  4),
    ]
    _ESTATE_BOUNDS = [
        (_dt.date(2014,1,1),   _dt.date(2017,5,11),   1),
        (_dt.date(2017,5,12),  _dt.date(2021,12,31),  2),
        (_dt.date(2022,1,1),   _dt.date(2023,12,31),  3),
        (_dt.date(2024,1,1),   _dt.date(2024,12,31),  4),
        (_dt.date(2025,1,1),   _dt.date(9999,12,31),  5),
    ]

    # === 物件生命週期 ===
    def __init__(self, memory: Optional[MemoryStore] = None):
        # memory 允許由外部（例如 app_gradio 的 session 管理器）傳入；
        # 若未提供則退回 BaseAgent 的預設 MEMORY。
        super().__init__(name="CallerAgent", memory=memory)
        self._sig_cache: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}


    # === 流程閘門（Step Gate） ===
    def _is_next(self, s: str) -> bool:
        return bool(self._NEXT_PAT.search((s or "").strip()))
    def _is_done(self, s: str) -> bool:
        return bool(self._DONE_PAT.search((s or "").strip()))
    def _wants_skip(self, text: str) -> bool:
        t = (text or "").strip().lower()
        return bool(re.search(r"(default\s*0|skip)", t))

    def _wants_direct_compute(self, text: str) -> bool:
        """使用者要求『直接計算』：跳過階段頁，直接開始工具計算。"""
        t = (text or "").strip().lower()
        return bool(re.search(r"(?:compute\s*now|run\s*now|calculate\s*now|calculate\s*immediately|compute\s*immediately|run\s*calculation|direct\s*calc|direct\s*calculation|calculate\s*directly|run\s*calculation\s*now)",
    t, flags=re.I
))

    def _explicit_correction(self, text: str) -> bool:
        return bool(re.search(r"(?:correct|fix|revise|modify|change|update|edit|adjust)", text or ""))

    # === 模式、文字正規化 & 小工具 ===
    def _infer_mode(self, text: str):
        t = str(text).lower()
        if any(k in t for k in ["maximize", "maximize_qty"]): return "maximize"
        if any(k in t for k in ["minimize"]):               return "minimize"
        return None

    def _norm_text(self, s: str) -> str:
        s = str(s or "")
        s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
        s = re.sub(r"[\s\u3000\u00A0]+", "", s)
        s = re.sub(r"[，。！？、：:；;（）()「」『』《》〈〉…—\-~·•]+", "", s)
        s = re.sub(r"[\U00010000-\U0010ffff]", "", s)
        return s

    def _is_reopen_constraints(self, text: str) -> bool:
        if not text: return False
        if self._REOPEN_PAT.search(text): return True
        t = self._norm_text(text)
        return t in {"Add more conditions","add a few more conditions","continue adding conditions","add more conditions", "continue adding", "add more conditions", "add more conditions"}

    def _to_int_token(self, s: str) -> Optional[int]:
        if not s: return None
        s = str(s).strip()
        s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
        cn = {"一":1, "二":2, "兩":2, "三":3, "四":4}
        if s.isdigit(): return int(s)
        return cn.get(s)

    def _coerce_rows(self, slots: Dict[str, Any]) -> None:
        if "rows" not in slots: return
        val = slots.get("rows")
        if isinstance(val, str):
            try:
                slots["rows"] = json.loads(val)
            except Exception:
                return
        if slots.get("rows") is not None and not isinstance(slots["rows"], list):
            slots.pop("rows", None)

    def _coerce_boolean_fields(self, slots: Dict[str, Any], bool_fields: Optional[set] = None) -> None:
        true_set = {"true", "t", "yes", "y", "1", "是", "有"}
        false_set = {"false", "f", "no", "n", "0", "否", "沒有", "無"}
        for k, v in list(slots.items()):
            if bool_fields is not None:
                if k not in bool_fields: continue
            else:
                if not re.match(r"^(is_|use_)", str(k)): continue
            if isinstance(v, bool): continue
            if isinstance(v, (int, float)):
                slots[k] = bool(v)
            elif isinstance(v, str):
                s = v.strip().lower()
                if s in true_set: slots[k] = True
                elif s in false_set or s == "": slots[k] = False

    def _roc_date_to_period(self, text: str, bounds) -> Optional[int]:
        m = self._DATE_RE.search(text)
        if not m: return None
        y, mth, day = m.groups()
        try:
            g_year = int(y) + 1911
            month  = int(mth) if mth else 1
            day    = int(day) if day else 1
            dt = self._dt.date(g_year, month, day)
        except Exception:
            return None
        for start, end, code in bounds:
            if start <= dt <= end:
                return code
        return None

    def _extract_budget_hint(self, text: str, allow_bare_number: bool = False) -> Optional[int]:
        s = str(text or "").strip()
        m = re.search(r"(budget(?:_tax)?|target(?:_tax)?)\s*[:=：]?\s*([^\s，。；;]+)", s, flags=re.I)
        if m:
            cand = m.group(2)
            val = parse_cn_amount(cand, allow_percent=False)
            if isinstance(val, (int, float)):
                return int(val)
        if allow_bare_number:
            val2 = parse_cn_amount(s, allow_percent=False)
            if isinstance(val2, (int, float)):
                return int(val2)
        return None

    # === 簽章解析（工具 entry func、型別、預設值） ===
    def _base_of(self, ann):
        if ann is inspect._empty: return None
        origin = get_origin(ann)
        if origin is None: return ann
        if origin is Annotated:
            args = get_args(ann)
            return self._base_of(args[0]) if args else None
        if origin in (typing.Union, types.UnionType):
            args = [a for a in get_args(ann) if a is not type(None)]
            return self._base_of(args[0]) if args else None
        return origin

    def _resolve_entry_function(self, tool_meta: Dict[str, Any], op: Optional[str]) -> Any:
        entry = tool_meta.get("entry_func")
        if isinstance(entry, dict):
            fn_name = entry.get(op or "minimize") or next(iter(entry.values()))
        else:
            fn_name = entry
        module = __import__(tool_meta["module"], fromlist=["*"])
        return getattr(module, fn_name)

    def _get_sig_info(self, tool_meta: Dict[str, Any], op: Optional[str]) -> Dict[str, Any]:
        key = (tool_meta["name"], op)
        if key in self._sig_cache:
            return self._sig_cache[key]

        func = self._resolve_entry_function(tool_meta, op)
        sig = inspect.signature(func)
        types: Dict[str, str] = {}
        defaults: Dict[str, Any] = {}
        bool_fields: set[str] = set()

        for name, p in sig.parameters.items():
            if name in {"self", "args", "kwargs"} or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if p.default is not inspect._empty:
                defaults[name] = p.default
            t = None
            if isinstance(p.default, bool):
                t = "bool"
            elif isinstance(p.default, (int, float)):
                t = "number"
            else:
                base = self._base_of(p.annotation)
                if base is bool:           t = "bool"
                elif base in (int, float): t = "number"
            types[name] = t or "other"
            if types[name] == "bool":
                bool_fields.add(name)

        info = {"func": func, "types": types, "defaults": defaults, "bool_fields": bool_fields}
        self._sig_cache[key] = info
        return info

    def _default_from_sig(self, field: str, sig_info: Dict[str, Any]):
        if field in sig_info["defaults"]:
            return sig_info["defaults"][field]
        t = sig_info["types"].get(field)
        if t == "bool":   return False
        if t == "number": return 0
        return 0

    # === 工具排名 / UI 組字 ===
    def _rank_tools(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        sims = INDEX.similarity_search_with_relevance_scores(text, k=k)
        return [(doc.page_content.split(" :::")[0], score) for doc, score in sims]

    def _fmt_vars_overview(self, tool_name: str) -> str:
        meta = TOOL_MAP.get(tool_name, {}) or {}
        labels: Dict[str, str] = meta.get("field_labels", {}) or {}
        required = list(meta.get("required_fields", []) or [])
        budget_field = meta.get("budget_field")

        def _lab(k: str) -> str:
            import re
            s = labels.get(k, k)
            if not isinstance(s, str): return k
            s = re.sub(r"[（(].*?[）)]", "", s).replace("？", "").strip()
            return s or k

        lines = [f"### Current tax types:{meta.get('description', tool_name)}"]
        lines.append("Phase 1: Please enter your tax status for calculating the tax amount. If no value is provided, the system will automatically enter 0 or No.")
        if required:
            lines.append("**Input basic fields**")
            lines += [f"- {_lab(k)}（`{k}`）" for k in required]
        if budget_field:
            lines.append(f"\n**maximum tax budget field**\n- {_lab(budget_field)}（`{budget_field}`）")
        lines.append("\nPlease reply with the corresponding value directly in the text box. After sending, the system will tell you the value currently received.")
        return "\n".join(lines)

    # === 單一輸入頁預覽 ===
    def _compose_inputs_page(self, tool_name: str, merged_slots: Dict[str, Any]) -> str:
        """
        將「可輸入欄位導覽」與「目前已收到的輸入」合併到同一頁。
        使用者可持續補值；回覆「下一步」會進入條件階段。
        """
        header = self._fmt_vars_overview(tool_name)
        preview = self._inline_current_inputs(tool_name, merged_slots)
        tail = "\n\n> To add more variables, enter them directly; to complete the settings, reply 「Next step」; to calculate directly, reply 「Calculate Directly」."
        # 插入階段一規則說明
        rules = "\n\n" + RULES_NOTE_STAGE1
        return header + ("\n\n" + preview if preview else "") + rules + tail


    def _inline_current_inputs(self, tool_name: str, slots: Dict[str, Any]) -> str:
        meta = TOOL_MAP.get(tool_name, {}) or {}
        labels: Dict[str, str] = meta.get("field_labels", {}) or {}

        def _lab(k: str) -> str:
            import re
            s = labels.get(k, k)
            if not isinstance(s, str): 
                return k
            # 移除括號內的說明，保留精簡標籤
            return re.sub(r"[（(].*?[）)]", "", s).replace("？", "").strip() or k

        def _fmt_val(v):
            if isinstance(v, bool):
                return "yes" if v else "no"
            try:
                f = float(v)
                return f"{f:,.0f}" if f.is_integer() else f"{f:,}"
            except Exception:
                return str(v)

        # 先收集一般欄位（排除 rows / 空值）
        basics = [(k, v) for k, v in slots.items() if k != "rows" and v not in (None, "")]
        lines: List[str] = []
        if basics:
            lines.append("——\n**Input received so far**")
            for k, v in basics:
                lines.append(f"- {_lab(k)}：{_fmt_val(v)}")

        # ★ 若此工具有 budget_field，另外特別顯示（且避免重覆）
        budget_key = meta.get("budget_field")
        if isinstance(budget_key, str) and budget_key:
            # 可能 slots 裡面還沒有這個鍵（例如你只有存放在別處）
            # 這裡先單純從 slots 讀；若 CallerAgent 也把它放到 slots，就能顯示
            bud_val = slots.get(budget_key)
            if bud_val not in (None, ""):
                # 若上面 basics 已經涵蓋，就不重覆顯示
                if all(k != budget_key for k, _ in basics):
                    if not lines:
                        lines.append("——\n**Input received so far**")
                    lines.append(f"- {_lab(budget_key)}：{_fmt_val(bud_val)}")

        return "\n".join(lines)

    def _fmt_input_review(self, tool_name: str, merged_slots: Dict[str, Any]) -> str:
        meta = TOOL_MAP.get(tool_name, {}) or {}
        labels: Dict[str, str] = meta.get("field_labels", {}) or {}
        row_fields = list(meta.get("row_fields", []) or [])

        def _lab(k: str) -> str:
            import re
            s = labels.get(k, k)
            if not isinstance(s, str): return k
            return re.sub(r"[（(].*?[）)]", "", s).replace("？", "").strip() or k

        def _fmt_money(x):
            try: return f"{float(x):,.0f}"
            except Exception: return str(x)

        lines = [f"### Current Input Overview — {meta.get('description', tool_name)}"]
        basics = [(k,v) for k,v in merged_slots.items() if k!="rows"]
        if basics:
            lines.append("**General fields**")
            for k,v in basics:
                if v in (None, ""): continue
                lines.append(f"- {_lab(k)}：{_fmt_money(v)}（`{k}`）")

        lines.append("\nIf you need to add more fields, enter them directly; if you have already completed this step, reply with 「Finish」 or 「Next step」.")
        return "\n".join(lines)

    # === RAG：早期條件建議 ===
    async def _early_condition_tips(self, tool_name: str, merged_slots: dict, op: str | None, user_msg: str) -> str:
        meta = TOOL_MAP.get(tool_name, {}) or {}
        labels = meta.get("field_labels", {}) or {}
        required = meta.get("required_fields", []) or []
        constraint_fields = meta.get("constraint_fields", []) or []
        budget_field = meta.get("budget_field")
        row_fields = list(meta.get("row_fields") or [])
        mode_hint = (op or "").lower()

        base_allowed = sorted(set(required) | set(constraint_fields) | ({budget_field} if budget_field else set()))
        rows = (merged_slots or {}).get("rows") or []
        nrows = len(rows) if isinstance(rows, list) else 0
        row_allowed = []
        if nrows and row_fields:
            for i in range(nrows):
                for f in row_fields:
                    row_allowed.append(f"row{i}.{f}")
        allowed_vars_for_prompt = base_allowed + row_allowed

        def _load_rag():
            try:
                persist_dir = os.getenv("RAG_CHROMA_DIR", "rag/chroma")
                collection  = os.getenv("RAG_COLLECTION", "tax_handbook")
                if not os.path.isdir(persist_dir): return None
                return Chroma(
                    collection_name=collection,
                    persist_directory=persist_dir,
                    embedding_function=OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"))
                )
            except Exception:
                return None

        db = getattr(self, "_rag_db", None)
        if db is None:
            db = _load_rag()
            setattr(self, "_rag_db", db)

        def _mk_query() -> str:
            lab_tokens = [str(v) for v in labels.values() if isinstance(v, str)]
            extra = " ".join(lab_tokens[:20])
            mode_hint = f" 模式:{op}" if op else ""
            return f"{tool_name} 條件 建議 稅務 {mode_hint} {user_msg} {extra}"

        sources = []
        if db:
            try:
                q = _mk_query()
                hits = db.similarity_search(q, k=6)
                for h in (hits or []):
                    page = (h.metadata or {}).get("page") or (h.metadata or {}).get("loc")
                    title = (h.metadata or {}).get("title") or (h.metadata or {}).get("source") or "手冊"
                    url = (h.metadata or {}).get("url")
                    txt = str(h.page_content or "").strip().replace("\n", " ")
                    if len(txt) > 320: txt = txt[:320] + "…"
                    sources.append({
                        "title": str(title),
                        "page": page,
                        "url": url,
                        "chunk": txt
                    })
            except Exception:
                sources = []

        context = {
            "tool_name": tool_name,
            "mode": (op or "").lower(),
            "allowed_vars": allowed_vars_for_prompt,
            "field_labels": labels,
            "current_params": merged_slots,
            "row_count": nrows,
            "row_fields": row_fields,
            "budget_field": budget_field,
            "user_msg": user_msg,
            "sources": sources,
        }

        policy_lines = []
        if not budget_field:
            policy_lines.append("The tool does not define the budget_field, therefore **it is prohibited** from making any suggestions related to 'tax cap/budget cap/budget'.")
        if mode_hint != "maximize":
            policy_lines.append("In this round, the mode is not maximized, and **suggestions related to 'tax cap/budget cap/budget' are **prohibited**.")

        sys_prompt = (
            "You are a tax-optimization 'condition design consultant'. Based on the provided tool-field whitelist, the current parameters, and (if available) the manual excerpts, "
            "propose 1–3 'actionable conditional suggestions' to be used as a pre-optimization setup.\n"
            "Rules:\n"
            "1) Only list each suggestion's title and rationale; **do not** output any JSON patch, free_vars, or constraints.\n"
            "2) If mode=maximize and budget_field is missing, you may suggest adding a tax/expense upper bound (rough estimates are allowed).\n"
            "3) If sources are provided, suggestions should be **as compatible as possible** with them or inspired by them (but must not clearly contradict sources).\n"
            "4) Output strict JSON only: {\"suggestions\":[{\"title\":\"...\",\"rationale\":\"...\"}, ...]}.\n"
            "5) title ≤ 20 chars; rationale ≤ 60 chars; avoid fluff and repetition; be practical and actionable.\n"
             + ("\n".join(policy_lines) if policy_lines else "")
        )
        user_prompt = json.dumps(context, ensure_ascii=False)
        raw = await self._chat(sys_prompt, user_prompt, temperature=0.2)

        try:
            obj = json.loads(raw) if isinstance(raw, str) else {}
        except Exception:
            obj = {}
        suggestions = obj.get("suggestions") if isinstance(obj, dict) else None
        if not isinstance(suggestions, list):
            suggestions = []

        cleaned = []
        for it in suggestions[:3]:
            title = it.get("title") if isinstance(it, dict) else None
            rationale = it.get("rationale") if isinstance(it, dict) else None
            if isinstance(title, str) and isinstance(rationale, str):
                title = title.strip(); rationale = rationale.strip()
                if title and rationale:
                    cleaned.append({"title": title, "rationale": rationale})

        if not cleaned:
            cleaned = [
                {
                    "title": "Add ratio bounds to avoid extremes",
                    "rationale": "Use inter-variable ratio constraints; they’re more robust than fixed constants."
                }
            ]

        md = ["#### Conditional suggestions (review first; add only if needed)"]
        for s in cleaned:
            md.append(f"**{s['title']}**\n- reason:{s['rationale']}")
        return "\n\n".join(md)

    # === NL 解析器整合（把自然語言塞進 slots） ===
    def _apply_nl_parser(self, tool_name: str, tool_meta: Dict[str, Any], user_msg: str,
                         merged_slots: Dict[str, Any], allowed_fields: set) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        parser_path = tool_meta.get("nl_parser")
        _append_debug(self.memory, "PARSER:", parser_path)

        if not parser_path:
            _append_debug(self.memory, "PARSER_SKIPPED_NO_PATH")
            return {}

        func_name = None
        if ":" in parser_path:
            module_path, func_name = parser_path.split(":", 1)
        else:
            module_path = parser_path

        try:
            _mod = __import__(module_path, fromlist=["*"])
            if func_name:
                _fn = getattr(_mod, func_name, None)
            else:
                _fn = (
                    getattr(_mod, "nl_to_payload", None) or
                    getattr(_mod, f"nl_to_{tool_name}_payload", None) or
                    getattr(_mod, "nl_to_nvat_payload", None) or
                    getattr(_mod, "nl_to_cargo_payload", None)
                )
            if callable(_fn):
                parsed = _fn(user_msg) or {}
            else:
                _append_debug(self.memory, "PARSER_FN_NOT_FOUND:", parser_path)
        except Exception as e:
            import traceback
            _append_debug(self.memory, f"PARSER_IMPORT_ERROR: {parser_path} -> {e}")
            _append_debug(self.memory, traceback.format_exc())

        _append_debug(self.memory, "PARSED_RAW:", parsed)

        parsed_rows = parsed.get("rows")
        if "rows" in allowed_fields and isinstance(parsed_rows, list) and len(parsed_rows) > 0:
            merged_slots["rows"] = parsed_rows

        if "free_vars" in allowed_fields and (merged_slots.get("free_vars") in (None, [], "")) and (parsed.get("free_vars") is not None):
            merged_slots["free_vars"] = parsed["free_vars"]

        if "constraints" in allowed_fields:
            parsed_constraints = parsed.get("constraints")
            if (merged_slots.get("constraints") in (None, {}, "")) and (parsed_constraints is not None):
                merged_slots["constraints"] = parsed_constraints

        budget_field = tool_meta.get("budget_field")
        _append_debug(self.memory, "BUDGET_FIELD:", budget_field, "VAL:", parsed.get("target_tax"))
        if budget_field in allowed_fields and (parsed.get("target_tax") is not None) and (merged_slots.get(budget_field) in (None, "", 0)):
            merged_slots[budget_field] = parsed["target_tax"]

        _append_debug(self.memory, "MERGED_AFTER_PARSER:", json.dumps(merged_slots, ensure_ascii=False, indent=2))
        return parsed

    # === 語意框架抽取（Intent + Slots） ===
    async def _extract_semantic_frame(self, user_msg: str, allowed_tools: Optional[List[str]] = None) -> Dict[str, Any]:
        tool_list = [t for t in TOOLS if (not allowed_tools or t["name"] in allowed_tools)]
        intents = [t["name"] for t in tool_list]
        req_fields_by_tool = {t["name"]: list(t.get("required_fields", [])) for t in tool_list}
        union_required_fields = sorted({fld for flds in req_fields_by_tool.values() for fld in flds})
        union_field_list = ", ".join(union_required_fields) if union_required_fields else ""

        mapping_doc = {}
        for t in tool_list:
            name = t["name"]
            reqs = req_fields_by_tool[name]
            labels_all = t.get("field_labels", {}) or {}
            labels_subset = {k: labels_all.get(k, "") for k in reqs}
            alias_map = t.get("alias", {}) or {}
            mapping_doc[name] = {
                "required_fields": reqs,
                "field_labels": labels_subset,
                "aliases": alias_map,
            }

        sys_prompt = (
           "You are a semantic parser. Your task is to convert the user’s tax-related description into a JSON semantic frame.\n"
            f"Please choose exactly one intent that best matches from the following intents: {(', '.join(intents) if intents else '(none)')}.\n"
            "Note: You must select one intent first, and then fill ONLY the required_fields of that intent; "
            "fields from other intents must NOT appear in slots (if not mentioned, set to null or omit).\n"
            f"Union list of all visible field names (for naming-style awareness only; not all are necessarily available this turn): {union_field_list}\n"
            "\n"
            "[Field definitions and Chinese alias mapping for each intent (reference only)]\n"
            + json.dumps(mapping_doc, ensure_ascii=False, indent=2)
            + "\n\n"
            "[Entity attribution rules (general; avoid putting 'self' info into dep)]\n"
            "1) If the text includes 'I / myself / me' → use *_self fields.\n"
            "2) If the text includes 'spouse ...' → use *_spouse.\n"
            "3) If the text includes 'dependents / children / parents ...' → use *_dep.\n"
            "4) If no subject is explicitly stated but the meaning refers to the user → default to *_self.\n"
            "5) If the same phrase matches multiple aliases, choose the most specific one.\n"
            "\n"
            "[Amount and unit normalization rules] …(same as the original)…\n"
            "\n"
            "[Output format]\n"
            "{ \"intent\": <intent>, \"slots\": {<field>: <value or null>} }\n"
            "Output ONLY the single JSON object above. No extra explanations."
        )
        raw = await self._chat(sys_prompt, user_msg)

        try:
            frame = json.loads(raw)
        except Exception:
            frame = {"intent": None, "slots": {}}

        intent = frame.get("intent")
        slots = frame.get("slots", {}) or {}

        if intent in req_fields_by_tool:
            allowed_for_intent = set(req_fields_by_tool[intent])
            slots = {k: v for k, v in slots.items() if k in allowed_for_intent}
            for req in allowed_for_intent:
                if req not in slots:
                    slots[req] = None
            frame["slots"] = slots
        else:
            frame = {"intent": None, "slots": {}}

        return frame

    # === 追問文案產生（缺欄位時） ===
    async def _generate_followup_question(self, tool_name: str, missing: List[str], filled: Dict[str, Any]) -> str:
        def _normalize_text(s):
            if not isinstance(s, str):
                return "" if s is None else str(s)
            return s.replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n").replace("\\t", "\t")

        labels: Dict[str, str] = TOOL_MAP[tool_name].get("field_labels", {}) or {}
        prompts = [_normalize_text(labels.get(f, f"please provide {f} value?")) for f in missing]
        base_q = "、".join(prompts)
        return (
            f"Phase 1: Please enter your tax status for calculating the tax amount.{base_q}。"
            "If no value is provided, the system will automatically substitute 0 or 「No」. To calculate using the default value, reply 「Skip」."
        )

    # === 核心：處理一個使用者輸入 ===
    async def handle(self, user_msg: str) -> Dict[str, Any]:
        def _is_next_local(s: str) -> bool:
            return bool(re.search(r"^(?:next step|next|go|continue|ok)$", (s or "").strip(), flags=re.I))

        # —— 0) 若使用者要「再加條件 / 繼續加 / 補幾個條件」→ 直接交給 ConstraintAgent 重開上一輪 —— 
        try:
            if self._is_reopen_constraints(user_msg):
                ctx = self.memory.get("last_exec_payload") or {}
                tool = ctx.get("tool_name") or self.memory.get("last_tool")
                if tool:
                    # 讓 ConstraintAgent 能即刻使用
                    self.memory.set("pending_constraint_payload", ctx.get("payload") or {
                        "tool_name": tool,
                        "user_params": ctx.get("user_params") or {},
                        "op": ctx.get("op"),
                    })
                    self.memory.set("pending_tool_for_constraints", tool)
                    return {"type": "reopen_constraints"}
                else:
                    # 沒有可重開的上下文，就請使用者先跑一次計算
                    return {
                        "type": "follow_up",
                        "question": "The calculation context of the previous round could not be found; please select the tax type and run the calculation once, then reply 「Add more conditions」."
                    }
        except Exception:
            # 容錯：就算這段失敗也不要影響後續正常流程
            pass
        
        # —— 1) 正常流程：解析語意框架（Intent + Slots）——
        conv_history = self.memory.get("conversation", [])
        conv_history.append({"role": "user", "content": user_msg})
        self.memory.set("conversation", conv_history)

        sticky_tool = self.memory.get("pending_tool")
        prefer_tool = _route_by_keywords(user_msg)

        # 使用者有沒有明講要換稅別？
        explicit_switch = re.search(
            r"(?:switch to|recalculate as|use instead|change to|change into|use a different|"
            r"don't use (?:the current|this).*?(?:tax)?|"
            r"i (?:want|would like) to calculate.*?tax)",
            user_msg,
            flags=re.IGNORECASE
        )

        # 「工具鎖」：如果之前已經有 last_tool，且這次沒有明講要換，就把它當成鎖定工具
        last_tool = self.memory.get("last_tool")
        locked_tool = None
        if last_tool and not explicit_switch:
            locked_tool = last_tool

        # 若當前 stage 已經有 pending_tool，就優先用 pending_tool 當鎖
        if sticky_tool:
            locked_tool = sticky_tool

        allowed_tools: Optional[List[str]] = None
        if locked_tool:
            # 已有「鎖定工具」時，semantic frame 只看這一個工具
            allowed_tools = [locked_tool]
        elif prefer_tool and prefer_tool in TOOL_MAP:
            # 只有在「沒有鎖」的情況下，才允許 keyword route 直接指定工具
            allowed_tools = [prefer_tool]

        frame = await self._extract_semantic_frame(user_msg, allowed_tools=allowed_tools)
        intent = frame.get("intent")
        slots: Dict[str, Any] = frame.get("slots", {}) or {}
        _append_debug(self.memory, "FRAME:", frame)

        if locked_tool:
            # 這裡包含 sticky_tool 以及 last_tool（只要沒有 explicit_switch）
            if intent != locked_tool:
                _append_debug(self.memory, f"LOCKED_TOOL_OVERRIDE: {intent} -> {locked_tool}")
            tool_name = locked_tool
            intent = locked_tool
        elif prefer_tool and prefer_tool in TOOL_MAP:
            # 只有在沒有鎖定、且使用者有 keyword cue（例如一開始那句「我要算營業稅」）時才用
            if intent != prefer_tool:
                _append_debug(self.memory, f"KEYWORD_ROUTE_OVERRIDE: {intent} -> {prefer_tool}")
            tool_name = prefer_tool
            intent = prefer_tool
        else:
            if intent in TOOL_MAP:
                tool_name = intent
            else:
                ranked = self._rank_tools(user_msg, k=1)
                if not ranked:
                    return {
                        "type": "follow_up",
                        "question": "Which type of tax would you like to calculate? You can say “I want to calculate Individual Income Tax,” or click one of the examples on the right."
                    }
                cand, score = ranked[0]
                if score is None or score < 0.25:
                    return {
                        "type": "follow_up",
                        "question": "Which type of tax would you like to calculate? You can say “I want to calculate Individual Income Tax,” or click one of the examples on the right."
                    }
                tool_name = cand
                intent = tool_name

        tool_meta = TOOL_MAP[tool_name]
        required = tool_meta.get("required_fields", [])
        constraint_fields = tool_meta.get("constraint_fields", []) or []
        budget_field = tool_meta.get("budget_field")
        allowed_fields = set(required) | set(constraint_fields) | ({budget_field} if budget_field else set())

        _append_debug(self.memory, f"TOOL: {tool_name} (intent={intent})")
        _append_debug(self.memory, "META:", tool_meta)

        last_tool = self.memory.get("last_tool")
        switched_tool = bool(last_tool and last_tool != tool_name and not sticky_tool)
        if switched_tool:
            self.memory.set("filled_slots", merged_slots)
            self.memory.set("pending_missing", None)
            self.memory.set("pending_constraint_payload", None)
            self.memory.set("pending_tool_for_constraints", None)
            self.memory.set("last_exec_payload", None)
            self.memory.set("op", None)
            _append_debug(self.memory, f"CLEAR_CTX_ON_TOOL_SWITCH: last={last_tool} -> now={tool_name}")
        self.memory.set("last_tool", tool_name)

        prev_slots_all = self.memory.get("filled_slots", {}) or {}
        prev_slots = {k: v for k, v in prev_slots_all.items() if k in allowed_fields}
        curr_slots = {k: v for k, v in slots.items() if (v is not None and k in allowed_fields)}

        pending_missing = self.memory.get("pending_missing") or []
        if pending_missing and not self._explicit_correction(user_msg):
            curr_slots = {k: v for k, v in curr_slots.items() if k in pending_missing}

        merged_slots = {**prev_slots, **curr_slots}
        # 把 budget_field（如 budget_tax）也塞進 merged_slots，便於第一階段預覽顯示 ===
        budget_field = tool_meta.get("budget_field")
        if budget_field and (budget_field in allowed_fields):
            # 先取使用者這輪文字中的「稅額上限」提示（支援：稅額上限/上限/budget_tax/target_tax 等）
            bud_hint = self._extract_budget_hint(user_msg, allow_bare_number=False)

            # 取既有 slots（之前輪已經提供過的值）
            bud_curr = curr_slots.get(budget_field)
            bud_prev = prev_slots.get(budget_field)

            # 以「本輪文字 > 既有已填」為優先，填入 merged_slots
            bud_val = None
            if isinstance(bud_hint, (int, float)) and bud_hint > 0:
                bud_val = int(bud_hint)
            elif isinstance(bud_curr, (int, float)) and bud_curr > 0:
                bud_val = int(bud_curr)
            elif isinstance(bud_prev, (int, float)) and bud_prev > 0:
                bud_val = int(bud_prev)

            if bud_val is not None:
                merged_slots[budget_field] = bud_val
        self._coerce_rows(merged_slots)

        # 特殊期間欄位正規化
        if tool_name == "gift_tax":
            fld, bounds, valid = "period_choice", self._GIFT_BOUNDS, (1, 2, 3, 4)
        elif tool_name == "estate_tax":
            fld, bounds, valid = "death_period",  self._ESTATE_BOUNDS, (1, 2, 3, 4, 5)
        else:
            fld = None
        if fld:
            val = merged_slots.get(fld)
            if val is not None:
                try:
                    code = int(str(val))
                    if code not in valid: raise ValueError
                    merged_slots[fld] = code
                except ValueError:
                    maybe = self._roc_date_to_period(str(val), bounds)
                    if maybe: merged_slots[fld] = maybe

        # === 階段一：可設定欄位總覽（不吃變數） ===
        force_direct = self._wants_direct_compute(user_msg)
        stage = self.memory.get("stage")
        if switched_tool or stage is None:
            # 先存一份目前解析到的值，讓下一輪可接續累積
            self.memory.set("pending_tool", tool_name)
            self.memory.set("filled_slots", merged_slots)

            # 若使用者輸入『直接計算』，不要停在階段一頁面，直接往下建立 tool_request
            if force_direct:
                self.memory.set("stage", "constraints")
                stage = "constraints"
            else:
                self.memory.set("stage", "inputs")
                return {
                    "type": "follow_up",
                    "stage": "inputs",
                    "tool_name": tool_name,
                    "question": self._compose_inputs_page(tool_name, merged_slots),
                }

        # === 單一輸入階段：吃變數直到下一步 ===
        if stage == "inputs":
            if force_direct:
                self.memory.set("stage", "constraints")
                self.memory.set("filled_slots", merged_slots)
            elif _is_next_local(user_msg) or self._wants_skip(user_msg):
                self.memory.set("stage", "constraints")
                self.memory.set("filled_slots", merged_slots)
            else:
                self.memory.set("filled_slots", merged_slots)
                return {
                    "type": "follow_up",
                    "stage": "inputs",
                    "tool_name": tool_name,
                    "question": self._compose_inputs_page(tool_name, merged_slots),
                }

        # === 進入加條件/檢缺流程（單入口工具） ===
        self._apply_nl_parser(tool_name, tool_meta, user_msg, merged_slots, allowed_fields)
        self._coerce_rows(merged_slots)

        sig_info = self._get_sig_info(tool_meta, op=None)
        bool_fields = sig_info["bool_fields"]
        self._coerce_boolean_fields(merged_slots, bool_fields)

        # 參數齊全 → 建立 tool_request
        self.memory.set("filled_slots", {})
        self.memory.set("pending_tool", None)
        self.memory.set("pending_missing", None)

        user_params = {k: v for k, v in merged_slots.items() if k in allowed_fields}
        self._coerce_boolean_fields(user_params, bool_fields)
        request_payload = {
            "tool_name": tool_name,
            "user_params": user_params,
            "raw_query": user_msg,
            "timestamp": datetime.utcnow().isoformat(),
            "__prev_tax__": None,
        }

        try:
            prev_tax_mem = self.memory.get("__prev_tax__")
            if isinstance(prev_tax_mem, (int, float)):
                request_payload["__prev_tax__"] = float(prev_tax_mem)
        except Exception:
            pass

        # 先推斷/取得 mode（若沒有就 None）
        op_text = self._infer_mode(user_msg)          # 使用者在文字有沒有說「最大化/最小化」
        op = op_text or self.memory.get("op")         # 沒說就用記憶中的

        # 若工具有 budget_field 且收到正的上限值，且使用者沒有明說「最小化」→ 預設最大化
        budget_val = merged_slots.get(budget_field) if budget_field else None
        if (not op_text or op_text != "minimize") and isinstance(budget_val, (int, float)) and budget_val > 0:
            op = "maximize"

        # 使用者要求『直接計算』：讓 ConstraintAgent 直接進入執行
        if force_direct:
            request_payload["__direct_execute__"] = True

        self.memory.set("op", op)


        early_md = await self._early_condition_tips(tool_name, merged_slots, op, user_msg)
        if early_md:
            request_payload["early_tips_md"] = early_md

        ltm = LTR_MEM.load()
        usage_count = ltm.get("tool_usage", {}).get(tool_name, 0) + 1
        ltm.setdefault("tool_usage", {})[tool_name] = usage_count
        LTR_MEM.save(ltm)
        return {"type": "tool_request", "payload": request_payload}
    
# ---------------------------------------------------------------------------
# ConstraintAgent (refactored layout)
# ---------------------------------------------------------------------------
class ConstraintAgent(BaseAgent):
    """
    第二層：接收 Caller 的 tool_request，
    追問是否要加 free_vars / constraints（支援自然語言 / JSON / 簡易 DSL），
    並把兩者寫回 payload["user_params"]。
    """

    # === A. 常數與正則 ===
    _CN_NUM = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    _RHS_VAR_OP_NUM = re.compile(r"^\s*(?P<var>[A-Za-z_][A-Za-z0-9_\.]*)\s*(?P<op>[*/])\s*(?P<num>-?\d+(?:\.\d+)?)\s*$")
    _RHS_NUM_OP_VAR = re.compile(r"^\s*(?P<num>-?\d+(?:\.\d+)?)\s*(?P<op>[*/])\s*(?P<var>[A-Za-z_][A-Za-z0-9_\.]*)\s*$")

    def _parse_linear_rhs(self, s: str) -> Optional[str]:
        """
        接受:
        - var * num  -> 'var * num'
        - var / num  -> 'var / num'
        - num * var  -> 轉成 'var * num'
        拒絕:
        - num / var  -> 不是線性比例，返回 None
        也要允許 row 變數: row\d+\.[A-Za-z_]\w*
        """
        s = (s or "").strip()
        # 先允許 row 變數
        _row_or_var = r"(?:row\d+\.[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*)"
        m1 = re.match(rf"^\s*(?P<var>{_row_or_var})\s*(?P<op>[*/])\s*(?P<num>-?\d+(?:\.\d+)?)\s*$", s)
        if m1:
            var, op, num = m1.group("var"), m1.group("op"), m1.group("num")
            return f"{var} {op} {float(num)}".rstrip("0").rstrip(".") if "." in num else f"{var} {op} {num}"

        m2 = re.match(rf"^\s*(?P<num>-?\d+(?:\.\d+)?)\s*(?P<op>[*/])\s*(?P<var>{_row_or_var})\s*$", s)
        if m2:
            num, op, var = m2.group("num"), m2.group("op"), m2.group("var")
            if op == "*":  # 2 * var -> var * 2
                return f"{var} * {float(num)}".rstrip("0").rstrip(".") if "." in num else f"{var} * {num}"
            else:
                # 'num / var' 非線性，拒絕
                return None
        return None


    # === B. 通用工具 ===
    def _to_number(self, s: str) -> Optional[float]:
        """共用：支援百/千/萬/億/兆與複合片語；此處不允許百分比。"""
        return parse_cn_amount(s, allow_percent=False)

    @staticmethod
    def _to_list(csv_or_list):
        if not csv_or_list:
            return []
        if isinstance(csv_or_list, list):
            return [str(x).strip() for x in csv_or_list if str(x).strip()]
        return [t.strip() for t in str(csv_or_list).split(",") if t.strip()]

    def _json_loads_with_merge(self, s: str) -> dict | None:
        """以 object_pairs_hook 解析 JSON，**合併重複鍵**（dict 做淺合併，否則後者覆蓋）。"""
        try:
            def _hook(pairs):
                d = {}
                for k, v in pairs:
                    if k not in d:
                        d[k] = v
                    else:
                        if isinstance(d[k], dict) and isinstance(v, dict):
                            d[k] = {**d[k], **v}
                        else:
                            d[k] = v
                return d
            return json.loads(s, object_pairs_hook=_hook)
        except Exception:
            try:
                return json.loads(s)
            except Exception:
                return None

    def _merge_fv(self, acc: List[str], new_fv):
        if not new_fv:
            return
        if isinstance(new_fv, str):
            new = [t.strip() for t in new_fv.split(",") if t.strip()]
        else:
            new = [str(t).strip() for t in new_fv if str(t).strip()]
        for t in new:
            if t not in acc:
                acc.append(t)

    def _merge_cons(self, acc: Dict[str, Any], new_cons: Dict[str, Any] | None):
        if not isinstance(new_cons, dict):
            return
        for k, v in new_cons.items():
            if not isinstance(v, dict) or not v:
                continue
            dst = acc.setdefault(k, {})
            for op, val in v.items():
                dst[op] = val

    # === C. 參數與條件「中文預覽」 ===
    def _fmt_params_preview(self, tool_name: str, payload: Dict[str, Any],
                            *, mask_free_and_constrained: bool = False) -> str:
        """顯示：稅種＋目前系統接收到的參數（含摘要）"""
        meta = TOOL_MAP.get(tool_name, {}) or {}
        labels: Dict[str, str] = meta.get("field_labels", {}) or {}
        row_fields = list(meta.get("row_fields", []) or [])
        desc = meta.get("description", tool_name)

        def _lab(k: str) -> str:
            s = labels.get(k, k)
            if not isinstance(s, str):
                return k
            s = re.sub(r"[（(].*?[）)]", "", s).replace("？", "").strip()
            return s or k

        def _fmt_money(x):
            try:
                f = float(x)
                return f"{f:,.0f}" if f.is_integer() else f"{f:,}"
            except Exception:
                return str(x)

        user_params = (payload.get("user_params") or {}) if isinstance(payload, dict) else {}

        # ——— Stage 3 專用：若變數已在 free_vars 或出現在約束中，遮蔽為「由求解器決定」 ———
        free_set: set[str] = set()
        cons_set: set[str] = set()
        if mask_free_and_constrained:
            fv_raw = user_params.get("free_vars") or []
            if isinstance(fv_raw, list):
                for t in fv_raw:
                    try:
                        s = str(t).strip()
                        if not s:
                            continue
                        # 只排除真正的 row 變數（例如 row1.quantity），其他像 cigar_new.quantity 要保留
                        if re.match(r"^row\d+\.", s):
                            continue
                        free_set.add(s)
                    except Exception:
                        pass
            cons_raw = user_params.get("constraints") or {}
            basics_keys = {k for k in user_params.keys() if k not in ("rows","constraints","free_vars")}
            if isinstance(cons_raw, dict) and basics_keys:
                try:
                    alt = "|".join(map(re.escape, basics_keys))
                    rx = re.compile(rf"\b({alt})\b")
                    for lhs in cons_raw.keys():
                        if not isinstance(lhs, str): continue
                        for m in rx.finditer(lhs):
                            cons_set.add(m.group(1))
                except Exception:
                    pass

        lines = [f"### Tax type:{desc}", "**Current variables(overview)**"]
        basics = [(k, v) for k, v in user_params.items() if k != "rows" and k not in ("free_vars", "constraints")]
        for k, v in basics:
            if v in (None, ""):
                continue
            vv = ("yes" if v else "no") if isinstance(v, bool) else ("Determined by the solver（free）" if k in free_set else ("Determined by the solver（constrained）" if k in cons_set else _fmt_money(v)))
 
            lines.append(f"- {_lab(k)}：{vv}（`{k}`）")
        return "\n".join(lines)

    def _fmt_constraints_preview_zh(self, tool_name: str, payload: Dict[str, Any]) -> str:
        """
        以中文顯示 free_vars 與 constraints（含 slug.field 與 'slug field' 兩種寫法、以及 row{i}.field）：
        - 例如：cement_white quantity + cement_portland_I quantity >= 1200
            → 白水泥- 數量 + 卜特蘭I型水泥- 數量 >= 1200
        - row 變數會顯示成「第 N 筆 的『欄位』」或用品項名稱別名。
        """
        meta = TOOL_MAP.get(tool_name, {}) or {}
        labels: Dict[str, str] = meta.get("field_labels", {}) or {}
        user_params = (payload.get("user_params") or {}) if isinstance(payload, dict) else {}
        rows = user_params.get("rows") or []

        # 蒐集 row 別名（優先：payload.row_aliases / user_params.row_aliases → 其次：rows 名稱）
        row_aliases: Dict[str, str] = {}
        if isinstance(payload.get("row_aliases"), dict):
            for k, v in payload["row_aliases"].items():
                if isinstance(k, str) and isinstance(v, str) and v.startswith("row"):
                    row_aliases[v] = k
        if isinstance(user_params.get("row_aliases"), dict):
            for k, v in user_params["row_aliases"].items():
                if isinstance(k, str) and isinstance(v, str) and v.startswith("row"):
                    row_aliases[v] = k
        if isinstance(rows, list):
            name_keys = ("main_name", "sub_name", "tax_item", "name", "item", "category")
            for i, r in enumerate(rows):
                if not isinstance(r, dict):
                    continue
                nm = None
                for nk in name_keys:
                    v = r.get(nk)
                    if isinstance(v, str) and v.strip():
                        nm = v.strip()
                        break
                row_aliases.setdefault(f"row{i}", nm or f"第 {i+1} 筆")

        def _lab(k: str) -> str:
            s = labels.get(k, k)
            if not isinstance(s, str):
                return k
            s = re.sub(r"[（(].*?[）)]", "", s).replace("？", "").strip()
            return s or k

        def _fmt_num(x) -> str:
            if isinstance(x, (int, float)):
                return str(int(x)) if (not isinstance(x, float) or abs(x - int(x)) <= 1e-9) else str(x)
            try:
                xs = str(x).strip()
                if re.match(r"^[0-9]+(?:\.[0-9]+)?$", xs.replace(",", "")):
                    return xs[:-2] if xs.endswith(".0") else xs
            except Exception:
                pass
            return str(x)
        
        def _humanize_slug_field(expr: str) -> str:
            fields = sorted(field_set_for_tool(tool_name), key=len, reverse=True)
            if not fields:
                return str(expr or "")

            fld_alt = "|".join(map(re.escape, fields))
            token_re = re.compile(
                rf"(?P<slug>[A-Za-z0-9_]+)\.(?P<field>{fld_alt})|(?P<slug2>[A-Za-z0-9_]+)\s+(?P<field2>{fld_alt})"
            )

            def _repl(m):
                if m.group("slug") and m.group("field"):
                    key = f"{m.group('slug')}.{m.group('field')}"
                else:
                    key = f"{m.group('slug2')}.{m.group('field2')}"
                return labels.get(key, key)

            s = token_re.sub(_repl, str(expr or ""))

            # ② 也把「裸變數鍵」換成中文（例如 salary_self → 本人薪資所得）
            try:
                single_vars = [k for k in labels.keys()
                               if isinstance(k, str)
                               and "." not in k
                               and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k)]
                if single_vars:
                    sv_alt = "|".join(map(re.escape, single_vars))
                    bare_re = re.compile(rf"\b({sv_alt})\b")
                    s = bare_re.sub(lambda m: labels.get(m.group(1), m.group(1)), s)
            except Exception:
                pass

            return s

        def _fmt_expr(expr: str) -> str:
            # 先把 slug/field 變中文，再處理 row 與運算子
            s = _humanize_slug_field(expr)
            return s

        # 自由變數
        fv_raw = user_params.get("free_vars") or []
        fv_tokens = [t.strip() for t in (fv_raw if isinstance(fv_raw, list) else str(fv_raw).split(",")) if str(t).strip()]

        def _fmt_free_var(tok: str) -> str:
            tok = tok.strip()
            # 其餘（含 slug.field 或 "slug field"）直接走人性化
            return _humanize_slug_field(tok)

        fv_cn = "、".join([_fmt_free_var(t) for t in fv_tokens]) if fv_tokens else "None"

        # 條件清單
        cons = user_params.get("constraints") or {}
        lines: List[str] = []
        lines.append("#### Current constraints preview")
        lines.append(f"free variable(s):{fv_cn}")

        cond_lines: List[str] = []
        if isinstance(cons, dict) and cons:
            OP_ORDER = ["==", ">=", "<=", ">", "<"]
            for lhs in sorted(cons.keys(), key=lambda x: str(x)):
                ops = cons.get(lhs) or {}
                lhs_cn = _fmt_expr(lhs)
                for op in OP_ORDER:
                    if op in ops:
                        vals = ops[op] if isinstance(ops[op], (list, tuple)) else [ops[op]]
                        for rhs in vals:
                            rhs_cn = _fmt_expr(rhs) if isinstance(rhs, str) else _fmt_num(rhs)
                            cond_lines.append(f"{lhs_cn} {'=' if op == '==' else op} {rhs_cn}")
        if cond_lines:
            lines.append("Constraints:")
            lines += [f"{i}. {s}" for i, s in enumerate(cond_lines, 1)]
        else:
            lines.append("Constraints: none")
        return "\n".join(lines)

    # === D. 欄位/別名解析 ===
    def _zhnum_to_int(self, s: str) -> Optional[int]:
        s = s.strip()
        if not s:
            return None
        if s in self._CN_NUM:
            return self._CN_NUM[s]
        if "十" in s:
            parts = s.split("十")
            a = self._CN_NUM.get(parts[0], 1 if parts[0] == "" else None)
            b = self._CN_NUM.get(parts[1], 0 if len(parts) == 1 or parts[1] == "" else None)
            if a is None or b is None:
                return None
            return a * 10 + b
        val = 0
        for ch in s:
            if ch not in self._CN_NUM:
                return None
            val = val * 10 + self._CN_NUM[ch]
        return val
    

    # === E. 條件調校建議產生器 ===
    async def _condition_refiner_tips(self, tool_name: str, payload: Dict[str, Any]) -> str:
        """
        依『目前條件 + 自由變數 + 模式/上限』回傳 3 條可操作的中文建議（Markdown）。
        - 僅使用本地上下文，不依賴外部服務。
        - 輸出格式： "#### 條件調校建議\n- ...\n- ...\n- ..."
        """
        meta = TOOL_MAP.get(tool_name, {}) or {}
        labels = meta.get("field_labels", {}) or {}
        required = meta.get("optimizable_fields", []) or []
        constraint_fields = meta.get("constraint_fields", []) or []
        budget_field = meta.get("budget_field")
        row_fields: List[str] = list(meta.get("row_fields") or [])

        allowed = sorted(set(required) | set(constraint_fields) | ({budget_field} if budget_field else set()))
        user_params = (payload.get("user_params") or {}) if isinstance(payload, dict) else {}
        free_vars = user_params.get("free_vars") or []
        constraints = user_params.get("constraints") or {}
        rows = user_params.get("rows") or []
        mode = (payload.get("op") or self.memory.get("op") or "").lower()
        budget_val = user_params.get(budget_field) if budget_field else None

        # rows → 展開 allowed 的 row{i}.<field>
        nrows = len(rows) if isinstance(rows, list) else 0
        if nrows and row_fields:
            allowed += [f"row{i}.{f}" for i in range(nrows) for f in row_fields]

        # 供模型參考的「條件中文預覽」
        cons_preview_md = self._fmt_constraints_preview_zh(tool_name, payload)

        # 系統提示
        sys_prompt = (
            "You are a **Tax Optimization Constraint Tuning Advisor**. Based on the current free variables and constraints, generate **3 actionable, practical** next-step suggestions. Avoid fluff. Each suggestion must be **≤ 50 characters**.\n"
            "Rules:\n"
            "1) Only suggest **how to adjust constraints / free variables / upper bounds**; do not repeat summaries.\n"
            "2) Prioritize pointing out: whether equalities are too tight, whether a tax cap is missing (for maximization), whether there are too few free variables, whether any variables are stuck at bounds and should be loosened, and whether any currently fixed fields should be added to free_vars.\n"
            "3) Do not use any new field names that are not in allowed_vars.\n"
            "4) If mentioning fields or variables, **prefer using the English names in field_labels**.\n"
            "5) Output **JSON only**: {\"tips\":[\"…\",\"…\",\"…\",\"…\"]}"
        )

        # 使用者內容（上下文）
        ctx = {
            "tool_name": tool_name,
            "mode": mode,
            "budget_field": budget_field,
            "budget_val": budget_val,
            "allowed_vars": allowed,
            "free_vars": free_vars if isinstance(free_vars, list) else [],
            "constraints": constraints if isinstance(constraints, dict) else {},
            "constraints_preview_md": cons_preview_md,
            "field_labels": labels,
            "row_fields": row_fields,
            "row_count": nrows,
        }
        import json as _json
        user_prompt = _json.dumps(ctx, ensure_ascii=False)

        # 先跑 LLM 取得 3 條建議；失敗則回退到啟發式。
        tips: List[str] = []
        try:
            raw = await self._chat(sys_prompt, user_prompt, temperature=0.2)
            obj = _json.loads(raw) if isinstance(raw, str) else {}
            arr = obj.get("tips")
            if isinstance(arr, list):
                tips = [str(s).strip() for s in arr if str(s).strip()]
        except Exception:
            tips = []

        # 簡單啟發式後備（確保至少 3 條）
        def _fmt_money(x):
            try: 
                f = float(x)
                return f"{int(f):,}" if abs(f-int(f))<1e-9 else f"{f:,}"
            except Exception:
                return str(x)

        if len(tips) < 3:
            eq_count = 0
            try:
                for ops in (constraints or {}).values():
                    if isinstance(ops, dict) and "==" in ops:
                        eq_count += 1
            except Exception:
                pass

            if mode == "maximize" and budget_field and not budget_val:
                tips.append(f"It is recommended to maximize the tax amount cap.（`{budget_field}`),such as 5,000,000.")
            if not free_vars or (isinstance(free_vars, list) and len(free_vars) < 1):
                # 優先建議能動的欄位（如 rows.quantity / 單價 等）
                cand = [v for v in allowed if v.endswith(".quantity") or v.endswith(".price") or v.endswith(".assessed_price")]
                demo = cand[:2] or allowed[:2]
                if demo:
                    tips.append(f"Allow adjustable fields (free_vars) to be opened, such as:{', '.join(demo)}。")
            if eq_count >= 2:
                tips.append("There are too many equality constraints. It is recommended to change them to ranges (≥/≤) to avoid having no solution or being stuck at the boundary.")
            if isinstance(budget_val, (int, float)):
                tips.append(f"The tax limit is{ _fmt_money(budget_val) },this could be changed to increasing the upper limit or widening the field that is aligned with the boundary.")
            if not tips:
                tips.append("Add proportions or upper and lower bounds to avoid using only equations; allow for reconfiguration of the model.")

        tips = tips[:3] if len(tips) >= 3 else (
            tips + ["More fields can be opened as free_vars. ", "Change the equation to a range to increase feasible solutions.", "Set a tax cap to maximize the potential."]
        )[:3]

        # ===  將 tips 中出現的欄位 key 改成中文 label ===
        def _clean_label_text(lbl: str) -> str:
            """去掉括號與問號，做成短版中文標籤。"""
            if not isinstance(lbl, str):
                return str(lbl)
            s = re.sub(r"[（(].*?[）)]", "", lbl)
            s = s.replace("？", "").strip()
            return s or lbl

        def _apply_labels_to_text(text: str) -> str:
            if not labels:
                return text
            s = str(text)
            try:
                # 依 key 長度由長到短替換，避免子字串彼此吃掉
                for key in sorted(labels.keys(), key=len, reverse=True):
                    if not isinstance(key, str):
                        continue
                    lbl = labels.get(key)
                    if not isinstance(lbl, str) or not lbl.strip():
                        continue
                    # 僅在 key 前後不是英數底線時替換，避免誤傷
                    pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])")
                    s = pat.sub(_clean_label_text(lbl), s)
            except Exception:
                pass
            return s

        tips = [_apply_labels_to_text(t) for t in tips]

        md = "#### Condition adjustment suggestions\n" + "\n".join(f"- {t}" for t in tips)
        return md


    def _resolve_field(self, tool_name: str, zh: str, payload: Dict[str, Any]) -> Optional[str]:
        zh = (zh or "").strip()
        if not zh:
            return None
        meta = TOOL_MAP.get(tool_name, {}) or {}
        alias_map: Dict[str, List[str]] = meta.get("alias", {}) or {}
        labels: Dict[str, str] = meta.get("field_labels", {}) or {}
        for key, phrases in alias_map.items():
            for p in (phrases or []):
                p = str(p).strip()
                if p and (zh == p or p in zh):
                    return key
        for key, lbl in labels.items():
            if not isinstance(lbl, str):
                continue
            tokens = [t for t in re.findall(r"[\u4e00-\u9fff]+", lbl) if t.strip()]
            for t in tokens:
                if t and t in zh:
                    return key
        return None

    def _alias_to_var(self, tool_name: str, phrase: str) -> Optional[str]:
        phrase = (phrase or "").strip()
        if not phrase:
            return None
        phrase = re.sub(r"[（(].*?[）)]", "", phrase).strip()

        meta = TOOL_MAP.get(tool_name, {}) or {}
        labels: Dict[str, str] = meta.get("field_labels", {}) or {}
        alias_map: Dict[str, List[str]] = meta.get("alias", {}) or {}

        rev: Dict[str, str] = {}
        for k, v in labels.items():
            if isinstance(v, str):
                for t in [t for t in re.findall(r"[\u4e00-\u9fff]+", v) if t.strip()]:
                    rev[t] = k
        for k, arr in alias_map.items():
            for a in (arr or []):
                a = str(a).strip()
                if a:
                    rev[a] = k

        if phrase in rev:
            return rev[phrase]
        for zh, key in rev.items():
            if zh and zh in phrase:
                return key
        return None

    # === E. 條件解析器（JSON/DSL、單變數界、等式比例、不等式比例、LLM） ===
    def _parse_reply_json_or_dsl(self, text: str, user_params: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        s = (text or "").strip()
        if not s:
            return None, None
        if any(k in s.lower() for k in ["next","none", "no"]):
            return None, None

        if s.lstrip().startswith("{"):
            try:
                obj = json.loads(s)
                fv = obj.get("free_vars")
                con = obj.get("constraints") or obj.get("constraints_json")
                if isinstance(fv, list):
                    fv = ",".join(str(x).strip() for x in fv)
                return (fv if fv else None), (con if isinstance(con, dict) else None)
            except Exception:
                pass

        fv = None
        cons: Dict[str, Any] = {}
        parts = [p.strip() for p in re.split(r"[;\n]+", s) if p.strip()]
        for p in parts:
            if p.lower().startswith("free"):
                m = re.search(r"free\s*[:=：]\s*(.+)$", p, flags=re.I)
                if m:
                    fv = ",".join([t.strip() for t in re.split(r"[,，\s]+", m.group(1).strip()) if t.strip()])
            elif p.lower().startswith("constraints") or any(ch in p for ch in ["<", ">", "=", "≥", "≤"]):
                body = re.sub(r"^\s*constraints\s*[:=：]\s*", "", p, flags=re.I)
                items = [x.strip() for x in re.split(r"[,，]+", body) if x.strip()]
                for it in items:
                    m2 = re.match(r"([A-Za-z0-9_\.]+)\s*(<=|>=|=|<|>)\s*([0-9][0-9,\.萬億]*)", it)
                    if not m2:
                        continue
                    var, op, num = m2.groups()
                    val = self._to_number(num)
                    if val is None:
                        continue
                    cons.setdefault(var, {})["==" if op == "=" else op] = val

        if fv:
            valid_keys = set(user_params.keys())
            keep = []
            tokens = [t.strip() for t in (fv if isinstance(fv, list) else fv.split(",")) if t.strip()]
            for v in tokens:
                if "." in v or v in valid_keys:
                    keep.append(v)
                else:
                    _append_debug(self.memory, f"[ConstraintAgent] WARN free_var '{v}' not in payload")
            fv = keep or None

        return fv, (cons or None)

    def _parse_single_bound_generic(self, tool_name: str, text: str, payload: Dict[str, Any]):
        """泛用：『第N筆/品項名』+ 欄位 + 至多/至少 + 數字"""
        meta = TOOL_MAP.get(tool_name, {}) or {}
        row_fields: List[str] = list(meta.get("row_fields") or [])
        if not row_fields:
            return None, None

        user_params = payload.get("user_params", {}) or {}

        labels: Dict[str, str] = meta.get("field_labels", {}) or {}
        alias_map: Dict[str, List[str]] = meta.get("alias", {}) or {}

        def _field_key_from_phrase(ph: str) -> Optional[str]:
            ph = (ph or "").strip("「」『』").strip()
            if not ph:
                return None
            for k, arr in alias_map.items():
                if k in row_fields:
                    for a in (arr or []):
                        a = (a or "").strip()
                        if a and (a in ph):
                            return k
            for k, lab in labels.items():
                if k in row_fields and isinstance(lab, str) and lab:
                    toks = re.findall(r"[\u4e00-\u9fff]+", lab)
                    if any(t and t in ph for t in toks):
                        return k
            return None

        op_map = {
            "no more than": "<=",
            "must not exceed": "<=",
            "at most": "<=",
            "not higher than": "<=",
            "≦": "<=",
            "≤": "<=",
            "<=": "<=",

            "at least": ">=",
            "not lower than": ">=",
            "no less than": ">=",
            "≧": ">=",
            "≥": ">=",
            ">=": ">="
        }

        rx = re.compile(
            r"(?:item\s*(?P<idx>\d+)|(?P<name>[A-Za-z0-9_()]+))\s*"
            r"(?:'s)?\s*(?P<field>\"[^\"]+\"|'[^']+'|[A-Za-z0-9_]+)?\s*"
            r"(?P<op>no more than|must not exceed|at most|not higher than|≦|≤|<=|"
            r"at least|not lower than|no less than|≧|≥|>=)\s*"
            r"(?P<num>[0-9][0-9,\.KMBT]*)"
        )
        s = (text or "").strip()
        m = rx.search(s)
        if not m:
            return None, None

        idx_s = m.group("idx")
        name = m.group("name") or ""
        field_ph = (m.group("field") or "").strip()
        op_kw = m.group("op")
        num_s = m.group("num")

        field_key = _field_key_from_phrase(field_ph) or (row_fields[0] if row_fields else None)
        if not field_key:
            return None, None

        op = op_map.get(op_kw)
        if op not in {"<=", ">="}:
            return None, None
        val = self._to_number(num_s)
        if val is None:
            return None, None

    def _parse_cn_ratio(self, tool_name: str, text: str, payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        s = (text or "").strip()
        m = re.search(r"([\u4e00-\u9fa5A-Za-z0-9_（）()]+)\s*(?:為|是|等於|=)\s*([\u4e00-\u9fa5A-Za-z0-9_（）()]+)\s*的\s*([\u4e00-\u9fa5A-Za-z0-9_]+)", s)
        if not m:
            return None, None
        lhs_zh, rhs_zh, frac_zh = m.groups()

        def _sanitize(p: str) -> str:
            return re.sub(r"[（(].*?[）)]", "", p).strip()

        lhs_zh = _sanitize(lhs_zh); rhs_zh = _sanitize(rhs_zh)
        fm = re.match(r"([\u4e00-\u9fa5\d]+)分之([\u4e00-\u9fa5\d]+)", frac_zh)

        def parse_int(token: str) -> Optional[int]:
            return int(token) if token.isdigit() else self._zhnum_to_int(token)

        lhs = self._resolve_field(tool_name, lhs_zh, payload)
        rhs = self._resolve_field(tool_name, rhs_zh, payload)
        if not lhs or not rhs:
            return None, None

        rhs_expr = None
        if fm:
            den_zh, num_zh = fm.groups()
            den = parse_int(den_zh); num = parse_int(num_zh)
            if not den or num is None:
                return None, None
            rhs_expr = f"{rhs} / {den}" if num == 1 else f"{rhs} * {num/den}"
        else:
            m_pct = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*$", frac_zh)
            if m_pct:
                r = float(m_pct.group(1)) / 100.0
                rhs_expr = f"{rhs} * {r}"
            else:
                specials = {"一半": (1, 2), "二分之一": (1, 2), "三分之一": (1, 3), "四分之一": (1, 4),
                            "五分之一": (1, 5), "六分之一": (1, 6), "七分之一": (1, 7), "八分之一": (1, 8), "九分之一": (1, 9),
                            "三成": (3, 10), "四成": (4, 10), "五成": (5, 10), "六成": (6, 10), "七成": (7, 10), "八成": (8, 10), "九成": (9, 10)}
                if frac_zh in specials:
                    num, den = specials[frac_zh]
                    rhs_expr = f"{rhs} / {den}" if num == 1 else f"{rhs} * {num/den}"
        if not rhs_expr:
            return None, None

        cons = {lhs: {"==": rhs_expr}}
        fv_list = list(dict.fromkeys([lhs, rhs]))
        return ",".join(fv_list), cons

    def _parse_ratio_constraint(
        self,
        tool_name: str,
        text: str
    ) -> Tuple[Optional[str], Optional[Dict[str, Dict[str, Any]]]]:
        """
        Parse equality ratio constraints from English text, e.g.
        - "A equals B's 30%"
        - "A is B's 1/3"
        - "A equals B's half"
        - "A equals B's 70%"  -> A == B * 0.7

        Returns:
        (free_vars_csv, constraints_dict) or (None, None)
        """
        s = (text or "").strip()
        if not s:
            return None, None

        # Examples matched:
        #   "<lhs> (is|equals|=) <rhs>'s <ratio>"
        m = re.search(
            r"(?P<lhs>[A-Za-z0-9_()]+?)\s*(?:is|equals|equal to|=|==)\s*"
            r"(?P<rhs>[A-Za-z0-9_()]+?)\s*(?:'s)?\s*"
            r"(?P<ratio>[^,.;\s]+)",
            s,
            flags=re.IGNORECASE,
        )
        if not m:
            return None, None

        lhs_raw = re.sub(r"[（(].*?[）)]", "", m.group("lhs")).strip()
        rhs_raw = re.sub(r"[（(].*?[）)]", "", m.group("rhs")).strip()
        ratio = (m.group("ratio") or "").strip().lower()

        lhs = self._alias_to_var(tool_name, lhs_raw)
        rhs = self._alias_to_var(tool_name, rhs_raw)
        if not lhs or not rhs:
            return None, None

        # 1) Fraction: "a/b"
        m_frac = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", ratio)
        if m_frac:
            num = int(m_frac.group(1))
            den = int(m_frac.group(2))
            if den == 0:
                return None, None
            rhs_expr = f"{rhs} / {den}" if num == 1 else f"{rhs} * {num/den}"
            return f"{lhs},{rhs}", {lhs: {"==": rhs_expr}}

        # 2) Percent: "12.5%"
        m_pct = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*$", ratio)
        if m_pct:
            r = float(m_pct.group(1)) / 100.0
            return f"{lhs},{rhs}", {lhs: {"==": f"{rhs} * {r}"}}

        # 3) Words: "half", "third", etc. (minimal set)
        specials = {
            "half": (1, 2),
            "one-half": (1, 2),
            "1/2": (1, 2),
            "one-third": (1, 3),
            "third": (1, 3),
            "one-quarter": (1, 4),
            "quarter": (1, 4),
        }
        if ratio in specials:
            num, den = specials[ratio]
            rhs_expr = f"{rhs} / {den}" if num == 1 else f"{rhs} * {num/den}"
            return f"{lhs},{rhs}", {lhs: {"==": rhs_expr}}

        return None, None


    def _parse_inequality_ratio_constraint(
        self,
        tool_name: str,
        text: str
    ) -> Tuple[Optional[str], Optional[Dict[str, Dict[str, Any]]]]:
        """
        Parse inequality/equality ratio constraints from English text, e.g.
        - "A no more than B's 30%"   -> A <= B * 0.3
        - "A at least B's 1/3"       -> A >= B * (1/3)
        - "A equals B's half"        -> A == B * 0.5

        Returns:
        (free_vars_csv, constraints_dict) or (None, None)
        """
        s = (text or "").strip()
        if not s:
            return None, None

        op_map = {
            "no more than": "<=",
            "at most": "<=",
            "not more than": "<=",
            "must not exceed": "<=",
            "not higher than": "<=",
            "<=": "<=",
            "≤": "<=",

            "at least": ">=",
            "no less than": ">=",
            "not less than": ">=",
            "not lower than": ">=",
            ">=": ">=",
            "≥": ">=",

            "equals": "==",
            "equal to": "==",
            "is": "==",
            "=": "==",
            "==": "==",
        }

        # Capture multi-word ops by ordering longer phrases first.
        op_pattern = (
            r"no more than|not more than|must not exceed|at most|not higher than|<=|≤|"
            r"no less than|not less than|not lower than|at least|>=|≥|"
            r"equal to|equals|==|=|is"
        )

        m = re.search(
            rf"(?P<lhs>[A-Za-z0-9_()]+?)\s*"
            rf"(?P<op>{op_pattern})\s*"
            rf"(?P<rhs>[A-Za-z0-9_()]+?)\s*(?:'s)?\s*"
            rf"(?P<ratio>"
            rf"[\d\.]+\s*%|"
            rf"[\d]+\s*/\s*[\d]+|"
            rf"half|third|quarter|one-half|one-third|one-quarter"
            rf")",
            s,
            flags=re.IGNORECASE,
        )
        if not m:
            return None, None

        lhs_raw = re.sub(r"[（(].*?[）)]", "", m.group("lhs")).strip()
        rhs_raw = re.sub(r"[（(].*?[）)]", "", m.group("rhs")).strip()
        op_kw = (m.group("op") or "").strip().lower()
        ratio = (m.group("ratio") or "").strip().lower()

        lhs = self._alias_to_var(tool_name, lhs_raw)
        rhs = self._alias_to_var(tool_name, rhs_raw)
        if not lhs or not rhs:
            return None, None

        # Build RHS expression for the ratio
        m_pct = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*$", ratio)
        if m_pct:
            r = float(m_pct.group(1)) / 100.0
            rhs_expr = f"{rhs} * {r}"
        else:
            m_frac = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", ratio)
            if m_frac:
                num = int(m_frac.group(1))
                den = int(m_frac.group(2))
                if den == 0:
                    return None, None
                rhs_expr = f"{rhs} / {den}" if num == 1 else f"{rhs} * {num/den}"
            else:
                specials = {
                    "half": (1, 2),
                    "one-half": (1, 2),
                    "third": (1, 3),
                    "one-third": (1, 3),
                    "quarter": (1, 4),
                    "one-quarter": (1, 4),
                }
                if ratio in specials:
                    num, den = specials[ratio]
                    rhs_expr = f"{rhs} / {den}" if num == 1 else f"{rhs} * {num/den}"
                else:
                    return None, None

        op = op_map.get(op_kw)
        if op not in {"<=", ">=", "=="}:
            return None, None

        return f"{lhs},{rhs}", {lhs: {op: rhs_expr}}


    def _vars_in_expr(self, expr: str, allowed: set[str]) -> List[str]:
        """在運算式中擷取有效變數（包含 row 變數）。"""
        s = str(expr or ""); seen, out = set(), []
        tok_pat = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]*")
        for t in tok_pat.findall(s):
            if t.startswith("row"):
                continue
            if t in allowed and t not in seen:
                seen.add(t); out.append(t)
        return out

    async def _nl_to_constraints(self, tool_name: str, user_text: str, payload: Dict[str, Any]) -> Tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
        meta = TOOL_MAP.get(tool_name, {}) or {}
        labels: Dict[str, str] = meta.get("field_labels", {}) or {}
        required = meta.get("optimizable_fields", []) or []
        constraint_fields = meta.get("constraint_fields", []) or []
        budget_field = meta.get("budget_field")
        base_allowed_vars = list(set(required) | set(constraint_fields) | ({budget_field} if budget_field else set()))
        alias = meta.get("alias", {}) or {}
        user_params = payload.get("user_params", {}) or {}
        extra_aliases = {}
        if isinstance(payload.get("row_aliases"), dict):
            extra_aliases.update(payload["row_aliases"])
        if isinstance(user_params.get("row_aliases"), dict):
            extra_aliases.update(user_params["row_aliases"])
        allowed_vars_for_prompt = base_allowed_vars
        sys_prompt = (
            "You are a 'Tax Constraint Compiler'. Your task is to convert the user's tax-related description into **strict JSON** (with `free_vars` and `constraints`).\n"
            f"1) You may ONLY use the following field keys: {', '.join(allowed_vars_for_prompt)}.\n"
            '2) Allowed operators: "==", ">=", "<=", ">", "<".\n'
            '3) Convert written numbers into Arabic numerals.\n'
            '4) You may use compound expressions (e.g., a + b).\n'
            '5) If you use an expression like a+b, `free_vars` must include every variable involved.\n'
            '6) If the same variable has multiple operators, merge them under the same key.\n'
            '7) Output EXACTLY one JSON object and nothing else.\n'
        )
        mapping_doc: Dict[str, Any] = {"field_labels": labels, "aliases": alias}
       
        examples = [
            {"in": "The total of my salary and my wife's salary must be 3,000,000",
            "out": {"free_vars": ["salary_self", "salary_spouse"],
                    "constraints": {"salary_self + salary_spouse": {"==": 3000000}}}},
            {"in": "Itemized deduction must not exceed 300,000",
            "out": {"free_vars": [], "constraints": {"itemized_deduction": {"<=": 300000}}}},
            {"in": "House transaction gain must be at least 200,000, and other income must be between 0 and 50,000",
            "out": {"free_vars": [], "constraints": {"house_transaction_gain": {">=": 200000},
                                                    "other_income": {">=": 0, "<=": 50000}}}},
            {"in": "None",
            "out": {"free_vars": [], "constraints": {}}},
        ]

        user_prompt = (
            "Field keys and alias reference (JSON):\n"
            + json.dumps(mapping_doc, ensure_ascii=False, indent=2)
            + "\n\nExamples (JSON):\n"
            + json.dumps(examples, ensure_ascii=False, indent=2)
            + "\n\nConvert the following description into JSON (output JSON only):\n"
            + str(user_text)
        )

        raw = await self._chat(sys_prompt, user_prompt, temperature=0.0)
        _append_debug(self.memory, "[ConstraintAgent] LLM constraints RAW:", raw)

        obj = self._json_loads_with_merge(raw) or {}
        if not isinstance(obj, dict):
            _append_debug(self.memory, "[ConstraintAgent] LLM constraints resolved failed（not dict）")
            return None, None

        def _to_list(x):
            if not x:
                return []
            if isinstance(x, list):
                return [str(t).strip() for t in x if str(t).strip()]
            return [t.strip() for t in str(x).split(",") if t.strip()]

        fv: List[str] = _to_list(obj.get("free_vars"))
        row_token_re = re.compile(r"row(\d+)\.([A-Za-z_][A-Za-z0-9_]*)")

        cons = obj.get("constraints")
        if not isinstance(cons, dict):
            cons = {}

        # 把頂層「非保留鍵」也視為 constraints 的展開
        TOPLEVEL_RESERVED = {"free_vars", "constraints", "budget_tax", "target_tax"}
        for k, v in list(obj.items()):
            if k in TOPLEVEL_RESERVED:
                continue
            if isinstance(v, dict):
                ops2 = {}
                for op, val in v.items():
                    op2 = "==" if op == "=" else op
                    if op2 not in {"==", ">=", "<=", ">", "<"}:
                        continue
                    if isinstance(val, (int, float)):
                        ops2[op2] = float(val)
                    elif isinstance(val, str):
                        s = val.strip()
                        rhs_expr = self._parse_linear_rhs(s)
                        if rhs_expr is not None:
                            # 例如 "row0.quantity * 0.3"
                            ops2[op2] = rhs_expr
                        else:
                            num = self._to_number(s)
                            if num is not None:
                                # 純數字字串
                                ops2[op2] = num
                            else:
                                # 保留一般變數 / 運算式字串（例如 "cigar_new.quantity"）
                                ops2[op2] = s
                if ops2:
                    cons.setdefault(k, {}).update(ops2)

        base_allowed_set = set(base_allowed_vars)
        allowed_set = base_allowed_set 

        # 只允許：
        # 1. 在 allowed_set（optimizable_fields + row_vars + budget 等）
        # 2. 且有 field_labels（會被轉成中文），或是 row{i}.field 形式
        label_keys = set(labels.keys())

        def _is_valid_fv_token(tok: str) -> bool:
            # 一般變數：必須在 allowed_set，且有對應的 field_labels
            return (tok in allowed_set) and (tok in label_keys)
        
        cleaned: Dict[str, Dict[str, Any]] = {}
        if cons:
            for expr, ops in (cons or {}).items():
                expr_str = str(expr)
                vars_in_expr = self._vars_in_expr(expr_str, allowed_set)
                if not vars_in_expr:
                    continue
                ops2: Dict[str, Any] = {}
                for op, val in (ops or {}).items():
                    if op == "=":
                        op = "=="
                    if op not in {"==", ">=", "<=", ">", "<"}:
                        continue
                    try:
                        if isinstance(val, str):
                            s2 = val.strip()
                            rhs_expr = self._parse_linear_rhs(s2)
                            if rhs_expr is not None:
                                # 驗證 RHS 變數是否合法
                                var_part = rhs_expr.split("*")[0].split("/")[0].strip()
                                valid_rhs = (var_part in base_allowed_set)
                                if not valid_rhs:
                                    _append_debug(self.memory, f"[ConstraintAgent] Discard illegal RHS variables: {rhs_expr}")
                                    continue
                                ops2[op] = rhs_expr
                            else:
                                num = self._to_number(s2)
                                if num is None:
                                    ops2[op] = num
                                ops2[op] = s2
                        else:
                            ops2[op] = float(val)
                    except Exception:
                        continue
                if ops2:
                    cleaned[expr_str] = ops2

        cons = cleaned or None

        auto_vars: List[str] = []
        if cons:
            for expr, ops in cons.items():
                auto_vars += self._vars_in_expr(expr, allowed_set)

        def _uniq(lst: List[str]) -> List[str]:
            return list(dict.fromkeys([t for t in lst if t]))

        fv = _uniq(fv + auto_vars)
        fv = [t for t in fv if _is_valid_fv_token(t)]
        return (fv if fv else None), cons

    async def _smart_parse(self, tool_name: str, text: str, payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        user_params = payload.get("user_params", {}) or {}
        fv_all: List[str] = []; cons_all: Dict[str, Any] = {}

        fv, cons = self._parse_reply_json_or_dsl(text, user_params)
        self._merge_fv(fv_all, fv); self._merge_cons(cons_all, cons)

        fv_sv, cons_sv = self._parse_single_bound_generic(tool_name, text, payload)
        self._merge_fv(fv_all, fv_sv); self._merge_cons(cons_all, cons_sv)

        fv2, cons2 = self._parse_ratio_constraint(tool_name, text)
        self._merge_fv(fv_all, fv2); self._merge_cons(cons_all, cons2)

        fv3, cons3 = self._parse_inequality_ratio_constraint(tool_name, text)
        self._merge_fv(fv_all, fv3); self._merge_cons(cons_all, cons3)

        fv4, cons4 = await self._nl_to_constraints(tool_name, text, payload)
        self._merge_fv(fv_all, fv4); self._merge_cons(cons_all, cons4)

        return (",".join(fv_all) if fv_all else None), (cons_all or None)

    # === F. 條件合併策略 ===
    def _merge_cons_with_policy(self, old_cons: dict | None, new_cons: dict | None,
                                *, user_text: str = "", default_policy: str = "auto") -> dict:
        """
        方案B（原本）：同一個 LHS、同一個 op 允許多個 RHS，最終用 AND。

        修補：
        - 針對「單一變數 + 單一運算子」且 RHS 全為常數的情況，
          採用 **最後收到的那個值覆蓋舊值**。
        - 適用所有運算子：==, >=, <=, >, <。
        """

        def _as_list(v):
            if v is None:
                return []
            if isinstance(v, (list, tuple)):
                return [x for x in v]
            return [v]

        def _is_atomic_var_lhs(lhs: str) -> bool:
            """
            單一變數（可含一個 dot 或 row 索引），不能含空白或 + - * / 等運算子。
            允許樣式：
            - salary_self
            - house_transaction_gain
            - cement_white.quantity
            - row0.quantity
            """
            s = str(lhs or "").strip()
            if not s or any(ch in s for ch in " +-*/()"):
                return False
            # rowN.field 或 slug.field 或 單一識別子
            pat = re.compile(r"^(?:row\d+\.[A-Za-z_]\w*|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)$")
            return bool(pat.match(s))

        def _pick_last_scalar(v):
            """從新值中挑最後一個標量（int/float）。"""
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, (list, tuple)) and v:
                for item in reversed(v):
                    if isinstance(item, (int, float)):
                        return float(item)
            return None

        def _is_scalar_only(v) -> bool:
            """
            判斷 value 是否「只包含數值」；
            只要混到字串運算式（例如 'b * 0.3'）就算 False。
            """
            if isinstance(v, (int, float)):
                return True
            if isinstance(v, (list, tuple)):
                if not v:
                    return False
                return all(isinstance(item, (int, float)) for item in v)
            return False

        old_cons = old_cons or {}
        new_cons = new_cons or {}
        final_cons: dict = {}

        all_lhs = set(old_cons.keys()) | set(new_cons.keys())
        for lhs in all_lhs:
            old_ops = old_cons.get(lhs, {}) or {}
            new_ops = new_cons.get(lhs, {}) or {}
            merged_ops: dict = {}

            is_atomic = _is_atomic_var_lhs(lhs)

            for op in ("==", ">=", "<=", ">", "<"):
                has_old = op in old_ops
                has_new = op in new_ops

                if not has_old and not has_new:
                    continue

                if is_atomic and has_new and _is_scalar_only(new_ops[op]):
                    last_scalar = _pick_last_scalar(new_ops[op])
                    if last_scalar is not None:
                        merged_ops[op] = last_scalar
                        continue  # 不要把舊值併進來

                # 一般模式：保留原本的 AND 行為（列表）
                if has_old and has_new:
                    merged_ops[op] = _as_list(old_ops[op]) + _as_list(new_ops[op])
                elif has_old:
                    merged_ops[op] = _as_list(old_ops[op])
                else:
                    merged_ops[op] = _as_list(new_ops[op])

                # 去重（保持穩定順序）
                try:
                    seen = set()
                    uniq = []
                    for item in merged_ops[op]:
                        key = f"{item:.12g}" if isinstance(item, (int, float)) else repr(item)
                        if key not in seen:
                            seen.add(key)
                            uniq.append(item)
                    merged_ops[op] = uniq
                except Exception:
                    pass

                # 如果某個 op 最後只剩一個元素，就還原成標量（美化輸出）
                if isinstance(merged_ops.get(op), list) and len(merged_ops[op]) == 1:
                    merged_ops[op] = merged_ops[op][0]

            if merged_ops:
                final_cons[lhs] = merged_ops

        return final_cons

    # === G. 互動說明（如何加條件） ===
    def _compose_question(self, tool_name: str, payload: dict | None = None) -> str:
        meta = TOOL_MAP.get(tool_name, {}) or {}
        desc = meta.get("description", tool_name)
        required_fields = list(meta.get("optimizable_fields", []) or [])
        field_labels = dict(meta.get("field_labels", {}) or {})
        row_fields = list(meta.get("row_fields", []) or [])

        def _lab(k: str) -> str:
            s = field_labels.get(k, k)
            if not isinstance(s, str):
                return k
            s = re.sub(r"[（(].*?[）)]", "", s).replace("？", "").replace("。", "").strip()
            return s or k

        base_vars_keys = list(dict.fromkeys(required_fields))
        base_section = ""
        if base_vars_keys:
            preview_n = 10
            show = base_vars_keys[:preview_n]
            rest = base_vars_keys[preview_n:]
            preview_lines = "\n".join(f"- {_lab(k)}" for k in show)
            if rest:
                rest_lines = "\n".join(f"- {_lab(k)}" for k in rest)
                base_section = (
                    "**Configurable fields (basic)**\n"
                    f"{preview_lines}\n"
                    f"- ……（there are {len(base_vars_keys)} Configurable fields）\n\n"
                    "<details>\n"
                    f"  <summary>expand to view all {len(base_vars_keys)} fields</summary>\n\n"
                    f"{rest_lines}\n"
                    "</details>\n"
                )
            else:
                base_section = "**Configurable fields (basic)**\n" + preview_lines

        payload = payload or {}

        examples: List[str] = []
        
       # Tool-specific examples (keep your original content structure)
        if tool_name == "income_tax":
            examples += [
                "• Salary split: salary_self + salary_spouse == 1,800,000",
                "• Itemized deduction cap: itemized_deduction <= 300,000",
                "• Rent deduction cap: rent_deduction <= 60% of dividend_income",
            ]
        elif tool_name == "foreigner_income_tax":
            examples += [
                "• Days of residence in Taiwan: at most 300 days",
                "• Other income range: other_income >= 50,000",
            ]
        elif tool_name == "business_income_tax":
            examples += [
                "• Expense ratio cap: operating_expenses_and_losses <= 30% of total_operating_revenue",
                "• Non-operating loss cap: non_operating_losses_and_expenses <= 1,000,000",
            ]
        elif tool_name == "vat_tax":
            examples += [
                "• Input VAT cap: input_vat <= 70,000,000",
                "• Output VAT must be at least double the input VAT: output_vat >= 2 * input_vat",
            ]
        elif tool_name == "nvat_tax":
            examples += [
                "• Nightclub sales minimum: nightclub_sales >= 8,000,000",
                "• Bar/hostess club + nightclub sales minimum: bar_sales + nightclub_sales >= 60,000,000",
            ]
        elif tool_name == "ta_tax":
            examples += [
                "• Cigarettes-new + cigars-new: at least 600 units",
                "• Brewed beer + reprocessed alcohol: at least 800 units",
            ]
        elif tool_name == "cargo_tax_minimize":
            examples += [
                "• Quantity cap: Item 1 'Quantity' <= 1,500",
                "• Dutiable value fixed: Item 2 'Dutiable Price' == 15,000",
            ]
        elif tool_name == "securities_tx_tax":
            examples += [
                "• Warrant exercise shares cap: Item 1 'Shares' <= 10,000",
                "• Amount ratio: Item 2 'Transaction Amount' is 30% of Item 1 'Transaction Amount'",
            ]
        elif tool_name == "futures_tx_tax":
            examples += [
                "• Contract amount range: Item 1 'Contract Amount' is between 100,000,000 and 500,000,000",
                "• Premium as a ratio: Item 2 'Premium Amount' is 2% of the 'Contract Amount' in the same row",
            ]
        elif tool_name == "special_goods_tax":
            examples += [
                "• Car unit price minimum: car_unit_price >= 3,000,000",
                "• Yacht quantity equals half the airplane quantity: yacht_quantity == airplane_quantity / 2",
            ]
        elif tool_name == "special_tax":
            examples += [
                "• Service sales minimum: membership_rights_sales >= 2,000,000",
            ]
        elif tool_name == "estate_tax":
            examples += [
                "• Agricultural deduction cap (years 6–9): agri_deduction_6_9_years <= 4,000,000",
                "• Outstanding debt cannot be negative: outstanding_debt_amount >= 0",
            ]
        elif tool_name == "gift_tax":
            examples += [
                "• Excluded house cap: excluded_house <= 1,000,000",
                "• Other gift liabilities cap: other_gift_liabilities <= 500,000",
            ]

        lines: List[str] = [f"### {desc} — Do you want to add constraints?"]
        if base_section:
            lines.append("\n" + base_section + "\n")
        if examples:
            lines.append("\n**More examples**")
            lines.append("\n".join(examples))
        lines.append("\n(Reminder: If you want to clear all constraints and start over, you can type \"Reset constraints\" anytime.)")
        return "\n".join(lines)


    # === H. 對外：重開與主流程 ===
    def reopen(self) -> Dict[str, Any]:
        ctx = self.memory.get("last_exec_payload") or {}
        tool = ctx.get("tool_name"); payload = ctx.get("payload")
        if not tool or not payload:
            return {"type": "error", "message": "The context of the previous round could not be found; please run the calculation again or provide new tools and parameters."}
        
        # === NEW: 將記憶中的 __prev_* 合併進 payload（保留既有非空值） ===
        mem_prev = {
            "__prev_tax__": self.memory.get("__prev_tax__"),
            "__prev_final_params__": self.memory.get("__prev_final_params__"),
            "__prev_constraints__": self.memory.get("__prev_constraints__"),
        }
        try:
            payload = merge_prev_into_payload(payload, mem_prev)
        except Exception:
            pass
        # -------------------------------------------------------------------

        self.memory.set("pending_constraint_payload", payload)
        self.memory.set("pending_tool_for_constraints", tool)

        header = "Phase Two: Please enter the conditions you wish to set according to your tax status (after completion, you will be prompted to 「Final Confirmation」 before the calculation begins)."
        params_md = self._fmt_params_preview(tool, payload)

        tips = payload.get("early_tips_md")
        tips_md = f"{tips}\n\n---" if isinstance(tips, str) and tips.strip() else ""

        howto_md = self._compose_question(tool, payload)
        preview_md = self._fmt_constraints_preview_zh(tool, payload)  # 初始預覽（多半為空）

        question = (
            f"{header}\n\n{params_md}\n\n{tips_md}\n\n{howto_md}\n\n{preview_md}\n\n"
            "Type constraints directly to add them. If you don't want to add any, reply \"None\".\n"
            "When you're ready to compute, reply \"Next\" (you will go through a final confirmation first).\n"
            "To clear all constraints and start over, type \"Reset constraints\".\n"
            "To compute immediately, reply \"Compute now\"."
        )
        return {"type": "follow_up", "stage": "constraints", "question": question, "tool_name": tool}

    async def handle(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        stage = obj.get("type")

        # 允許外部以 'reopen_constraints' 直接重啟
        if stage == "reopen_constraints":
            return self.reopen()

        # 進入條件設定
        if stage == "tool_request":
            payload = obj.get("payload") or {}
            tool = payload.get("tool_name")

            # 使用者要求『直接計算』：跳過條件/最終確認，直接執行工具
            if payload.get("__direct_execute__"):
                dbg = self.memory.get("debug_lines", [])
                self.memory.set("debug_lines", [])
                self.memory.set("await_execute_confirm", False)
                self.memory.set("pending_constraint_payload", None)
                self.memory.set("pending_tool_for_constraints", None)
                return {"type": "ready_for_execute", "payload": payload, "debug": dbg}

            # ---- 確保 __prev_tax__ 在進入條件階段就跟著 payload 走 ----
            if "__prev_tax__" not in payload:
                try:
                    prev_tax_mem = self.memory.get("__prev_tax__")
                    if isinstance(prev_tax_mem, (int, float)):
                        payload["__prev_tax__"] = float(prev_tax_mem)
                except Exception:
                    pass
            # ------------------------------------------------------

            # 進入條件階段前，清掉「等待最終確認」旗標
            self.memory.set("await_execute_confirm", False)
            self.memory.set("pending_constraint_payload", payload)
            self.memory.set("pending_tool_for_constraints", tool)

            header = (
                "Phase Two: Please enter the constraints you want to set based on your tax situation "
                "(after you finish, you will be prompted for a Final Confirmation before computation starts)."
            )
            params_md = self._fmt_params_preview(tool, payload)

            tips = payload.get("early_tips_md")
            tips_md = f"{tips}\n\n---" if isinstance(tips, str) and tips.strip() else ""

            howto_md = self._compose_question(tool, payload)
            preview_md = self._fmt_constraints_preview_zh(tool, payload)  # 初始預覽（多半為空）

            question = (
                f"{header}\n\n{params_md}\n\n{tips_md}\n\n{howto_md}\n\n{preview_md}\n\n"
                "Type constraints directly to add them. If you don't want to add any, reply \"None\".\n"
                "When you're ready to compute, reply \"Next\" (you will go through a final confirmation first).\n"
                "To clear all constraints and start over, type \"Reset constraints\".\n"
                "To compute immediately, reply \"Compute now\"."
            )
            return {"type": "follow_up", "stage": "constraints", "question": question, "tool_name": tool}

        # 條件輸入迴路
        if stage == "constraints_reply":
            text = (obj.get("text") or "").strip()

            # English-only "compute now" triggers
            if re.search(r"(?:^|\s)(compute\s*now|run\s*now|direct\s*calc)(?:$|\s)", text, flags=re.I):
                payload = self.memory.get("pending_constraint_payload") or {}
                dbg = self.memory.get("debug_lines", [])
                self.memory.set("debug_lines", [])
                self.memory.set("await_execute_confirm", False)
                self.memory.set("pending_constraint_payload", None)
                self.memory.set("pending_tool_for_constraints", None)
                return {"type": "ready_for_execute", "payload": payload, "debug": dbg}

            awaiting = bool(self.memory.get("await_execute_confirm"))
            payload = self.memory.get("pending_constraint_payload") or {}
            cur_tool = self.memory.get("pending_tool_for_constraints")
            last_tool = self.memory.get("last_tool")

            if cur_tool and last_tool and cur_tool != last_tool:
                self.memory.set("pending_constraint_payload", None)
                self.memory.set("pending_tool_for_constraints", None)
                return {
                    "type": "follow_up",
                    "stage": "constraints",
                    "tool_name": last_tool,
                    "question": (
                        "Detected that you switched to a different tax type. The previous round's constraints have been cleared. "
                        "Please re-enter the constraints you want to add, or reply \"None\"."
                    )
                }

            tool = self.memory.get("pending_tool_for_constraints")
            if not payload or not tool:
                return {"type": "ready_for_execute", "payload": obj.get("payload") or {}}

            low = text.lower()
            is_next = bool(re.search(r"(?:^|\s)(next|continue|go|ok)(?:$|\s)", text, flags=re.I))
            is_done = any(k in low for k in ["done", "finish", "stop"])

            #  在條件階段也支援「重設條件」→ 清空 free_vars / constraints
            if re.search(r"(reset\s*constraints?|clear\s*constraints?)", text, flags=re.I):
                user_params = payload.get("user_params") or {}
                # 只清空條件相關，保留使用者原本填的基本欄位
                user_params.pop("free_vars", None)
                user_params.pop("constraints", None)
                payload["user_params"] = user_params

                self.memory.set("pending_constraint_payload", payload)
                self.memory.set("await_execute_confirm", False)  # 回到條件輸入狀態

                params_md = self._fmt_params_preview(tool, payload)
                preview_md = self._fmt_constraints_preview_zh(tool, payload)
                howto_md = self._compose_question(tool, payload)
                tips_md = await self._condition_refiner_tips(tool, payload)

                reset_msg = (
                    "All constraints have been cleared. You're back in the constraints setup phase.\n\n"
                    f"{params_md}\n\n{howto_md}\n\n{preview_md}\n\n{tips_md}\n\n"
                    "To add constraints again, type them directly. When you're finished, reply \"Next\"."
                )
                return {
                    "type": "follow_up",
                    "stage": "constraints",
                    "tool_name": tool,
                    "question": reset_msg,
                }

            # A) 若已在等待最終確認狀態
            if awaiting:
                # 使用者確認開算
                if is_next or is_done:
                    self.memory.set("await_execute_confirm", False)
                    self.memory.set("pending_constraint_payload", None)
                    self.memory.set("pending_tool_for_constraints", None)
                    dbg = self.memory.get("debug_lines", [])

                    # ---- 在回傳 ready_for_execute 前保險補上 __prev_tax__ ----
                    if "__prev_tax__" not in payload:
                        try:
                            prev_tax_mem = self.memory.get("__prev_tax__")
                            if isinstance(prev_tax_mem, (int, float)):
                                payload["__prev_tax__"] = float(prev_tax_mem)
                        except Exception:
                            pass
                    # ---------------------------------------------------

                    self.memory.set("debug_lines", [])
                    return {
                        "type": "ready_for_execute",
                        "payload": payload,
                        "constraints_preview": {
                            "free_vars": (payload.get("user_params") or {}).get("free_vars"),
                            "constraints": (payload.get("user_params") or {}).get("constraints"),
                        },
                        "debug": dbg,
                    }

                # 使用者要求返回條件頁
                if any(k in low for k in ["back", "return"]):
                    self.memory.set("await_execute_confirm", False)
                    header = (
                        "You're back on the constraints page. Type constraints directly to add them, "
                        "or reply \"Next\" to enter final confirmation again."
                    )
                    params_md = self._fmt_params_preview(tool, payload)
                    howto_md = self._compose_question(tool, payload)
                    preview_md = self._fmt_constraints_preview_zh(tool, payload)
                    return {
                        "type": "follow_up",
                        "stage": "constraints",
                        "tool_name": tool,
                        "question": f"{header}\n\n{params_md}\n\n{howto_md}\n\n{preview_md}",
                    }

                # 其他輸入：維持在「最終確認」頁，不變更狀態，提示可用指令
                # 保持 await_execute_confirm = True
                self.memory.set("await_execute_confirm", True)

                # 重新渲染最終確認頁（與第一次進入確認時一致）
                params_md = self._fmt_params_preview(tool, payload, mask_free_and_constrained=True)
                preview_md = self._fmt_constraints_preview_zh(tool, payload)
                tips_md = await self._condition_refiner_tips(tool, payload)

                confirm_text = (
                    "### Phase Three: Final Confirmation (Before Execution)\n"
                    "You are currently on the final confirmation page. Reply \"Next\" to start computing, "
                    "or reply \"Back\" to return to the constraints page and keep adjusting.\n\n"
                    f"{params_md}\n\n{preview_md}\n\n{tips_md}\n\n"
                    "Available commands: \"Next\" to compute, or \"Back\" to return to constraints."
                )

                return {
                    "type": "follow_up",
                    "stage": "pre_execute_review",
                    "tool_name": tool,
                    "question": confirm_text,
                }

            # B) 尚未進入確認 — 第一次按「下一步」先顯示確認頁
            if is_next or is_done:
                self.memory.set("await_execute_confirm", True)
                # 第三階段顯示：若變數在 free_vars 或受約束，覆蓋輸入值為「由求解器決定」
                params_md = self._fmt_params_preview(tool, payload, mask_free_and_constrained=True)
                preview_md = self._fmt_constraints_preview_zh(tool, payload)
                tips_md = await self._condition_refiner_tips(tool, payload)
                confirm_text = (
                    "### Phase Three: Final Confirmation (Before Execution)\n"
                    "Below are your current **inputs and constraints**:\n\n"
                    f"{params_md}\n\n{preview_md}\n\n{tips_md}\n\n"
                    "Reply \"Next\" to start computing; reply \"Back\" to return to the constraints page and keep adjusting. "
                    "To clear all constraints and start over, type \"Reset constraints\"."
                )
                return {"type": "follow_up", "stage": "pre_execute_review", "tool_name": tool, "question": confirm_text}

            # 2) 不加條件（留在本階段，提示用「下一步」開始執行）
            if not text or any(k in low for k in ["none", "no", "skip"]):
                preview_md = self._fmt_constraints_preview_zh(tool, payload)
                tips_md = await self._condition_refiner_tips(tool, payload)
                dbg = self.memory.get("debug_lines", [])
                self.memory.set("debug_lines", [])
                return {
                    "type": "follow_up",
                    "stage": "constraints",
                    "tool_name": tool,
                    "question": (
                        f"{preview_md}\n\n{tips_md}\n\n"
                        "To add more constraints, type them directly. When you're finished, reply \"Next\".\n"
                        "To clear all constraints and start over, type \"Reset constraints\"."
                    ),
                    "debug": dbg,
                }

            # 3) 解析 + 合併條件
            user_params = payload.get("user_params", {}) or {}
            fv, cons = self._parse_reply_json_or_dsl(text, user_params)
            if fv is None and cons is None:
                try:
                    fv, cons = self._parse_cn_ratio(tool, text, payload)
                except Exception as e:
                    _append_debug(self.memory, f"[ConstraintAgent] _parse_cn_ratio error: {e}")
            if fv is None and cons is None:
                fv, cons = self._parse_inequality_ratio_constraint(tool, text)
            if fv is None and cons is None:
                fv, cons = await self._nl_to_constraints(tool, text, payload)

            _append_debug(self.memory, f"[ConstraintAgent] PARSED -> free_vars={fv} constraints={cons}")

            if fv:
                merged = list(dict.fromkeys(self._to_list(user_params.get("free_vars")) + self._to_list(fv)))
                if merged:
                    user_params["free_vars"] = merged

            if cons:
                old_cons = user_params.get("constraints") or {}
                if not isinstance(old_cons, dict):
                    old_cons = {}
                merged = self._merge_cons_with_policy(old_cons, cons, user_text=text, default_policy="auto")
                user_params["constraints"] = merged

            payload["user_params"] = user_params
            self.memory.set("pending_constraint_payload", payload)
            self.memory.set("pending_tool_for_constraints", tool)
            # 加條件後，強制回到未確認狀態
            self.memory.set("await_execute_confirm", False)

            # ---- 每次用戶更新條件，也把 __prev_tax__ 保持在 payload 中 ----
            if "__prev_tax__" not in payload:
                try:
                    prev_tax_mem = self.memory.get("__prev_tax__")
                    if isinstance(prev_tax_mem, (int, float)):
                        payload["__prev_tax__"] = float(prev_tax_mem)
                except Exception:
                    pass
            # ------------------------------------------------------

            dbg = self.memory.get("debug_lines", [])
            self.memory.set("debug_lines", [])
            preview_md = self._fmt_constraints_preview_zh(tool, payload)
            tips_md = await self._condition_refiner_tips(tool, payload)

            return {
                "type": "follow_up",
                "stage": "constraints",
                "tool_name": tool,
                "question": (
                    f"{preview_md}\n\n{tips_md}\n\n"
                    "To add more constraints, type them directly. When you're finished, reply \"Next\".\n"
                    "To clear all constraints and start over, type \"Reset constraints\"."
                ),
                "debug": dbg,
            }

        return obj


class ExecuteAgent(BaseAgent):
    async def handle(self, payload: Dict[str, Any]):
        tool_name = payload.get("tool_name")
        params = (payload.get("user_params") or {})  # dict
        op = payload.get("op")

        # 允許把「前一輪」的 __prev_tax__ 從記憶體補進來，但**不要**覆蓋 payload 既有值
        try:
            prev_tax_mem = self.memory.get("__prev_tax__")
            if "__prev_tax__" not in payload and isinstance(prev_tax_mem, (int, float)):
                payload["__prev_tax__"] = float(prev_tax_mem)
        except Exception:
            pass

        if not tool_name:
            raise RuntimeError("ExecuteAgent: missing tool_name")

        meta = TOOL_MAP[tool_name]
        module = __import__(meta["module"], fromlist=["*"])
        entry = meta["entry_func"]

        # 智慧選擇：若未指定 op 且有正的 budget，上 maximize（前提：工具真的有 maximize 入口）
        if not op and isinstance(entry, dict):
            budget_field = meta.get("budget_field") or "budget_tax"
            bval = (params or {}).get(budget_field)
            if isinstance(bval, (int, float)) and bval > 0 and "maximize" in entry:
                op = "maximize"

        # 仍尊重顯式指定；沒指定就回退 minimize
        fn_name = entry.get(op or "minimize") if isinstance(entry, dict) else entry

        func = getattr(module, fn_name)

        def _stash_for_reopen(_tool: str, _params: Dict[str, Any], _op: Optional[str], *, prev_pack: Optional[dict] = None):
            ctx_payload = {
                "tool_name": _tool,
                "user_params": _params,
                "op": _op,
            }

            #  把 Caller 生成且這一輪帶進來的 early_tips_md 一起保留，供「重設條件 / reopen」時使用
            tips = (payload or {}).get("early_tips_md")
            if isinstance(tips, str) and tips.strip():
                ctx_payload["early_tips_md"] = tips

            # 把「本輪」結果打包成『下一輪的上一輪』，僅寫入 last_exec_payload/pending_constraint_payload
            if isinstance(prev_pack, dict):
                for k in ("__prev_tax__", "__prev_final_params__", "__prev_constraints__"):
                    v = prev_pack.get(k)
                    if v not in (None, {}, []):
                        ctx_payload[k] = v

            self.memory.set("last_exec_payload", {
                "tool_name": _tool,
                "user_params": _params,
                "op": _op,
                "payload": dict(ctx_payload),
            })
            self.memory.set("last_tool", _tool)
            self.memory.set("pending_constraint_payload", dict(ctx_payload))
            self.memory.set("pending_tool_for_constraints", _tool)

        try:
            # 嘗試以關鍵字參數呼叫
            result = func(**params)

            # === NEW: 從本輪 result 萃取上一輪快照，合併到下一輪 payload 並寫入記憶 ===
            try:
                prev_pack = extract_prev_from_result(result)
            except Exception:
                prev_pack = {}

            # 回傳給 ReasoningAgent 的 payload 保留「進入本輪前」的 __prev_*，不要混入 prev_pack
            report_payload = dict(payload or {})
            report_payload["op"] = op

            # 將「本輪快照」只塞到**重開用**的 context（給下一輪），不要動 memory.__prev_*（讓 ReasoningAgent 結尾統一 commit）
            _stash_for_reopen(tool_name, params, op, prev_pack=prev_pack)

            return {
                "result": result,
                "tool_name": tool_name,
                "payload": report_payload,
            }

        except TypeError as e:
            # 若簽名只有單一參數且允許位置引數，改用單一參數呼叫
            sig = inspect.signature(func)
            params_list = list(sig.parameters.values())
            single_positional = (
                len(params_list) == 1 and
                params_list[0].kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD
                )
            )
            if single_positional:
                result = func(params)

                # === NEW: 同步於單一參數模式也帶入上一輪快照 ===
                try:
                    prev_pack = extract_prev_from_result(result)
                except Exception:
                    prev_pack = {}

                exec_payload = {"user_params": params, "op": op}
                try:
                    exec_payload = merge_prev_into_payload(exec_payload, prev_pack)
                except Exception:
                    pass

                try:
                    if isinstance(prev_pack.get("__prev_tax__"), (int, float)):
                        self.memory.set("__prev_tax__", float(prev_pack["__prev_tax__"]))
                    if isinstance(prev_pack.get("__prev_final_params__"), dict):
                        self.memory.set("__prev_final_params__", prev_pack["__prev_final_params__"])
                    if isinstance(prev_pack.get("__prev_constraints__"), dict):
                        self.memory.set("__prev_constraints__", prev_pack["__prev_constraints__"])
                    self.memory.set("last_result", result)
                except Exception:
                    pass

                _stash_for_reopen(tool_name, params, op)
                return {
                    "result": result,
                    "tool_name": tool_name,
                    "payload": exec_payload,
                }

            # 非上述情況就把原錯拋出（交給上層顯示）
            raise



class ReasoningAgent(BaseAgent):
    # ===== Perf tracing (ReasoningAgent) =====
    def _perf_reset(self) -> None:
        self._perf_spans: list[tuple[str, float]] = []
        self._perf_handle_t0 = None

    def _perf_add(self, phase: str, dt: float) -> None:
        try:
            self._perf_spans.append((str(phase), float(dt)))
        except Exception:
            pass

    from contextlib import contextmanager

    @contextmanager
    def _perf_span(self, phase: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._perf_add(phase, time.perf_counter() - t0)

    def _perf_finalize_spans(self) -> list[tuple[str, float]]:
        try:
            t0 = getattr(self, "_perf_handle_t0", None)
            spans = getattr(self, "_perf_spans", None)
            if isinstance(spans, list) and isinstance(t0, (int, float)):
                if not any((p[0] == "handle_total") for p in spans):
                    spans.append(("handle_total", float(time.perf_counter() - t0)))
            out = list(spans) if isinstance(spans, list) else []
            try:
                self.memory.set("perf_spans_last", out)
            except Exception:
                pass
            return out
        except Exception:
            return []

    async def _chat_traced(self, phase: str, sys_prompt: str, user_msg: str, **kwargs) -> str:
        with self._perf_span(f"llm:{phase}"):
            return await self._chat(sys_prompt, user_msg, **kwargs)

    """
    Spec-driven generic report engine (single-pass LLM drafting)
    """

    SPEC_DEFAULT = {
        "paths": {
            "baseline":       ["baseline", "base_tax", "tax_baseline"],
            "optimized":      ["optimized", "optimized_total_tax", "total_tax", "tax", "net_tax", "optimized_tax"],
            "mode":           ["mode"],
            "status":         ["status", "solve_status", "baseline_status"],
            "diff":           ["diff", "param_diff"],
            "final_params":   ["final_params", "chosen_params"],
            "input_params":   ["input_params"],
            "constraints":    ["constraints"],
            "optimized_sales": ["optimized_total_sales", "total_sales", "sales",
                                "optimized_total_quantity", "total_quantity", "quantity", "max_quantity"],
            "budget":         ["budget", "budget_tax", "tax_budget"],
        },
        "payload_fallbacks": {
            "mode":         ["op"],
            "constraints":  ["user_params.constraints"],
            "input_params": ["user_params"],
        },
        "sections": ["summary", "kpis", "diff", "constraints", "input_params", "final_params", "conclusion", "compliance"],
        "suggestion_rules": [
            {
                "if": "delta_tax is not None and delta_tax == 0",
                "then": "The optimal result matches the baseline. Consider allowing more variables to change or adding structural constraints (e.g., a tax budget, ratios, or bounds) to expand the search space."
            },
            {
                "if": "delta_tax is not None and delta_tax < 0",
                "then": "A lower-tax combination was found. Please review feasibility and compliance (supporting documents and recognition basis)."
            },
            {
                "if": "delta_tax is not None and delta_tax > 0",
                "then": "An optimized solution that satisfies your requirements was found."
            },
            {
                "if": "delta_tax is not None and delta_tax > 0 and mode_str == 'maximize'",
                "then": "Objective is 'maximize': the system found the best solution under your budget and constraints. A higher tax is the expected tradeoff for higher output (quantity/sales)."
            },
            {
                "if": "delta_tax is not None and delta_tax > 0 and has_constraints",
                "then": "All existing and additional constraints are satisfied. A higher tax vs. baseline is a reasonable consequence of meeting those constraints (not an optimization failure)."
            },
            {
                "if": "delta_tax is not None and delta_tax > 0 and not has_constraints and mode_str != 'maximize'",
                "then": "Optimization completed under the current setup. A higher tax than baseline may come from different objectives/parameters or discrete/legal minimums. If you want alternative tradeoffs, create a comparison run with different constraints."
            }
        ],
    }

    SPEC_BY_TOOL = {
        "income_tax": {
            "suggestion_rules": [
                {
                    "if": 'delta_tax == 0 and has_constraint("salary_self + salary_spouse")',
                    "then": "You set 'self salary + spouse salary = fixed total'. Reallocating only between the two usually won’t reduce tax much. Consider loosening deduction/exemption-related variables or adding structural constraints like a tax budget."
                },
                {
                    "if": "delta_tax == 0 and missing_or_zero(['itemized_deduction','rent_deduction','education_count','preschool_count','long_term_care_count','disability_count','property_loss_deduction'])",
                    "then": "Consider providing or loosening: itemized deductions, rent deduction, education/preschool/LTC/disability deductions, and property-loss deduction to expand the optimization space."
                },
            ]
        },
        "vat_tax": {
            "suggestion_rules": [
                {
                    "if": "delta_tax == 0",
                    "then": "To reduce net tax, consider allowing some input/output VAT entries to vary and add reasonable upper bounds or ratio constraints for input VAT to give the model room to adjust."
                }
            ]
        },
        "nvat_tax": {
            "suggestion_rules": [
                {
                    "if": "mode_str == 'maximize' and budget_missing",
                    "then": "You selected 'maximize' but did not provide a tax budget. Add **budget_tax** so the system can maximize sales within the budget."
                }
            ]
        },
        "ta_tax": {
            "suggestion_rules": [
                {
                    "if": "mode_str == 'maximize' and budget_missing",
                    "then": "You selected 'maximize' but did not provide a tax budget. Add **budget_tax** so the system can maximize quantity within the budget."
                }
            ]
        },
    }

    def __init__(self, memory: Optional[MemoryStore] = None):
        super().__init__(name="ReasoningAgent", memory=memory)

    def _fmt_money(self, x):
        try:
            return f"{float(x):,.0f}"
        except Exception:
            return str(x)

    def _trend(self, delta):
        try:
            d = float(delta)
            if d > 0:
                return f"↑ {self._fmt_money(abs(d))}"
            if d < 0:
                return f"↓ {self._fmt_money(abs(d))}"
            return "—"
        except Exception:
            return "—"

    def _label(self, tool_name: str, var_key: str) -> str:
        labels = TOOL_MAP.get(tool_name, {}).get("field_labels", {}) or {}
        s = labels.get(var_key, var_key)
        if not isinstance(s, str):
            return var_key
        s = re.sub(r"[（(].*?[）)]", "", s).replace("？", "").strip()
        return s or var_key

    def _get_by_path(self, root: dict, dotted: str):
        try:
            cur = root
            for part in dotted.split("."):
                if isinstance(cur, list) and part.isdigit():
                    cur = cur[int(part)]
                elif isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return None
            return cur
        except Exception:
            return None

    def _first_hit(self, obj: dict, paths: List[str]):
        for p in paths or []:
            v = self._get_by_path(obj, p)
            if v is not None:
                return v
        return None

    def _format_constraints(self, tool_name: str, constraints: dict | str | None) -> str:
        if not constraints:
            return "(Not set)"

        def _fmt_num(x):
            try:
                f = float(x)
                return f"{int(f):,}" if abs(f - int(f)) < 1e-9 else f"{f:,}"
            except Exception:
                return str(x)

        def _fmt_rhs(rhs):
            if isinstance(rhs, (list, tuple)) and rhs:
                rhs = rhs[0]
            if isinstance(rhs, (int, float)):
                return _fmt_num(rhs)
            if isinstance(rhs, str):
                s = self._humanize_expr(tool_name, rhs)
                return s.replace("*", "×").replace("/", "÷")
            return str(rhs)

        def _fmt_op(op):
            return "=" if op == "==" else op

        if isinstance(constraints, str):
            return self._humanize_expr(tool_name, constraints)

        if isinstance(constraints, dict):
            lines = []
            OP_ORDER = ["==", ">=", "<=", ">", "<"]
            for lhs, ops in constraints.items():
                lhs_cn = self._humanize_expr(tool_name, lhs)
                if not isinstance(ops, dict):
                    lines.append(f"- {lhs_cn} {ops}")
                    continue
                for op in OP_ORDER:
                    if op not in ops:
                        continue
                    vals = ops[op]
                    vals = vals if isinstance(vals, (list, tuple)) else [vals]
                    for rhs in vals:
                        rhs_cn = _fmt_rhs(rhs)
                        lines.append(f"- {lhs_cn} {_fmt_op(op)} {rhs_cn}")
            return "\n".join(lines) if lines else "(Not set)"

        return self._humanize_expr(tool_name, str(constraints))

    def _format_diff(self, tool_name: str, diff: dict | None) -> str:
        if not isinstance(diff, dict) or not diff:
            return "(No parameter changes, or the tool did not return a diff.)"
        out = []
        for k, v in diff.items():
            if not isinstance(v, dict):
                continue
            orig = v.get("original")
            opt = v.get("optimized")
            delta = v.get(
                "difference",
                (opt if isinstance(opt, (int, float)) else 0) - (orig if isinstance(orig, (int, float)) else 0),
            )
            out.append(
                f"- {self._label(tool_name, k)}: {self._fmt_money(orig)} → {self._fmt_money(opt)} ({self._trend(delta)})"
            )
        return "\n".join(out) if out else "(No parameter changes, or the tool did not return a diff.)"

    def _eval_rules(self, rules: List[dict], ctx: dict) -> List[str]:
        out = []

        def _safe_eval(expr: str) -> bool:
            try:
                return bool(eval(expr, {"__builtins__": {}}, ctx))
            except Exception:
                return False

        for r in rules or []:
            cond = r.get("if")
            msg = r.get("then")
            if cond and msg and _safe_eval(cond):
                out.append(msg)
        return out

    def _labels(self, tool_name: str) -> dict:
        return (TOOL_MAP.get(tool_name, {}) or {}).get("field_labels", {}) or {}

    def _humanize_key(self, tool_name: str, key: str) -> str:
        labels = self._labels(tool_name)
        k = (key or "").strip()
        if " " in k and "." not in k:
            parts = k.split()
            if len(parts) == 2 and parts[1] in ("quantity", "assessed_price", "tp", "ep", "sc", "ca", "pa"):
                k = f"{parts[0]}.{parts[1]}"
        return labels.get(k, k)

    def _field_set_for_tool(self, tool_name: str) -> set[str]:
        labels = (TOOL_MAP.get(tool_name, {}) or {}).get("field_labels", {}) or {}
        right_side = set()
        for k in labels.keys():
            if isinstance(k, str) and "." in k:
                _, fld = k.split(".", 1)
                right_side.add(fld.strip())
        right_side |= {"quantity", "assessed_price", "tp", "ep", "sc", "ca", "pa", "price", "unit_price"}
        return {f for f in right_side if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", f)}

    def _humanize_expr(self, tool_name: str, expr: str) -> str:
        labels = self._labels(tool_name)
        fields = sorted(self._field_set_for_tool(tool_name), key=len, reverse=True)
        if not fields:
            return str(expr or "")

        fld_alt = "|".join(map(re.escape, fields))
        token_re = re.compile(rf"([A-Za-z0-9_]+)\.({fld_alt})|([A-Za-z0-9_]+)\s+({fld_alt})")

        s = str(expr or "")

        def _repl(m):
            if m.group(1) and m.group(2):
                key = f"{m.group(1)}.{m.group(2)}"
            else:
                key = f"{m.group(3)}.{m.group(4)}"
            return labels.get(key, key)

        s = token_re.sub(_repl, s)

        try:
            single_vars = [
                k
                for k in labels.keys()
                if isinstance(k, str) and "." not in k and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k)
            ]
            if single_vars:
                sv_alt = "|".join(map(re.escape, single_vars))
                bare_re = re.compile(rf"\b({sv_alt})\b")
                s = bare_re.sub(lambda m: labels.get(m.group(1), m.group(1)), s)
        except Exception:
            pass

        return s

    # -------- RAG-enhanced advice (using optimization result as context) --------
    def _load_rag_db(self):
        import os
        try:
            if getattr(self, "_rag_db", None) is not None:
                return getattr(self, "_rag_db")
            persist_dir = os.getenv("RAG_CHROMA_DIR", "rag/chroma")
            collection = os.getenv("RAG_COLLECTION", "tax_handbook")
            if not os.path.isdir(persist_dir):
                setattr(self, "_rag_db", None)
                return None
            db = Chroma(
                collection_name=collection,
                persist_directory=persist_dir,
                embedding_function=OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small")),
            )
            setattr(self, "_rag_db", db)
            return db
        except Exception:
            setattr(self, "_rag_db", None)
            return None

    def _mk_rag_query_from_ctx(
        self,
        tool_name: str,
        mode,
        budget,
        constraints,
        diff,
        input_params,
        final_params: dict | None = None,
    ) -> str:
        """
        Compress optimization context into a query string, including:
        - tool_name / tax saving / compliance / optimization / mode
        - tax budget (if any)
        - constraint keys (first few)
        - adjusted variables from diff (first few)
        - key fields from final_params (top 6): label=value; if constrained, add '(bound)'
        (Row-related content has been removed.)
        """
        parts = [tool_name, "tax saving", "compliance", "optimization", f"mode:{str(mode or '')}"]

        try:
            if isinstance(budget, (int, float)) and budget > 0:
                parts.append(f"tax_budget:{int(budget)}")
        except Exception:
            pass

        cons_keys = []
        if isinstance(constraints, dict) and constraints:
            cons_keys = list(constraints.keys())[:3]
            if cons_keys:
                parts.append("constraints:" + ";".join(cons_keys))

        if isinstance(diff, dict) and diff:
            changed = [k for k in diff.keys()]
            if changed:
                parts.append("adjusted_vars:" + ",".join(changed[:5]))

        try:
            cons_lhs = set(cons_keys or [])
            if isinstance(constraints, dict):
                cons_lhs |= {str(k) for k in constraints.keys()}

            def _is_key_interesting(k: str) -> bool:
                hit_suffix = (
                    k.endswith(".quantity")
                    or k.endswith(".price")
                    or k.endswith(".assessed_price")
                    or k.endswith(".unit_price")
                    or k.endswith(".tp")
                    or k.endswith(".ep")
                    or k.endswith(".sc")
                    or k.endswith(".ca")
                    or k.endswith(".pa")
                )
                return hit_suffix or ("." in k)

            def _coerce_num(x):
                if isinstance(x, (int, float)):
                    return x
                if isinstance(x, str):
                    s = x.strip().replace(",", "").replace("，", "")
                    try:
                        return float(s)
                    except Exception:
                        return None
                return None

            if isinstance(final_params, dict) and final_params:
                cand = []
                for k, meta in final_params.items():
                    if not isinstance(k, str) or not isinstance(meta, dict):
                        continue
                    v = meta.get("value", None)
                    if v is None:
                        continue
                    num = _coerce_num(v)
                    if num is not None and _is_key_interesting(k):
                        cand.append((k, num))

                def _score(item):
                    k, num = item
                    w = 0
                    if k.endswith(".quantity"):
                        w += 3
                    if k.endswith(".price") or k.endswith(".assessed_price") or k.endswith(".unit_price"):
                        w += 2
                    if k.endswith((".tp", ".ep", ".sc", ".ca", ".pa")):
                        w += 1
                    return (w, abs(num))

                cand.sort(key=_score, reverse=True)
                top = cand[:6]

                finals_frag = []
                for k, num in top:
                    label = self._label(tool_name, k)
                    val_txt = f"{int(num):,}" if abs(num - int(num)) < 1e-9 else f"{num:,}"
                    bound_note = " (bound)" if k in cons_lhs else ""
                    finals_frag.append(f"{label}={val_txt}{bound_note}")
                if finals_frag:
                    parts.append("final:" + ";".join(finals_frag))
        except Exception:
            pass

        return " ".join(parts)

    async def _advice_from_result_no_rag(self, tool_name: str, result: dict, payload: dict, field_labels: dict):
        import json, re
        baseline = result.get("baseline")
        optimized = result.get("optimized") or result.get("tax") or result.get("optimized_tax")
        mode = (result.get("mode") or payload.get("op") or "").lower()
        diff = result.get("diff") or {}
        constraints = result.get("constraints") or (payload.get("user_params") or {}).get("constraints") or {}
        final_params = result.get("final_params") or {}
        labels = field_labels or {}

        def lab(k):
            s = labels.get(k, k)
            return re.sub(r"[（(].*?[）)]", "", s or "").replace("？", "").strip()

        deltas = []
        for k, v in (diff or {}).items():
            if not isinstance(v, dict):
                continue
            orig = v.get("original")
            opt = v.get("optimized")
            if isinstance(orig, (int, float)) and isinstance(opt, (int, float)):
                try:
                    chg = abs(opt - orig)
                    if chg >= 50000 or (abs(orig) > 0 and chg / abs(orig) >= 0.3):
                        deltas.append(f"{lab(k)}: {orig} → {opt}")
                except Exception:
                    pass

        ctx = {
            "tool_name": tool_name,
            "mode": mode,
            "baseline": baseline,
            "optimized": optimized,
            "deltas": deltas[:6],
            "constraints": constraints,
            "final_params": final_params,
            "field_labels": labels,
        }

        sys = (
            "You are an actionable-advice generator for Taiwan tax optimization reports.\n"
            "You MUST base advice ONLY on ctx.deltas. Any evidence text (if provided) can only be used to explain WHY tax changes, and must not introduce new variables.\n"
            "Strict rules:\n"
            "1) Each advice item must cite at least one delta (label + original→optimized). Do NOT mention variables not present in ctx.deltas.\n"
            "2) Each advice item must include the underlying mechanism and explain the tax/payment number and how it is computed. Use concrete, non-speculative wording.\n"
            "3) Do NOT guess the user's situation. Describe only what this run changed and why it affects tax.\n"
            "4) Output in English. One advice item per line.\n"
            "5) If ctx.deltas is empty: still produce a small amount of advice. Do NOT output: "
            "\"There were no adjustable variable changes, therefore no adjustment advice can be provided.\"\n"
            "   - First explain: optimal equals baseline / no better solution under current constraints.\n"
            "   - Then give a few bullet points explaining the tax computation workflow, and propose next steps to expand the optimization space (e.g., allow more free_vars / add structural constraints).\n"
            "Output MUST be strict JSON: {\"advice\":[\"...\"]}. No extra text."
        )
        user = json.dumps(ctx, ensure_ascii=False)

        txt = await self._chat_traced("advice_json_basic", sys, user, temperature=0.2)
        try:
            obj = json.loads(txt) if isinstance(txt, str) else {}
            adv = obj.get("advice")
            if not isinstance(adv, list):
                return []
            adv = [str(a).strip() for a in adv if str(a).strip()]
            return adv[:5]
        except Exception:
            return []

    async def _rag_advice_from_result(self, tool_name: str, result: dict, payload: dict, field_labels: dict):
        import os, json
        baseline = result.get("baseline")
        optimized = result.get("optimized") or result.get("tax") or result.get("optimized_tax")
        mode = (result.get("mode") or payload.get("op") or "").lower()
        diff = result.get("diff") or {}
        constraints = result.get("constraints") or (payload.get("user_params") or {}).get("constraints") or {}
        final_params = result.get("final_params") or {}
        labels = field_labels or {}

        main_q = self._mk_rag_query_from_ctx(
            tool_name=tool_name,
            mode=mode,
            budget=(payload.get("user_params") or {}).get(TOOL_MAP.get(tool_name, {}).get("budget_field"))
            or self._first_hit(result, self.SPEC_DEFAULT["paths"]["budget"]),
            constraints=constraints,
            diff=diff,
            input_params=(payload.get("user_params") or {}),
            final_params=final_params,
        )

        queries = [
            main_q,
            f"{main_q} documentation recognition cap",
            f"{main_q} exemption deduction ratio limit",
        ]

        persist_dir = os.getenv("RAG_CHROMA_DIR", "rag/chroma")
        collection = os.getenv("RAG_COLLECTION", "tax_handbook")
        if not os.path.isdir(persist_dir):
            no_rag = await self._advice_from_result_no_rag(tool_name, result, payload, labels)
            return (no_rag[:5], [])

        try:
            db = Chroma(
                collection_name=collection,
                persist_directory=persist_dir,
                embedding_function=OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small")),
            )
        except Exception:
            no_rag = await self._advice_from_result_no_rag(tool_name, result, payload, labels)
            return (no_rag[:5], [])

        k = int(os.getenv("RAG_K", "8"))
        min_score_default = float(os.getenv("RAG_MIN_SCORE", "0.15"))
        sources, chunks = [], []

        def _push_hit(h, score=None):
            meta = h.metadata or {}
            page = meta.get("page") or meta.get("loc")
            title = meta.get("title") or meta.get("source") or "Handbook"
            url = meta.get("url")
            txt = (h.page_content or "").strip().replace("\n", " ")
            if len(txt) > 500:
                txt = txt[:500] + "…"
            sources.append({"title": str(title), "page": page, "url": url, "chunk": txt})
            chunks.append(txt)

        try:
            t_sim0 = time.perf_counter()

            for q in queries:
                try:
                    mmr_hits = db.max_marginal_relevance_search(q, k=k, fetch_k=k * 3)
                    for h in mmr_hits:
                        _push_hit(h)
                except Exception:
                    pass

            for q in queries:
                try:
                    scored = db.similarity_search_with_score(q, k=k)
                    scores = [float(s) for _, s in scored if isinstance(s, (int, float))]
                    thresh = min_score_default
                    if scores and min(scores) < 0.05:
                        thresh = 0.0
                    for h, s in scored:
                        try:
                            if float(s) >= thresh:
                                _push_hit(h, s)
                        except Exception:
                            _push_hit(h, None)
                except Exception:
                    try:
                        for h in db.similarity_search(q, k=k):
                            _push_hit(h, None)
                    except Exception:
                        pass

            self._perf_add("rag:similarity_total", time.perf_counter() - t_sim0)

            uniq = []
            seen = set()
            for src in sources:
                key = src["chunk"][:120]
                if key not in seen:
                    seen.add(key)
                    uniq.append(src)
            sources = uniq[:8]
            chunks = [s["chunk"] for s in sources]

        except Exception:
            sources, chunks = [], []

        if chunks:
            diffs = result.get("diff") or {}
            final_params = result.get("final_params") or {}
            deltas = []
            if isinstance(diffs, dict):
                for k, meta in diffs.items():
                    if not isinstance(meta, dict):
                        continue
                    orig = meta.get("original")
                    optv = meta.get("optimized")
                    delta = meta.get("difference")
                    if not isinstance(orig, (int, float)) or not isinstance(optv, (int, float)):
                        continue
                    deltas.append(
                        {
                            "key": k,
                            "label": labels.get(k, k),
                            "original": orig,
                            "optimized": optv,
                            "delta": delta if isinstance(delta, (int, float)) else (optv - orig),
                        }
                    )
            ctx = {
                "tool_name": tool_name,
                "mode": mode,
                "baseline": baseline,
                "optimized": optimized,
                "deltas": deltas,
                "constraints": constraints,
                "final_params": final_params,
                "field_labels": labels,
                "handbook_evidence": chunks[:6],
            }
            sys = (
                "You are an actionable-advice generator for Taiwan tax optimization reports.\n"
                "You MUST base advice ONLY on ctx.deltas. Evidence text can only explain WHY tax changes, and must not introduce new variables.\n"
                "Strict rules:\n"
                "1) Each advice item must cite at least one delta (label + original→optimized). Do NOT mention variables not present in ctx.deltas.\n"
                "2) Each advice item must include the underlying mechanism and explain the tax/payment number and how it is computed.\n"
                "3) Do NOT guess the user's situation. Describe only what this run changed and why it affects tax.\n"
                "4) Output in English. One advice item per line.\n"
                "5) If ctx.deltas is empty: still produce a small amount of advice. Do NOT output: "
                "\"There were no adjustable variable changes, therefore no adjustment advice can be provided.\"\n"
                "   - First explain: optimal equals baseline / no better solution under current constraints.\n"
                "   - Then outline the computation workflow and propose next steps (allow more free_vars / add structural constraints).\n"
                "Output MUST be strict JSON: {\"advice\":[\"...\"]}. No extra text."
            )
            user = json.dumps(ctx, ensure_ascii=False)

            txt = await self._chat_traced("advice_json_basic", sys, user, temperature=0.2)
            try:
                obj = json.loads(txt) if isinstance(txt, str) else {}
                adv = obj.get("advice")
                if isinstance(adv, list):
                    adv = [str(a).strip() for a in adv if str(a).strip()]
                    if adv:
                        return (adv[:5], sources)
            except Exception:
                pass

        no_rag = await self._advice_from_result_no_rag(tool_name, result, payload, labels)
        return (no_rag[:5], [])

    def _effective_params(self, final_params: dict | None, input_params: dict | None) -> dict:
        """
        Override input_params with final_params.value to get an "effective parameter view" for making suggestions.
        """
        base = dict(input_params or {})
        if isinstance(final_params, dict):
            for k, meta in final_params.items():
                if not isinstance(k, str) or not isinstance(meta, dict):
                    continue
                if "value" in meta:
                    base[k] = meta["value"]
        return base

    async def _render_once_with_llm(self, draft_md: str, result: dict, tool_name: str, field_labels: dict) -> str:
        import re, json

        def _extract_numbers(s: str):
            return list(dict.fromkeys(re.findall(r"\d[\d,]*(?:\.\d+)?", s or "")))

        def _extract_var_like_tokens(s: str):
            toks = re.findall(r"[A-Za-z_][A-Za-z0-9_\.]*", s or "")
            ban = {"http", "https", "json"}
            return [t for t in list(dict.fromkeys(toks)) if t.lower() not in ban]

        must_keep = _extract_numbers(draft_md) + _extract_numbers(json.dumps(result, ensure_ascii=False))
        must_keep = list(dict.fromkeys(must_keep))

        allowed_vars = _extract_var_like_tokens(draft_md)

        def _clean_label(ss: str) -> str:
            return re.sub(r"[（(].*?[）)]", "", ss or "").replace("？", "").strip() or (ss or "")

        if field_labels:
            allowed_vars += [_clean_label(v) for v in field_labels.values() if isinstance(v, str)]
        allowed_vars = list(dict.fromkeys([t for t in allowed_vars if t]))

        sys = (
            "You are a professional tax report writer.\n"
            "Rewrite the provided 'Draft Markdown' into the final **conclusion report** (Markdown).\n"
            "Do NOT add any new numbers. The following numbers MUST be preserved verbatim.\n"
            "Do NOT fabricate facts.\n"
            "Section order is fixed: Summary → Key Metrics → Parameter Changes → Constraints → Conclusion → Compliance Notes.\n"
            "In the 'Conclusion' section, you MUST include:\n"
            "(A) Tax computation workflow: describe the path from income/tax base → deductions/credits → tax due/net tax, "
            "using only fields and numbers already in the draft.\n"
            "(A-2) Explain how each tax amount is computed: for every tax-related number in the draft "
            "(tax due, net tax, baseline tax, optimized tax, difference, etc.), explain its source and formula.\n"
            "Write each tax amount using explicit arithmetic/formulas (e.g., A = B − C − D, or A = (B × rate) − credit), "
            "and substitute numbers that already appear in the draft.\n"
            "If the draft lacks enough detail to fully expand the arithmetic, state which fields/rates/credits it depends on "
            "and explicitly list what is missing.\n"
            "(B) Rationale for parameter changes: explain how each change affects tax, tied to the relevant mechanism "
            "(deduction caps, exemptions, brackets, dividend scheme comparison, etc.). Do not speculate.\n"
            "If there are no parameter changes, you must still explain the workflow, why optimal equals baseline under current constraints, "
            "and next steps to expand the optimization space (allow more free_vars / add structural constraints).\n"
            "Do NOT output the sentence: "
            "\"There were no adjustable variable changes, therefore no adjustment advice can be provided.\"\n"
            "Tone: consultant-like, precise, concise.\n"
            "Output ONLY the Markdown body."
        )

        user = (
            "【Numbers that must be kept verbatim】\n"
            + json.dumps(must_keep, ensure_ascii=False)
            + "\n\n【Allowed variable names】\n"
            + json.dumps(allowed_vars, ensure_ascii=False)
            + "\n\n【Field labels (may be used to translate variables)】\n"
            + json.dumps(field_labels or {}, ensure_ascii=False)
            + "\n\n【Draft Markdown】\n"
            + (draft_md or "")
        )

        raw = await self._chat(sys, user, temperature=0.1)
        final_md = raw.strip() if isinstance(raw, str) else ""

        final_nums = set(_extract_numbers(final_md))
        if not set(must_keep).issubset(final_nums):
            _append_debug(self.memory, "[ReasoningAgent] single-pass number guard failed -> fallback")
            return ""
        return final_md

    def _call_external_renderer(self, tool_name: str, result: dict, payload: dict, field_labels: dict | None):
        meta = TOOL_MAP.get(tool_name, {}) or {}
        rr_path = meta.get("report_renderer")
        if not rr_path:
            return None
        try:
            mod, fn = (rr_path.split(":", 1) + ["render_report"])[:2]
            _mod = __import__(mod, fromlist=["*"])
            rr = getattr(_mod, fn)
            md = rr(result=result, payload=payload or {}, tool_name=tool_name, field_labels=field_labels or {})
            if isinstance(md, str) and md.strip():
                return md
        except Exception as e:
            _append_debug(self.memory, f"[ReasoningAgent] external report_renderer error: {e}")
        return None

    def _payload_with_constraints_reset(self, payload: dict) -> dict:
        """
        For returning to the constraints setup phase:
        - Clear current constraints / free_vars
        - Keep other user_params inputs (income, amounts, etc.)
        - Keep existing early_tips_md only (no recalculation / overwrite)
        """
        new_payload = dict(payload or {})
        up = dict((new_payload.get("user_params") or {}))

        # Clear constraints/adjustable fields
        up["constraints"] = {}
        up["free_vars"] = []
        # If you also want to clear bounds, uncomment:
        # up["bounds"] = {}

        new_payload["user_params"] = up

        # Keep existing early_tips_md only; do not recalc or add flags
        if "early_tips_md" in payload and isinstance(payload["early_tips_md"], str):
            new_payload["early_tips_md"] = payload["early_tips_md"]
        else:
            new_payload.pop("early_tips_md", None)

        # Mark interaction mode: reset (for upstream/UI logic)
        new_payload["__constraint_mode__"] = "reset"

        # Ensure no leftover "regenerate" flag
        new_payload.pop("__regenerate_early_tips__", None)

        return new_payload

    async def handle(self, exec_output: Dict[str, Any]) -> Dict[str, Any]:
        self._perf_reset()
        self._perf_handle_t0 = time.perf_counter()

        def safe_json(obj: _Any, *, max_len: int = 800, max_keys: int = 50) -> str:
            """Best-effort JSON for debug with truncation."""
            try:
                if isinstance(obj, dict):
                    keys = sorted(list(obj.keys()))[:max_keys]
                    slim = {k: obj.get(k) for k in keys}
                    s = _json.dumps(slim, ensure_ascii=False, default=str)
                elif isinstance(obj, list):
                    s = _json.dumps(obj[:max_keys], ensure_ascii=False, default=str)
                else:
                    s = _json.dumps(obj, ensure_ascii=False, default=str)
            except Exception:
                s = str(obj)
            if len(s) > max_len:
                s = s[:max_len] + f"... <truncated {len(s)-max_len} chars>"
            return s

        def dbg(tag: str, data: _Any = None):
            if data is None:
                print(f"[ReasoningAgentDBG] {tag}")
            else:
                print(f"[ReasoningAgentDBG] {tag}: {safe_json(data)}")

        def _coerce_num(x):
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, str):
                s = x.strip()
                if not s:
                    return None
                s = s.replace(",", "").replace("，", "").replace(" ", "")
                try:
                    return float(s)
                except Exception:
                    return None
            return None

        dbg("ENTER handle()")
        dbg(
            "exec_output.head",
            {
                "has_result": isinstance(exec_output.get("result"), dict),
                "has_payload": isinstance(exec_output.get("payload"), dict),
                "tool_name": exec_output.get("tool_name"),
            },
        )

        # === Freeze previous snapshot at the very beginning (read-only) ===
        hist = self.memory.get("history") or []
        prev_frozen = hist[-1] if isinstance(hist, list) and hist else {}
        dbg("history.last.exists", bool(prev_frozen))

        # Fallback: legacy __prev_* (or external prev_run_snapshot)
        if not prev_frozen:
            prev_frozen = {
                "tax": self.memory.get("__prev_tax__"),
                "final_params": self.memory.get("__prev_final_params__"),
                "constraints": self.memory.get("__prev_constraints__"),
            } or {}
            dbg("prev_frozen.from_legacy_mem", prev_frozen)

        _payload0 = exec_output.get("payload") or {}
        if not prev_frozen.get("final_params"):
            pf = _payload0.get("__prev_final_params__")
            if isinstance(pf, dict):
                prev_frozen["final_params"] = pf
                dbg("prev_frozen.set.final_params.from_exec_payload")
        if not prev_frozen.get("constraints"):
            pc = _payload0.get("__prev_constraints__")
            if isinstance(pc, dict):
                prev_frozen["constraints"] = pc
                dbg("prev_frozen.set.constraints.from_exec_payload")
        if prev_frozen.get("tax") is None:
            pt = _payload0.get("__prev_tax__")
            pt_num = _coerce_num(pt)
            if isinstance(pt_num, (int, float)):
                prev_frozen["tax"] = float(pt_num)
                dbg("prev_frozen.set.tax.from_exec_payload", prev_frozen["tax"])

        try:
            lep = self.memory.get("last_exec_payload") or {}
            lep_payload = lep.get("payload") or {}
            if isinstance(lep_payload, dict):
                if prev_frozen.get("tax") is None and isinstance(lep_payload.get("__prev_tax__"), (int, float)):
                    prev_frozen["tax"] = float(lep_payload["__prev_tax__"])
                    dbg("prev_frozen.set.tax.from_last_exec_payload", prev_frozen["tax"])
                if not isinstance(prev_frozen.get("final_params"), dict) and isinstance(
                    lep_payload.get("__prev_final_params__"), dict
                ):
                    prev_frozen["final_params"] = lep_payload["__prev_final_params__"]
                    dbg("prev_frozen.set.final_params.from_last_exec_payload")
                if not isinstance(prev_frozen.get("constraints"), dict) and isinstance(
                    lep_payload.get("__prev_constraints__"), dict
                ):
                    prev_frozen["constraints"] = lep_payload["__prev_constraints__"]
                    dbg("prev_frozen.set.constraints.from_last_exec_payload")
        except Exception as e:
            dbg("last_exec_payload.lookup.error", str(e))

        try:
            pend = self.memory.get("pending_constraint_payload") or {}
            if isinstance(pend, dict):
                if prev_frozen.get("tax") is None and isinstance(pend.get("__prev_tax__"), (int, float)):
                    prev_frozen["tax"] = float(pend["__prev_tax__"])
                    dbg("prev_frozen.set.tax.from_pending")
                if not isinstance(prev_frozen.get("final_params"), dict) and isinstance(
                    pend.get("__prev_final_params__"), dict
                ):
                    prev_frozen["final_params"] = pend["__prev_final_params__"]
                    dbg("prev_frozen.set.final_params.from_pending")
                if not isinstance(prev_frozen.get("constraints"), dict) and isinstance(
                    pend.get("__prev_constraints__"), dict
                ):
                    prev_frozen["constraints"] = pend["__prev_constraints__"]
                    dbg("prev_frozen.set.constraints.from_pending")
        except Exception as e:
            dbg("pending_constraint_payload.lookup.error", str(e))

        try:
            snap = self.memory.get("prev_run_snapshot") or {}
            if isinstance(snap, dict):
                if prev_frozen.get("tax") is None and isinstance(snap.get("tax"), (int, float)):
                    prev_frozen["tax"] = float(snap["tax"])
                    dbg("prev_frozen.set.tax.from_prev_run_snapshot")
                if not isinstance(prev_frozen.get("final_params"), dict) and isinstance(snap.get("final_params"), dict):
                    prev_frozen["final_params"] = snap["final_params"]
                    dbg("prev_frozen.set.final_params.from_prev_run_snapshot")
                if not isinstance(prev_frozen.get("constraints"), dict) and isinstance(snap.get("constraints"), dict):
                    prev_frozen["constraints"] = snap["constraints"]
                    dbg("prev_frozen.set.constraints.from_prev_run_snapshot")
        except Exception as e:
            dbg("prev_run_snapshot.lookup.error", str(e))

        dbg(
            "prev_frozen.summary",
            {
                "has_prev": bool(prev_frozen),
                "prev_tax": prev_frozen.get("tax"),
                "fp_keys": list((prev_frozen.get("final_params") or {}).keys())[:8],
                "has_constraints": isinstance(prev_frozen.get("constraints"), dict),
            },
        )

        result = exec_output.get("result") or {}
        tool_name = exec_output.get("tool_name", "")
        payload = exec_output.get("payload") or self.memory.get("pending_constraint_payload") or {}

        dbg("tool_name", tool_name)
        dbg("result.keys", list(result.keys()) if isinstance(result, dict) else type(result).__name__)
        dbg("payload.keys", list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__)

        _payload_prev_tax = _coerce_num((exec_output.get("payload") or {}).get("__prev_tax__"))
        _payload_prev_fp = (exec_output.get("payload") or {}).get("__prev_final_params__")
        _payload_prev_cons = (exec_output.get("payload") or {}).get("__prev_constraints__")

        try:
            prev_tax_external = float(_payload_prev_tax) if isinstance(_payload_prev_tax, (int, float)) else None
        except Exception:
            prev_tax_external = None

        prev_final_params = _payload_prev_fp if isinstance(_payload_prev_fp, dict) else None
        prev_constraints = _payload_prev_cons if isinstance(_payload_prev_cons, dict) else None

        if prev_tax_external is None:
            try:
                ptm = prev_frozen.get("tax")
                prev_tax_external = float(ptm) if isinstance(ptm, (int, float)) else None
            except Exception:
                prev_tax_external = None
        if not isinstance(prev_final_params, dict):
            prev_final_params = prev_frozen.get("final_params") if isinstance(prev_frozen.get("final_params"), dict) else None
        if not isinstance(prev_constraints, dict):
            prev_constraints = prev_frozen.get("constraints") if isinstance(prev_frozen.get("constraints"), dict) else None

        dbg(
            "prev_external",
            {
                "tax_from_payload": (exec_output.get("payload") or {}).get("__prev_tax__"),
                "effective_prev_tax": prev_tax_external,
                "has_prev_final_params": isinstance(prev_final_params, dict),
                "has_prev_constraints": isinstance(prev_constraints, dict),
            },
        )

        labels = (TOOL_MAP.get(tool_name, {}).get("field_labels", {}) or {}).copy()
        res_labels = (result or {}).get("field_labels")
        if isinstance(res_labels, dict):
            for k, v in res_labels.items():
                if isinstance(k, str) and isinstance(v, str):
                    labels[k] = v

        spec = dict(self.SPEC_DEFAULT)
        spec.update(self.SPEC_BY_TOOL.get(tool_name, {}))
        paths = spec["paths"]
        pfb = spec.get("payload_fallbacks", {})

        dbg("paths", paths)
        dbg("payload_fallbacks", pfb)

        baseline = self._first_hit(result, paths.get("baseline", []))
        optimized = self._first_hit(result, paths.get("optimized", []))
        optimized_sales = self._first_hit(result, paths.get("optimized_sales", []))
        mode = self._first_hit(result, paths.get("mode", []))
        status = self._first_hit(result, paths.get("status", []))
        diff = self._first_hit(result, paths.get("diff", []))
        final_params = self._first_hit(result, paths.get("final_params", []))
        input_params = self._first_hit(result, paths.get("input_params", []))
        constraints = self._first_hit(result, paths.get("constraints", []))

        dbg("raw.kpis", {"baseline": baseline, "optimized": optimized, "mode": mode, "status": status})
        dbg("raw.final_params.exists", isinstance(final_params, dict))
        dbg("raw.constraints.exists", isinstance(constraints, dict))

        if constraints is None:
            constraints = self._first_hit(payload, pfb.get("constraints", []))
            dbg("constraints.from_payload_fallback", isinstance(constraints, dict))
        if input_params is None:
            input_params = self._first_hit(payload, pfb.get("input_params", []))
            dbg("input_params.from_payload_fallback", isinstance(input_params, dict))
        if mode is None:
            mode = self._first_hit(payload, pfb.get("mode", []))
            dbg("mode.from_payload_fallback", mode)

        optimized_sales = _coerce_num(optimized_sales)

        fallback_total_qty = None
        try:
            if (optimized_sales is None) and isinstance(final_params, dict) and final_params:
                s = 0.0
                hit = False
                for k, meta in final_params.items():
                    if isinstance(k, str) and k.endswith(".quantity") and isinstance(meta, dict):
                        v = _coerce_num(meta.get("value"))
                        if v is not None:
                            s += v
                            hit = True
                if hit:
                    fallback_total_qty = s
        except Exception as e:
            dbg("fallback_total_qty.error", str(e))
            fallback_total_qty = None

        status_str = str((result or {}).get("status") or status or "").lower()
        no_solution_flag = bool((result or {}).get("no_solution")) or status_str in {"infeasible", "unsat", "infeasible_or_unbounded"}
        has_any_kpis = (
            isinstance(optimized, (int, float))
            or isinstance(optimized_sales, (int, float))
            or isinstance(baseline, (int, float))
            or (isinstance(final_params, dict) and len(final_params) > 0)
        )
        dbg("solution.flags", {"no_solution_flag": no_solution_flag, "has_any_kpis": has_any_kpis})

        if no_solution_flag and not has_any_kpis:
            budget_field = TOOL_MAP.get(tool_name, {}).get("budget_field")
            user_params = (payload.get("user_params") or {}) if isinstance(payload, dict) else {}
            budget = user_params.get(budget_field)
            cons_for_view = (result.get("constraints") or constraints or user_params.get("constraints"))
            in_params = (result.get("input_params") or input_params or user_params)
            dbg("no_solution.context", {"budget_field": budget_field, "budget": budget})

            def fmt(x):
                try:
                    return f"{float(x):,.0f}"
                except Exception:
                    return str(x)

            md = []
            md.append(f"### {TOOL_MAP.get(tool_name, {}).get('description', tool_name)}")
            md.append(
                f"**No feasible solution**: Under the current constraints and tax budget "
                f"({fmt(budget) if budget else 'not provided'}), there is no combination that satisfies all constraints "
                "while staying within the budget."
            )
            if isinstance(in_params, dict) and len(in_params) > 0:
                pruned = {k: v for k, v in in_params.items() if k != "constraints"}
                if pruned:
                    md.append("#### Your key inputs (excerpt)")
                    md.append("```json\n" + _json.dumps(pruned, ensure_ascii=False, indent=2, default=str) + "\n```")

            md.append("#### Applied constraints")
            md.append(self._format_constraints(tool_name, cons_for_view))

            advice = [
                "Allow the necessary variables to vary so the solver has room to adjust.",
                f"Increase the tax budget (current: {fmt(budget)})." if budget else "Provide a tax budget (e.g., `tax budget 5,000,000`).",
                "Or switch to 'minimize' mode first to estimate the minimum tax under these constraints.",
            ]
            md.append("#### Conclusion\n" + "\n".join(f"- {s}" for s in advice))

            self.memory.set("last_tool", tool_name)
            self.memory.set("last_exec_payload", {"tool_name": tool_name, "payload": payload})
            dbg("RETURN.no_solution")
            return {
                "type": "final_feedback",
                "text": "\n\n".join(md),
                "perf_spans": self._perf_finalize_spans(),
                "perf_spans": self._perf_finalize_spans(),
                "raw_result": result,
                "next_actions_hint": (
                    "Want to change constraints? Reply \"Add constraints\" to add new constraints on top of the current ones; "
                    "reply \"Reset constraints\" to clear everything and return to setup. "
                    "If you want to export this run's report, type \"Finalize report\"."
                ),
            }

        external_md = self._call_external_renderer(tool_name, result, payload, labels)
        dbg("external_md.exists", bool(external_md))

        budget_field = TOOL_MAP.get(tool_name, {}).get("budget_field")
        user_params = (payload.get("user_params") or {}) if isinstance(payload, dict) else {}
        budget_from_params = user_params.get(budget_field) if budget_field else None
        budget_from_result = self._first_hit(result, spec["paths"].get("budget", []))
        budget_val = budget_from_params if budget_from_params is not None else budget_from_result
        dbg("budget", {"field": budget_field, "from_params": budget_from_params, "from_result": budget_from_result, "final": budget_val})

        is_maximize = (str(mode).lower() == "maximize")
        dbg("mode.is_maximize", is_maximize)

        compare_base = None
        compare_label = ""
        if isinstance(prev_tax_external, (int, float)):
            compare_base = float(prev_tax_external)
            compare_label = "previous run"
        elif isinstance(baseline, (int, float)):
            compare_base = float(baseline)
            compare_label = "baseline"
        dbg("compare_base.selected", {"label": compare_label, "value": compare_base})

        def _fm(x):
            try:
                return f"{float(x):,.0f}"
            except Exception:
                return str(x)

        _free_vars = result.get("free_vars")
        if not isinstance(_free_vars, (list, tuple)):
            _free_vars = user_params.get("free_vars") if isinstance(user_params.get("free_vars"), list) else []
        free_vars_list = list(_free_vars or [])
        dbg("free_vars", free_vars_list)

        if is_maximize:
            if len(free_vars_list) == 0:
                msg = []
                msg.append(f"### {TOOL_MAP.get(tool_name, {}).get('description', tool_name)}")
                if isinstance(baseline, (int, float)) and isinstance(budget_val, (int, float)):
                    if baseline > budget_val:
                        gap = float(baseline) - float(budget_val)
                        msg.append("**No feasible solution**: No variables are allowed to change, and the unadjusted input tax already exceeds your budget.")
                        msg.append(f"- Input tax: {_fm(baseline)}")
                        msg.append(f"- Tax budget: {_fm(budget_val)}")
                        msg.append(f"- Exceeds by: {_fm(gap)}")
                    else:
                        msg.append("**Cannot increase output**: No variables are allowed to change, so the model cannot adjust the combination to increase total output while staying within the budget.")
                        msg.append(f"- Input tax: {_fm(baseline)} (within budget {_fm(budget_val)})")
                else:
                    msg.append("**Cannot increase output**: No variables are allowed to change, so the model cannot adjust any amounts/quantities.")
                dbg("RETURN.maximize.no_free_vars")
                return {"type": "final_feedback", "text": "\n".join(msg), "perf_spans": self._perf_finalize_spans(), "raw_result": result}

            if isinstance(baseline, (int, float)) and isinstance(budget_val, (int, float)) and baseline > budget_val:
                gap = float(baseline) - float(budget_val)
                msg = [
                    f"### {TOOL_MAP.get(tool_name, {}).get('description', tool_name)}",
                    "**No feasible solution**: Even without any adjustments, the tax already exceeds your budget.",
                    f"- Input tax (no adjustment): {_fm(baseline)}",
                    f"- Tax budget: {_fm(budget_val)}",
                    f"- Exceeds by: {_fm(gap)}",
                    "#### Conclusion",
                    "- First run in 'minimize tax' mode to estimate the minimum tax under the current constraints. If the minimum still exceeds the budget, you must raise the budget or relax constraints.",
                ]
                dbg("RETURN.maximize.baseline_over_budget")
                return {
                    "type": "final_feedback",
                    "text": "\n".join(msg),
                    "perf_spans": self._perf_finalize_spans(),
                    "perf_spans": self._perf_finalize_spans(),
                    "raw_result": result,
                    "next_actions_hint": (
                        "Want to change constraints? Reply \"Add constraints\" to add new constraints on top of the current ones; "
                        "reply \"Reset constraints\" to clear everything and return to setup. "
                        "If you want to export this run's report, type \"Finalize report\"."
                    ),
                }

        shown_total_qty = optimized_sales if optimized_sales is not None else fallback_total_qty
        dbg("shown_total_qty", shown_total_qty)

        if is_maximize and isinstance(shown_total_qty, (int, float)):
            used_tax = optimized if isinstance(optimized, (int, float)) else None
            metric_name = "Max quantity" if tool_name in {"cargo_tax", "ta_tax"} else "Max sales"
            if isinstance(used_tax, (int, float)) and isinstance(budget_val, (int, float)) and budget_val > 0:
                used_pct = used_tax / budget_val
                headroom = max(budget_val - used_tax, 0)
                summary = (
                    f"Under the current assumptions and constraints, **{metric_name}** is **{self._fmt_money(shown_total_qty)}**. "
                    f"Tax used: {self._fmt_money(used_tax)} / budget {self._fmt_money(budget_val)} "
                    f"(usage {used_pct:.1%}, remaining {self._fmt_money(headroom)})."
                )
            else:
                tail = f"; tax used {self._fmt_money(used_tax)}" if isinstance(used_tax, (int, float)) else ""
                summary = f"Under the current assumptions and constraints, **{metric_name}** is **{self._fmt_money(shown_total_qty)}**{tail}."
        else:
            if isinstance(optimized, (int, float)) and isinstance(compare_base, (int, float)):
                delta_tax = float(optimized) - float(compare_base)
                dbg("compare_base.delta", {"label": compare_label, "value": compare_base, "delta": delta_tax})
                summary = (
                    f"Under the current assumptions and constraints, the optimal tax is **{self._fmt_money(optimized)}** "
                    f"(vs. {compare_label} {self._fmt_money(compare_base)}, {self._trend(delta_tax)})."
                )
            elif isinstance(optimized, (int, float)):
                summary = f"Under the current assumptions and constraints, the optimal tax is **{self._fmt_money(optimized)}**."
            elif isinstance(baseline, (int, float)):
                summary = f"Under the current assumptions and constraints, the input tax is **{self._fmt_money(baseline)}**."
            else:
                summary = "Under the current assumptions and constraints, tax cannot be computed yet."

        budget_missing = (str(mode).lower() == "maximize" and not user_params.get(budget_field))
        effective_params = self._effective_params(final_params, input_params)
        dbg("effective_params.exists", isinstance(effective_params, dict))
        dbg("budget_missing", budget_missing)

        def has_constraint(substr: str):
            import json
            return substr in json.dumps(constraints, ensure_ascii=False) if constraints else False

        def missing_or_zero(keys: list[str]):
            d = effective_params or {}
            for k in keys:
                v = d.get(k)
                if v is None:
                    return True
                if isinstance(v, (int, float)) and abs(v) < 1e-9:
                    return True
                if isinstance(v, str):
                    try:
                        f = float(v.replace(",", ""))
                        if abs(f) < 1e-9:
                            return True
                    except Exception:
                        if not v.strip():
                            return True
            return False

        _delta_tax = None
        if isinstance(optimized, (int, float)) and isinstance(compare_base, (int, float)):
            _delta_tax = float(optimized) - float(compare_base)

        ctx = {
            "delta_tax": _delta_tax,
            "mode_str": str(mode or ""),
            "budget_missing": budget_missing,
            "has_constraint": has_constraint,
            "missing_or_zero": missing_or_zero,
            "has_constraints": bool(constraints),
        }
        dbg("ctx", ctx)

        suggestions = []
        suggestions += self._eval_rules(self.SPEC_DEFAULT.get("suggestion_rules", []), ctx)
        tool_rules = self.SPEC_BY_TOOL.get(tool_name, {}).get("suggestion_rules")
        if tool_rules:
            suggestions += self._eval_rules(tool_rules, ctx)
        dbg("suggestions.count", len(suggestions))

        def _safe_label(key: str) -> str:
            try:
                return self._label(tool_name, key)
            except Exception:
                try:
                    tail = key.split(".")[-1]
                    return self._label(tool_name, tail)
                except Exception:
                    return key

        def _extract_val(meta_or_scalar):
            if isinstance(meta_or_scalar, dict):
                v = meta_or_scalar.get("value")
                try:
                    return float(v)
                except Exception:
                    return None
            try:
                return float(meta_or_scalar)
            except Exception:
                return None

        prev_diff_lines = []
        keys_debug_sample = []
        if isinstance(final_params, dict) and isinstance(prev_final_params, dict):
            eps = 1e-6
            keys = set(final_params.keys()) | set(prev_final_params.keys())
            keys_debug_sample = sorted(list(keys))[:10]
            for k in sorted(keys):
                try:
                    cur_v = _extract_val(final_params.get(k))
                    prv_v = _extract_val(prev_final_params.get(k))
                    if cur_v is None and prv_v is None:
                        continue
                    is_diff = (cur_v is None) or (prv_v is None) or (abs(float(cur_v) - float(prv_v)) > eps)
                    if is_diff:
                        lab = _safe_label(k)
                        try:
                            delta = (0.0 if cur_v is None else float(cur_v)) - (0.0 if prv_v is None else float(prv_v))
                        except Exception:
                            delta = 0.0
                        left = self._fmt_money(prv_v) if prv_v is not None else str(prv_v)
                        right = self._fmt_money(cur_v) if cur_v is not None else str(cur_v)
                        trend = self._trend(delta)
                        prev_diff_lines.append(f"- {lab}: {left} → {right} ({trend})")
                except Exception as e:
                    dbg("prev_diff.item.error", {"key": k, "err": str(e)})

        dbg("prev_diff.keys.sample", keys_debug_sample)
        dbg("prev_diff.count", len(prev_diff_lines))

        if not prev_diff_lines:
            prev_diff_lines = ["(No previous run to compare.)"]

        def _binding_constraints_list(tool: str, cons: dict | None, finals: dict | None, limit: int = 4) -> list[str]:
            if not isinstance(cons, dict) or not isinstance(finals, dict):
                return []
            eps2 = 1e-6
            hits = []
            for expr, ops in cons.items():
                if not isinstance(ops, dict):
                    continue
                var = str(expr).strip()
                meta = finals.get(var) if isinstance(finals.get(var), dict) else None
                if not isinstance(meta, dict):
                    continue
                val = meta.get("value")
                if not isinstance(val, (int, float)):
                    continue
                for op, b in ops.items():
                    if not isinstance(b, (int, float)):
                        continue
                    tight = (abs(val - b) <= eps2) if op in {"==", ">=", "<="} else False
                    if tight:
                        label = self._label(tool, var.split(".")[-1])
                        hits.append(f"{label} is tight at {op} {self._fmt_money(b)} (current {self._fmt_money(val)})")
                        break
                if len(hits) >= limit:
                    break
            return hits

        if is_maximize and isinstance(budget_val, (int, float)):
            better: list[str] = []
            if isinstance(optimized, (int, float)) and budget_val > 0:
                headroom = max(budget_val - optimized, 0)
                used_pct = optimized / budget_val
                better.append(f"Budget usage: ~{used_pct:.1%}; remaining headroom {self._fmt_money(headroom)}.")
                if status_str in {"unsat", "infeasible", "infeasible_or_unbounded"} or headroom > 0:
                    better.append(
                        "This is already the maximum output within your constraints and budget; the remaining headroom is not enough to add another taxable unit under current pricing/tax rules."
                    )
            binds = _binding_constraints_list(tool_name, constraints, final_params, limit=4)
            if binds:
                better.append("These variables are binding (at your boundary), which may be the main reason you cannot increase output:")
                better.extend(f"  - {b}" for b in binds)
            better.append("Practical levers to increase output:")
            better.append("  - Increase the tax budget (if policy/budget allows).")
            better.append("  - Relax binding boundaries (turn equalities into ranges, adjust min/max), or allow them as free_vars.")
            better.append("  - Reduce the tax base (e.g., price/spec/ABV/category) so each additional unit costs less tax.")
            better.append("  - Allow more fields to vary for cross-variable reallocation (e.g., quantity / price).")
            if better:
                suggestions = better

        if not suggestions:
            suggestions = [
                "If you have a specific goal (e.g., total tax ≤ X), add a tax budget or structural constraints like ratios/bounds to let the solver explore more flexible combinations."
            ]

        try:
            rag_sugs, rag_sources = await self._rag_advice_from_result(tool_name, result, payload, labels)
        except Exception as e:
            dbg("rag_advice.error", str(e))
            rag_sugs, rag_sources = [], []

        final_suggestions = rag_sugs if rag_sugs else suggestions
        dbg("final_suggestions.count", len(final_suggestions))

        ref_md = ""
        if rag_sources:
            ref_lines = ["\n> References (retrieval snippets)"]
            for i, src in enumerate(rag_sources[:6], 1):
                loc = f" p.{src['page']}" if src.get("page") is not None else ""
                tail = f" ({src['url']})" if src.get("url") else ""
                ref_lines.append(f"> - [{i}] {src['title']}{loc}: {src['chunk']}{tail}")
            ref_md = "\n".join(ref_lines)

        kpi_lines = []
        if mode is not None:
            kpi_lines.append(f"- **Mode**: {mode}")
        if status is not None:
            kpi_lines.append(f"- **Solve status**: {status}")
        if baseline is not None:
            kpi_lines.append(f"- **Input tax**: {self._fmt_money(baseline)}")
        if isinstance(compare_base, (int, float)) and compare_label:
            kpi_lines.append(f"- **Comparison base**: {compare_label} {self._fmt_money(compare_base)}")
        if optimized is not None:
            kpi_lines.append(f"- **Optimized tax / Used tax**: {self._fmt_money(optimized)}")

        if is_maximize:
            if isinstance(shown_total_qty, (int, float)):
                metric_name = "Max quantity" if tool_name in {"cargo_tax", "ta_tax"} else "Max sales"
                kpi_lines.append(f"- **{metric_name}**: {self._fmt_money(shown_total_qty)}")
            if isinstance(budget_val, (int, float)):
                kpi_lines.append(f"- **Tax budget**: {self._fmt_money(budget_val)}")
                if isinstance(optimized, (int, float)) and budget_val > 0:
                    used_pct = optimized / budget_val
                    headroom = max(budget_val - optimized, 0)
                    kpi_lines.append(f"- **Budget usage**: {used_pct:.1%}")
                    kpi_lines.append(f"- **Remaining headroom**: {self._fmt_money(headroom)}")
            if (str(self._first_hit(result, ["baseline_status"]) or "").lower() in {"unsat", "infeasible"}):
                kpi_lines.append("- **Note**: Even if the baseline is infeasible, an optimized feasible solution may exist under your constraints.")

        input_param_lines: list[str] = []
        if isinstance(input_params, dict):
            labels2 = self._labels(tool_name)
            for k in labels2.keys():
                if k in ("constraints", "free_vars"):
                    continue
                v = input_params.get(k)
                if v is None or v == "":
                    continue
                if isinstance(v, bool):
                    shown = "Yes" if v else "No"
                else:
                    shown = self._fmt_money(v)
                input_param_lines.append(f"- {self._label(tool_name, k)}: {shown}")

        final_lines = []
        if isinstance(final_params, dict):
            for k, meta in final_params.items():
                if isinstance(meta, dict):
                    v = meta.get("value")
                    t = meta.get("type", "fixed")
                    final_lines.append(
                        f"- {self._label(tool_name, k)}: {self._fmt_money(v)} ({'recommended' if t == 'free' else 'default'})"
                    )

        blocks = {
            "summary": f"### {TOOL_MAP.get(tool_name, {}).get('description', tool_name)}\n{summary}",
            "kpis": ("#### Key Metrics\n" + "\n".join(kpi_lines)) if kpi_lines else "",
            "diff": ("#### Parameter Changes (vs. input)\n" + self._format_diff(tool_name, diff)),
            "prev_diff": ("#### Parameter Changes (vs. previous run)\n" + ("\n".join(prev_diff_lines) if prev_diff_lines else "(No previous run to compare.)")),
            "constraints": ("#### Applied Constraints\n" + self._format_constraints(tool_name, constraints)),
            "input_params": ("#### Input Parameters\n" + ("\n".join(input_param_lines) if input_param_lines else "(Input parameters unavailable, or tool did not return them.)")),
            "final_params": ("#### Final Parameters (Optimized)\n" + ("\n".join(final_lines) if final_lines else "(Final parameters unavailable, or no key changes.)")),
            "conclusion": ("#### Conclusion\n" + ("\n".join(f"- {s}" for s in (final_suggestions or [])))) + ref_md,
            "compliance": (
                "#### Risk & Compliance Notes\n"
                "- This report is a **model-derived estimate**. Actual tax outcomes depend on authoritative rules and filing data.\n"
                "- For deductions/recognition/salary structure, follow regulations and supporting documents to avoid compliance risks."
            ),
        }

        dividend_compare_md = ""
        if tool_name == "income_tax" and isinstance(final_params, dict):
            try:
                def _v(k: str):
                    meta = final_params.get(k)
                    return meta.get("value") if isinstance(meta, dict) else meta

                scheme = _v("dividend_scheme_best")
                if scheme:
                    _fm2 = self._fmt_money
                    c_base = _v("tax_combined_progressive_with_dividend")
                    c_credit = _v("tax_combined_dividend_credit_capped")
                    c_net = _v("tax_combined_net")
                    c_due = _v("tax_combined_tax_due")
                    c_ref = _v("tax_combined_refund")

                    s_non = _v("tax_separate_progressive_without_dividend")
                    s_div = _v("tax_separate_dividend_tax_28")
                    s_net = _v("tax_separate_net")

                    dividend_compare_md = (
                        "#### Dividend scheme comparison (combined credit vs. 28% separate)\n"
                        f"- **Best scheme (chosen by comparing both within this run)**: {scheme}\n"
                        f"- Combined: tax {_fm2(c_base)} − credit {_fm2(c_credit)} = net {_fm2(c_net)} (due {_fm2(c_due)} / refund {_fm2(c_ref)})\n"
                        f"- Separate: other-income tax {_fm2(s_non)} + dividend tax {_fm2(s_div)} = net {_fm2(s_net)}\n"
                    )

                    div_solutions = result.get("dividend_scheme_solutions") if isinstance(result, dict) else None
                    if isinstance(div_solutions, dict):
                        def _summ_solution(tag: str, node: dict) -> str:
                            taxv = node.get("optimized")
                            fp = node.get("final_params") or {}
                            free_kv = []
                            if isinstance(fp, dict):
                                for kk, mm in fp.items():
                                    if isinstance(mm, dict) and mm.get("type") == "free":
                                        free_kv.append(f"{self._label(tool_name, kk)}={_fm2(mm.get('value'))}")
                            free_txt = ("; " + ", ".join(free_kv)) if free_kv else ""
                            return f"- {tag}: optimal tax {_fm2(taxv)}{free_txt}"

                        lines = ["\n#### Per-scheme optimization (same assumptions/constraints)"]
                        if "combined_only" in div_solutions:
                            lines.append(_summ_solution("Combined credit only", div_solutions["combined_only"]))
                        if "separate_only" in div_solutions:
                            lines.append(_summ_solution("28% separate only", div_solutions["separate_only"]))
                        dividend_compare_md = dividend_compare_md + "\n" + "\n".join(lines) + "\n"

            except Exception as e:
                dbg("dividend_compare.error", str(e))

        if dividend_compare_md:
            if blocks.get("kpis"):
                blocks["kpis"] = (blocks["kpis"].rstrip() + "\n\n" + dividend_compare_md).strip()
            else:
                blocks["kpis"] = dividend_compare_md.strip()

        sections = list(spec.get("sections", []))
        if "prev_diff" not in sections:
            try:
                di = sections.index("diff")
                sections.insert(di + 1, "prev_diff")
            except Exception:
                sections.append("prev_diff")

        _external_md = self._call_external_renderer(tool_name, result, payload, labels)
        if _external_md:
            draft_md = "\n\n".join([blocks.get("summary", ""), blocks.get("kpis", "")]).strip() + "\n\n" + _external_md
        else:
            draft_md = "\n\n".join([blocks[s] for s in sections if blocks.get(s)])

        dbg("draft_md.len", len(draft_md) if isinstance(draft_md, str) else None)

        try:
            final_once = await self._render_once_with_llm(draft_md, result, tool_name, labels)
            final_md = final_once if final_once else draft_md
        except Exception as e:
            dbg("render_once_with_llm.error", str(e))
            final_md = draft_md

        final_md = final_md.rstrip()
        dbg("final_md.len", len(final_md))

        try:
            self.memory.set("last_report_md", final_md)

            kpis_snapshot = {
                "baseline": baseline if isinstance(baseline, (int, float)) else None,
                "optimized": optimized if isinstance(optimized, (int, float)) else None,
                "mode": mode,
                "status": status,
            }
            title = f"{TOOL_MAP.get(tool_name, {}).get('description', tool_name)} report {datetime.now():%Y-%m-%d %H:%M:%S}"
            latest_bundle = {
                "md": final_md,
                "json": result,
                "kpis": kpis_snapshot,
                "tool_name": tool_name,
                "title": title,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            self.memory.set("__latest_report__", latest_bundle)
            dbg("persist.memory.__latest_report__.set", {"has_md": True, "has_json": bool(result)})

            base_dir = "reports/last_run"
            os.makedirs(base_dir, exist_ok=True)
            md_path = os.path.join(base_dir, "last.md")
            json_path = os.path.join(base_dir, "last.json")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(final_md)
            with open(json_path, "w", encoding="utf-8") as f:
                _json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            dbg("persist.files.written", {"md": md_path, "json": json_path})
        except Exception as e:
            dbg("persist.error", str(e))

        self.memory.set("last_report_md", final_md)
        self.memory.set("last_result", result)
        self.memory.set("last_payload", payload)

        try:
            self.memory.set("__latest_report__", {"md": final_md, "json": result, "ts": __import__("time").time()})
        except Exception:
            pass

        try:
            cur_tax = None
            if isinstance(optimized, (int, float)):
                cur_tax = float(optimized)
            elif isinstance(baseline, (int, float)):
                cur_tax = float(baseline)
            dbg("commit.cur_tax", cur_tax)

            cur_snapshot = {
                "tax": cur_tax,
                "final_params": final_params if isinstance(final_params, dict) else None,
                "constraints": constraints if isinstance(constraints, dict) else None,
            }
            dbg("commit.cur_snapshot.head", {"has_fp": isinstance(cur_snapshot["final_params"], dict), "has_cons": isinstance(cur_snapshot["constraints"], dict)})

            hist = self.memory.get("history") or []
            if not isinstance(hist, list):
                hist = []
            hist.append(cur_snapshot)
            if len(hist) > 5:
                hist = hist[-5:]
            self.memory.set("history", hist)
            dbg("commit.history.size", len(hist))

            if isinstance(cur_tax, float):
                self.memory.set("__prev_tax__", cur_tax)
            if isinstance(final_params, dict):
                self.memory.set("__prev_final_params__", final_params)
            if isinstance(constraints, dict):
                self.memory.set("__prev_constraints__", constraints)

            try:
                last_ctx = self.memory.get("last_exec_payload") or {}
                lp = last_ctx.get("payload") or {}
                lp["__prev_tax__"] = cur_tax if isinstance(cur_tax, float) else lp.get("__prev_tax__")
                if isinstance(final_params, dict):
                    lp["__prev_final_params__"] = final_params
                if isinstance(constraints, dict):
                    lp["__prev_constraints__"] = constraints
                last_ctx["payload"] = lp
                self.memory.set("last_exec_payload", last_ctx)
                dbg("commit.last_exec_payload.updated", True)
            except Exception as e:
                dbg("commit.last_exec_payload.error", str(e))
        except Exception as e:
            dbg("commit.history.error", str(e))

        dbg("EXIT handle()")
        return {
            "type": "final_feedback",
            "text": final_md,
            "perf_spans": self._perf_finalize_spans(),
            "perf_spans": self._perf_finalize_spans(),
            "raw_result": result,
            "next_actions_hint": (
                "Want to change constraints? Reply \"Add constraints\" to add new constraints on top of the current ones; "
                "reply \"Reset constraints\" to clear everything and return to setup. "
                "If you want to export this run's report, type \"Finalize report\"."
            ),
        }