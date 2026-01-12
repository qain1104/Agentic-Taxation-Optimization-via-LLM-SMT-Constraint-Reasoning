#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tax_agent_pipeline.py

Tax SMT Codegen + Self-Repair Agent (RAG -> LLM -> Local Exec -> Auto Repair) with traces.

Enhancements
-----------
1) Aux reference injection (line-numbered): load local .txt files and inject into BOTH codegen + repair prompts.
2) Prompt dumps: save exact prompts sent to the LLM for codegen/repair (for audit).
3) Better convergence:
   - Automatic probes (counterfactual diagnostics) on mismatch (requires oracle).
   - Loop breaker: repeated same failure signature => escalate model/reasoning, allow rewrite.
4) Repair prompt includes BOTH RAG clauses + aux refs (not only code + error).
5) Web search default ON (codegen + repair), can disable via --no-web-search.
6) Multi-mismatch feedback: on mismatch, collect up to N mismatches and inject a table into repair prompt
   to avoid overfitting a single sample.

Example
-------
export OPENAI_API_KEY=sk-...
python tax_agent_pipeline.py \
  --input income_tax_input.txt \
  --samples income_tax_samples.json \
  --chroma-dir chroma --collection laws_collection --k 15 \
  --schema income_tax \
  --extra-ref 114_numbers.txt \
  --web-allowed-domains mof.gov.tw,laws.moj.gov.tw
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import z3  # noqa: F401

# NOTE: sync OpenAI client; we wrap streaming calls using asyncio.to_thread.
from openai import OpenAI


# ---------- Defaults (can be overridden by CLI) ----------
DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_CODEGEN_MODEL = "gpt-5.2"
DEFAULT_REPAIR_MODEL = "gpt-5-mini"

# Escalation defaults
DEFAULT_ESCALATE_MODEL = "gpt-5.2"          # stronger model for hard repairs
DEFAULT_ESCALATE_REASONING = "high"       # for loop breaker
DEFAULT_ESCALATE_AFTER_SAME_SIG = 2       # repeated same failure signature
DEFAULT_REWRITE_AFTER_SAME_SIG = 2        # when to allow rewrite
DEFAULT_MAX_PROBES = 12                   # cap probe count
DEFAULT_MAX_MISMATCHES = 8                # cap mismatch list shown to LLM


# ---------- Schemas / contracts (extend over time) ----------
SCHEMA_REQUIRED_KEYS: Dict[str, List[str]] = {
    "income_tax": [
        "is_married",
        "salary_self",
        "salary_spouse",
        "salary_dep",
        "interest_income",
        "stock_dividend",
        "house_transaction_gain",
        "other_income",
        "cnt_under_70",
        "cnt_over_70",
        "use_itemized",
        "itemized_deduction",
        "property_loss_deduction",
        "education_count",
        "education_fee",
        "preschool_count",
        "long_term_care_count",
        "rent_deduction",
        "disability_count",
    ]
}


DEFAULT_CODE_CONTRACT = """
0) 你必須在程式碼最前面加上一段註解區塊，格式如下（用繁體中文、最多 3 點）：
   # PATCH_NOTES:
   # - ...
   # END_PATCH_NOTES
   這段只寫「你這版做了哪些修正 / 依據哪個條文或哪個 mismatch 修」，不要寫逐步推理。
1) 你必須輸出「單一段 Python 程式碼」，不要加任何解釋文字或 markdown。
2) 程式碼必須 `import z3` 並定義 `def compute_tax(**kwargs) -> int:`
3) 你必須用 Z3 建模並求解（用 z3.Solver 或 z3.Optimize 都可以），最後回傳 int。
4) 程式碼必須可執行且不依賴外部檔案。
5) 所有金額以新台幣整數計算；如果需要四捨五入/截尾，請清楚寫出（並註解依據）。
6) 若你使用【補充參考】中的任何年度數字/門檻/扣除額上限，必須在程式碼旁用註解標註來源行號，例如：
   # REF:114_numbers.txt L12
   若使用 RAG 文檔，也請標註條號或原文短句。
7) 若你使用 web_search 找到數字/門檻，也要在程式碼旁註解來源網域，例如：
   # WEB:mof.gov.tw (關鍵字: ...)
【Hard Ban: 禁止針對 sample 特例硬編碼】
- 禁止使用任何形式的「樣本特例」或「oracle override」：
  - 不得用 input 向量組成 key（tuple/dict/str hash）去查表回傳固定 expected。
  - 不得根據某一筆樣本的精確數值（例如 salary_self==1326459 且 ...）做 if-else 直接回傳。
  - 不得把 sample idx、expected、mismatch table 的某列內容寫進程式碼作為條件。
- 產出程式必須對「未出現過的新隨機樣本」也能合理計算。
- 若違反，視為不合格，即使 samples 全部 PASS 也算 FAIL。

【SMT 程式生成強制規範（請嚴格遵守，違反視為不合格）】
    A) 型別一致性（最重要）
    - 所有「金額/稅額/人數/門檻/級距」一律使用 z3.Int。
    - 布林條件可用 Python bool 做 sanitize，但進入 SMT 後：
    - 若用 z3.Bool：請用 If(bool_expr, ..., ...)
    - 若用 Int(0/1)：請全程一致，不得混用 Bool 與 Int 當條件。
    - 禁止在關鍵公式中使用 Python float（例如 0.05, 0.1）。百分比計算必須使用整數分子/分母：
    - floor/trunc: (amount * num) // den
    - round: (amount * num + den//2) // den   （amount 必須先約束為 >=0）

    B) Z3 表達式規則
    - 禁止 Python if/and/or/min/max 直接作用在 Z3 表達式上。
    必須使用 z3.If / z3.And / z3.Or。
    - 任何 cap/min/max/clamp 都必須用 If 寫出（例如 max(0,x) = If(x>=0,x,0)）。

    C) 輸入綁定與健壯性
    - compute_tax(**kwargs) 必須對 schema keys 做容錯讀取：
    - 缺值 => 0
    - 轉型失敗 => 0
    - 負值 => clamp 成 0
    - 每個輸入都必須「綁定」到 SMT 變數並加上非負約束：
    x_py = sanitize(kwargs["x"])
    x = Int("x"); s.add(x == x_py, x >= 0)

    D) 建模結構（可修復、可稽核）
    - 必須建立清楚的中間變數（例如 gross_income, deductions, taxable_income, tax_before_credit, tax）
    並用等式約束逐步串起來；禁止把整個稅法流程塞進單一超長 expression。
    - 最終稅額必須是 Int 變數 tax 並約束 tax >= 0，最後從 model 取值回傳：
    assert s.check() == sat
    return m.eval(tax, model_completion=True).as_long()

    E) 可追溯註解
    - 任何門檻/扣除額上限/稅率級距數字都必須在程式碼旁註解來源：
    - REF:xxx.txt Lyy 或 RAG 條號/短句 或 WEB:domain

""".strip()


PROMPT_TEMPLATE = r"""
你現在是一位同時精通 Python 與 SMT (Z3) 的稅法專家。
我想要用 Z3 來計算各種稅金，並將各步驟都寫成 constraints 後再求解。
下列是與稅法相關的參考文檔內容、補充參考資料，以及使用者的情境與問題。

【文檔內容（RAG）】
{clauses}

【補充參考（可能包含年度數字/門檻/FAQ；優先視為權威數字來源；已加行號）】
{aux_refs}

【使用者的問題】
{user_input}

註：
- 使用者的問題中若有 _，代表他是個變數
- ? 則是你要依據法條找到對應的數字（若補充參考有年度數字，優先使用）
- ?? 是此 function 應 return 的數字
- 請在程式碼中註解該 ? 取自哪一條相關文檔（用原文短句或條號即可），或註解「REF:xxx.txt Lyy」。
- 若所有輸入皆為已給定的固定值，請使用 z3.Solver（不要使用 Optimize / minimize / maximize）。
- 只有在題目要求「求最佳解/上下界」或存在 free variable 時才可用 Optimize。
【Code Contract / Hard Requirements】
{code_contract}
""".strip()


# ============== Utility: code fence stripping ==============
_CODE_FENCE_RE = re.compile(r"^\s*```(?:python)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def strip_code_fences(s: str) -> str:
    return re.sub(_CODE_FENCE_RE, "", s).strip()


class TransientLLMError(RuntimeError):
    """Retryable LLM/transport error (stream interrupted, connection dropped, etc.)."""


_CTRL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_llm_text(s: str) -> str:
    """Remove control characters that can appear when a stream is interrupted."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return _CTRL_CHAR_RE.sub("", s)


def extract_patch_notes(code: str) -> str:
    lines = code.splitlines()
    in_block = False
    notes: List[str] = []
    for ln in lines:
        s = ln.strip()
        if s == "# PATCH_NOTES:":
            in_block = True
            continue
        if in_block:
            if s == "# END_PATCH_NOTES":
                break
            if s.startswith("#"):
                notes.append(s[1:].strip())
    return "\n".join([n for n in notes if n])


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


class TraceLogger:
    """Console + JSONL trace."""

    def __init__(self, run_dir: Path, *, quiet: bool = False) -> None:
        self.run_dir = run_dir
        self.quiet = quiet
        self.t0 = time.perf_counter()
        self._fp = (run_dir / "trace.jsonl").open("a", encoding="utf-8")

    def emit(self, event: str, **payload: Any) -> None:
        rec = {
            "event": event,
            "wall_time": time.time(),
            "elapsed_s": round(time.perf_counter() - self.t0, 6),
            **payload,
        }
        self._fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fp.flush()
        msg = payload.get("msg")
        if (not self.quiet) and msg:
            print(msg, flush=True)

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


# ============== RAG retrieval via ChromaDB (direct chromadb) ==============
def retrieve_clauses_chromadb(
    *,
    query: str,
    persist_dir: str,
    collection_name: str,
    k: int,
    embed_model: str,
) -> str:
    """Retrieve top-k clauses from a ChromaDB persistent collection."""
    try:
        import chromadb
        from chromadb.config import Settings
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    except Exception as e:
        raise RuntimeError("chromadb is required for retrieval. Install with: pip install chromadb") from e

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    ef = OpenAIEmbeddingFunction(api_key=openai_key, model_name=embed_model)
    col = client.get_or_create_collection(name=collection_name, embedding_function=ef)

    res = col.query(query_texts=[query], n_results=k, include=["documents", "metadatas"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    parts: List[str] = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        head: List[str] = []
        if isinstance(meta, dict):
            if meta.get("law_name"):
                head.append(str(meta.get("law_name")))
            if meta.get("article_id"):
                head.append(str(meta.get("article_id")))
            if meta.get("title"):
                head.append(str(meta.get("title")))
        header = " / ".join(head).strip()
        if header:
            parts.append(f"[{i+1}] {header}\n{doc}")
        else:
            parts.append(f"[{i+1}]\n{doc}")
    return "\n\n".join(parts).strip()


def load_samples(path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Samples JSON must be a list.")
    for s in data:
        if isinstance(s, dict) and "expected" not in s and "expected_tax" in s:
            s["expected"] = s["expected_tax"]
    return data


# ============== Execute generated code ==============
def run_generated_solver_once(code_text: str, params: Dict[str, Any]) -> Tuple[bool, Optional[int], str]:
    """Returns (ok, result_if_ok, error_report)."""
    try:
        import z3 as _z3
    except Exception as e:
        return False, None, f"Missing dependency z3-solver: {e}"

    g: Dict[str, Any] = {"__builtins__": __builtins__, "z3": _z3}
    for name in [
        "Solver",
        "Optimize",
        "Int",
        "Real",
        "Bool",
        "If",
        "And",
        "Or",
        "Not",
        "Implies",
        "Sum",
        "ToReal",
        "ToInt",
        "sat",
        "unsat",
        "unknown",
    ]:
        if hasattr(_z3, name):
            g[name] = getattr(_z3, name)

    l: Dict[str, Any] = {}
    try:
        exec(code_text, g, l)
    except Exception:
        return False, None, "Exec/compile error:\n" + traceback.format_exc()

    fn = l.get("compute_tax") or g.get("compute_tax")
    if not callable(fn):
        return False, None, "Generated code does not define callable compute_tax(**kwargs)->int"

    try:
        res = fn(**params)
        if not isinstance(res, int):
            return False, None, f"compute_tax returned non-int: {type(res)} value={res}"
        return True, int(res), ""
    except Exception:
        return False, None, "Runtime error in compute_tax:\n" + traceback.format_exc()


def verify_against_samples(
    code_text: str,
    samples: List[Dict[str, Any]],
) -> Tuple[bool, Optional[int], Optional[int], Optional[Dict[str, Any]], str]:
    """Returns (ok, last_result, failing_index, failing_sample, report)."""
    last_result: Optional[int] = None
    for idx, s in enumerate(samples, start=1):
        inp = s.get("inputs", {})
        expected_val = s.get("expected", s.get("expected_tax", None))

        ok, got, err = run_generated_solver_once(code_text, inp)
        if not ok:
            return False, None, idx, s, err

        last_result = got
        if int(got) != int(expected_val):
            report = (
                f"Mismatch on sample {idx}: expected={expected_val} got={got}\n"
                f"Inputs: {json.dumps(inp, ensure_ascii=False)}"
            )
            return False, None, idx, s, report

    return True, last_result, None, None, ""


def verify_against_samples_multi(
    code_text: str,
    samples: List[Dict[str, Any]],
    *,
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
    schema: str = "income_tax",
) -> Tuple[
    bool,                         # ok
    Optional[int],                # last_result
    Optional[str],                # fatal_report (compile/runtime/oracle error) if any
    List[Dict[str, Any]],         # mismatches (up to max_mismatches)
]:
    """
    - compile/runtime/oracle error => fatal_report != None, mismatches=[]
    - mismatch => fatal_report=None, mismatches=[...]
    - pass => ok=True, fatal_report=None, mismatches=[]
    """
    _ = schema  # reserved for future schema-specific checks

    last_result: Optional[int] = None
    mismatches: List[Dict[str, Any]] = []

    for idx, s in enumerate(samples, start=1):
        inp = s.get("inputs", {})
        expected_val = s.get("expected", s.get("expected_tax", None))

        ok, got, err = run_generated_solver_once(code_text, inp)
        if not ok:
            return False, None, err, []

        last_result = got
        if int(got) != int(expected_val):
            mismatches.append(
                {
                    "idx": idx,
                    "id": s.get("id", idx),
                    "expected": int(expected_val),
                    "got": int(got),
                    "diff": int(got) - int(expected_val),
                    "inputs": inp,
                }
            )
            if len(mismatches) >= max_mismatches:
                break

    ok_all = (len(mismatches) == 0)
    return ok_all, last_result, None, mismatches


def format_mismatch_table(mismatches: List[Dict[str, Any]], *, keys: List[str]) -> str:
    if not mismatches:
        return "(無 mismatch)"

    lines = []
    header = ["idx", "expected", "got", "diff"] + keys
    lines.append(" | ".join(header))
    lines.append(" | ".join(["---"] * len(header)))

    for m in mismatches:
        inp = m.get("inputs", {}) or {}
        row = [
            str(m.get("idx")),
            str(m.get("expected")),
            str(m.get("got")),
            str(m.get("diff")),
        ] + [str(inp.get(k, "")) for k in keys]
        lines.append(" | ".join(row))

    return "\n".join(lines)


def get_mismatch_table_keys(schema: str) -> List[str]:
    # 盡量短，但能定位：婚姻/扶養/扣除額/所得類型/人數
    if schema == "income_tax":
        return [
            "is_married",
            "cnt_under_70",
            "cnt_over_70",
            "use_itemized",
            "itemized_deduction",
            "rent_deduction",
            "long_term_care_count",
            "preschool_count",
            "disability_count",
            "education_count",
            "education_fee",
            "property_loss_deduction",
            "interest_income",
            "stock_dividend",
            "house_transaction_gain",
            "other_income",
            "salary_self",
            "salary_spouse",
            "salary_dep",
        ]
    # fallback：若其他 schema 就抓 required keys（最多 16 個避免 prompt 太長）
    return (SCHEMA_REQUIRED_KEYS.get(schema, []) or [])[:16]


# ============== LLM calls (Responses API streaming) ==============
def _event_get(ev: Any, key: str, default: Any = None) -> Any:
    if hasattr(ev, key):
        return getattr(ev, key)
    if isinstance(ev, dict):
        return ev.get(key, default)
    return default


def responses_generate_code_streaming(
    *,
    prompt: str,
    model: str,
    reasoning_effort: str,
    verbosity: str,
    stream_to_stdout: bool,
    trace: TraceLogger,
    tag: str,
    enable_web_search: bool = False,
    web_allowed_domains: Optional[List[str]] = None,
) -> str:
    client = OpenAI()
    trace.emit(
        "llm.start",
        tag=tag,
        model=model,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        web_search=enable_web_search,
        msg=f"[LLM:{tag}] start model={model} (reasoning={reasoning_effort}, verbosity={verbosity}, web_search={enable_web_search})",
    )

    tools = None
    if enable_web_search:
        tool: Dict[str, Any] = {"type": "web_search"}
        if web_allowed_domains:
            tool["filters"] = {"allowed_domains": web_allowed_domains}
        tools = [tool]

    buf: List[str] = []
    completed = False

    try:
        stream = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You output ONLY executable Python code (no markdown, no prose). "
                        "Ensure syntax is valid. Always include PATCH_NOTES block at top. "
                        "Do not use placeholders like TODO. "
                        "Do not omit required function signature."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            tools=tools,
            stream=True,
            reasoning={"effort": reasoning_effort},
            text={"verbosity": verbosity},
        )

        if stream_to_stdout:
            print("", flush=True)

        for ev in stream:
            etype = _event_get(ev, "type", "")
            if etype == "response.output_text.delta":
                delta = _event_get(ev, "delta", "")
                if delta:
                    buf.append(delta)
                    if stream_to_stdout:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
            elif etype in ("response.completed", "response.output_text.done", "response.done"):
                completed = True
                break
            elif etype in ("response.failed", "response.incomplete", "error"):
                err_msg = _event_get(ev, "error", None) or _event_get(ev, "message", None) or str(ev)
                raise TransientLLMError(f"stream ended with {etype}: {err_msg}")
            else:
                continue

    except Exception as e:
        if type(e).__name__ in {"BadRequestError", "AuthenticationError", "PermissionDeniedError"}:
            raise
        raise TransientLLMError(str(e)) from e

    if stream_to_stdout:
        sys.stdout.write("\n")
        sys.stdout.flush()

    out = sanitize_llm_text("".join(buf))
    if (not completed) or (not out.strip()):
        raise TransientLLMError("stream did not complete cleanly (empty or incomplete output)")

    trace.emit("llm.done", tag=tag, model=model, out_chars=len(out), msg=f"[LLM:{tag}] done (chars={len(out)})")
    return out


async def llm_generate_code(
    *,
    prompt: str,
    model: str,
    reasoning_effort: str,
    verbosity: str,
    stream_to_stdout: bool,
    trace: TraceLogger,
    tag: str,
    enable_web_search: bool = False,
    web_allowed_domains: Optional[List[str]] = None,
) -> str:
    # Retry on transient network / streaming issues.
    max_retries = 5
    backoff = 1.0
    last_exc: Optional[BaseException] = None

    for attempt in range(1, max_retries + 1):
        try:
            return await asyncio.to_thread(
                responses_generate_code_streaming,
                prompt=prompt,
                model=model,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
                stream_to_stdout=stream_to_stdout,
                trace=trace,
                tag=tag,
                enable_web_search=enable_web_search,
                web_allowed_domains=web_allowed_domains,
            )
        except BaseException as e:
            last_exc = e
            retryable = isinstance(e, TransientLLMError)
            if (not retryable) or attempt == max_retries:
                raise
            trace.emit(
                "llm.retry",
                tag=tag,
                model=model,
                attempt=attempt,
                max_retries=max_retries,
                backoff=round(backoff, 3),
                err_type=type(e).__name__,
                err=str(e),
                msg=f"[LLM:{tag}] retry {attempt}/{max_retries} after {backoff:.1f}s ({type(e).__name__}: {e})",
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 8.0)

    raise RuntimeError(f"llm_generate_code failed after retries: {last_exc}")


@dataclass
class PipelineResult:
    code: str
    rounds: int
    ok: bool
    last_error: str
    total_seconds: float
    run_dir: str


# ============== Probes (counterfactual diagnostics) ==============
def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def build_income_tax_probes(base: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    probes: List[Tuple[str, Dict[str, Any]]] = []

    def add(label: str, patch: Dict[str, Any]) -> None:
        v = dict(base)
        v.update(patch)
        probes.append((label, v))

    if "use_itemized" in base:
        add("use_itemized=False, itemized_deduction=0", {"use_itemized": False, "itemized_deduction": 0})
        add(
            "use_itemized=True (keep itemized_deduction)",
            {"use_itemized": True, "itemized_deduction": _safe_int(base.get("itemized_deduction", 0))},
        )

    for k in [
        "rent_deduction",
        "property_loss_deduction",
        "education_fee",
        "stock_dividend",
        "interest_income",
        "house_transaction_gain",
        "other_income",
    ]:
        if k in base:
            add(f"{k}=0", {k: 0})

    if "education_count" in base:
        add("education_count=0, education_fee=0", {"education_count": 0, "education_fee": 0})

    for k in ["preschool_count", "long_term_care_count", "disability_count", "cnt_under_70", "cnt_over_70"]:
        if k in base:
            add(f"{k}=0", {k: 0})

    if "is_married" in base and "salary_spouse" in base:
        if bool(base.get("is_married")):
            add("is_married=False, salary_spouse=0", {"is_married": False, "salary_spouse": 0})
        else:
            add(
                "is_married=True (keep salary_spouse)",
                {"is_married": True, "salary_spouse": _safe_int(base.get("salary_spouse", 0))},
            )

    return probes


def run_probes_table(
    *,
    base_inputs: Dict[str, Any],
    code_text: str,
    schema: str,
    max_probes: int,
) -> str:
    if schema != "income_tax":
        return "(Probes 未實作於此 schema)"

    base_ok, base_got, base_err = run_generated_solver_once(code_text, base_inputs)
    base_got_i = int(base_got) if (base_ok and base_got is not None) else None

    probes = build_income_tax_probes(base_inputs)[:max_probes]

    rows: List[str] = []
    rows.append("probe | oracle_tax | solver_tax | oracle_delta | solver_delta | note")
    rows.append("---|---:|---:|---:|---:|---")

    for label, variant in probes:
        ok, got, err = run_generated_solver_once(code_text, variant)
        if ok and got is not None:
            s = int(got)
            od = o - base_oracle
            sd = (s - base_got_i) if base_got_i is not None else 0
            rows.append(f"{label} | {s} | {od:+d} | {sd:+d} | ")
        else:
            err1 = (err.splitlines()[0] if err else "solver_error").strip()
            od = o - base_oracle
            rows.append(f"{label} | ERR | {od:+d} | ERR | {err1}")

    base_line = (
        f"BASE | {base_oracle} | {base_got_i if base_got_i is not None else 'ERR'} | +0 | +0 | "
        f"{('' if base_ok else (base_err.splitlines()[0] if base_err else ''))}"
    )
    return "\n".join([base_line, ""] + rows).strip()


# ============== Prompt builders ==============
def build_repair_prompt(
    *,
    mode: str,  # "minimal" or "rewrite"
    prev_code: str,
    error_report: str,
    failing_sample: Optional[Dict[str, Any]],
    expected: Optional[int],
    got: Optional[int],
    clauses: str,
    aux_refs: str,
    code_contract: str,
    probes_table: str,
    mismatch_table: Optional[str] = None,
) -> str:
    fail_inputs = (failing_sample or {}).get("inputs", {})

    if mode == "rewrite":
        header = "你之前生成的 Python+Z3 程式碼與 oracle/samples 不吻合。請你『允許大改/重寫』來修正（不必最小修改）。"
        extra = (
            "【重要】你必須讓所有 samples 全部吻合（10/10 PASS），並根據 probes 差異定位是哪個扣除額/門檻/稅率段/排富條款錯了。"
        )
    else:
        header = "你之前生成的 Python+Z3 程式碼在本地端執行時失敗了，請你做『最小修改』修復它。"
        extra = "優先修語法/執行錯誤；若是 mismatch，請針對差異做精準修改。"

    diff_val: Optional[int] = None
    try:
        if got is not None and expected is not None:
            diff_val = int(got) - int(expected)
    except Exception:
        diff_val = None

    analysis_block = f"""
【差距分析（請先在腦中完成，勿輸出文字）】
我現在的程式碼算出來的結果是：{got}
但 ground truth（oracle/expected）是：{expected}
diff = {diff_val}

如果有多筆 mismatch，請以 mismatch table 為主，同時滿足所有列，不得只針對單一 failing case 特化。

請你先在腦中推導以下三件事（不要輸出推導過程）：
1) 這些差距最可能對應到哪一段邏輯（例如：免稅額/扣除額上限/排富門檻/稅率級距/所得分類/特別扣除額的適用條件）
2) 從 probes 的 delta（oracle_delta vs solver_delta）判斷是哪個變數的邏輯缺漏或門檻值錯誤
3) 目前程式缺少的關鍵條件是什麼（例如：某扣除額必須先判斷適用資格、某項需 min/max 上限、某年度門檻、某分離課稅規則）
4) 如果目前提供的chroma RAG文件中沒有提及相關規定，請自行上網搜集相關資訊（不需要輸出搜尋過程）。

接著只做「最小必要修改」讓所有 samples PASS。
""".strip()

    mismatch_block = ""
    if mismatch_table and str(mismatch_table).strip():
        mismatch_block = f"""
【Mismatch Table（多筆 mismatch 摘要；你必須同時修正全部列）】
{mismatch_table}
""".strip()

    return f"""
{header}
請只輸出修復後的完整 Python 程式碼（不要解釋文字、不要 markdown）。

{extra}

【文檔內容（RAG）】
{clauses}

【補充參考（可能包含年度數字/門檻/FAQ；已加行號）】
{aux_refs}

【錯誤報告 / mismatch】
{error_report}

【Failing inputs】
{json.dumps(fail_inputs, ensure_ascii=False)}

【Expected vs Got】
expected={expected}
got={got}

{analysis_block}

{mismatch_block}

【診斷 Probes（BASE 與多個只改單一因素的變體；oracle vs solver 的 delta）】
{probes_table}

【必須維持的 Code Contract】
{code_contract}

【你上一版的程式碼】
{prev_code}
""".strip()


def read_text_files_line_numbered(paths: List[str]) -> str:
    parts: List[str] = []
    for p in paths:
        pp = Path(p).expanduser()
        if not pp.exists():
            parts.append(f"[MISSING] {p}")
            continue
        try:
            txt = pp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            txt = pp.read_text(encoding="utf-8", errors="replace")
        txt = txt.strip()
        if not txt:
            continue
        lines = txt.splitlines()
        numbered = "\n".join([f"L{i+1}: {line}" for i, line in enumerate(lines)])
        parts.append(f"=== {pp.name} ===\n{numbered}")
    return "\n\n".join(parts).strip() or "(無)"


def build_code_contract(schema_names: List[str], contract_file: Optional[str]) -> str:
    base = DEFAULT_CODE_CONTRACT
    if contract_file:
        p = Path(contract_file).expanduser()
        if p.exists():
            try:
                base = p.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                base = p.read_text(encoding="utf-8", errors="replace").strip()

    keys: List[str] = []
    for s in schema_names:
        keys.extend(SCHEMA_REQUIRED_KEYS.get(s, []))
    keys = sorted(set(keys))
    if keys:
        base = base + "\n\n" + "\n".join(
            [
                "【Schema Keys】",
                "compute_tax(**kwargs) 至少要能處理下列 keys（至少讀取/容錯，不一定都影響結果）：",
            ]
            + [f"- {k}" for k in keys]
        )
    return base.strip()


def classify_failure(report: str) -> str:
    r = (report or "").lstrip()
    if r.startswith("Exec/compile error:"):
        return "compile"
    if r.startswith("Runtime error in compute_tax:"):
        return "runtime"
    if r.startswith("Mismatch on sample"):
        return "mismatch"
    return "other"


def build_fail_sig(*, fail_type: str, failing_index: Optional[int], expected: Any, got: Any, short_line: str) -> str:
    return f"{fail_type}|idx={failing_index}|exp={expected}|got={got}|{short_line[:120]}"


async def synthesize_with_auto_repair(
    *,
    user_problem: str,
    rag_query: Optional[str],
    persist_dir: str,
    collection_name: str,
    k: int,
    samples: List[Dict[str, Any]],
    max_attempts: int,  # 0 => unlimited
    run_dir: Path,
    stream_llm: bool,
    quiet: bool,
    show_llm_notes: bool,
    show_llm_head: int,
    reasoning_codegen: str,
    reasoning_repair: str,
    verbosity: str,
    enable_web_search: bool,
    web_allowed_domains: Optional[List[str]],
    aux_refs: str,
    code_contract: str,
    schema: str,
    embed_model: str,
    codegen_model: str,
    repair_model: str,
    escalate_model: str,
    escalate_reasoning: str,
    escalate_after_same_sig: int,
    rewrite_after_same_sig: int,
    max_probes: int,
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
) -> PipelineResult:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未設定。請 export OPENAI_API_KEY=sk-... 或建立 .env")

    trace = TraceLogger(run_dir, quiet=quiet)
    trace.emit("init", msg=f"[INIT] run_dir={run_dir}")

    # ---- RAG ----
    query = rag_query.strip() if rag_query else user_problem
    trace.emit(
        "rag.start",
        k=k,
        collection=collection_name,
        persist_dir=persist_dir,
        embed_model=embed_model,
        msg=f"[RAG] retrieving top-{k} clauses from '{collection_name}' (embed={embed_model}) ...",
    )
    rag_t0 = time.perf_counter()
    clauses = retrieve_clauses_chromadb(
        query=query,
        persist_dir=persist_dir,
        collection_name=collection_name,
        k=k,
        embed_model=embed_model,
    )
    rag_dt = time.perf_counter() - rag_t0
    (run_dir / "retrieved_clauses.txt").write_text(clauses, encoding="utf-8")
    (run_dir / "aux_refs.txt").write_text(aux_refs, encoding="utf-8")
    (run_dir / "code_contract.txt").write_text(code_contract, encoding="utf-8")
    trace.emit(
        "rag.done",
        seconds=round(rag_dt, 3),
        out_chars=len(clauses),
        msg=f"[RAG] done in {rag_dt:.2f}s | chars={len(clauses)} (saved retrieved_clauses.txt)",
    )

    prompt = PROMPT_TEMPLATE.format(
        clauses=clauses or "(RAG 無命中)",
        aux_refs=aux_refs,
        user_input=user_problem,
        code_contract=code_contract,
    )

    # ---- prompt dump (audit) ----
    (run_dir / "prompt_codegen.txt").write_text(prompt, encoding="utf-8")

    # ---- web_search compatibility ----
    # NOTE: web_search requires reasoning.effort != minimal
    if enable_web_search and reasoning_codegen == "minimal":
        trace.emit("warn", msg="[WARN] web_search requires reasoning-codegen != minimal; auto-bump to 'low'.")
        reasoning_codegen = "low"
    if enable_web_search and reasoning_repair == "minimal":
        trace.emit("warn", msg="[WARN] web_search requires reasoning-repair != minimal; auto-bump to 'low'.")
        reasoning_repair = "low"

    # ---- initial codegen ----
    t0 = time.perf_counter()
    code = await llm_generate_code(
        prompt=prompt,
        model=codegen_model,
        reasoning_effort=reasoning_codegen,
        verbosity=verbosity,
        stream_to_stdout=stream_llm,
        trace=trace,
        tag="codegen",
        enable_web_search=enable_web_search,
        web_allowed_domains=web_allowed_domains,
    )
    code = strip_code_fences(code)
    (run_dir / "round_0_codegen.py").write_text(code, encoding="utf-8")

    if show_llm_notes:
        notes = extract_patch_notes(code)
        if notes:
            print(f"[LLM:codegen] PATCH_NOTES:\n{notes}\n", flush=True)
    if show_llm_head > 0:
        head = "\n".join(code.splitlines()[:show_llm_head])
        print(f"[LLM:codegen] HEAD({show_llm_head}):\n{head}\n", flush=True)

    last_error = ""
    attempt = 0
    sig_counts: Dict[str, int] = {}

    while True:
        attempt += 1

        trace.emit("verify.start", attempt=attempt, samples=len(samples), msg=f"[VERIFY {attempt}] running {len(samples)} samples ...")
        verify_start = time.perf_counter()

        ok, last_result, fatal_report, mismatches = verify_against_samples_multi(
            code,
            samples,
            max_mismatches=max_mismatches,
            schema=schema,
        )

        verify_dt = time.perf_counter() - verify_start

        status = "PASS" if ok else "FAIL"
        trace.emit(
            "verify.done",
            attempt=attempt,
            status=status,
            seconds=round(verify_dt, 3),
            msg=f"[VERIFY {attempt}] => {status} ({verify_dt:.2f}s)",
        )

        if ok:
            total_elapsed = time.perf_counter() - t0
            trace.emit("done", rounds=attempt, total_seconds=round(total_elapsed, 3), msg=f"[DONE] all samples passed | rounds={attempt} total={total_elapsed:.2f}s")
            trace.close()
            return PipelineResult(code=code, rounds=attempt, ok=True, last_error="", total_seconds=total_elapsed, run_dir=str(run_dir))

        # -------- FAIL: build a richer report --------
        failing_index: Optional[int] = None
        failing_sample: Optional[Dict[str, Any]] = None
        failing_inputs: Dict[str, Any] = {}
        expected: Optional[int] = None
        got: Optional[int] = None
        mismatch_table = ""

        if fatal_report:
            report = fatal_report
        else:
            # choose primary mismatch: the one with largest |diff| (more informative)
            primary = None
            if mismatches:
                primary = sorted(mismatches, key=lambda m: abs(int(m.get("diff", 0))), reverse=True)[0]

            if primary:
                failing_index = int(primary.get("idx", 0)) or None
                expected = int(primary.get("expected")) if primary.get("expected") is not None else None
                got = int(primary.get("got")) if primary.get("got") is not None else None
                failing_inputs = dict(primary.get("inputs", {}) or {})
                if failing_index and 1 <= failing_index <= len(samples):
                    failing_sample = samples[failing_index - 1]

            keys = get_mismatch_table_keys(schema)
            mismatch_table = format_mismatch_table(mismatches, keys=keys)

            # keep the prefix "Mismatch on sample" so classify_failure() works
            report = (
                f"Mismatch on sample {failing_index}: expected={expected} got={got}\n"
                f"(showing up to {len(mismatches)} mismatches)\n\n"
                f"{mismatch_table}"
            )

        short_line = report.splitlines()[0] if report else "(no report)"
        trace.emit(
            "verify.fail",
            attempt=attempt,
            failing_index=failing_index,
            msg=f"[FAIL {attempt}] sample #{failing_index}: {short_line}" if failing_index else f"[FAIL {attempt}] error: {short_line}",
        )

        last_error = report
        (run_dir / f"round_{attempt}_fail_report.txt").write_text(report, encoding="utf-8")

        fail_type = classify_failure(report)

        # Try to compute got if not known and we have inputs
        if got is None and failing_inputs:
            run_ok, run_got, run_err = run_generated_solver_once(code, failing_inputs)
            if run_ok and run_got is not None:
                got = int(run_got)
            elif fail_type != "mismatch" and (not report.strip()):
                report = run_err
                last_error = report

        sig = build_fail_sig(
            fail_type=fail_type,
            failing_index=failing_index,
            expected=expected,
            got=got,
            short_line=short_line,
        )
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
        same_sig_count = sig_counts[sig]

        probes_table = "(無 probes：需要 oracle 才能產生診斷表)"
        if fail_type == "mismatch" and failing_inputs:
            try:
                probes_table = run_probes_table(
                    base_inputs=failing_inputs,
                    code_text=code,
                    schema=schema,
                    max_probes=max_probes,
                )
            except Exception as e:
                probes_table = f"(probes 產生失敗: {type(e).__name__}: {e})"

        selected_model = repair_model
        selected_reasoning = reasoning_repair
        repair_mode = "minimal"

        if same_sig_count >= rewrite_after_same_sig:
            repair_mode = "rewrite"
        if same_sig_count >= escalate_after_same_sig:
            selected_model = escalate_model
            selected_reasoning = escalate_reasoning

        # ensure web_search compatibility
        if enable_web_search and selected_reasoning == "minimal":
            selected_reasoning = "low"

        trace.emit(
            "repair.plan",
            attempt=attempt,
            fail_type=fail_type,
            same_sig_count=same_sig_count,
            repair_mode=repair_mode,
            repair_model=selected_model,
            repair_reasoning=selected_reasoning,
            web_search=enable_web_search,
            msg=f"[REPAIR PLAN] type={fail_type} same_sig={same_sig_count} mode={repair_mode} model={selected_model} reasoning={selected_reasoning} web_search={enable_web_search}",
        )

        repair_prompt = build_repair_prompt(
            mode=repair_mode,
            prev_code=code,
            error_report=report,
            failing_sample=failing_sample,
            expected=expected,
            got=got,
            clauses=clauses or "(RAG 無命中)",
            aux_refs=aux_refs,
            code_contract=code_contract,
            probes_table=probes_table,
            mismatch_table=mismatch_table,
        )

        # ---- prompt dump (audit) ----
        (run_dir / f"prompt_repair_{attempt}.txt").write_text(repair_prompt, encoding="utf-8")

        # ---- repair (web_search default ON too) ----
        code = await llm_generate_code(
            prompt=repair_prompt,
            model=selected_model,
            reasoning_effort=selected_reasoning,
            verbosity=verbosity,
            stream_to_stdout=stream_llm,
            trace=trace,
            tag=f"repair_{attempt}",
            enable_web_search=enable_web_search,
            web_allowed_domains=web_allowed_domains,
        )
        code = strip_code_fences(code)
        (run_dir / f"round_{attempt}_repaired.py").write_text(code, encoding="utf-8")

        if show_llm_notes:
            notes = extract_patch_notes(code)
            if notes:
                print(f"[LLM:repair_{attempt}] PATCH_NOTES:\n{notes}\n", flush=True)
        if show_llm_head > 0:
            head = "\n".join(code.splitlines()[:show_llm_head])
            print(f"[LLM:repair_{attempt}] HEAD({show_llm_head}):\n{head}\n", flush=True)

        if max_attempts and attempt >= max_attempts:
            total_elapsed = time.perf_counter() - t0
            trace.emit(
                "stop",
                rounds=attempt,
                total_seconds=round(total_elapsed, 3),
                reason="max_attempts_reached",
                msg=f"[STOP] max_attempts reached ({max_attempts}). last_error saved.",
            )
            trace.close()
            return PipelineResult(code=code, rounds=attempt, ok=False, last_error=last_error, total_seconds=total_elapsed, run_dir=str(run_dir))


# ============== CLI ==============
async def _amain() -> None:
    parser = argparse.ArgumentParser(description="Tax SMT agent (RAG + OpenAI + local exec + auto repair + traces).")
    parser.add_argument("--input", required=True, help="Path to the natural-language task text.")
    parser.add_argument("--rag-query", default=None, help="Optional explicit RAG query; defaults to input text.")
    parser.add_argument("--chroma-dir", default="chroma_db", help="Chroma persistence directory for RAG (persist_dir).")
    parser.add_argument("--collection", default="tax_laws", help="Chroma collection name.")
    parser.add_argument("--k", type=int, default=15, help="Top-k clauses to retrieve.")
    parser.add_argument("--out", default="generated_tax_solver.py", help="Where to write the final generated solver code.")
    parser.add_argument("--max-attempts", type=int, default=0, help="Max codegen+repair attempts. 0 = unlimited.")

    parser.add_argument("--stream-llm", action="store_true", help="Stream LLM output to console.")
    parser.add_argument("--quiet", action="store_true", help="Less console output (still writes trace.jsonl).")
    parser.add_argument("--run-dir", default=None, help="Optional run directory. Default: runs/<timestamp>")

    parser.add_argument("--show-llm-notes", action="store_true", help="Print LLM PATCH_NOTES after each generation.")
    parser.add_argument("--show-llm-head", type=int, default=0, help="Print first N lines of generated code each round.")

    parser.add_argument("--reasoning-codegen", default="low", choices=["minimal", "low", "medium", "high", "xhigh"], help="Reasoning effort for codegen.")
    parser.add_argument("--reasoning-repair", default="medium", choices=["minimal", "low", "medium", "high", "xhigh"], help="Reasoning effort for repair.")
    parser.add_argument("--verbosity", default="low", choices=["low", "medium", "high"], help="Verbosity for text output.")

    # web_search default ON, can disable via --no-web-search
    parser.set_defaults(enable_web_search=True)
    parser.add_argument("--web-search", dest="enable_web_search", action="store_true", help="Enable web_search (default: on).")
    parser.add_argument("--no-web-search", dest="enable_web_search", action="store_false", help="Disable web_search.")
    parser.add_argument("--web-allowed-domains", default="mof.gov.tw,laws.moj.gov.tw", help="Comma-separated allowlist for web_search.")

    parser.add_argument("--samples", default=None, help="JSON file path: list of {inputs: {...}, expected: int} samples.")

    parser.add_argument("--sample-n", type=int, default=10, help="How many random samples to generate.")
    parser.add_argument("--sample-seed", type=int, default=0, help="RNG seed for sample generation.")
    parser.add_argument("--samples-out", default=None, help="If generating samples, write them to this JSON file.")

    parser.add_argument("--extra-ref", action="append", default=[], help="Path to a local .txt reference file. Can be repeated.")
    parser.add_argument("--schema", default="income_tax", help=f"Schema name (for probes & key hints). Available: {sorted(SCHEMA_REQUIRED_KEYS)}")
    parser.add_argument("--contract-file", default=None, help="Optional path to a full Code Contract text (overrides default).")

    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model used by Chroma query embedding.")
    parser.add_argument("--codegen-model", default=DEFAULT_CODEGEN_MODEL, help="LLM model for initial code generation.")
    parser.add_argument("--repair-model", default=DEFAULT_REPAIR_MODEL, help="LLM model for repair attempts.")
    parser.add_argument("--escalate-model", default=DEFAULT_ESCALATE_MODEL, help="Stronger model when loop breaker triggers.")

    parser.add_argument("--escalate-reasoning", default=DEFAULT_ESCALATE_REASONING, choices=["minimal", "low", "medium", "high", "xhigh"])
    parser.add_argument("--escalate-after-same-sig", type=int, default=DEFAULT_ESCALATE_AFTER_SAME_SIG, help="Escalate model/reasoning after N repeated same failures.")
    parser.add_argument("--rewrite-after-same-sig", type=int, default=DEFAULT_REWRITE_AFTER_SAME_SIG, help="Allow rewrite after N repeated same failures.")
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES, help="Max number of probes per mismatch.")
    parser.add_argument("--max-mismatches", type=int, default=DEFAULT_MAX_MISMATCHES, help="Max mismatch rows to show in repair prompt.")

    args = parser.parse_args()


    # --- run_dir naming: <schema>_<timestamp> ---
    ts = _now_tag()
    tax_name = Path(args.input).stem  # income_tax (from income_tax.txt)
    run_dir = Path(args.run_dir) if args.run_dir else Path("source_code/code_synthesis/runs") / f"{tax_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] run_dir: {run_dir}", flush=True)

    samples: Optional[List[Dict[str, Any]]] = None
    if args.samples:
        samples = load_samples(args.samples)

    if samples is None:
        raise RuntimeError("You must provide --samples.")

    user_problem = Path(args.input).read_text(encoding="utf-8")

    # web search settings
    web_allowed_domains: Optional[List[str]] = None
    if getattr(args, "web_allowed_domains", ""):
        parts = [p.strip() for p in str(args.web_allowed_domains).split(",")]
        web_allowed_domains = [p for p in parts if p]
    enable_web_search = bool(getattr(args, "enable_web_search", True))

    # extra refs + contract (line-numbered refs)
    aux_refs = read_text_files_line_numbered(list(args.extra_ref or []))
    code_contract = build_code_contract([args.schema], args.contract_file)

    # Save run inputs for reproducibility
    (run_dir / "task_input.txt").write_text(user_problem, encoding="utf-8")
    (run_dir / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "extra_ref_paths.json").write_text(json.dumps(list(args.extra_ref or []), ensure_ascii=False, indent=2), encoding="utf-8")

    res = await synthesize_with_auto_repair(
        user_problem=user_problem,
        rag_query=args.rag_query,
        persist_dir=args.chroma_dir,
        collection_name=args.collection,
        k=args.k,
        samples=samples,
        max_attempts=args.max_attempts,
        run_dir=run_dir,
        stream_llm=args.stream_llm,
        quiet=args.quiet,
        show_llm_notes=args.show_llm_notes,
        show_llm_head=args.show_llm_head,
        reasoning_codegen=args.reasoning_codegen,
        reasoning_repair=args.reasoning_repair,
        verbosity=args.verbosity,
        enable_web_search=enable_web_search,
        web_allowed_domains=web_allowed_domains,
        aux_refs=aux_refs,
        code_contract=code_contract,
        schema=args.schema,
        embed_model=args.embed_model,
        codegen_model=args.codegen_model,
        repair_model=args.repair_model,
        escalate_model=args.escalate_model,
        escalate_reasoning=args.escalate_reasoning,
        escalate_after_same_sig=args.escalate_after_same_sig,
        rewrite_after_same_sig=args.rewrite_after_same_sig,
        max_probes=args.max_probes,
        max_mismatches=args.max_mismatches,
    )

    Path(args.out).write_text(res.code, encoding="utf-8")
    if res.ok:
        print(f"[OK] All samples passed. Runs={res.rounds}, total={res.total_seconds:.2f}s. Wrote solver code to: {args.out}", flush=True)
    else:
        print(f"[FAIL] Runs={res.rounds}, total={res.total_seconds:.2f}s. Last error:\n{res.last_error}\nWrote last code to: {args.out}", flush=True)

    print(
        f"[TRACE] run_dir={res.run_dir} (trace.jsonl, retrieved_clauses.txt, aux_refs.txt, prompt_codegen.txt, prompt_repair_*.txt, per-round code saved)",
        flush=True,
    )


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
