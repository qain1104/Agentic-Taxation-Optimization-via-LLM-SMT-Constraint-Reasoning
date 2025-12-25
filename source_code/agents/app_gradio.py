# -*- coding: utf-8 -*-
from __future__ import annotations

import sys, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uuid
import time
import json
import traceback
import re
from collections import defaultdict
import functools

import gradio as gr
import logging

logging.basicConfig(
    level=logging.INFO,  # 想更安靜就改 WARNING
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

from multi_agent_tax_system import (
    CallerAgent,
    ConstraintAgent,
    ExecuteAgent,
    ReasoningAgent,
    MemoryStore,
    TOOL_MAP,
    _trigger_fin_export,
)

# 每個 session 一組獨立的 MemoryStore + agents
SESSIONS: dict[str, dict] = {}

# ===== 用隱藏標籤綁定對話 Session（避免用 id(history) 每次都變） =====
_SESSION_TAG_RE = re.compile(r"<!--\s*SESSION:([0-9a-fA-F-]{8,})\s*-->")

def _get_or_create_session_key(history) -> str:
    """從 history 內倒序尋找 SESSION 標記；若沒有，生成新的 UUID。"""
    if isinstance(history, list):
        for msg in reversed(history):
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, str):
                continue
            m = _SESSION_TAG_RE.search(content)
            if m:
                return m.group(1)
    return str(uuid.uuid4())

def _attach_session_tag(text: str, session_key: str) -> str:
    """在回覆文字末尾附加 <!-- SESSION:... -->，避免重複附加。"""
    if not isinstance(text, str):
        text = str(text)
    if _SESSION_TAG_RE.search(text):
        return text
    return text + f"\n\n<!-- SESSION:{session_key} -->"

def _get_session_bundle(session_key: str) -> dict:
    """依 session_key 取得或建立一組 session 專用的 agents + memory。"""
    bundle = SESSIONS.get(session_key)
    if bundle is None:
        mem = MemoryStore()
        bundle = {
            "memory": mem,
            "caller": CallerAgent(memory=mem),
            "constraint": ConstraintAgent(memory=mem),
            "executor": ExecuteAgent(memory=mem),
            "reasoner": ReasoningAgent(memory=mem),
            # 用來計算「系統問 → 使用者回」的跨 request 等待時間
            "awaiting_user": None,  # {"agent":..., "phase":..., "t0":...}
        }
        SESSIONS[session_key] = bundle
    return bundle

def _dump_debug_and_clear(caller_agent):
    lines = caller_agent.memory.get("debug_lines", []) or []
    caller_agent.memory.set("debug_lines", [])
    if not lines:
        return ""
    return (
        "\n\n<details><summary>DEBUG</summary>\n\n```\n"
        + "\n".join(str(x) for x in lines)
        + "\n```\n</details>"
    )

def _strip_inline_tips(md: str) -> str:
    """讓報告本體乾淨：剝掉 ReasoningAgent 最後附加的互動提示"""
    if not isinstance(md, str):
        return md
    tip = "想變更條件？回覆「再加條件」可在現有基礎上加新限制；回覆「重設條件」會清空所有條件並回到設定階段。"
    md = md.replace("\n\n> " + tip, "")
    md = md.replace("\n> " + tip, "")
    md = md.replace("> " + tip, "")
    md = md.replace(tip, "")
    return md.strip()

_TUNING_TIPS_BLOCK_RE = re.compile(
    r"\n*條件調校建議\s*\n"          # block header
    r"(?:.*\n)*?"                   # block body (non-greedy)
    r"(?=\n(?:若要再加條件|若完成設定|若要清空|目前條件|第三階段|回覆「下一步」|$))",
    re.M
)

def _strip_condition_tuning_tips(md: str) -> str:
    """移除 ConstraintAgent 的『條件調校建議』區塊，保留 early_tips_md。"""
    if not isinstance(md, str):
        return md
    return _TUNING_TIPS_BLOCK_RE.sub("\n", md).strip()


def _ui_footer_tip() -> str:
    """報告下方的 UI 操作說明（不放進報告本體）"""
    return (
        "\n\n> **下一步**\n"
        "> • 要調整條件：回覆「再加條件」，或回覆「重設條件」回到設定階段。\n"
        "> • 若要**以此輪報告作為輸出報告**，請輸入 **「計算完成」**。\n"
    )

def _details_text(title: str, lines) -> str:
    if not lines:
        return ""
    return (
        f"\n\n<details><summary>{title}</summary>\n\n```\n"
        + "\n".join(str(x) for x in lines)
        + "\n```\n</details>"
    )

# =========================
# Perf helpers (NEW)
# =========================
def _perf_new():
    # perf[agent][phase] = seconds
    return defaultdict(lambda: defaultdict(float))

def _perf_add(perf, agent: str, phase: str, dt: float):
    try:
        perf[agent][phase] += float(dt)
    except Exception:
        pass

def _perf_to_plain_dict(perf) -> dict:
    return {a: dict(ph) for a, ph in perf.items()}

def _format_perf_breakdown(perf) -> str:
    """本輪（turn）perf 統計：用 handle_total 當作 agent wall-clock，避免 nested spans 重複加總。"""
    if not perf:
        return ""

    rows = []
    for agent, phases in perf.items():
        if not isinstance(phases, dict):
            continue
        for phase, sec in phases.items():
            rows.append((agent, str(phase), float(sec)))
    rows.sort(key=lambda x: (-x[2], x[0], x[1]))

    totals = {}
    # NOTE: phases like `llm:*` / `rag:*` are nested spans inside an agent call.
    # For wall-clock turn time we only count the top-level `handle_total` per agent (if present).
    for agent, phases in perf.items():
        if isinstance(phases, dict) and "handle_total" in phases:
            totals[agent] = float(phases.get("handle_total") or 0.0)
        elif isinstance(phases, dict):
            totals[agent] = float(sum(float(v) for v in phases.values()))
        else:
            totals[agent] = 0.0

    # Hide agents that did not run in this turn (total ~ 0) to avoid confusing attribution.
    totals = {a: t for a, t in totals.items() if float(t) > 1e-9}
    keep_agents = set(totals.keys())
    rows = [r for r in rows if r[0] in keep_agents]

    total_all = float(sum(totals.values()))

    md = []
    md.append(f"\n\n**⏱️ 思考時間（本輪）≈ {total_all:.3f}s**")
    md.append("\n<details><summary>詳細耗時（點我展開）</summary>\n")
    md.append("\n| Agent | Phase | Time (s) | Meaning |")
    md.append("|---|---|---:|---|")
    for agent, phase, sec in rows[:200]:
        md.append(f"| {agent} | {phase} | {sec:.3f} | {_phase_explain(agent, phase)} |")

    md.append("\n**Agent 總計**")
    md.append("\n| Agent | Total (s) |")
    md.append("|---|---:|")
    for a, t in sorted(totals.items(), key=lambda kv: -kv[1]):
        md.append(f"| {a} | {t:.3f} |")

    md.append("\n</details>")
    md.append(_format_perf_explain(perf))
    return "\n".join(md)

def _phase_explain(agent: str, phase: str) -> str:
    """
    將 perf phase 轉成「這段時間在做什麼」的簡短說明（用於 debug / 論文 latency 解釋）。
    """
    p = (phase or "").strip()

    # Top-level
    if p == "handle_total":
        return "此 Agent 本輪處理的整體 wall-clock（避免把 nested span 重複加總）"

    # LLM spans
    if p.startswith("llm:"):
        name = p.split(":", 1)[1]
        mapping = {
            "caller_frame": "Caller：LLM 解析自然語言 → intent/slots（稅種判斷、欄位抽取）",
            "caller_suggest": "Caller：LLM 生成追問/補欄位建議（缺哪些欄位、怎麼問）",
            "constraint_suggest": "Constraint：LLM 產生條件式建議（可放寬/可最佳化方向）",
            "constraint_parse": "Constraint：LLM 將自然語言條件轉成可求解的 constraint JSON",
            "advice_json_basic": "Reasoning：LLM 依最佳化結果產生簡易建議（不引入新變數）",
            "render_final_report": "Reasoning：LLM 改寫草稿為最終報告（更長、更慢）",
            "render_once_with_llm": "Reasoning：LLM 將草稿精修為最終報告（單次；可能較慢）",
        }
        return mapping.get(name, f"LLM 呼叫：{name}")

    # RAG spans
    if p.startswith("rag:"):
        name = p.split(":", 1)[1]
        mapping = {
            "build_queries": "RAG：依稅種/約束/變動欄位組出檢索 query",
            "check_store": "RAG：檢查向量庫資料夾/collection 是否可用",
            "init_vectorstore": "RAG：初始化 Chroma + Embeddings（可能含 IO/連線）",
            "mmr_search": "RAG：MMR 檢索（多樣性搜尋；通常會做 embedding + 相似度計算）",
            "similarity_search": "RAG：相似度檢索（with_score / fallback）",
            "dedup": "RAG：去重與截斷 evidence chunks（避免重複內容）",
            "compose_ctx": "RAG：把 deltas/constraints/evidence 組成給 LLM 的 ctx",
        }
        return mapping.get(name, f"RAG 步驟：{name}")


    # Render / IO spans
    if p.startswith("render:"):
        name = p.split(":", 1)[1]
        mapping = {
            "external_renderer": "渲染：外部 renderer 產生報告版型（可能含額外格式化）",
        }
        return mapping.get(name, f"渲染步驟：{name}")

    if p.startswith("io:"):
        name = p.split(":", 1)[1]
        mapping = {
            "persist_report_files": "IO：將報告寫入檔案（md/json）",
        }
        return mapping.get(name, f"IO：{name}")

    # Tool calls
    if p.startswith("tool_call_total:"):
        tool = p.split(":", 1)[1]
        return f"工具執行：{tool}（例如 SMT/最佳化求解）"

    # Fallback
    return ""

def _agent_total_from_phases(phases: dict) -> float:
    """Turn 的 wall-clock：優先用 handle_total（避免 nested span 重複加總）"""
    if not isinstance(phases, dict):
        return 0.0
    if "handle_total" in phases:
        return float(phases.get("handle_total") or 0.0)
    return float(sum(float(v) for v in phases.values()))

def _format_session_perf(executor, session_key: str, current_turn_perf=None) -> str:
    """聚合 executor.memory['perf_trace']，印出本 session（整題）累積時間。"""
    try:
        hist = executor.memory.get("perf_trace") or []
        if not isinstance(hist, list) or not hist:
            return ""

        agg = {}
        n = 0
        for item in hist:
            if not isinstance(item, dict):
                continue
            if item.get("session") != session_key:
                continue
            perf = item.get("perf") or {}
            if not isinstance(perf, dict):
                continue
            n += 1
            for agent, phases in perf.items():
                agg[agent] = agg.get(agent, 0.0) + _agent_total_from_phases(phases)

        # include current turn (so the session total shown in UI matches "as of this response")
        if isinstance(current_turn_perf, dict) and current_turn_perf:
            n += 1
            for agent, phases in current_turn_perf.items():
                agg[agent] = agg.get(agent, 0.0) + _agent_total_from_phases(phases)

        if n == 0:
            return ""

        rows = sorted(agg.items(), key=lambda kv: -kv[1])
        total = float(sum(agg.values()))

        md = []
        md.append(f"\n\n<details><summary>📌 本題累積耗時（跨 {n} 輪）≈ {total:.3f}s</summary>\n")
        md.append("\n| Agent | Total (s) |")
        md.append("|---|---:|")
        for a, t in rows:
            md.append(f"| {a} | {t:.3f} |")
        md.append("\n</details>")
        return "\n".join(md)
    except Exception:
        return ""

def _format_perf_explain(perf) -> str:
    """列出本輪出現過的 phase 的中文說明（方便 debug / paper）。"""
    try:
        if not perf or not isinstance(perf, dict):
            return ""
        uniq = []
        seen = set()
        for agent, phases in perf.items():
            if not isinstance(phases, dict):
                continue
            for phase in phases.keys():
                key = (agent, str(phase))
                if key in seen:
                    continue
                seen.add(key)
                uniq.append((agent, str(phase), _phase_explain(agent, str(phase))))

        if not uniq:
            return ""

        md = []
        md.append("\n\n<details><summary>🧩 耗時細項說明（本輪出現的 phase 都在做什麼）</summary>\n")
        md.append("\n| Agent | Phase | 說明 |")
        md.append("|---|---|---|")
        for a, p, e in uniq[:200]:
            md.append(f"| {a} | {p} | {e or ''} |")
        md.append("\n</details>")
        return "\n".join(md)
    except Exception:
        return ""

def _persist_perf_snapshot(executor, session_key: str, turn_perf, meta: dict | None = None):
    """將本輪 perf trace 存入 executor.memory['perf_trace']（最多 50 筆），方便回溯/匯出。"""
    try:
        perf_plain = _perf_to_plain_dict(turn_perf)
        item = {
            "ts": time.time(),
            "session": session_key,
            "perf": perf_plain,
        }
        if isinstance(meta, dict):
            item.update(meta)
        hist = executor.memory.get("perf_trace") or []
        if not isinstance(hist, list):
            hist = []
        hist.append(item)
        if len(hist) > 50:
            hist = hist[-50:]
        executor.memory.set("perf_trace", hist)
    except Exception:
        pass

# =========================

def _preserve_reopen_context_from_exec(exec_out: dict, caller, constraint, executor):
    """把工具執行結果存入各 Agent 的記憶，供『再加條件 / 重設條件』續接使用。"""
    try:
        tool = exec_out.get("tool_name")
        pay  = exec_out.get("payload") or {}
        if not tool or not isinstance(pay, dict):
            return

        ctx_payload = {
            "tool_name": tool,
            "user_params": (pay.get("user_params") or {}),
            "op": pay.get("op"),
        }

        pending_from_caller = caller.memory.get("pending_constraint_payload") or {}
        pending_from_cons   = constraint.memory.get("pending_constraint_payload") or {}
        tips = (
            pending_from_caller.get("early_tips_md")
            or pending_from_cons.get("early_tips_md")
            or pay.get("early_tips_md")
        )
        if isinstance(tips, str) and tips.strip():
            ctx_payload["early_tips_md"] = tips

        constraint.memory.set("pending_tool_for_constraints", tool)
        constraint.memory.set("pending_constraint_payload", ctx_payload)
        constraint.memory.set("last_exec_payload", {"tool_name": tool, "payload": ctx_payload})

        caller.memory.set("pending_tool_for_constraints", tool)
        caller.memory.set("pending_constraint_payload", ctx_payload)
        caller.memory.set("last_tool", tool)

        executor.memory.set("last_exec_payload", {"tool_name": tool, "payload": ctx_payload})

    except Exception:
        pass

def _persist_run_and_get_prev(exec_out: dict, executor):
    try:
        history = executor.memory.get("history_runs") or []
    except Exception:
        history = []

    payload = {
        "ts": time.time(),
        "tool_name": exec_out.get("tool_name"),
        "mode": (exec_out.get("result") or {}).get("mode"),
        "baseline": (exec_out.get("result") or {}).get("baseline"),
        "optimized": (exec_out.get("result") or {}).get("optimized"),
        "status": (exec_out.get("result") or {}).get("status"),
        "final_params": (exec_out.get("result") or {}).get("final_params") or {},
        "constraints": (exec_out.get("result") or {}).get("constraints") or {},
    }

    history.append(payload)
    executor.memory.set("history_runs", history)
    executor.memory.set("last_run", payload)

    prev_run = history[-2] if len(history) >= 2 else None
    executor.memory.set("prev_run", prev_run)
    return prev_run

def _persist_report_markdown(exec_out: dict, report_md: str, executor):
    try:
        tool = exec_out.get("tool_name") or (exec_out.get("payload") or {}).get("tool_name")
        if not tool or not isinstance(report_md, str) or not report_md.strip():
            return
        res = exec_out.get("result") or {}
        payload = exec_out.get("payload") or {}
        user_params = (payload.get("user_params") or {}) if isinstance(payload, dict) else {}

        budget_field = TOOL_MAP.get(tool, {}).get("budget_field")
        budget_val = user_params.get(budget_field) if budget_field else None
        if budget_val is None:
            for k in ("budget", "budget_tax", "tax_budget"):
                if isinstance(res.get(k), (int, float)):
                    budget_val = res.get(k); break

        item = {
            "ts": time.time(),
            "mode": (res.get("mode") or payload.get("op")),
            "status": res.get("status"),
            "baseline": res.get("baseline"),
            "optimized": (res.get("optimized") or res.get("optimized_total_tax")
                          or res.get("total_tax") or res.get("tax") or res.get("optimized_tax")),
            "budget": budget_val,
            "md": report_md,
        }
        hist = executor.memory.get("report_history") or {}
        arr = hist.get(tool, [])
        arr.append(item)
        if len(arr) > 20:
            arr = arr[-20:]
        hist[tool] = arr
        executor.memory.set("report_history", hist)
    except Exception:
        pass

def _save_last_run_files(tool_name: str | None, final_md: str, result: dict, payload: dict):
    import re as _re, json as _json, time as _time

    if not isinstance(final_md, str) or not final_md.strip():
        return

    tool = tool_name or "unknown_tool"
    tool_slug = _re.sub(r"[^A-Za-z0-9_-]+", "_", str(tool))

    out_dir = os.path.join("reports", "last_run")
    os.makedirs(out_dir, exist_ok=True)

    md_path_tool  = os.path.join(out_dir, f"last_{tool_slug}.md")
    json_path_tool = os.path.join(out_dir, f"last_{tool_slug}.json")
    md_path_latest  = os.path.join(out_dir, "last.md")
    json_path_latest = os.path.join(out_dir, "last.json")

    mode = (result or {}).get("mode") or (payload or {}).get("op")
    status = (result or {}).get("status")
    baseline = (result or {}).get("baseline")
    optimized = (
        (result or {}).get("optimized")
        or (result or {}).get("optimized_total_tax")
        or (result or {}).get("total_tax")
        or (result or {}).get("tax")
        or (result or {}).get("optimized_tax")
    )
    budget = None
    up = (payload or {}).get("user_params") or {}
    for k in ("budget", "budget_tax", "tax_budget"):
        if isinstance((result or {}).get(k), (int, float)):
            budget = (result or {}).get(k); break
        if isinstance(up.get(k), (int, float)):
            budget = up.get(k); break

    pack = {
        "ts": int(_time.time()),
        "tool_name": tool,
        "mode": mode,
        "status": status,
        "baseline": baseline,
        "optimized": optimized,
        "budget": budget,
        "result": result,
        "payload": payload,
        "markdown": final_md
    }

    with open(md_path_tool, "w", encoding="utf-8") as f:
        f.write(final_md)
    with open(json_path_tool, "w", encoding="utf-8") as f:
        _json.dump(pack, f, ensure_ascii=False, indent=2)

    with open(md_path_latest, "w", encoding="utf-8") as f:
        f.write(final_md)
    with open(json_path_latest, "w", encoding="utf-8") as f:
        _json.dump(pack, f, ensure_ascii=False, indent=2)

def _should_finish(s: str) -> bool:
    s = (s or "").strip().lower()
    return any(k in s for k in [
        "計算完成", "完成計算",
        "出建議報告", "產生建議報告",
        "出結論報告", "產生結論報告", "產出結論",
        "匯總", "總結", "產出建議", "final report", "finish & advise"
    ])

def _reset_session_state(caller, constraint, executor, reasoner):
    try:
        last_ctx = executor.memory.get("last_exec_payload") or {}
        tool = last_ctx.get("tool_name")
        payload = last_ctx.get("payload")
    except Exception:
        tool, payload = None, None

    for a in (caller, constraint, reasoner):
        try:
            a.memory.clear()
        except Exception:
            pass

    if tool and isinstance(payload, dict):
        try:
            constraint.memory.set("pending_tool_for_constraints", tool)
            constraint.memory.set("pending_constraint_payload", payload)
            constraint.memory.set("last_exec_payload", {"tool_name": tool, "payload": payload})

            caller.memory.set("pending_tool_for_constraints", tool)
            caller.memory.set("pending_constraint_payload", payload)
            caller.memory.set("last_tool", tool)
        except Exception:
            pass

def _hard_reset_all_states():
    SESSIONS.clear()

def _on_hard_reset():
    _hard_reset_all_states()
    return ([{"role": "assistant", "content": INTRO_MSG}], "")

INTRO_MSG = """**👋 歡迎使用《114年度台灣稅務 Agentic Service》**

**請先告訴系統你要算什麼稅，目前支援：**
- 綜所稅、外僑所得稅、營利事業所得稅
- 遺產稅、贈與稅
- 加值型營業稅、非加值型營業稅
- 貨物稅、菸酒稅
- 證券 / 期貨交易稅
- 特種貨物及勞務稅

**系統會先判斷你要計算的稅種，再循序漸進地協助你補齊欄位、設定條件、最佳化稅負，最後產出報告。**
- 完成多輪比較後，輸入 **「計算完成」**，系統會以**此輪報告**作為**結論報告**並存檔。

> 本系統結果為估算，實際稅負仍以主管機關規定與申報資料為準。
"""

async def chat_logic(
    user_msg: str,
    history,
    show_debug: bool = False,
    auto_reset: bool = True,
    report_fast: bool = False,
):
    session_key = _get_or_create_session_key(history)
    bundle = _get_session_bundle(session_key)
    caller = bundle["caller"]
    constraint = bundle["constraint"]
    executor = bundle["executor"]
    reasoner = bundle["reasoner"]

    # Report mode (full vs fast) stored in session memory for ReasoningAgent & CallerAgent early tips.
    try:
        bundle["memory"].set("report_mode", "fast" if report_fast else "full")
    except Exception:
        pass

    turn_perf = _perf_new()

    # 0) 跨 request 的 user wait time（上一輪系統提問 -> 本輪 user 回覆）
    #    注意：此等待時間不應計入「思考時間」，所以不寫入 turn_perf
    wait_state = bundle.get("awaiting_user")
    if isinstance(wait_state, dict) and isinstance(wait_state.get("t0"), (int, float)):
        dt_wait = time.perf_counter() - wait_state["t0"]
        # 若你想留存等待時間，可放到 memory / perf_trace meta（可選）
        # executor.memory.set("last_user_wait_sec", float(dt_wait))
    bundle["awaiting_user"] = None

    def _should_reset_constraints(s: str) -> bool:
        s = (s or "").strip().lower()
        return any(key in s for key in ["重設條件", "重置條件", "reset constraints", "clear constraints"])

    def has_latest_report() -> bool:
        try:
            if reasoner and (reasoner.memory.get("last_report_md") or reasoner.memory.get("__latest_report__")):
                return True
        except Exception:
            pass
        try:
            if executor and (executor.memory.get("last_report_md") or executor.memory.get("__latest_report__")):
                return True
        except Exception:
            pass
        return os.path.exists("reports/last_run/last.md") or os.path.exists("reports/last_run/last.json")

    def _should_hard_reset(s: str) -> bool:
        s = (s or "").strip().lower()
        if "條件" in s:
            return False
        exact = {"重置", "清空", "reset", "重新開始", "restart", "硬重置", "hard reset"}
        if s in exact:
            return True
        return s in {"reset()", "reset all", "clear all"}

    async def _do_reset_constraints_and_reopen(sess_key: str):
        last_ctx = executor.memory.get("last_exec_payload") or {}
        tool = last_ctx.get("tool_name") or caller.memory.get("pending_tool_for_constraints")
        payload0 = (
            last_ctx.get("payload")
            or caller.memory.get("pending_constraint_payload")
            or constraint.memory.get("pending_constraint_payload")
            or {}
        )
        if not tool or not isinstance(payload0, dict):
            return _attach_session_tag("⚠️ 找不到上一輪上下文，請先指定要計算的稅種或執行一次計算。", sess_key)

        new_payload = reasoner._payload_with_constraints_reset(payload0)

        try:
            constraint.memory.clear()
        except Exception:
            pass

        constraint.memory.set("pending_tool_for_constraints", tool)
        constraint.memory.set("pending_constraint_payload", new_payload)
        caller.memory.set("pending_tool_for_constraints", tool)
        caller.memory.set("pending_constraint_payload", new_payload)
        executor.memory.set("last_exec_payload", {"tool_name": tool, "payload": new_payload})

        t0 = time.perf_counter()
        ask = await constraint.handle({"type": "reopen_constraints"})
        _perf_add(turn_perf, "ConstraintAgent", "handle_total", time.perf_counter() - t0)


        # Merge nested spans from ConstraintAgent (e.g., llm:constraint_parse) into this turn's perf
        try:
            spans = constraint.memory.get("perf_spans_last:ConstraintAgent")
            if isinstance(spans, list):
                for it in spans:
                    if isinstance(it, (list, tuple)) and len(it) == 2:
                        ph, sec = it
                    elif isinstance(it, dict):
                        ph, sec = it.get("phase"), it.get("time")
                    else:
                        continue
                    if str(ph) == "handle_total":
                        continue
                    _perf_add(turn_perf, "ConstraintAgent", str(ph), float(sec))
        except Exception:
            pass
        cons_dbg_html = _details_text("DEBUG（ConstraintAgent）", ask.get("debug") or [])
        q = _strip_condition_tuning_tips(ask.get("question") or "（沒有問題文字）")
        debug_block = _dump_debug_and_clear(caller) if show_debug else ""

        msg = q + (cons_dbg_html if show_debug else "") + debug_block + _format_perf_breakdown(turn_perf) + _format_session_perf(executor, session_key, turn_perf)
        _persist_perf_snapshot(executor, sess_key, turn_perf, meta={"type": "reset_constraints_reopen"})
        return _attach_session_tag(msg, sess_key)

    # 1)「重設條件」
    if _should_reset_constraints(user_msg):
        return await _do_reset_constraints_and_reopen(session_key)

    # 2)「硬重置」
    if _should_hard_reset(user_msg):
        for a in (caller, constraint, executor, reasoner):
            try:
                a.memory.clear()
            except Exception:
                pass
        bundle["awaiting_user"] = None
        _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "hard_reset"})
        return _attach_session_tag(INTRO_MSG, session_key)

    # 3)「計算完成」
    if _should_finish(user_msg):
        if not has_latest_report():
            _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "finish_no_report"})
            return _attach_session_tag("目前尚未完成任何稅額試算，請先選擇稅種並完成至少一次計算。", session_key)

        base = "reports/last_run"
        sent_title = ""
        t0 = time.perf_counter()
        try:
            info = await _trigger_fin_export(executor.memory)
            if isinstance(info, dict):
                sent_title = info.get("title") or ""
            else:
                sent_title = str(info) if info is not None else ""
        except Exception as e:
            sent_title = f"(匯出程序略過：{e})"
        _perf_add(turn_perf, "ExecuteAgent", "fin_export_total", time.perf_counter() - t0)

        msg = (
            f"✅ 最終**結論報告**已自動儲存：\n"
            f"- {base}/last.md\n- {base}/last.json\n\n"
            f"（每次「計算完成」都會覆寫為最新），已送出報告：{sent_title}"
            + _format_perf_breakdown(turn_perf) + _format_session_perf(executor, session_key, turn_perf)
        )
        _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "finish"})
        return _attach_session_tag(msg, session_key)

    try:
        pending_for_cons = (
            constraint.memory.get("pending_tool_for_constraints")
            or constraint.memory.get("pending_constraint_payload")
            or caller.memory.get("pending_tool_for_constraints")
            or caller.memory.get("pending_constraint_payload")
        )
        if pending_for_cons:
            # ConstraintAgent path: user is replying constraints
            t0 = time.perf_counter()
            parsed = await constraint.handle({"type": "constraints_reply", "text": user_msg})
            _perf_add(turn_perf, "ConstraintAgent", "handle_total", time.perf_counter() - t0)


            # Merge nested spans from ConstraintAgent (e.g., llm:constraint_parse) into this turn's perf
            try:
                spans = constraint.memory.get("perf_spans_last:ConstraintAgent")
                if isinstance(spans, list):
                    for it in spans:
                        if isinstance(it, (list, tuple)) and len(it) == 2:
                            ph, sec = it
                        elif isinstance(it, dict):
                            ph, sec = it.get("phase"), it.get("time")
                        else:
                            continue
                        if str(ph) == "handle_total":
                            continue
                        _perf_add(turn_perf, "ConstraintAgent", str(ph), float(sec))
            except Exception:
                pass
            cons_dbg_html = _details_text("DEBUG（ConstraintAgent）", parsed.get("debug") or [])

            if parsed.get("type") == "reset_constraints":
                return await _do_reset_constraints_and_reopen(session_key)

            if parsed.get("type") == "ready_for_execute":
                payload = parsed.get("payload") or {}

                t0 = time.perf_counter()
                exec_out = await executor.handle(payload)
                _perf_add(turn_perf, "ExecuteAgent", f"tool_call_total:{payload.get('tool_name','unknown')}", time.perf_counter() - t0)

                _preserve_reopen_context_from_exec(exec_out, caller, constraint, executor)

                prev_run = _persist_run_and_get_prev(exec_out, executor)
                try:
                    exec_out.setdefault("payload", {})
                    if prev_run:
                        prev_tax = prev_run.get("optimized")
                        if not isinstance(prev_tax, (int, float)):
                            prev_tax = prev_run.get("baseline")
                        if isinstance(prev_tax, (int, float)):
                            exec_out["payload"]["__prev_tax__"] = float(prev_tax)
                        if isinstance(prev_run.get("final_params"), dict):
                            exec_out["payload"]["__prev_final_params__"] = prev_run["final_params"]
                        if isinstance(prev_run.get("constraints"), dict):
                            exec_out["payload"]["__prev_constraints__"] = prev_run["constraints"]
                except Exception:
                    pass

                t0 = time.perf_counter()
                fb = await reasoner.handle(exec_out)
                _perf_add(turn_perf, "ReasoningAgent", "handle_total", time.perf_counter() - t0)

                # finer spans if ReasoningAgent provides them
                try:
                    spans = fb.get("perf_spans") if isinstance(fb, dict) else None
                    if not spans:
                        spans = reasoner.memory.get("perf_spans_last:ReasoningAgent")
                    if isinstance(spans, list):
                        for item in spans:
                            if isinstance(item, (list, tuple)) and len(item) == 2:
                                name = str(item[0])
                                if name != "handle_total":
                                    _perf_add(turn_perf, "ReasoningAgent", name, float(item[1]))
                            elif isinstance(item, dict) and "name" in item and "sec" in item:
                                name = str(item["name"])
                                if name != "handle_total":
                                    _perf_add(turn_perf, "ReasoningAgent", name, float(item["sec"]))
                except Exception:
                    pass

                # persist outputs
                try:
                    _save_last_run_files(
                        exec_out.get("tool_name"),
                        (fb.get("text", "") or "") if isinstance(fb, dict) else "",
                        exec_out.get("result") or {},
                        exec_out.get("payload") or {},
                    )
                except Exception as e:
                    dbg_lines = caller.memory.get("debug_lines", []) or []
                    dbg_lines.append(f"[last-run-save] ERROR: {e}")
                    caller.memory.set("debug_lines", dbg_lines)

                try:
                    if isinstance(fb, dict):
                        _persist_report_markdown(exec_out, fb.get("text", ""), executor)
                except Exception:
                    pass

                debug_block = _dump_debug_and_clear(caller) if show_debug else ""
                payload_fmt = json.dumps(payload, ensure_ascii=False, indent=2)
                raw_result_fmt = json.dumps(exec_out.get("result"), ensure_ascii=False, indent=2)
                report_md = _strip_inline_tips((fb.get("text", "") or "") if isinstance(fb, dict) else "")

                debug_details = ""
                if show_debug:
                    debug_details = (
                        "\n\n<details><summary>執行參數 payload</summary>\n\n```json\n"
                        + payload_fmt + "\n```\n</details>"
                        + "\n\n<details><summary>工具原始回傳 result</summary>\n\n```json\n"
                        + raw_result_fmt + "\n```\n</details>"
                    )

                msg = (
                    report_md
                    + _ui_footer_tip()
                    + debug_details
                    + (cons_dbg_html if show_debug else "")
                    + debug_block
                    + _format_perf_breakdown(turn_perf) + _format_session_perf(executor, session_key, turn_perf)
                )
                _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "execute_via_constraint_reply"})
                bundle["awaiting_user"] = None
                if auto_reset:
                    _reset_session_state(caller, constraint, executor, reasoner)
                    bundle["awaiting_user"] = None
                return _attach_session_tag(msg, session_key)

            if parsed.get("type") == "follow_up":
                q = _strip_condition_tuning_tips(parsed.get("question") or "（沒有問題文字）")
                debug_block = _dump_debug_and_clear(caller) if show_debug else ""
                msg = q + (cons_dbg_html if show_debug else "") + debug_block + _format_perf_breakdown(turn_perf) + _format_session_perf(executor, session_key, turn_perf)
                _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "constraint_follow_up"})
                return _attach_session_tag(msg, session_key)

            debug_block = _dump_debug_and_clear(caller) if show_debug else ""
            msg = (
                "⚠️ 未知 ConstraintAgent 回覆：\n```json\n"
                + json.dumps(parsed, ensure_ascii=False, indent=2)
                + "\n```"
                + (cons_dbg_html if show_debug else "")
                + debug_block
                + _format_perf_breakdown(turn_perf) + _format_session_perf(executor, session_key, turn_perf)
            )
            _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "constraint_unknown"})
            return _attach_session_tag(msg, session_key)

        # ---- General path: CallerAgent ----
        t0 = time.perf_counter()
        result = await caller.handle(user_msg)
        _perf_add(turn_perf, "CallerAgent", "handle_total", time.perf_counter() - t0)


        # Merge nested spans from CallerAgent (e.g., llm:caller_frame) into this turn's perf
        try:
            spans = caller.memory.get("perf_spans_last:CallerAgent")
            if isinstance(spans, list):
                for it in spans:
                    if isinstance(it, (list, tuple)) and len(it) == 2:
                        ph, sec = it
                    elif isinstance(it, dict):
                        ph, sec = it.get("phase"), it.get("time")
                    else:
                        continue
                    if str(ph) == "handle_total":
                        continue
                    _perf_add(turn_perf, "CallerAgent", str(ph), float(sec))
        except Exception:
            pass
        if isinstance(result, dict) and result.get("type") == "reopen_constraints":
            t0 = time.perf_counter()
            ask = await constraint.handle({"type": "reopen_constraints"})
            _perf_add(turn_perf, "ConstraintAgent", "handle_total", time.perf_counter() - t0)


            # Merge nested spans from ConstraintAgent (e.g., llm:constraint_parse) into this turn's perf
            try:
                spans = constraint.memory.get("perf_spans_last:ConstraintAgent")
                if isinstance(spans, list):
                    for it in spans:
                        if isinstance(it, (list, tuple)) and len(it) == 2:
                            ph, sec = it
                        elif isinstance(it, dict):
                            ph, sec = it.get("phase"), it.get("time")
                        else:
                            continue
                        if str(ph) == "handle_total":
                            continue
                        _perf_add(turn_perf, "ConstraintAgent", str(ph), float(sec))
            except Exception:
                pass
            cons_dbg_html = _details_text("DEBUG（ConstraintAgent）", ask.get("debug") or [])
            q = _strip_condition_tuning_tips(ask.get("question") or "（沒有問題文字）")
            debug_block = _dump_debug_and_clear(caller) if show_debug else ""
            msg = q + (cons_dbg_html if show_debug else "") + debug_block + _format_perf_breakdown(turn_perf) + _format_session_perf(executor, session_key, turn_perf)
            _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "reopen_constraints"})
            return _attach_session_tag(msg, session_key)

        if isinstance(result, dict) and result.get("type") == "reset_constraints":
            return await _do_reset_constraints_and_reopen(session_key)

        if result is None:
            debug_block = _dump_debug_and_clear(caller) if show_debug else ""
            msg = "⚠️ 系統回傳空結果（None）。" + debug_block + _format_perf_breakdown(turn_perf) + _format_session_perf(executor, session_key, turn_perf)
            _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "caller_none"})
            return _attach_session_tag(msg, session_key)
        if not isinstance(result, dict):
            debug_block = _dump_debug_and_clear(caller) if show_debug else ""
            msg = f"⚠️ 非預期回傳型別：{type(result).__name__}\n{result!r}" + debug_block + _format_perf_breakdown(turn_perf) + _format_session_perf(executor, session_key, turn_perf)
            _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "caller_bad_type"})
            return _attach_session_tag(msg, session_key)

        rtype = result.get("type")

        if rtype == "follow_up":
            msg = result.get("question") or "（沒有問題文字）"
            if result.get("stage") == "constraints":
                try:
                    pc_payload = caller.memory.get("pending_constraint_payload")
                    pc_tool = caller.memory.get("pending_tool_for_constraints")
                    if pc_payload or pc_tool:
                        constraint.memory.set("pending_constraint_payload", pc_payload or {})
                        constraint.memory.set("pending_tool_for_constraints", pc_tool)
                    else:
                        constraint.memory.set("pending_constraint_payload", result.get("payload") or {})
                        constraint.memory.set("pending_tool_for_constraints", result.get("tool_name"))
                except Exception:
                    pass

            debug_block = _dump_debug_and_clear(caller) if show_debug else ""
            msg2 = msg + debug_block + _format_perf_breakdown(turn_perf) + _format_session_perf(executor, session_key, turn_perf)
            _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "caller_follow_up"})
            return _attach_session_tag(msg2, session_key)

        if rtype == "tool_request":
            payload = result.get("payload", {}) or {}

            t0 = time.perf_counter()
            ask = await constraint.handle(result)
            _perf_add(turn_perf, "ConstraintAgent", "handle_total", time.perf_counter() - t0)


            # Merge nested spans from ConstraintAgent (e.g., llm:constraint_parse) into this turn's perf
            try:
                spans = constraint.memory.get("perf_spans_last:ConstraintAgent")
                if isinstance(spans, list):
                    for it in spans:
                        if isinstance(it, (list, tuple)) and len(it) == 2:
                            ph, sec = it
                        elif isinstance(it, dict):
                            ph, sec = it.get("phase"), it.get("time")
                        else:
                            continue
                        if str(ph) == "handle_total":
                            continue
                        _perf_add(turn_perf, "ConstraintAgent", str(ph), float(sec))
            except Exception:
                pass
            payload_fmt = json.dumps(payload, ensure_ascii=False, indent=2)
            cons_dbg_html = _details_text("DEBUG（ConstraintAgent）", ask.get("debug") or [])

            if ask.get("type") == "follow_up":
                q = _strip_condition_tuning_tips(ask.get("question") or "（沒有問題文字）")
                debug_block = _dump_debug_and_clear(caller) if show_debug else ""
                msg = q + (cons_dbg_html if show_debug else "") + debug_block + _format_perf_breakdown(turn_perf) + _format_session_perf(executor, session_key, turn_perf)
                _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "constraint_follow_up_after_tool_request"})
                return _attach_session_tag(msg, session_key)

            if ask.get("type") == "ready_for_execute":
                payload2 = ask.get("payload") or payload

                t0 = time.perf_counter()
                exec_out = await executor.handle(payload2)
                _perf_add(turn_perf, "ExecuteAgent", f"tool_call_total:{payload2.get('tool_name','unknown')}", time.perf_counter() - t0)

                _preserve_reopen_context_from_exec(exec_out, caller, constraint, executor)

                prev_run = _persist_run_and_get_prev(exec_out, executor)
                try:
                    exec_out.setdefault("payload", {})
                    if prev_run:
                        prev_tax = prev_run.get("optimized")
                        if not isinstance(prev_tax, (int, float)):
                            prev_tax = prev_run.get("baseline")
                        if isinstance(prev_tax, (int, float)):
                            exec_out["payload"]["__prev_tax__"] = float(prev_tax)
                        if isinstance(prev_run.get("final_params"), dict):
                            exec_out["payload"]["__prev_final_params__"] = prev_run["final_params"]
                        if isinstance(prev_run.get("constraints"), dict):
                            exec_out["payload"]["__prev_constraints__"] = prev_run["constraints"]
                except Exception:
                    pass

                t0 = time.perf_counter()
                fb = await reasoner.handle(exec_out)
                _perf_add(turn_perf, "ReasoningAgent", "handle_total", time.perf_counter() - t0)

                try:
                    spans = fb.get("perf_spans") if isinstance(fb, dict) else None
                    if not spans:
                        spans = reasoner.memory.get("perf_spans_last:ReasoningAgent")
                    if isinstance(spans, list):
                        for item in spans:
                            if isinstance(item, (list, tuple)) and len(item) == 2:
                                _perf_add(turn_perf, "ReasoningAgent", str(item[0]), float(item[1]))
                            elif isinstance(item, dict) and "name" in item and "sec" in item:
                                _perf_add(turn_perf, "ReasoningAgent", str(item["name"]), float(item["sec"]))
                except Exception:
                    pass

                try:
                    _save_last_run_files(
                        exec_out.get("tool_name"),
                        (fb.get("text", "") or "") if isinstance(fb, dict) else "",
                        exec_out.get("result") or {},
                        exec_out.get("payload") or {},
                    )
                except Exception as e:
                    dbg_lines = caller.memory.get("debug_lines", []) or []
                    dbg_lines.append(f"[last-run-save] ERROR: {e}")
                    caller.memory.set("debug_lines", dbg_lines)

                try:
                    if isinstance(fb, dict):
                        _persist_report_markdown(exec_out, fb.get("text", ""), executor)
                except Exception:
                    pass

                raw_result_fmt = json.dumps(exec_out.get("result"), ensure_ascii=False, indent=2)
                report_md = _strip_inline_tips((fb.get("text", "") or "") if isinstance(fb, dict) else "")

                debug_details = ""
                if show_debug:
                    debug_details = (
                        "\n\n<details><summary>執行參數 payload</summary>\n\n```json\n"
                        + payload_fmt + "\n```\n</details>"
                        + "\n\n<details><summary>工具原始回傳 result</summary>\n\n```json\n"
                        + raw_result_fmt + "\n```\n</details>"
                    )

                msg = (
                    report_md
                    + _ui_footer_tip()
                    + debug_details
                    + (cons_dbg_html if show_debug else "")
                    + _format_perf_breakdown(turn_perf) + _format_session_perf(executor, session_key, turn_perf)
                )
                debug_block = _dump_debug_and_clear(caller) if show_debug else ""
                _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "execute_via_tool_request"})
                bundle["awaiting_user"] = None
                if auto_reset:
                    _reset_session_state(caller, constraint, executor, reasoner)
                    bundle["awaiting_user"] = None
                return _attach_session_tag(msg + debug_block, session_key)

            debug_block = _dump_debug_and_clear(caller) if show_debug else ""
            msg = (
                "⚠️ 未知 ConstraintAgent 回覆：\n```json\n"
                + json.dumps(ask, ensure_ascii=False, indent=2)
                + "\n```"
                + (cons_dbg_html if show_debug else "")
                + debug_block
                + _format_perf_breakdown(turn_perf) + _format_session_perf(executor, session_key, turn_perf)
            )
            _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "constraint_unknown_after_tool_request"})
            return _attach_session_tag(msg, session_key)

        debug_block = _dump_debug_and_clear(caller) if show_debug else ""
        msg = "⚠️ 未知 CallerAgent 回覆：\n```json\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n```" + debug_block + _format_perf_breakdown(turn_perf) + _format_session_perf(executor, session_key, turn_perf)
        _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "caller_unknown"})
        return _attach_session_tag(msg, session_key)

    except Exception as exc:
        debug_block = _dump_debug_and_clear(caller) if show_debug else ""
        tb = traceback.format_exc()
        msg = f"⚠️ 系統錯誤：{exc}\n\n```\n{tb}\n```" + debug_block + _format_perf_breakdown(turn_perf) + _format_session_perf(executor, session_key, turn_perf)
        _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "exception", "error": str(exc)})
        return _attach_session_tag(msg, session_key)

# =========================
# UI
# =========================
with gr.Blocks(
    title="Taiwan Tax Agentic Service Demo",
    theme=gr.themes.Soft(),
    css=r"""
    /* ==== 自訂聊天框：移除右上角垃圾桶（Clear） ==== */
    #tax-chatbot .icon-button-wrapper.top-panel { display: none !important; }

    #tax-chatbot button[aria-label="Clear"],
    #tax-chatbot button[aria-label*="Clear"],
    #tax-chatbot [data-testid="clear-button"],
    #tax-chatbot button:has(svg.lucide-trash),
    #tax-chatbot button:has(svg[class*="trash"]) {
        display: none !important;
    }
    """
) as demo:

    with gr.Row(elem_id="header-card"):
        gr.Markdown(
            "### Taiwan Tax Multi-Agent Demo\n"
            "以多代理架構自動解析意圖→補齊稅務變數→最佳化→報告輸出",
            elem_classes=["glass-card"],
        )

    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": INTRO_MSG}],
                type="messages",
                height=560,
                show_copy_button=True,
                label="對話",
                elem_id="tax-chatbot",
            )
            msg = gr.Textbox(
                placeholder="輸入完指令後，按住 shift + Enter 可送出，Enter 換行",
                lines=2,
            )
            with gr.Row():
                send = gr.Button("🚀 送出", variant="primary")
                clear = gr.Button("🧹 清空輸入/對話（硬重置）")
                clear.click(_on_hard_reset, inputs=None, outputs=[chatbot, msg], queue=False)

        with gr.Column(scale=5):
            with gr.Group(elem_classes=["glass-card"]):
                gr.Markdown("**⚙️ 執行選項**")
                with gr.Row():
                    show_debug = gr.Checkbox(value=False, label="顯示 DEBUG 區塊")
                    auto_reset = gr.Checkbox(value=True, label="每輪結束自動軟重置（保留續接）")
                report_fast = gr.Checkbox(value=False, label="快速報告（略過 RAG / early_tips / 縮短建議）")

            with gr.Group(elem_classes=["glass-card"]):
                gr.Markdown("**🧭 請選擇以下稅種 **（點一下自動填入）")
                ex1 = gr.Button("我想計算綜合所得稅", elem_classes=["example-chip"])
                ex2 = gr.Button("我想計算外僑所得稅", elem_classes=["example-chip"])
                ex3 = gr.Button("我想計算營利事業所得稅", elem_classes=["example-chip"])
                ex4 = gr.Button("我想計算遺產稅", elem_classes=["example-chip"])
                ex5 = gr.Button("我想計算贈與稅", elem_classes=["example-chip"])
                ex6 = gr.Button("我想計算加值型營業稅", elem_classes=["example-chip"])
                ex7 = gr.Button("我想計算非加值型營業稅", elem_classes=["example-chip"])
                ex8 = gr.Button("我想計算貨物稅", elem_classes=["example-chip"])
                ex9 = gr.Button("我想計算菸酒稅", elem_classes=["example-chip"])
                ex10 = gr.Button("我想計算證券交易稅", elem_classes=["example-chip"])
                ex11 = gr.Button("我想計算期貨交易稅", elem_classes=["example-chip"])
                ex12 = gr.Button("我想計算特種貨物稅", elem_classes=["example-chip"])
                ex13 = gr.Button("我想計算特種勞務稅", elem_classes=["example-chip"])

            with gr.Accordion("📘 使用說明（點我展開）", open=False, elem_classes=["glass-card"]):
                gr.Markdown(
                    """請先輸入欲計算的稅種、系統會引導您補齊變數、加入條件、生成報告。

**輸入格式建議**：
- 支援「萬 / 億」單位，系統會自動轉換成「元」。
- 支援民國日期，如「112/3/15」。
- 多筆資料請用「; / ， / ；」分隔，每筆可用「x / X」表示數量，如「名車 800 萬 x 2」。
- 可用「→ 最大 / 最小」表示優化目標，或是直接指定目標稅額，如「總稅額 500000」。
- 可用「+ - * /」表達運算，如「土地 7000 萬 + 房屋 3000 萬」。
- 可用「> / < / >= / <= / =」表示條件，如「土地 ≥ 5000 萬」。
- 可用「%」表示百分比，如「持股 20%」。
- 可用「約 / 大約 / 左右」表示模糊數字，如「遺產總額 1 億左右」。
- 可用「至 / 到」表示區間，如「期間 111 年至 113 年」。
"""
                )

    def jump_to_tax(tool_name: str, history, show_dbg=False, auto_rst=True):
        """側邊稅種按鈕：不走 LLM 判斷，直接進入該稅種『階段一（inputs）』。"""
        session_key = _get_or_create_session_key(history)
        bundle = _get_session_bundle(session_key)
        mem = bundle.get("memory")
        caller = bundle.get("caller")

        # 清掉舊上下文（但不清全域 SESSIONS）
        try:
            if mem:
                mem.clear()
        except Exception:
            pass

        # 初始化到指定稅種的階段一
        try:
            if mem:
                mem.set("stage", "inputs")
                mem.set("pending_tool", tool_name)
                mem.set("last_tool", tool_name)
                mem.set("filled_slots", {})
                mem.set("pending_missing", None)
                mem.set("pending_constraint_payload", None)
                mem.set("pending_tool_for_constraints", None)
                mem.set("last_exec_payload", None)
                mem.set("op", None)
        except Exception:
            pass

        try:
            q = caller._compose_inputs_page(tool_name, {})
        except Exception:
            q = f"已切換稅種：{tool_name}（但無法載入欄位導覽，請檢查 TOOL_MAP / tools_registry）"

        q = _strip_condition_tuning_tips(q)
        msg = _attach_session_tag(q, session_key)

        # 只顯示第一階段頁面（不保留舊對話）
        new_history = [{"role": "assistant", "content": msg}]
        bundle["awaiting_user"] = {"t0": time.perf_counter(), "agent": "CallerAgent", "phase": "user_wait"}
        return new_history, ""

    async def on_submit(user_text, history, show_dbg, auto_rst, report_fast_flag):
        bot_text = await chat_logic(user_text, history, show_dbg, auto_rst, report_fast_flag)
        new_history = (history or []) + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": bot_text},
        ]
        return new_history, ""

    send.click(on_submit, inputs=[msg, chatbot, show_debug, auto_reset, report_fast], outputs=[chatbot, msg])
    msg.submit(on_submit, inputs=[msg, chatbot, show_debug, auto_reset, report_fast], outputs=[chatbot, msg])

    def _attach_quick_pick(btn: gr.Button, tool_name: str):
        # 直接切到該稅種的『階段一』，不走 LLM 判斷，也不需要先送出訊息
        return btn.click(
            functools.partial(jump_to_tax, tool_name),
            inputs=[chatbot, show_debug, auto_reset],
            outputs=[chatbot, msg],
            queue=False,
        )


    _attach_quick_pick(ex1,  "income_tax")
    _attach_quick_pick(ex2,  "foreigner_income_tax")
    _attach_quick_pick(ex3,  "business_income_tax")
    _attach_quick_pick(ex4,  "estate_tax")
    _attach_quick_pick(ex5,  "gift_tax")
    _attach_quick_pick(ex6,  "vat_tax")
    _attach_quick_pick(ex7,  "nvat_tax")
    _attach_quick_pick(ex8,  "cargo_tax")
    _attach_quick_pick(ex9,  "ta_tax")
    _attach_quick_pick(ex10, "securities_tx_tax")
    _attach_quick_pick(ex11, "futures_tx_tax")
    _attach_quick_pick(ex12, "special_goods_tax")
    _attach_quick_pick(ex13, "special_tax")

    demo.load(_on_hard_reset, inputs=None, outputs=[chatbot, msg], queue=False)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=32770,
        share=False,
        debug=True,
        show_api=False,
    )
