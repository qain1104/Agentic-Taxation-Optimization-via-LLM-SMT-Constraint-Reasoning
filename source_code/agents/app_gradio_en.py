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

from source_code.agents.multi_agent_tax_system_en import (
    CallerAgent,
    ConstraintAgent,
    ExecuteAgent,
    ReasoningAgent,
    MemoryStore,
    TOOL_MAP,
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
    tip = "Want to change the conditions? Reply「Add Condition」 to add new restrictions to the existing ones; reply 「Reset Condition」to clear all conditions and return to the setting stage."
    md = md.replace("\n\n> " + tip, "")
    md = md.replace("\n> " + tip, "")
    md = md.replace("> " + tip, "")
    md = md.replace(tip, "")
    return md.strip()

_TUNING_TIPS_BLOCK_RE = re.compile(
    r"\n*Condition adjustment suggestions\s*\n"          # block header
    r"(?:.*\n)*?"                   # block body (non-greedy)
    r"(?=\n(?:To add more conditions | To complete settings | To clear | Current conditions | Stage 3 | Reply「Next step」|$))",
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
        "\n\n> **Next Step**\n"
        "> • To adjust the conditions: reply 「Add conditions」 or reply 「Reset conditions」 to return to the setting stage.\n"
        "> • To use this round of reports as the output report, enter 「Calculation complete」.\n"
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
    if not perf:
        return ""

    rows = []
    for agent, phases in perf.items():
        for phase, sec in phases.items():
            rows.append((agent, phase, sec))
    rows.sort(key=lambda x: (-x[2], x[0], x[1]))

    totals = {}
    for agent, _, sec in rows:
        totals[agent] = totals.get(agent, 0.0) + sec
    total_all = sum(totals.values())

    md = []
    md.append(f"\n\n**⏱️ Thinking Time（this round）≈ {total_all:.3f}s**")
    md.append("\n<details><summary>Detailed time taken (click to expand)</summary>\n")
    md.append("\n| Agent | Phase | Time (s) |")
    md.append("|---|---|---:|")
    for agent, phase, sec in rows[:200]:
        md.append(f"| {agent} | {phase} | {sec:.3f} |")
    md.append("\n**Agent Total**")
    md.append("\n| Agent | Total (s) |")
    md.append("|---|---:|")
    for a, t in sorted(totals.items(), key=lambda kv: -kv[1]):
        md.append(f"| {a} | {t:.3f} |")
    md.append("\n</details>")
    return "\n".join(md)

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
        "Calculation Completed","Recommendation Report Generated",
        "Conclusion Report Generated", "Conclusions Produced",
        "Summary", "Recommendations Produced", "final report", "finish & advise"
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

INTRO_MSG = """**👋 Welcome to the ROC Year 114 (2025) Taiwan Tax Agentic Service**

**First, tell the system which tax you want to calculate. Currently supported:**
- Individual Income Tax, Foreigner Individual Income Tax, Business Income Tax
- Estate Tax, Gift Tax
- Value-Added Business Tax (VAT), Non-VAT Business Tax
- Commodity (Excise) Tax, Tobacco and Alcohol Tax
- Securities / Futures Transaction Tax
- Special Goods Tax / Special Services Tax

**The system will identify the tax type, then guide you step-by-step to fill fields, set constraints, optimize the tax outcome, and generate a report.**
- After you finish comparing scenarios, type **"finish calculation"** and the system will save the **current report** as the **final report**.
> Results are estimates. The actual tax liability is subject to the competent authority’s rules and your filing data.
"""

async def chat_logic(
    user_msg: str,
    history,
    show_debug: bool = False,
    auto_reset: bool = True
):
    session_key = _get_or_create_session_key(history)
    bundle = _get_session_bundle(session_key)
    caller = bundle["caller"]
    constraint = bundle["constraint"]
    executor = bundle["executor"]
    reasoner = bundle["reasoner"]

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
        return any(key in s for key in ["reset constraints", "clear constraints"])

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
        exact = {"restart", "hard reset"}
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
            return _attach_session_tag("⚠️ Previous context cannot be found, please specify the type of tax to be calculated or perform a calculation once.", sess_key)

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

        cons_dbg_html = _details_text("DEBUG（ConstraintAgent）", ask.get("debug") or [])
        q = _strip_condition_tuning_tips(ask.get("question") or "（沒有問題文字）")
        debug_block = _dump_debug_and_clear(caller) if show_debug else ""

        msg = q + (cons_dbg_html if show_debug else "") + debug_block + _format_perf_breakdown(turn_perf)
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
            return _attach_session_tag("No tax calculation has been completed yet. Please select the tax type and complete at least one calculation.", session_key)

        base = "reports/last_run"
        sent_title = ""
        t0 = time.perf_counter()
        _perf_add(turn_perf, "ExecuteAgent", "fin_export_total", time.perf_counter() - t0)

        msg = (
            f"✅ The final **conclusion report** has been automatically saved.:\n"
            f"- {base}/last.md\n- {base}/last.json\n\n"
            f"（Each time calculation is completed, the latest version will be overwritten. Report has been sent:{sent_title}"
            + _format_perf_breakdown(turn_perf)
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
                        spans = reasoner.memory.get("perf_spans_last")
                    if isinstance(spans, list):
                        for item in spans:
                            if isinstance(item, (list, tuple)) and len(item) == 2:
                                _perf_add(turn_perf, "ReasoningAgent", str(item[0]), float(item[1]))
                            elif isinstance(item, dict) and "name" in item and "sec" in item:
                                _perf_add(turn_perf, "ReasoningAgent", str(item["name"]), float(item["sec"]))
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
                        "\n\n<details><summary>payload</summary>\n\n```json\n"
                        + payload_fmt + "\n```\n</details>"
                        + "\n\n<details><summary>Tool raw return</summary>\n\n```json\n"
                        + raw_result_fmt + "\n```\n</details>"
                    )

                msg = (
                    report_md
                    + _ui_footer_tip()
                    + debug_details
                    + (cons_dbg_html if show_debug else "")
                    + debug_block
                    + _format_perf_breakdown(turn_perf)
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
                msg = q + (cons_dbg_html if show_debug else "") + debug_block + _format_perf_breakdown(turn_perf)
                _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "constraint_follow_up"})
                return _attach_session_tag(msg, session_key)

            debug_block = _dump_debug_and_clear(caller) if show_debug else ""
            msg = (
                "⚠️ 未知 ConstraintAgent 回覆：\n```json\n"
                + json.dumps(parsed, ensure_ascii=False, indent=2)
                + "\n```"
                + (cons_dbg_html if show_debug else "")
                + debug_block
                + _format_perf_breakdown(turn_perf)
            )
            _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "constraint_unknown"})
            return _attach_session_tag(msg, session_key)

        # ---- General path: CallerAgent ----
        t0 = time.perf_counter()
        result = await caller.handle(user_msg)
        _perf_add(turn_perf, "CallerAgent", "handle_total", time.perf_counter() - t0)

        if isinstance(result, dict) and result.get("type") == "reopen_constraints":
            t0 = time.perf_counter()
            ask = await constraint.handle({"type": "reopen_constraints"})
            _perf_add(turn_perf, "ConstraintAgent", "handle_total", time.perf_counter() - t0)

            cons_dbg_html = _details_text("DEBUG（ConstraintAgent）", ask.get("debug") or [])
            q = _strip_condition_tuning_tips(ask.get("question") or "（沒有問題文字）")
            debug_block = _dump_debug_and_clear(caller) if show_debug else ""
            msg = q + (cons_dbg_html if show_debug else "") + debug_block + _format_perf_breakdown(turn_perf)
            _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "reopen_constraints"})
            return _attach_session_tag(msg, session_key)

        if isinstance(result, dict) and result.get("type") == "reset_constraints":
            return await _do_reset_constraints_and_reopen(session_key)

        if result is None:
            debug_block = _dump_debug_and_clear(caller) if show_debug else ""
            msg = "⚠️ System returns None" + debug_block + _format_perf_breakdown(turn_perf)
            _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "caller_none"})
            return _attach_session_tag(msg, session_key)
        if not isinstance(result, dict):
            debug_block = _dump_debug_and_clear(caller) if show_debug else ""
            msg = f"⚠️ Unexpected return type:{type(result).__name__}\n{result!r}" + debug_block + _format_perf_breakdown(turn_perf)
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
            msg2 = msg + debug_block + _format_perf_breakdown(turn_perf)
            _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "caller_follow_up"})
            return _attach_session_tag(msg2, session_key)

        if rtype == "tool_request":
            payload = result.get("payload", {}) or {}

            t0 = time.perf_counter()
            ask = await constraint.handle(result)
            _perf_add(turn_perf, "ConstraintAgent", "handle_total", time.perf_counter() - t0)

            payload_fmt = json.dumps(payload, ensure_ascii=False, indent=2)
            cons_dbg_html = _details_text("DEBUG（ConstraintAgent）", ask.get("debug") or [])

            if ask.get("type") == "follow_up":
                q = _strip_condition_tuning_tips(ask.get("question") or "（沒有問題文字）")
                debug_block = _dump_debug_and_clear(caller) if show_debug else ""
                msg = q + (cons_dbg_html if show_debug else "") + debug_block + _format_perf_breakdown(turn_perf)
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
                        spans = reasoner.memory.get("perf_spans_last")
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
                        "\n\n<details><summary>payload</summary>\n\n```json\n"
                        + payload_fmt + "\n```\n</details>"
                        + "\n\n<details><summary>result</summary>\n\n```json\n"
                        + raw_result_fmt + "\n```\n</details>"
                    )

                msg = (
                    report_md
                    + _ui_footer_tip()
                    + debug_details
                    + (cons_dbg_html if show_debug else "")
                    + _format_perf_breakdown(turn_perf)
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
                "⚠️ Unknown ConstraintAgent apply:\n```json\n"
                + json.dumps(ask, ensure_ascii=False, indent=2)
                + "\n```"
                + (cons_dbg_html if show_debug else "")
                + debug_block
                + _format_perf_breakdown(turn_perf)
            )
            _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "constraint_unknown_after_tool_request"})
            return _attach_session_tag(msg, session_key)

        debug_block = _dump_debug_and_clear(caller) if show_debug else ""
        msg = "⚠️ Unknown CallerAgent apply:\n```json\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n```" + debug_block + _format_perf_breakdown(turn_perf)
        _persist_perf_snapshot(executor, session_key, turn_perf, meta={"type": "caller_unknown"})
        return _attach_session_tag(msg, session_key)

    except Exception as exc:
        debug_block = _dump_debug_and_clear(caller) if show_debug else ""
        tb = traceback.format_exc()
        msg = f"⚠️ System error:{exc}\n\n```\n{tb}\n```" + debug_block + _format_perf_breakdown(turn_perf)
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
            "Automatic intent parsing using a multi-agent architecture → supplementing tax variables → optimization → report output",
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
                placeholder="After entering the command, press Shift + Enter to send the command, and Enter will start a newline.",
                lines=2,
            )
            with gr.Row():
                send = gr.Button("🚀 Send", variant="primary")
                clear = gr.Button("🧹 Clear input/dialogue (hard reset)")
                clear.click(_on_hard_reset, inputs=None, outputs=[chatbot, msg], queue=False)

        with gr.Column(scale=5):
            with gr.Group(elem_classes=["glass-card"]):
                gr.Markdown("**⚙️ Execution Options**")
                with gr.Row():
                    show_debug = gr.Checkbox(value=False, label="Show DEBUG blocks")
                    auto_reset = gr.Checkbox(value=True, label="Automatic soft reset at the end of each round (with continuation preserved).")

            with gr.Group(elem_classes=["glass-card"]):
                gr.Markdown("**🧭 Pick a tax type** (click to auto-fill)")
                ex1 = gr.Button("Calculate Individual Income Tax", elem_classes=["example-chip"])
                ex2 = gr.Button("Calculate Foreigner Individual Income Tax", elem_classes=["example-chip"])
                ex3 = gr.Button("Calculate Business Income Tax", elem_classes=["example-chip"])
                ex4 = gr.Button("Calculate Estate Tax", elem_classes=["example-chip"])
                ex5 = gr.Button("Calculate Gift Tax", elem_classes=["example-chip"])
                ex6 = gr.Button("Calculate VAT (Value-Added Business Tax)", elem_classes=["example-chip"])
                ex7 = gr.Button("Calculate Non-VAT Business Tax", elem_classes=["example-chip"])
                ex8 = gr.Button("Calculate Commodity (Excise) Tax", elem_classes=["example-chip"])
                ex9 = gr.Button("Calculate Tobacco and Alcohol Tax", elem_classes=["example-chip"])
                ex10 = gr.Button("Calculate Securities Transaction Tax", elem_classes=["example-chip"])
                ex11 = gr.Button("Calculate Futures Transaction Tax", elem_classes=["example-chip"])
                ex12 = gr.Button("Calculate Special Goods Tax", elem_classes=["example-chip"])
                ex13 = gr.Button("Calculate Special Services Tax", elem_classes=["example-chip"])

            with gr.Accordion("📘 User guide (click to expand)", open=False, elem_classes=["glass-card"]):
                gr.Markdown(
                    """Enter the tax type first, and the system will guide you to fill variables, add constraints, and generate a report.

**Input format tips**:
- Supports Chinese units like  / ; the system will convert them into TWD automatically.
- Supports ROC dates, e.g., 112/3/15.
- For multiple items, separate with `;` / `,` / ``. Use `x` / `X` for quantities, e.g., `luxury car 8,000,000 x 2`.
- Use `→ max / min` to indicate an optimization goal, or specify a target tax directly, e.g., `total tax 500000`.
- You can use `+ - * /` for arithmetic, e.g., `land 70,000,000 + house 30,000,000`.
- Use `> / < / >= / <= / =` for constraints, e.g., `land >= 50,000,000`.
- Use `%` for percentages, e.g., `shareholding 20%`.
- Use `about / approx / ~` to express fuzzy numbers, e.g., `estate total about 100,000,000`.
- Use `to` for ranges, e.g., `period 111 to 113`.
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
            q = f"Tax type has been changed to {tool_name} (but the field navigation cannot be loaded; please check TOOL_MAP / tools_registry).）"

        q = _strip_condition_tuning_tips(q)
        msg = _attach_session_tag(q, session_key)

        # 只顯示第一階段頁面（不保留舊對話）
        new_history = [{"role": "assistant", "content": msg}]
        bundle["awaiting_user"] = {"t0": time.perf_counter(), "agent": "CallerAgent", "phase": "user_wait"}
        return new_history, ""

    async def on_submit(user_text, history, show_dbg, auto_rst):
        bot_text = await chat_logic(user_text, history, show_dbg, auto_rst)
        new_history = (history or []) + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": bot_text},
        ]
        return new_history, ""

    send.click(on_submit, inputs=[msg, chatbot, show_debug, auto_reset], outputs=[chatbot, msg])
    msg.submit(on_submit, inputs=[msg, chatbot, show_debug, auto_reset], outputs=[chatbot, msg])

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
