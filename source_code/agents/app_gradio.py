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
    _trigger_fin_export
)

# 每個 session 一組獨立的 MemoryStore + agents
SESSIONS: dict[str, dict] = {}

# ===== 用隱藏標籤綁定對話 Session（避免用 id(history) 每次都變） =====
_SESSION_TAG_RE = re.compile(r"<!--\s*SESSION:([0-9a-fA-F-]{8,})\s*-->")

def _get_or_create_session_key(history) -> str:
    """
    從 history 內倒序尋找 SESSION 標記；若沒有，生成新的 UUID。
    """
    if isinstance(history, list):
        for msg in reversed(history):
            if isinstance(msg, dict):
                content = msg.get("content")
            else:
                content = None
            if not isinstance(content, str):
                continue
            m = _SESSION_TAG_RE.search(content)
            if m:
                return m.group(1)
    # 沒找到就新建一個
    return str(uuid.uuid4())

def _attach_session_tag(text: str, session_key: str) -> str:
    """
    在回覆文字末尾附加 <!-- SESSION:... -->，避免重複附加。
    """
    if not isinstance(text, str):
        text = str(text)
    if _SESSION_TAG_RE.search(text):
        return text
    return text + f"\n\n<!-- SESSION:{session_key} -->"


def _get_session_bundle(session_key: str) -> dict:
    """
    依 session_key 取得或建立一組 session 專用的 agents + memory。
    """
    bundle = SESSIONS.get(session_key)
    if bundle is None:
        mem = MemoryStore()
        bundle = {
            "memory": mem,
            "caller": CallerAgent(memory=mem),
            "constraint": ConstraintAgent(memory=mem),
            "executor": ExecuteAgent(memory=mem),
            "reasoner": ReasoningAgent(memory=mem),
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

# --- 讓報告本體乾淨：剝掉 ReasoningAgent 最後附加的互動提示 ---
def _strip_inline_tips(md: str) -> str:
    if not isinstance(md, str):
        return md
    tip = "想變更條件？回覆「再加條件」可在現有基礎上加新限制；回覆「重設條件」會清空所有條件並回到設定階段。"
    # 報告內可能有前置的 "> " 與前後換行，逐一移除
    md = md.replace("\n\n> " + tip, "")
    md = md.replace("\n> " + tip, "")
    md = md.replace("> " + tip, "")
    md = md.replace(tip, "")
    return md.strip()

# --- 報告下方的 UI 操作說明（不放進報告本體） ---
def _ui_footer_tip() -> str:
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

def _preserve_reopen_context_from_exec(exec_out: dict, caller, constraint, executor):
    """把工具執行結果存入各 Agent 的記憶，供『再加條件 / 重設條件』續接使用。"""
    try:
        tool = exec_out.get("tool_name")
        pay  = exec_out.get("payload") or {}
        if not tool or not isinstance(pay, dict):
            return

        # 先組基本的 ctx_payload
        ctx_payload = {
            "tool_name": tool,
            "user_params": (pay.get("user_params") or {}),
            "op": pay.get("op"),
        }

        # 把當前 pending payload 中的 early_tips_md 也帶進保險箱
        pending_from_caller = caller.memory.get("pending_constraint_payload") or {}
        pending_from_cons   = constraint.memory.get("pending_constraint_payload") or {}
        tips = (
            pending_from_caller.get("early_tips_md")
            or pending_from_cons.get("early_tips_md")
            or pay.get("early_tips_md")
        )
        if isinstance(tips, str) and tips.strip():
            ctx_payload["early_tips_md"] = tips

        # 寫入 constraint / caller
        constraint.memory.set("pending_tool_for_constraints", tool)
        constraint.memory.set("pending_constraint_payload", ctx_payload)
        constraint.memory.set("last_exec_payload", {"tool_name": tool, "payload": ctx_payload})

        caller.memory.set("pending_tool_for_constraints", tool)
        caller.memory.set("pending_constraint_payload", ctx_payload)
        caller.memory.set("last_tool", tool)

        # ★ 同步到 executor（保險箱）
        executor.memory.set("last_exec_payload", {"tool_name": tool, "payload": ctx_payload})

    except Exception:
        pass


def _persist_run_and_get_prev(exec_out: dict, executor):
    """
    把本輪執行的稅額與參數存入 executor.memory 的歷史陣列，並回傳『上一輪』快照（若有）。
    結構：
    - history_runs: [ { ts, tool_name, mode, baseline, optimized, status, final_params, constraints } ... ]
    - last_run: 同上最後一筆
    - prev_run: 倒數第二筆（若存在）
    """
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

# ========== 專門存放每輪的報告 Markdown ==========
def _persist_report_markdown(exec_out: dict, report_md: str, executor):
    """
    將本輪 ReasoningAgent 產出的 Markdown 全文，持久化到 executor.memory['report_history']。
    結構：
    report_history: {
        <tool_name>: [
            {
                "ts": float, "mode": str|None, "status": str|None,
                "baseline": float|None, "optimized": float|None,
                "budget": float|None, "md": str
            }, ...
        ]
    }
    """
    try:
        tool = exec_out.get("tool_name") or (exec_out.get("payload") or {}).get("tool_name")
        if not tool or not isinstance(report_md, str) or not report_md.strip():
            return
        res = exec_out.get("result") or {}
        payload = exec_out.get("payload") or {}
        user_params = (payload.get("user_params") or {}) if isinstance(payload, dict) else {}

        # 嘗試抓 budget
        budget_field = TOOL_MAP.get(tool, {}).get("budget_field")
        budget_val = user_params.get(budget_field) if budget_field else None
        if budget_val is None:
            # 若工具有回傳 budget 欄位，也納入
            for k in ("budget", "budget_tax", "tax_budget"):
                if isinstance(res.get(k), (int, float)):
                    budget_val = res.get(k); break

        item = {
            "ts": __import__("time").time(),
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
        # 控制上限（例如保留最近 20 份）
        if len(arr) > 20:
            arr = arr[-20:]
        hist[tool] = arr
        executor.memory.set("report_history", hist)
    except Exception:
        pass


def _save_last_run_files(tool_name: str | None, final_md: str, result: dict, payload: dict):
    """
    將『本輪』的最終報告與原始結果落地存檔。
    - 只保留『最後一輪』語意：以固定檔名覆寫。
    - 產出 Markdown 與 JSON 兩份（API 端通常較愛吃 JSON，但你也有漂亮的 MD 可用）。
    目錄結構：
        reports/last_run/
        ├─ last_<tool>.md
        ├─ last_<tool>.json
        ├─ last.md          （全域最新，無論稅別）
        └─ last.json
    """
    import os, re, json, time as _time

    if not isinstance(final_md, str) or not final_md.strip():
        return

    tool = tool_name or "unknown_tool"
    tool_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(tool))

    out_dir = os.path.join("reports", "last_run")
    os.makedirs(out_dir, exist_ok=True)

    # 固定檔名（覆寫）——「只存最後一次」
    md_path_tool  = os.path.join(out_dir, f"last_{tool_slug}.md")
    json_path_tool = os.path.join(out_dir, f"last_{tool_slug}.json")
    md_path_latest  = os.path.join(out_dir, "last.md")
    json_path_latest = os.path.join(out_dir, "last.json")

    # 組 JSON：包含必要中繼資訊，方便 API 端直接打包上傳
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
        "result": result,    # 工具原始回傳（完整）
        "payload": payload,  # 回推用的上下文（含 user_params / constraints 等）
        "markdown": final_md # 方便有需要時一檔帶走
    }

    # ---- 落地存檔（覆寫即可）----
    with open(md_path_tool, "w", encoding="utf-8") as f:
        f.write(final_md)
    with open(json_path_tool, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)

    # 也同時覆寫全域 latest（看你要不要；通常好用）
    with open(md_path_latest, "w", encoding="utf-8") as f:
        f.write(final_md)
    with open(json_path_latest, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)


# ========== 辨識『計算完成』的指令 ==========
def _should_finish(s: str) -> bool:
    s = (s or "").strip().lower()
    return any(k in s for k in [
        "計算完成", "完成計算",
        "出建議報告", "產生建議報告",  # 舊指令仍支援
        "出結論報告", "產生結論報告", "產出結論",
        "匯總", "總結", "產出建議", "final report", "finish & advise"
    ])


# =========================
# 軟重置：清短期記憶，但回填上一輪上下文（讓「再加條件」能續接）
# =========================
def _reset_session_state(caller, constraint, executor, reasoner):
    try:
        last_ctx = executor.memory.get("last_exec_payload") or {}
        tool = last_ctx.get("tool_name")
        payload = last_ctx.get("payload")
    except Exception:
        tool, payload = None, None

    # 清掉短期記憶（保留 executor 的保險箱）
    for a in (caller, constraint, reasoner):
        try:
            a.memory.clear()
        except Exception:
            pass

    # 回填上一輪上下文，保證可直接「再加條件」
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

# =========================
# 硬重置：把所有 session 的記憶全部清空（真正回到全空）
# =========================
def _hard_reset_all_states():
    """
    硬重置：清空所有 session 的 agents 記憶。
    （下一輪 chat_logic 會自動為新的 history 建立新的 session bundle。）
    """
    SESSIONS.clear()


def _on_hard_reset():
    _hard_reset_all_states()
    return ([{"role": "assistant", "content": INTRO_MSG}], "")

def _format_thinking_time(tt: dict[str, float]) -> str:
    if not tt:
        return ""
    order = ["CallerAgent", "ConstraintAgent", "ExecuteAgent", "ReasoningAgent"]
    total = sum(tt.values())
    rows = ["| Agent | Time (s) |", "|---|---:|"]
    for k in order:
        if k in tt:
            rows.append(f"| {k} | {tt[k]:.3f} |")
    return "\n\n**🧠 思考時間**（本輪）≈ **{total:.3f}s**\n\n".format(total=total) + "\n".join(rows)


async def chat_logic(
    user_msg: str,
    history,
    show_debug: bool = False,
    auto_reset: bool = True
):
    # ===== 取得本輪對應的 session agents（用 hidden SESSION tag 綁定） =====
    session_key = _get_or_create_session_key(history)
    bundle = _get_session_bundle(session_key)
    caller = bundle["caller"]
    constraint = bundle["constraint"]
    executor = bundle["executor"]
    reasoner = bundle["reasoner"]

    # ===== 指令判斷器 =====
    def _should_reset_constraints(s: str) -> bool:
        s = (s or "").strip().lower()
        return any(key in s for key in ["重設條件", "重置條件", "reset constraints", "clear constraints"])

    def has_latest_report() -> bool:
        # 1) 看這個 session 的 ReasoningAgent / ExecuteAgent 記憶體
        try:
            if reasoner and (
                reasoner.memory.get("last_report_md") or reasoner.memory.get("__latest_report__")
            ):
                return True
        except Exception:
            pass
        try:
            if executor and (
                executor.memory.get("last_report_md") or executor.memory.get("__latest_report__")
            ):
                return True
        except Exception:
            pass

        # 2) 檔案 fallback（handle() 已寫入 reports/last_run/）
        return (
            os.path.exists("reports/last_run/last.md")
            or os.path.exists("reports/last_run/last.json")
        )

    def _should_hard_reset(s: str) -> bool:
        """
        硬重置採【精確比對】與少量同義詞；只要訊息包含「條件」兩字就不當硬重置。
        避免把「重設條件」誤判成整站重置。
        """
        s = (s or "").strip().lower()
        if "條件" in s:
            return False
        exact = {"重置", "清空", "reset", "重新開始", "restart", "硬重置", "hard reset"}
        if s in exact:
            return True
        # 接受幾個常見簡寫
        return s in {"reset()", "reset all", "clear all"}

    async def _do_reset_constraints_and_reopen(sess_key: str):
        # 優先用上一輪 executor 保留的上下文；退而求其次用 caller/constraint 的 pending
        last_ctx = executor.memory.get("last_exec_payload") or {}
        tool = last_ctx.get("tool_name") or caller.memory.get("pending_tool_for_constraints")
        payload0 = (
            last_ctx.get("payload")
            or caller.memory.get("pending_constraint_payload")
            or constraint.memory.get("pending_constraint_payload")
            or {}
        )
        if not tool or not isinstance(payload0, dict):
            return _attach_session_tag(
                "⚠️ 找不到上一輪上下文，請先指定要計算的稅種或執行一次計算。",
                sess_key,
            )

        # 用 ReasoningAgent 的 API 清空條件（constraints/free_vars/bounds）
        new_payload = reasoner._payload_with_constraints_reset(payload0)

        # **重點：清空 ConstraintAgent（避免沿用 constraints_preview / free_vars 快取）**
        try:
            constraint.memory.clear()
        except Exception:
            pass

        # 回寫 pending（讓 ConstraintAgent 重新發問）
        constraint.memory.set("pending_tool_for_constraints", tool)
        constraint.memory.set("pending_constraint_payload", new_payload)
        caller.memory.set("pending_tool_for_constraints", tool)
        caller.memory.set("pending_constraint_payload", new_payload)

        # 更新保險箱：以便後續「再加條件」仍能銜接這個全新狀態
        executor.memory.set("last_exec_payload", {"tool_name": tool, "payload": new_payload})

        # 重新開啟「條件設定」階段
        ask = await constraint.handle({"type": "reopen_constraints"})
        cons_dbg_html = _details_text("DEBUG（ConstraintAgent）", ask.get("debug") or [])
        q = ask.get("question") or "（沒有問題文字）"
        debug_block = _dump_debug_and_clear(caller) if show_debug else ""
        return _attach_session_tag(q + (cons_dbg_html if show_debug else "") + debug_block, sess_key)

    # 1)「重設條件」
    if _should_reset_constraints(user_msg):
        return await _do_reset_constraints_and_reopen(session_key)

    # 2)「硬重置」（精確比對） → 清掉這個 session 的記憶
    if _should_hard_reset(user_msg):
        for a in (caller, constraint, executor, reasoner):
            try:
                a.memory.clear()
            except Exception:
                pass
        return _attach_session_tag(INTRO_MSG, session_key)

    # 3)「計算完成」→ 彙總所有 Markdown 成建議報告
    if _should_finish(user_msg):
        if not has_latest_report():
            return _attach_session_tag(
                "目前尚未完成任何稅額試算，請先選擇稅種並完成至少一次計算。",
                session_key,
            )

        base = "reports/last_run"
        sent_title = ""
        try:
            info = await _trigger_fin_export(executor.memory)
            # info 可能是 dict 或其他型別，這裡保守取值
            if isinstance(info, dict):
                sent_title = info.get("title") or ""
            else:
                sent_title = str(info) if info is not None else ""
        except Exception as e:
            # 匯出（寄送）失敗不應阻擋使用者取得「已產出之最後報告」
            sent_title = f"(匯出程序略過：{e})"

        msg = (
            f"✅ 最終**結論報告**已自動儲存：\n"
            f"- {base}/last.md\n- {base}/last.json\n\n"
            f"（每次「計算完成」都會覆寫為最新），已送出報告：{sent_title}"
        )
        return _attach_session_tag(msg, session_key)

    # ===== 本輪思考時間累加器 =====
    thinking_times: dict[str, float] = {
        "CallerAgent": 0.0,
        "ConstraintAgent": 0.0,
        "ExecuteAgent": 0.0,
        "ReasoningAgent": 0.0,
    }

    try:
        # 同時檢查 ConstraintAgent 與 CallerAgent 的 pending 狀態
        pending_for_cons = (
            constraint.memory.get("pending_tool_for_constraints")
            or constraint.memory.get("pending_constraint_payload")
            or caller.memory.get("pending_tool_for_constraints")
            or caller.memory.get("pending_constraint_payload")
        )
        if pending_for_cons:
            # ConstraintAgent
            t0 = time.perf_counter()
            parsed = await constraint.handle({"type": "constraints_reply", "text": user_msg})
            thinking_times["ConstraintAgent"] += time.perf_counter() - t0

            cons_dbg_html = _details_text("DEBUG（ConstraintAgent）", parsed.get("debug") or [])

            # ✅ 若 ConstraintAgent 回傳 reset 訊號，也能處理
            if parsed.get("type") == "reset_constraints":
                return await _do_reset_constraints_and_reopen(session_key)

            if parsed.get("type") == "ready_for_execute":
                payload = parsed.get("payload") or {}

                # ExecuteAgent
                t0 = time.perf_counter()
                exec_out = await executor.handle(payload)
                thinking_times["ExecuteAgent"] += time.perf_counter() - t0

                # ★ 同步可續接上下文到 caller/constraint/executor
                _preserve_reopen_context_from_exec(exec_out, caller, constraint, executor)

                # ★★★ NEW：保存本輪與取得上一輪快照，並把上一輪關鍵值塞回 exec_out["payload"]
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

                # ReasoningAgent
                t0 = time.perf_counter()
                fb = await reasoner.handle(exec_out)
                # ★★★ 新增：把『本輪』結果落地存檔（只保留最後一次）
                try:
                    _save_last_run_files(
                        exec_out.get("tool_name"),
                        fb.get("text", "") or "",
                        exec_out.get("result") or {},
                        exec_out.get("payload") or {},
                    )
                except Exception as e:
                    # 不要中斷流程；寫到 debug 方便排查
                    dbg_lines = caller.memory.get("debug_lines", []) or []
                    dbg_lines.append(f"[last-run-save] ERROR: {e}")
                    caller.memory.set("debug_lines", dbg_lines)
                try:
                    _persist_report_markdown(exec_out, fb.get("text", ""), executor)
                except Exception:
                    pass
                thinking_times["ReasoningAgent"] += time.perf_counter() - t0

                debug_block = _dump_debug_and_clear(caller) if show_debug else ""
                payload_fmt = json.dumps(payload, ensure_ascii=False, indent=2)
                raw_result_fmt = json.dumps(exec_out.get("result"), ensure_ascii=False, indent=2)
                report_md = _strip_inline_tips(fb.get("text", "") or "")

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
                    + _ui_footer_tip()  # NEW: UI 端的操作說明顯示在報告之後
                    + debug_details
                    + (cons_dbg_html if show_debug else "")
                    + debug_block
                    + _format_thinking_time(thinking_times)
                )
                if auto_reset:
                    _reset_session_state(caller, constraint, executor, reasoner)
                return _attach_session_tag(msg, session_key)

            if parsed.get("type") == "follow_up":
                q = parsed.get("question") or "（沒有問題文字）"
                debug_block = _dump_debug_and_clear(caller) if show_debug else ""
                msg = q + (cons_dbg_html if show_debug else "") + debug_block
                return _attach_session_tag(msg, session_key)

            debug_block = _dump_debug_and_clear(caller) if show_debug else ""
            msg = (
                "⚠️ 未知 ConstraintAgent 回覆：\n```json\n"
                + json.dumps(parsed, ensure_ascii=False, indent=2)
                + "\n```"
                + (cons_dbg_html if show_debug else "")
                + debug_block
            )
            return _attach_session_tag(msg, session_key)

        # ---- 一般情況：交給 CallerAgent ----
        t0 = time.perf_counter()
        result = await caller.handle(user_msg)
        thinking_times["CallerAgent"] += time.perf_counter() - t0

        # ★ 支援 CallerAgent 的 reopen 訊號（例如使用者輸入「再加條件」）
        if isinstance(result, dict) and result.get("type") == "reopen_constraints":
            t0 = time.perf_counter()
            ask = await constraint.handle({"type": "reopen_constraints"})
            thinking_times["ConstraintAgent"] += time.perf_counter() - t0

            cons_dbg_html = _details_text("DEBUG（ConstraintAgent）", ask.get("debug") or [])
            q = ask.get("question") or "（沒有問題文字）"
            debug_block = _dump_debug_and_clear(caller) if show_debug else ""
            msg = q + (cons_dbg_html if show_debug else "") + debug_block
            return _attach_session_tag(msg, session_key)

        # 若 CallerAgent 直接回傳 reset_constraints，也能接住
        if isinstance(result, dict) and result.get("type") == "reset_constraints":
            return await _do_reset_constraints_and_reopen(session_key)

        if result is None:
            debug_block = _dump_debug_and_clear(caller) if show_debug else ""
            msg = "⚠️ 系統回傳空結果（None）。" + debug_block
            return _attach_session_tag(msg, session_key)
        if not isinstance(result, dict):
            debug_block = _dump_debug_and_clear(caller) if show_debug else ""
            msg = f"⚠️ 非預期回傳型別：{type(result).__name__}\n{result!r}" + debug_block
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
            return _attach_session_tag(msg + debug_block, session_key)

        if rtype == "tool_request":
            payload = result.get("payload", {}) or {}
            
            # ConstraintAgent（第一次，詢問/解析約束）
            t0 = time.perf_counter()
            ask = await constraint.handle(result)
            thinking_times["ConstraintAgent"] += time.perf_counter() - t0

            payload_fmt = json.dumps(payload, ensure_ascii=False, indent=2)
            cons_dbg_html = _details_text("DEBUG（ConstraintAgent）", ask.get("debug") or {})

            if ask.get("type") == "follow_up":
                q = ask.get("question") or "（沒有問題文字）"
                msg = (
                    f"{q}"
                    + (cons_dbg_html if show_debug else "")
                )
                debug_block = _dump_debug_and_clear(caller) if show_debug else ""
                return _attach_session_tag(msg + debug_block, session_key)

            if ask.get("type") == "ready_for_execute":
                payload2 = ask.get("payload") or payload
                # ExecuteAgent
                t0 = time.perf_counter()
                exec_out = await executor.handle(payload2)
                thinking_times["ExecuteAgent"] += time.perf_counter() - t0

                # ★ 同步可續接上下文到 caller/constraint/executor
                _preserve_reopen_context_from_exec(exec_out, caller, constraint, executor)

                # ★★★ NEW：保存本輪與取得上一輪快照，並把上一輪關鍵值塞回 exec_out["payload"]
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

                # ReasoningAgent
                t0 = time.perf_counter()
                fb = await reasoner.handle(exec_out)
                # ★★★ 新增：把『本輪』結果落地存檔（只保留最後一次）
                try:
                    _save_last_run_files(
                        exec_out.get("tool_name"),
                        fb.get("text", "") or "",
                        exec_out.get("result") or {},
                        exec_out.get("payload") or {},
                    )
                except Exception as e:
                    # 不要中斷流程；寫到 debug 方便排查
                    dbg_lines = caller.memory.get("debug_lines", []) or []
                    dbg_lines.append(f"[last-run-save] ERROR: {e}")
                    caller.memory.set("debug_lines", dbg_lines)

                thinking_times["ReasoningAgent"] += time.perf_counter() - t0
                
                raw_result_fmt = json.dumps(exec_out.get("result"), ensure_ascii=False, indent=2)
                report_md = _strip_inline_tips(fb.get("text", "") or "")

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
                    + _ui_footer_tip()  # NEW: UI 端的操作說明顯示在報告之後
                    + debug_details
                    + (cons_dbg_html if show_debug else "")
                    + _format_thinking_time(thinking_times)
                )
                debug_block = _dump_debug_and_clear(caller) if show_debug else ""
                if auto_reset:
                    _reset_session_state(caller, constraint, executor, reasoner)
                return _attach_session_tag(msg + debug_block, session_key)

            debug_block = _dump_debug_and_clear(caller) if show_debug else ""
            msg = (
                "⚠️ 未知 ConstraintAgent 回覆：\n```json\n"
                + json.dumps(ask, ensure_ascii=False, indent=2)
                + "\n```"
                + (cons_dbg_html if show_debug else "")
                + debug_block
            )
            return _attach_session_tag(msg, session_key)

        debug_block = _dump_debug_and_clear(caller) if show_debug else ""
        msg = "⚠️ 未知 CallerAgent 回覆：\n```json\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n```" + debug_block
        return _attach_session_tag(msg, session_key)

    except Exception as exc:
        debug_block = _dump_debug_and_clear(caller) if show_debug else ""
        tb = traceback.format_exc()
        msg = f"⚠️ 系統錯誤：{exc}\n\n```\n{tb}\n```" + debug_block
        return _attach_session_tag(msg, session_key)


INTRO_MSG = r"""
**👋 歡迎使用《114年度台灣稅務 Agentic Service》**

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

with gr.Blocks(
    title="Taiwan Tax Agentic Service Demo",
    theme=gr.themes.Soft(),
    css=r"""
    /* ==== 自訂聊天框：移除右上角垃圾桶（Clear） ==== */
    /* v5 可能出現的 selector 一次蓋掉，確保穩定 */
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
        gr.Markdown("### Taiwan Tax Multi-Agent Demo\n以多代理架構自動解析意圖→補齊稅務變數→最佳化→報告輸出", elem_classes=["glass-card"])

    with gr.Row():
        # 左：聊天區
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
                lines=2
            )
            with gr.Row():
                send = gr.Button("🚀 送出", variant="primary")
                clear = gr.Button("🧹 清空輸入/對話（硬重置）")
                # 按鈕 → 硬重置（全空）
                clear.click(_on_hard_reset, inputs=None, outputs=[chatbot, msg])

        # 右：控制面板
        with gr.Column(scale=5):
            with gr.Group(elem_classes=["glass-card"]):
                gr.Markdown("**⚙️ 執行選項**")
                with gr.Row():
                    show_debug = gr.Checkbox(value=False, label="顯示 DEBUG 區塊")
                    # ✅ 勾選時：每輪結束做「軟重置」（可續接再加條件）
                    auto_reset = gr.Checkbox(value=True, label="每輪結束自動軟重置（保留續接）")

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
                    """
            請先輸入欲計算的稅種、系統會引導您補齊變數、加入條件、生成報告。

            **輸入格式建議**：
            - 支援「萬 / 億」單位，系統會自動轉換成「元」。
            - 支援民國日期，如「112/3/15」。
            - 多筆資料請用「; / ， / ；」分隔，每筆可用「x / X」表示數量，如「名車 800 萬 x 2」。
            - 可用「→ 最大 / 最小」表示優化目標，或是直接指定目標稅額，如「總稅額 500000」。
            - 可用「+ - * /」表達運算，如「土地 7000 萬 + 房屋 3000 萬」。
            - 可用「> / < / >= / <= / =」表示條件，如「土地 ≥ 5000 萬」。
            - 可用「%」表示百分比，如「持股 20%」。
            - 可用「約 / 大約 / 左右」表示模糊數字，如「遺產總額 1 億左右」。
            - 可用「至 / 到」表示區間，如「期間 111 年至 113 年」。"""
            )

    # 事件處理（messages 版本）
    async def on_submit(user_text, history, show_dbg, auto_rst):
        bot_text = await chat_logic(user_text, history, show_dbg, auto_rst)
        new_history = (history or []) + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": bot_text},
        ]
        return new_history, ""

    send.click(on_submit, inputs=[msg, chatbot, show_debug, auto_reset], outputs=[chatbot, msg])
    msg.submit(on_submit, inputs=[msg, chatbot, show_debug, auto_reset], outputs=[chatbot, msg])

    # --- 快速範例：點了就送出 ---
    def _attach_quick_example(btn: gr.Button, text: str):
        # 任何自動填入前先【硬清空】（重置所有 session 的記憶與對話區）
        return (
            btn.click(_on_hard_reset, inputs=None, outputs=[chatbot, msg])
              .then(lambda: text, None, msg)  # 重置後再填入預設訊息
              .then(on_submit, [msg, chatbot, show_debug, auto_reset], [chatbot, msg])
        )

    _attach_quick_example(ex1,  "我想計算綜合所得稅")
    _attach_quick_example(ex2,  "我想計算外僑所得稅")
    _attach_quick_example(ex3,  "我想計算營利事業所得稅")
    _attach_quick_example(ex4,  "我想計算遺產稅")
    _attach_quick_example(ex5,  "我想計算贈與稅")
    _attach_quick_example(ex6,  "我想計算加值型營業稅")
    _attach_quick_example(ex7,  "我想計算非加值型營業稅")
    _attach_quick_example(ex8,  "我想計算貨物稅")
    _attach_quick_example(ex9,  "我想計算菸酒稅")
    _attach_quick_example(ex10, "我想計算證券交易稅")
    _attach_quick_example(ex11, "我想計算期貨交易稅")
    _attach_quick_example(ex12, "我想計算特種貨物稅")
    _attach_quick_example(ex13, "我想計算特種勞務稅")

    # 頁面載入時 → 硬重置（回到全空）
    demo.load(_on_hard_reset, inputs=None, outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=32770,
        share=False,
        debug=True,
        show_api=False,
    )