# agents/integrations/fintax_api.py
from __future__ import annotations
import asyncio
import os, json, re, time
from typing import Optional
from agents.multi_agent_tax_system import MEMORY
import logging, hashlib, datetime
import httpx

logger = logging.getLogger("tax.final_report")
def get_latest_report(memory=None):
    """
    取最後一份報告來源優先序：
    1) 來自傳入的 memory（如 ReasoningAgent.memory 或全域 MEMORY）
    2) 磁碟 fallback: reports/last_run/last.md / last.json
    回傳格式：{"md": str|None, "json": dict|None, "title": str, "ts": float}
    """
    md = None
    js = None
    ts = time.time()

    # 1) memory
    try:
        if memory:
            md = memory.get("last_report_md") or None
            latest = memory.get("__latest_report__") or {}
            if not md:
                md = latest.get("md")
            js = latest.get("json")
    except Exception:
        pass

    # 2) disk fallback
    base = "reports/last_run"
    md_path = os.path.join(base, "last.md")
    js_path = os.path.join(base, "last.json")

    if not md and os.path.exists(md_path):
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md = f.read()
        except Exception:
            md = None

    if js is None and os.path.exists(js_path):
        try:
            with open(js_path, "r", encoding="utf-8") as f:
                js = json.load(f)
        except Exception:
            js = None

    # derive a title from md
    title = ""
    if isinstance(md, str):
        m = re.search(r"^#{1,3}\s*(.+)$", md, flags=re.M)
        title = (m.group(1).strip() if m else "").strip()
    if not title:
        title = "稅務試算結論報告"

    return {"md": md, "json": js, "title": title, "ts": ts}

async def send_final_report(report: dict):
    """
    真的送出：在既有 logger/sent.log 的基礎上，呼叫 send_fintax_api。
    回傳包含 delivered / status_code / delivered_to 等欄位，讓前端可顯示。
    """
    md = (report or {}).get("md")
    js = (report or {}).get("json")
    title = (report or {}).get("title") or "稅務試算結論報告"

    if not isinstance(md, str) or not md.strip():
        raise RuntimeError("沒有可送出的報告內容（md 為空）。")

    md_len = len(md)
    md_sha = hashlib.sha256(md.encode("utf-8")).hexdigest()[:12]
    json_keys = len(list(js.keys())) if isinstance(js, dict) else 0
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("[SEND] title=%s md_len=%d md_sha=%s json_keys=%d ts=%s",
                title, md_len, md_sha, json_keys, ts)

    # 本地 sent.log
    try:
        base = "reports/last_run"; os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "last.sent.log"), "a", encoding="utf-8") as f:
            f.write(f"{ts} | title={title} | md_len={md_len} | md_sha={md_sha} | json_keys={json_keys}\n")
    except Exception as e:
        logger.warning("[SEND] write sent.log failed: %s", e)

    # === 真的送出到 FIN 後端 ===
    try:
        resp_info = await send_fintax_api(report=md, title=title)
        # 落地回應（利於現場排查）
        base = "reports/last_run"; os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "last.sent.response.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(resp_info, ensure_ascii=False, indent=2))
        if resp_info.get("ok"):
            logger.info("[SEND] webhook status=%s", resp_info.get("status_code"))
        else:
            logger.warning("[SEND] webhook failed: %s", resp_info.get("error"))
        return {
            "ok": bool(resp_info.get("ok")),
            "title": title,
            "delivered": bool(resp_info.get("ok")),
            "delivered_to": resp_info.get("delivered_to"),
            "status_code": resp_info.get("status_code"),
            "error": resp_info.get("error"),
        }
    except Exception as e:
        logger.warning("[SEND] exception: %s", e)
        return {"ok": False, "title": title, "delivered": False, "error": str(e)}

async def send_fintax_api(
    report: str,
    title: Optional[str] = None
):
    """
    依參考格式送出：
      POST http://fin-backend:6677/report/generated/fin-tax
      Content-Type: application/json
      body: {"report": string, "title": string|null}
    可用環境變數 FIN_BACKEND_BASEURL 覆寫 host（預設 http://fin-backend:6677）
    """
    if not isinstance(report, str) or not report.strip():
        raise ValueError("report 為必填且不可為空字串。")

    base = os.getenv("FIN_BACKEND_BASEURL", "http://fin-backend:6677").rstrip("/")
    url = f"{base}/report/generated/fin-tax"

    payload = {"report": report}
    if title is not None:
        payload["title"] = title

    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(url, json=payload)
        # 與參考範例一致：印出狀態與 json（方便你在後端 dbg）
        try:
            print(res.status_code, res.json())
        except Exception:
            print(res.status_code, res.text)

        info = {
            "ok": res.status_code >= 200 and res.status_code < 300,
            "status_code": res.status_code,
            "delivered_to": url
        }
        try:
            info["json"] = res.json()
        except Exception:
            info["text"] = res.text
        if not info["ok"]:
            info["error"] = f"HTTP {res.status_code}"
        return info



async def push_latest_report(title: Optional[str] = None):
    """方便在流程尾端直接呼叫（參考格式：只送 report 與 title）。"""
    report = get_latest_report(memory=MEMORY)
    return await send_fintax_api(
        report=report.get("md") or "",
        title=title or report.get("title"),
    )


if __name__ == "__main__":
    # 本地測試：python -m agents.integrations.fintax_api
    asyncio.run(push_latest_report("fintax-test"))

