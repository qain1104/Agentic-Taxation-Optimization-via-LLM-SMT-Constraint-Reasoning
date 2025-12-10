"""
app_fastapi.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
提供給工研院平台串接用的 REST API 服務：

* GET /health  ：健康檢查（Docker / K8s 存活 &就緒探針）
* POST /run    ：平台發出「稅務試算 / 最佳化」請求的核心端點
* POST /export ：平台在使用者按下「匯出報告」時呼叫，必要時順便推送到 FIN 後端

說明
----

1. 本檔案假設 multi-agent pipeline 已經在 multi_agent_tax_system.py 中實作，
   並提供一個同步函式：

       def run_tax_pipeline(payload: dict) -> dict:
           ...
           return {
               "status": "ok",
               "kpi": {...},
               "final_params": {...},
               "diff": {...},
               "report_md": "...",
               # 其他欄位不影響本 API
           }

   若你的實作函式叫別的名字（例如 run_multi_agent、run_from_api 等），
   可以到這個檔尾的 `_call_backend` 裡面調整。

2. /export 會透過 agents.integrations.fintax_api 存取最後一份報告，
   並在 push_to_fin = true 時，呼叫 FIN 的 webhook：

       POST {FIN_BACKEND_BASEURL}/report/generated/fin-tax

   FIN_BACKEND_BASEURL 請透過環境變數設定。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, root_validator

# ----------------------------------------------------------------------------
# 匯入 multi_agent_tax_system（pipeline 入口）與 FIN 整合工具
# ----------------------------------------------------------------------------

try:
    from agents import multi_agent_tax_system as mats  # type: ignore
except Exception:  # pragma: no cover - fallback 給目前 root 有 multi_agent_tax_system.py 的情況
    import multi_agent_tax_system as mats  # type: ignore

try:
    from agents.integrations.fintax_api import (  # type: ignore
        get_latest_report,
        push_latest_report,
    )
except Exception:  # pragma: no cover - fallback 給 integrations/fintax_api.py
    from integrations.fintax_api import get_latest_report, push_latest_report  # type: ignore


logger = logging.getLogger("tax.api")

# 若 multi_agent_tax_system 已經有做好 logging 設定，這裡只確保有 basicConfig
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ----------------------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------------------


class RunRequest(BaseModel):
    """
    /run 的 request body。

    工研院平台會直接以這個格式呼叫，字段對應如下：

        tool_name   ：指定稅務工具（例如 "income_tax"）
        user_params ：使用者情境欄位（是否已婚、所得、扣除額等）
        constraints ：可選，變數約束（例如 {"total_tax": {"<=": 80000}})
        free_vars   ：可選，允許最佳化調整的欄位名稱陣列
        op          ：可選，要做 "minimize" / "maximize"
        budget_tax  ：可選，稅額上限或預算
        raw_query   ：可選，原始自然語言敘述（方便後端紀錄 / debug）

    其實 CallerAgent / ConstraintAgent 若已經從對話流程組好 payload，
    也可以直接丟原始 dict 進來，只要欄位名稱對得上即可。
    """

    tool_name: str = Field(..., description="稅務工具名稱，例如 'income_tax'")
    user_params: Dict[str, Any] = Field(
        default_factory=dict, description="使用者輸入的情境欄位"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        default=None, description="變數約束條件（可選）"
    )
    free_vars: Optional[List[str]] = Field(
        default=None, description="允許最佳化調整的欄位名稱陣列（可選）"
    )
    op: Optional[str] = Field(
        default=None,
        description="最佳化方向：'minimize' / 'maximize'，若為 None 交由後端推斷",
    )
    budget_tax: Optional[float] = Field(
        default=None, description="稅額預算 / 上限（可選）"
    )
    raw_query: Optional[str] = Field(
        default=None, description="原始自然語言敘述（可選）"
    )

    class Config:
        extra = "allow"  # 容許平台附加其他欄位，不會被丟掉

    @root_validator
    def _normalize_op(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        op = values.get("op")
        if isinstance(op, str):
            op_norm = op.strip().lower()
            if op_norm in {"minimize", "max", "maximize", "min"}:
                if op_norm.startswith("min"):
                    values["op"] = "minimize"
                elif op_norm.startswith("max"):
                    values["op"] = "maximize"
                else:
                    values["op"] = None
            else:
                # 不合法就丟掉，交給後端自己推斷
                values["op"] = None
        return values


class ExportRequest(BaseModel):
    """
    /export 的 request body。

        format      ："md" / "json" / "both"（預設 "both"）
        push_to_fin ：是否要順便把報告推送到 FIN （預設 False）
        title       ：可選，覆寫報告標題
    """

    format: str = Field(
        default="both",
        description='匯出格式："md" / "json" / "both"',
    )
    push_to_fin: bool = Field(
        default=False, description="是否順便推送到 FIN 後端"
    )
    title: Optional[str] = Field(
        default=None, description="覆寫報告標題（可選）"
    )

    @root_validator
    def _normalize_format(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        fmt = (values.get("format") or "both").lower()
        if fmt not in {"md", "json", "both"}:
            fmt = "both"
        values["format"] = fmt
        return values


# ----------------------------------------------------------------------------
# FastAPI app 初始化
# ----------------------------------------------------------------------------

app = FastAPI(
    title="FIN Tax Multi-Agent Backend",
    description="工研院 FIN 稅務試算多代理服務的 REST API 介面",
    version="0.1.0",
)

# 若平台與此服務部署在不同 domain，可視需求調整 CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Helper：呼叫 multi_agent_tax_system 的 pipeline
# ----------------------------------------------------------------------------


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    """確保取得當前 thread 的 event loop（適用於 uvicorn / hypercorn 環境）。"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


async def _call_backend(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    統一封裝呼叫 multi_agent_tax_system 的邏輯
    """
    # 嘗試尋找可用的 pipeline 函式名稱，目前只有一個，但未來可擴充
    candidate_names = ["run_tax_pipeline"]
    backend_fn = None

    for name in candidate_names:
        fn = getattr(mats, name, None)
        if callable(fn):
            backend_fn = fn
            logger.info("Using backend function: %s.%s", mats.__name__, name)
            break

    if backend_fn is None:
        raise RuntimeError(
            "No backend pipeline function found in multi_agent_tax_system. "
            "Please implement one of: run_tax_pipeline(payload), "
            "run_multi_agent(payload), run_from_api(payload)."
        )

    # 支援 sync / async 兩種實作
    if inspect.iscoroutinefunction(backend_fn):
        return await backend_fn(payload)  # type: ignore[misc]

    # 同步函式的情況下，直接呼叫即可（通常會被 uvicorn 多 worker 分擔）
    loop = _ensure_event_loop()
    return await loop.run_in_executor(None, backend_fn, payload)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------


@app.get("/health")
async def health() -> Dict[str, Any]:
    """
    健康檢查端點：

    * 由 Docker / K8s 的 healthcheck 或平台後端定期呼叫
    * 無需輸入參數
    * 回傳 JSON 如：
        {"status":"ok","time":"2025-12-04T12:34:56.789012+00:00"}
    """
    now = datetime.now(timezone.utc).isoformat()
    return {"status": "ok", "time": now}


@app.post("/run")
async def run_tax(req: RunRequest) -> Dict[str, Any]:
    """
    平台呼叫的核心試算端點。

    Request JSON 例：

        {
          "tool_name": "income_tax",
          "user_params": { ... },
          "constraints": { ... },
          "free_vars": ["donation"],
          "op": "minimize",
          "budget_tax": 80000
        }

    Response JSON 例（成功）：

        {
          "status": "ok",
          "kpi": { ... },
          "final_params": { ... },
          "diff": { ... },
          "report_md": "## 本輪試算結果 ...",
          "...": "..."
        }
    """
    logger.info(
        "/run called: tool_name=%s, has_constraints=%s, has_free_vars=%s",
        req.tool_name,
        req.constraints is not None,
        req.free_vars is not None,
    )

    payload: Dict[str, Any] = req.dict()

    try:
        result = await _call_backend(payload)
    except Exception as e:
        logger.exception("Error while running tax pipeline: %s", e)
        # 對平台統一回傳 status:error 的 JSON，而不是直接 500
        return {
            "status": "error",
            "message": str(e),
            "tool_name": req.tool_name,
        }

    # 確保有 status 欄位，若後端沒給就補 "ok"
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail="Backend pipeline returned non-dict result.",
        )

    if "status" not in result:
        result["status"] = "ok"

    # 若沒有 report_md，仍然回傳，但前端就不要渲染報告
    if "report_md" not in result:
        result.setdefault("report_md", "")

    return result


@app.post("/export")
async def export_report(req: ExportRequest) -> Dict[str, Any]:
    """
    匯出最新一份報告，並可選擇是否推送到 FIN 後端。

    Request JSON 例：

        {
          "format": "both",          // "md" / "json" / "both"
          "push_to_fin": true,       // 是否要順便丟給 FIN
          "title": "自訂報告標題"     // 可選
        }

    Response JSON 例：

        {
          "status": "ok",
          "title": "綜合所得稅-2024-minimize-20251204-123456",
          "md": "## 本輪試算結果 ...",
          "json": { ... },
          "delivery": {
            "pushed": true,
            "url": "https://fin-backend.example.com/report/generated/fin-tax",
            "status": "ok",
            "status_code": 200
          }
        }
    """
    # 1) 先從 Memory 拿最後一份報告
    bundle = get_latest_report()
    md = (bundle.get("md") or "").strip()
    js = bundle.get("json") if isinstance(bundle.get("json"), dict) else None

    if not md and not js:
        raise HTTPException(
            status_code=404,
            detail="No report available. Please run /run at least once.",
        )

    title = req.title or bundle.get("title") or ""

    response: Dict[str, Any] = {
        "status": "ok",
        "title": title,
    }

    # 2) 依 format 回傳對應內容
    if req.format in {"md", "both"}:
        response["md"] = md
    if req.format in {"json", "both"}:
        response["json"] = js

    # 3) 視需要推送到 FIN 後端
    if req.push_to_fin:
        try:
            delivery_info = await push_latest_report(title=title or None)
        except Exception as e:
            logger.exception("Error while pushing report to FIN backend: %s", e)
            delivery_info = {
                "status": "exception",
                "error": str(e),
                "pushed": False,
                "url": None,
            }

        # 整理成平台看得懂的 delivery 區塊
        pushed_status = delivery_info.get("status")
        pushed_ok = pushed_status in {"ok", "skipped"}  # skipped = 未設定 FIN_BACKEND_BASEURL

        response["delivery"] = {
            "pushed": pushed_ok,
            "status": pushed_status,
            "status_code": delivery_info.get("status_code"),
            "url": delivery_info.get("url"),
        }

    return response


# ----------------------------------------------------------------------------
# 直接啟動（for docker / local 開發）
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # 本地開發可直接：
    #   python app_fastapi.py
    #
    # 也可以在 docker-compose 指定：
    #   command: uvicorn app_fastapi:app --host 0.0.0.0 --port 32770
    import uvicorn

    port = int(os.getenv("PORT", "32770"))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, reload=False)
