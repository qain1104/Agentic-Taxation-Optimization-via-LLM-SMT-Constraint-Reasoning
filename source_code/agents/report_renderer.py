# report_renderer.py

from typing import Dict, Any, List, Optional

# ------------------------ 基礎格式工具 ------------------------

def _fmt_money(x):
    try:
        n = int(round(float(x)))
    except Exception:
        return "-"
    return f"{n:,}"

def _trend(delta: Optional[float]) -> str:
    if delta is None:
        return "—"
    try:
        d = int(round(float(delta)))
    except Exception:
        return "—"
    if d > 0:  return f"↑ {abs(d):,}"
    if d < 0:  return f"↓ {abs(d):,}"
    return "—"

def _first_exist(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

# ------------------------ 欄位標籤 & 欄位白名單 ------------------------

_FIELD_LABEL = {
    # 共通
    "tax": "稅額",
    "quantity": "數量",
    "assessed_price": "課稅價格",
    "price": "單價",
    "mode": "稅式",
    "category": "類別",
    "subcategory": "子類",
    "output_tax_val": "應稅銷售額",
    "input_tax_val": "進項金額",
    "sales_val": "銷售額",
    "tp": "成交金額",
    "ep": "履約價格",
    "sc": "履約股數",
    "ca": "契約金額",
    "pa": "權利金",
    "alcohol_content": "酒精度",
    "tax_item": "稅目",
}

_ALLOWED_FIELDS = {
    # 逐筆「可顯示」欄位的候選清單（會按可得性篩掉不存在的）
    "cargo": ["mode", "quantity", "assessed_price"],
    "special_goods": ["price", "quantity"],
    "vat": ["output_tax_val", "input_tax_val"],
    "nvat": ["category", "sales_val"],
    "securities": ["tax_item", "tp", "ep", "sc"],
    "futures": ["tax_item", "ca", "pa"],
    "tobacco_alcohol": ["category", "subcategory", "quantity", "alcohol_content"],
}

# ------------------------ 逐筆名稱擷取 ------------------------

def _row_name(kind: str, idx: int, bi: Dict, oi: Dict, input_rows: Optional[List[Dict]]) -> str:
    # 1) 先從結果物件抓（優先 optimized，再 baseline）
    for src in (oi or {}, bi or {}):
        for k in ("main_name", "sub_name", "name", "item", "model"):
            v = src.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    # 2) 依稅種用原始 rows 補：品名或分類組合
    r = None
    if isinstance(input_rows, list) and 0 <= idx < len(input_rows):
        r = input_rows[idx]

    if r:
        if kind in ("cargo", "special_goods"):
            for k in ("main_name", "sub_name", "name", "item", "model"):
                v = r.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        if kind in ("tobacco_alcohol",):
            cat = r.get("category"); sub = r.get("subcategory")
            if cat:
                return f"{cat}（{sub}）" if sub is not None else str(cat)
        if kind in ("securities", "futures"):
            ti = r.get("tax_item")
            if ti:
                return str(ti)

    # 3) 都沒有就顯示「第i筆」
    return f"第{idx+1}筆"

# ------------------------ 逐筆欄位值取得 ------------------------

def _get_field_value(field: str, bi: Dict, oi: Dict, row: Optional[Dict]) -> (Any, Any):
    """回傳 (base_val, opt_val)。若缺就嘗試用原始 row 補 baseline。"""
    b = bi.get(field) if isinstance(bi, dict) else None
    o = oi.get(field) if isinstance(oi, dict) else None
    if b is None and isinstance(row, dict):
        b = row.get(field)
    return b, o

# ------------------------ 表格渲染（逐筆明細） ------------------------

def _render_items_table(kind: str,
                        baseline_items: Optional[List[Dict]],
                        optimized_items: Optional[List[Dict]],
                        input_rows: Optional[List[Dict]]) -> str:
    bis = baseline_items or []
    ois = optimized_items or []
    n = max(len(bis), len(ois))
    if n == 0:
        return "（工具未提供逐筆明細）"

    # 動態決定要顯示哪些欄位（存在於任一來源即可）
    allow = _ALLOWED_FIELDS.get(kind, [])
    cols = []
    for f in allow:
        exists = any((isinstance(x, dict) and f in x) for x in bis+ois) \
                 or any((isinstance(r, dict) and f in r) for r in (input_rows or []))
        if exists:
            cols.append(f)

    # 表頭
    headers = ["項目", "稅額（基準→最佳化 / 變化）"] + [_FIELD_LABEL.get(c, c) + "（基準→最佳化）" for c in cols]
    out = ["|" + "|".join(headers) + "|",
           "|" + "|".join(["---"]*len(headers)) + "|"]

    # 列資料
    for i in range(n):
        bi = bis[i] if i < len(bis) else {}
        oi = ois[i] if i < len(ois) else {}
        row_src = input_rows[i] if (isinstance(input_rows, list) and i < len(input_rows)) else None

        name = _row_name(kind, i, bi, oi, input_rows)

        bt = bi.get("tax"); ot = oi.get("tax")
        delta = None
        if bt is not None and ot is not None:
            delta = (ot - bt)
        tax_cell = f"{_fmt_money(bt)} → {_fmt_money(ot)} / {_trend(delta)}" if (bt is not None or ot is not None) else "—"

        cells = [name, tax_cell]
        for c in cols:
            b, o = _get_field_value(c, bi, oi, row_src)
            cell = f"{_fmt_money(b)} → {_fmt_money(o)}" if (b is not None or o is not None) else "—"
            # 非純數值欄（例如 category / mode / tax_item）不加千分位箭頭
            if c in ("category", "subcategory", "mode", "tax_item"):
                b_txt = str(b) if b is not None else "—"
                o_txt = str(o) if o is not None else "—"
                cell = f"{b_txt} → {o_txt}"
            cells.append(cell)

        out.append("|" + "|".join(cells) + "|")

    return "\n".join(out)

# ------------------------ 參數調整（param_diff / diff） ------------------------

def _render_param_diff(param_diff: Optional[Dict[str, Dict]], input_rows: Optional[List[Dict]], kind: str) -> str:
    if not isinstance(param_diff, dict) or not param_diff:
        return "（參數無調整或工具未回傳差異）"
    lines = []
    for k, v in param_diff.items():
        # k 可能像 "row0.quantity" / "row2.sales_val"
        try:
            row_idx = int(k.split(".")[0].replace("row", ""))
        except Exception:
            row_idx = None
        name = _row_name(kind, row_idx if row_idx is not None else 0, {}, {}, input_rows) if row_idx is not None else k
        field = k.split(".")[1] if "." in k else k
        label = _FIELD_LABEL.get(field, field)
        orig = v.get("original"); opt = v.get("optimized")
        delta = None
        if isinstance(orig, (int,float)) and isinstance(opt, (int,float)):
            delta = opt - orig
        lines.append(f"- **{name}／{label}**：{_fmt_money(orig)} → {_fmt_money(opt)}（{_trend(delta)}）")
    return "\n".join(lines)

def _render_diff(diff: Optional[Dict], input_rows: Optional[List[Dict]], kind: str) -> str:
    if not isinstance(diff, dict) or not diff:
        return "（參數無調整或工具未回傳差異）"
    lines = []
    for k, v in diff.items():
        # diff 的 key 可能就是 row index
        try:
            idx = int(k)
        except Exception:
            idx = None
        name = _row_name(kind, idx if idx is not None else 0, {}, {}, input_rows) if idx is not None else str(k)
        orig, opt = v.get("original"), v.get("optimized")
        delta = None
        if isinstance(orig, (int,float)) and isinstance(opt, (int,float)):
            delta = opt - orig
        lines.append(f"- **{name}**：稅額 { _fmt_money(orig) } → { _fmt_money(opt) }（{ _trend(delta) }）")
    return "\n".join(lines)

# ------------------------ 約束摘要 ------------------------

def _render_constraints(constraints: Optional[Dict[str, Dict]]) -> str:
    if not isinstance(constraints, dict) or not constraints:
        return "（未設定額外約束）"
    lines = []
    for k, ops in constraints.items():
        # ops 可能是 {"<=": 30} 或 {"=": "row0.x * 1.2"}
        parts = []
        for op, rhs in ops.items():
            parts.append(f"{op} {rhs}")
        lines.append(f"- `{k}` " + "，".join(parts))
    return "\n".join(lines)

# ------------------------ 主渲染 API ------------------------

def render_tax_report(*,
                      kind: str,                    # "cargo" / "special_goods" / "vat" / "nvat" / "securities" / "futures" / "tobacco_alcohol"
                      result: Dict[str, Any],       # 工具回傳的 dict
                      input_params: Optional[Dict] = None,   # 原始輸入（至少包含 rows）
                      title: Optional[str] = None) -> str:
    """
    回傳 Markdown 字串，一致格式呈現各稅法多筆明細。
    """
    rows = (input_params or {}).get("rows")
    baseline = result.get("baseline")

    # 不同工具對「最佳化後稅額」的命名不一致，這裡做 Path 容錯
    optimized_tax = _first_exist(result, ["optimized", "optimized_total_tax", "optimized_tax"])
    optimized_qty = _first_exist(result, ["optimized_total_qty", "optimized_total_sales"])
    budget_or_target = _first_exist(result, ["budget_tax", "target_tax"])
    status = _first_exist(result, ["status", "baseline_status"])

    # ---- Summary & KPI ----
    title = title or {
        "cargo": "貨物稅（多筆）",
        "special_goods": "特種貨物稅（多筆）",
        "vat": "加值型營業稅",
        "nvat": "非加值型營業稅",
        "securities": "證券交易稅",
        "futures": "期貨交易稅",
        "tobacco_alcohol": "菸酒稅",
    }.get(kind, f"{kind} 報告")

    kpi_lines = []
    if baseline is not None:
        kpi_lines.append(f"- **輸入稅額**：{_fmt_money(baseline)}")
    if optimized_tax is not None:
        kpi_lines.append(f"- **最佳解稅額**：{_fmt_money(optimized_tax)}")
        if baseline is not None:
            kpi_lines.append(f"- **稅額變化**：{_trend((optimized_tax - baseline) if isinstance(optimized_tax,(int,float)) else None)}")
    if optimized_qty is not None:
        kpi_lines.append(f"- **最佳化後總量**：{_fmt_money(optimized_qty)}")
    if budget_or_target is not None:
        kpi_lines.append(f"- **稅額上限/目標**：{_fmt_money(budget_or_target)}")
    if status:
        kpi_lines.append(f"- **狀態**：{status}")

    # ---- Items ----
    baseline_items  = result.get("baseline_items") or result.get("baseline_with_constraints_items")
    optimized_items = result.get("optimized_items")

    items_block = _render_items_table(kind, baseline_items, optimized_items, rows)

    # ---- Diff / Param Diff ----
    diff_block       = _render_diff(result.get("diff"), rows, kind)
    param_diff_block = _render_param_diff(result.get("param_diff"), rows, kind)

    # ---- Constraints ----
    constraints_block = _render_constraints(result.get("constraints"))

    # ---- Final report ----
    report = []
    report.append(f"# {title}")
    if kpi_lines:
        report.append("## 重點數據")
        report.append("\n".join(kpi_lines))
    report.append("## 逐筆明細")
    report.append(items_block)
    if optimized_tax is not None or optimized_qty is not None:
        report.append("## 參數調整")
        report.append(param_diff_block if result.get("param_diff") else diff_block)
    report.append("## 套用的約束")
    report.append(constraints_block)
    report.append("## 風險與合規備註")
    report.append("- 本報告為模型推導之**估算**，實際稅負仍以主管機關規定與申報資料為準。")
    report.append("- 涉及扣除認列、薪資結構等請務必依據法規與憑證（避免規避稅捐風險）。")

    return "\n\n".join(report)

# tool_name -> 你 renderer 的 kind 映射
_KIND_BY_TOOL = {
    "cargo_tax_minimize": "cargo",
    "special_goods_tax": "special_goods",
    "vat_tax": "vat",
    "nvat_tax": "nvat",
    "securities_tx_tax": "securities",
    "futures_tx_tax": "futures",
    "ta_tax": "tobacco_alcohol",
    # 其他沒有多筆明細的工具就不要列，會自動回退
}

def render_report(*, result: Dict[str, Any], payload: Dict[str, Any], tool_name: str, field_labels: Dict[str, Any] | None = None) -> Optional[str]:
    """
    提供給 ReasoningAgent 外掛用的薄適配器：
    - 僅在 rows 是多筆時輸出 Markdown；否則回 None 讓系統走原本模板。
    - 補上 payload 裡的 constraints（若工具結果未帶）。
    """
    kind = _KIND_BY_TOOL.get(tool_name)
    if not kind:
        return None  # 非你支援的稅種 → 回退

    up = (payload or {}).get("user_params") or {}
    rows = up.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return None  # 單筆不走外部 renderer，讓通用模板處理

    # 確保 constraints 至少能從 payload 呈現
    res = dict(result or {})
    if "constraints" not in res:
        cons = up.get("constraints")
        if cons:
            res["constraints"] = cons

    # 交給你現成的主渲染器
    return render_tax_report(kind=kind, result=res, input_params=up, title=None)