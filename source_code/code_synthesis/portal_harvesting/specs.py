from __future__ import annotations

from typing import Any, Dict, List

PORTAL_SCHEMA_CONFIG: Dict[str, Dict[str, Any]] = {
    "business_income": {
        "title": "營利事業所得稅",
        "source_url": "https://www.etax.nat.gov.tw/etwmain/etw158w/35",
        "render_style": "linear_equation",
        "field_aliases": {
            "營業收入總額": "OperatingRevenueTotal_input",
            "銷貨退回": "SalesReturn_input",
            "銷貨折讓": "SalesAllowance_input",
            "營業成本": "OperatingCost_input",
            "營業費用及損失總額": "OperatingExpensesLosses_input",
            "非營業收入總額": "NonOperatingRevenueTotal_input",
            "非營業損失及費用總額": "NonOperatingLossExpenses_input",
            "前10年核定虧損本年度扣除額(藍色及簽證申報適用)": "Prev10LossDeduction_input",
            "合於獎勵規定之免稅所得(附計算表)": "TaxIncentiveExempt_input",
            "停徵之證券、期貨交易所得(損失)": "ExemptSecuritiesIncome_input",
            "免徵所得稅之出售土地增益(損失)": "ExemptLandIncome_input",
            "交易符合所得稅法第4條之4規定房屋、土地、房屋使用權、預售屋及其坐落基地暨股份或出資額之所得(損失)": "Article4_4HouseLandGain_input",
            "營業期間滿一年者": "is_full_year",
            "月份數": "m_partial",
        },
        "formula_hints": [
            "營業收入總額 - 銷貨退回 - 銷貨折讓 = 營業收入淨額",
            "營業收入淨額 - 營業成本 = 營業毛利",
            "營業毛利 - 營業費用及損失總額 = 營業淨利",
            "營業淨利 + 非營業收入總額 - 非營業損失及費用總額 = 全年所得額",
            "全年所得額 - 前10年核定虧損本年度扣除額(藍色及簽證申報適用) - 合於獎勵規定之免稅所得(附計算表) - 停徵之證券、期貨交易所得(損失) - 免徵所得稅之出售土地增益(損失) - 交易符合所得稅法第4條之4規定房屋、土地、房屋使用權、預售屋及其坐落基地暨股份或出資額之所得(損失) = 課稅所得額",
        ],
        "note_hints": ["營業期間滿1年者", "營業期間滿一年者", "分開計算稅額合併報繳", "附註"],
        "branch_hints": ["營業期間滿1年者"],
        "output_hints": ["應納稅額"],
        "format_guidance": {
            "summary": "整理成公式導向模板。每個公式之後列出對應輸入欄位為 '_'，稅率門檻未知處保留 '?'，最終回傳目標保留 '??'。",
        },
    },
    "business_tax": {
        "title": "營業稅",
        "source_url": "",
        "render_style": "branch_template",
        "field_aliases": {
            "銷項稅額": "output_tax_val",
            "進項稅額": "input_tax_val",
            "銷售額": "sales_val",
            "類別": "category",
        },
        "formula_hints": ["銷項稅額 - 進項稅額 = 應納營業稅額", "銷售額 × 稅率 = 應納稅額"],
        "note_hints": ["加值型", "非加值型", "稅率", "應納營業稅額"],
        "branch_hints": ["加值型", "非加值型", "若為"],
        "output_hints": ["應納營業稅額"],
        "format_guidance": {
            "summary": "整理成分支模板。先分加值型與非加值型，再在各分支下列公式、必要輸入欄位與稅率/例外說明。未知稅率與門檻保留 '?'，最終目標保留 '??'。",
            "branch_order": ["加值型", "非加值型"],
        },
    },
    "security_futures": {
        "title": "證券交易稅 / 期貨交易稅",
        "source_url": "",
        "render_style": "dispatch_template",
        "field_aliases": {
            "稅目": "tax_item",
            "證券成交價格": "tp",
            "每股發行價格": "ep",
            "股數": "sc",
            "契約金額": "ca",
            "口數": "pa",
        },
        "formula_hints": ["證券交易稅 = 成交價格 × 稅率", "期貨交易稅 = 契約金額 × 口數 × 稅率"],
        "note_hints": ["證券交易稅", "期貨交易稅", "股票", "公司債", "期貨"],
        "branch_hints": ["證券", "期貨", "若為"],
        "output_hints": ["應納稅額"],
        "format_guidance": {
            "summary": "整理成同一個 compute_tax 需處理兩類稅目的 dispatch 模板。先列 tax_item，再分證券與期貨各自所需欄位與稅率說明。未知稅率保留 '?'，最終目標保留 '??'。",
            "branch_order": ["證券交易稅", "期貨交易稅"],
        },
    },
}


def get_schema_spec(schema: str) -> Dict[str, Any]:
    if schema not in PORTAL_SCHEMA_CONFIG:
        raise KeyError(f"Unsupported portal schema: {schema}")
    return PORTAL_SCHEMA_CONFIG[schema]


def list_supported_schemas() -> List[str]:
    return sorted(PORTAL_SCHEMA_CONFIG.keys())
