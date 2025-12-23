# tools_registry.py

from importlib import import_module
from typing import Callable, List, Dict, TypedDict, Union, Optional
from tax_calculators.tobacco_alcohol_tax import FIELD_LABELS as TA_FIELD_LABELS
TA_FIELD_LABELS_REG = {**TA_FIELD_LABELS, "budget_tax": "稅額上限（元）"}

class ToolMeta(TypedDict):
    # 必填
    name: str
    description: str
    module: str
    entry_func: Union[str, Dict[str, str]]  # ← 支援 dict
    keywords: List[str]
    required_fields: List[str]
    optimizable_fields: List[str]
    # 選填
    constraint_fields: List[str]
    field_labels: Dict[str, str]
    budget_field: str            # ← 最大化分支的上限欄位名稱（預設 budget_tax）
    nl_parser: str               # ← 自然語言解析器模組路徑，例如 "agents.cargo_nl_parser"

TOOLS: List[ToolMeta] = [
    # 綜合所得稅計算工具
    {
        "name": "income_tax",
        "description": "綜合所得稅",
        "module": "tax_calculators.income_tax",
        "entry_func": "calculate_comprehensive_income_tax",
        "keywords": ["綜合所得稅", "income", "個人稅"],
        "required_fields": [
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
            "property_loss_deduction",
            "education_fee",
            "preschool_count",
            "long_term_care_count",
            "rent_deduction",
            "education_count",
            "disability_count",
            "itemized_deduction",
        ],
        "optimizable_fields": [
            "salary_self",
            "salary_spouse",
            "salary_dep",
            "interest_income",
            "stock_dividend",
            "house_transaction_gain",
            "other_income",
            "cnt_under_70",
            "cnt_over_70",
            "property_loss_deduction",
            "education_fee",
            "preschool_count",
            "long_term_care_count",
            "rent_deduction",
            "education_count",
            "disability_count",
            "itemized_deduction",
        ],
        "constraint_fields": ["free_vars", "constraints"],
        "field_labels": {
            "is_married": "您是否已婚（是/否）",
            "salary_self": "本人薪資所得（元）",
            "salary_spouse": "配偶薪資所得（元）",
            "salary_dep": "扶養親屬薪資所得（元）",
            "interest_income": "利息所得（元）",
            "stock_dividend": "股票股利所得（元）",
            "house_transaction_gain": "房屋交易所得（元）",
            "other_income": "其他所得（元）",
            "cnt_under_70": "70 歲以下受扶養人數",
            "cnt_over_70": "70 歲以上受扶養人數",
            "property_loss_deduction": "財產損失扣除額（元）",
            "education_fee": "實際教育費用（元）",
            "preschool_count": "幼兒人數",
            "long_term_care_count": "長照適用人數",
            "rent_deduction": "房屋租金支出（元）",
            "education_count": "教育扣除適用人數",
            "disability_count": "身障者人數",
            "itemized_deduction": "列舉扣除額（元）",
            "dividend_scheme_best": "股利課稅選擇",
            "tax_combined_progressive_with_dividend": "合併計稅：稅額-累進，含股利",
            "tax_combined_dividend_credit_raw": "合併計稅：股利抵減額 股利×8.5%",
            "tax_combined_dividend_credit_capped": "合併計稅：股利抵減額 套上限80,000後",
            "tax_combined_net": "合併計稅：抵減後差額",
            "tax_combined_tax_due": "合併計稅：實際應繳",
            "tax_combined_refund": "合併計稅：實際可退",
            "tax_separate_progressive_without_dividend": "分開計稅：其他所得稅額（累進，不含股利）",
            "tax_separate_dividend_tax_28": "分開計稅：股利稅額 股利×28%",
            "tax_separate_net": "分開計稅：合計應納",
            "tax_separate_tax_due": "分開計稅：實際應繳",
            "tax_separate_refund": "分開計稅：實際可退（通常為0）",
        },
        "alias": {
            "is_married": ["是否已婚", "已婚", "婚姻狀況"],
            "salary_self": ["本人薪資", "我(的)?薪資", "自己薪資", "本人所得", "我(的)?所得"],
            "salary_spouse": ["配偶薪資", "太太薪資", "老婆薪資", "先生薪資", "老公薪資", "另一半薪資", "配偶所得"],
            "salary_dep": ["扶養親屬薪資", "受扶養薪資", "小孩薪資", "子女薪資"],
            "interest_income": ["利息所得", "存款利息"],
            "stock_dividend": ["股票股利所得", "股利所得", "配股"],
            "house_transaction_gain": ["房屋交易所得", "賣房所得", "房地交易所得", "不動產交易所得"],
            "other_income": ["其他所得", "其他收入"],
            "cnt_under_70": ["70歲以下受扶養人數", "未滿70受扶養人數", "受扶養人數(未滿70)"],
            "cnt_over_70": ["70歲以上受扶養人數", "年滿70受扶養人數", "受扶養人數(70以上)"],
            "property_loss_deduction": ["財產損失扣除額", "財損扣除", "災損扣除"],
            "education_fee": ["實際教育費用", "教育費用"],
            "preschool_count": ["幼兒人數", "學齡前人數"],
            "long_term_care_count": ["長照適用人數", "長照人數"],
            "rent_deduction": ["房屋租金支出", "租金扣除額", "房租支出"],
            "education_count": ["教育扣除適用人數", "教育扣除人數"],
            "disability_count": ["身障者人數", "殘障人數"],
            "itemized_deduction": ["列舉扣除", "列舉扣除額", "列舉"]
        }
    },
    # 外僑綜合所得稅計算工具
    {
        "name": "foreigner_income_tax",
        "description": "外僑綜合所得稅",
        "module": "tax_calculators.foreigner_income_tax",
        "entry_func": "calculate_foreigner_income_tax",
        "keywords": ["外僑綜所稅", "foreigner tax", "non‑resident"],
        "required_fields": [
            "days_of_stay",
            "is_departure",
            "is_married",
            "salary_self",
            "salary_spouse",
            "salary_dep",
            "interest_income",
            "interest_spouse",
            "interest_dep",
            "other_income",
            "other_income_spouse",
            "other_income_dep",
            "cnt_under_70",
            "cnt_over_70",
            "use_itemized",
            "itemized_deduction",
            "property_loss_deduction",
            "disability_count",
            "education_count",
            "education_fee",
            "preschool_count",
            "long_term_care_count",
            "rent_deduction",
        ],
        "optimizable_fields": [
            "days_of_stay",
            "salary_self",
            "salary_spouse",
            "salary_dep",
            "interest_income",
            "interest_spouse",
            "interest_dep",
            "other_income",
            "other_income_spouse",
            "other_income_dep",
            "cnt_under_70",
            "cnt_over_70",
            "use_itemized",
            "itemized_deduction",
            "property_loss_deduction",
            "disability_count",
            "education_count",
            "education_fee",
            "preschool_count",
            "long_term_care_count",
            "rent_deduction",
        ],
        "constraint_fields": ["free_vars", "constraints"],
        "field_labels": {          # 中文提示字典 —— 只在這裡維護一次
            "days_of_stay": "當年度在台居留天數",
            "is_departure": "是否為當年度離境 (是/否)",
            "is_married": "您是否已婚（是/否）",
            "salary_self": "本人薪資所得（元）",
            "salary_spouse": "配偶薪資所得（元）",
            "salary_dep": "扶養親屬薪資所得（元）",
            "interest_income": "利息所得（元）",
            "interest_spouse": "配偶利息所得（元）",
            "interest_dep": "扶養親屬利息所得（元）",
            "other_income": "其他所得（元）",
            "other_income_spouse": "配偶其他所得（元）",
            "other_income_dep": "扶養親屬其他所得（元）",
            "cnt_under_70": "70 歲以下受扶養人數",
            "cnt_over_70": "70 歲以上受扶養人數",
            "use_itemized": "是否採用列舉扣除 (是/否)",
            "itemized_deduction": "列舉扣除額（元）",
            "property_loss_deduction": "財產損失扣除額（元）",
            "disability_count": "身障者人數",
            "education_count": "教育扣除適用人數",
            "education_fee": "實際教育費用（元）",
            "preschool_count": "幼兒人數",
            "long_term_care_count": "長照適用人數",
            "rent_deduction": "房屋租金支出（元）",
        },
        "alias": {
            "days_of_stay": ["在台居留天數", "停留天數", "居留天數"],
            "is_departure": ["是否當年度離境", "是否離境", "離境與否"],
            "is_married": ["是否已婚", "已婚", "婚姻狀況"],
            "salary_self": ["本人薪資", "我(的)?薪資", "自己薪資", "本人所得"],
            "salary_spouse": ["配偶薪資", "太太薪資", "先生薪資", "老公薪資", "老婆薪資"],
            "salary_dep": ["扶養親屬薪資", "受扶養薪資"],
            "interest_income": ["利息所得", "存款利息"],
            "interest_spouse": ["配偶利息所得", "配偶存款利息"],
            "interest_dep": ["扶養親屬利息所得"],
            "other_income": ["其他所得", "其他收入"],
            "other_income_spouse": ["配偶其他所得", "配偶其他收入"],
            "other_income_dep": ["扶養親屬其他所得"],
            "cnt_under_70": ["70歲以下受扶養人數"],
            "cnt_over_70": ["70歲以上受扶養人數"],
            "use_itemized": ["是否採列舉扣除", "採用列舉扣除"],
            "itemized_deduction": ["列舉扣除額", "列舉扣除"],
            "property_loss_deduction": ["財產損失扣除額", "災損扣除"],
            "disability_count": ["身障者人數"],
            "education_count": ["教育扣除適用人數"],
            "education_fee": ["實際教育費用", "教育費用"],
            "preschool_count": ["幼兒人數", "學齡前人數"],
            "long_term_care_count": ["長照適用人數", "長照人數"],
            "rent_deduction": ["房屋租金支出", "租金扣除額"]
        }
    },
    # 營利事業所得稅計算工具
    {
        "name": "business_income_tax",
        "description": "營利事業所得稅",
        "module": "tax_calculators.business_income_tax",   # ← 依實際路徑調整
        "entry_func": "calculate_business_income_tax",  # ← 公開函式
        "keywords": ["營利事業所得稅", "corporate income tax", "business tax"],
        "required_fields": [
            "OperatingRevenueTotal",
            "SalesReturn",
            "SalesAllowance",
            "OperatingCost",
            "OperatingExpensesLosses",
            "NonOperatingRevenueTotal",
            "NonOperatingLossExpenses",
            "Prev10LossDeduction",
            "TaxIncentiveExempt",
            "ExemptSecuritiesIncome",
            "ExemptLandIncome",
            "Article4_4HouseLandGain",
            "is_full_year",
            "m_partial",
        ],
        "optimizable_fields": [
            "OperatingRevenueTotal",
            "SalesReturn",
            "SalesAllowance",
            "OperatingCost",
            "OperatingExpensesLosses",
            "NonOperatingRevenueTotal",
            "NonOperatingLossExpenses",
            "Prev10LossDeduction",
            "TaxIncentiveExempt",
            "ExemptSecuritiesIncome",
            "ExemptLandIncome",
            "Article4_4HouseLandGain",
        ],
        "constraint_fields": ["free_vars", "constraints"],
        "field_labels": {
            "OperatingRevenueTotal":   "營業收入總額（元）",
            "SalesReturn":             "銷貨退回（元）",
            "SalesAllowance":          "銷貨折讓（元）",
            "OperatingCost":           "營業成本（元）",
            "OperatingExpensesLosses": "營業費用及損失（元）",
            "NonOperatingRevenueTotal":"營業外收入總額（元）",
            "NonOperatingLossExpenses":"營業外損失及費用（元）",
            "Prev10LossDeduction":     "可扣抵之前 10 年虧損（元）",
            "TaxIncentiveExempt":      "租稅優惠免稅額（元）",
            "ExemptSecuritiesIncome":  "免稅證券所得（元）",
            "ExemptLandIncome":        "免稅土地所得（元）",
            "Article4_4HouseLandGain": "4‑4 條房地交易所得（元）",
            "is_full_year":            "是否全年營業（是/否）",
            "m_partial":               "本年度營業月數（1‑12）",
        },
        "alias": {
            "OperatingRevenueTotal":   ["營業收入總額", "營業收入", "營業收入合計"],
            "SalesReturn":             ["銷貨退回"],
            "SalesAllowance":          ["銷貨折讓"],
            "OperatingCost":           ["營業成本", "成本"],
            "OperatingExpensesLosses": ["營業費用及損失", "營業費用", "費用及損失"],
            "NonOperatingRevenueTotal":["營業外收入", "業外收入"],
            "NonOperatingLossExpenses":["營業外損失及費用", "營業外損失"],
            "Prev10LossDeduction":     ["可扣抵之前10年虧損", "前十年虧損扣抵", "10年虧損扣抵"],
            "TaxIncentiveExempt":      ["租稅優惠免稅額", "租稅優惠"],
            "ExemptSecuritiesIncome":  ["免稅證券所得"],
            "ExemptLandIncome":        ["免稅土地所得"],
            "Article4_4HouseLandGain": ["4-4條房地交易所得", "房地合一4-4所得"],
            "is_full_year":            ["是否全年營業", "全年營業與否"],
            "m_partial":               ["本年度營業月數", "營業月數"]
        }
    },
    # 貨物稅
    {
        "name": "cargo_tax",
        "description": "貨物稅",
        "module": "tax_calculators.cargo_tax",
        "entry_func": { "minimize": "minimize_cargo_tax", "maximize": "maximize_cargo_qty" },
        "keywords": ["貨物稅", "excise", "goods tax", "出廠數量", "完稅價格"],
        "required_fields": [
            "cement_white.quantity", "cement_portland_I.quantity", "cement_blast_furnace.quantity", "cement_other.quantity",
            "oil_gasoline.quantity", "oil_diesel.quantity", "oil_kerosene.quantity", "oil_jetfuel.quantity",
            "oil_fueloil.quantity", "oil_solvent.quantity", "lpg.quantity",
            "tire_bus_truck.assessed_price","tire_bus_truck.quantity",
            "tire_other.assessed_price","tire_other.quantity",
            "tire_inner_solid.assessed_price","tire_inner_solid.quantity",
            "drink_diluted_juice.assessed_price","drink_diluted_juice.quantity",
            "drink_other.assessed_price","drink_other.quantity",
            "drink_pure_juice.assessed_price","drink_pure_juice.quantity",
            "glass_plain.assessed_price","glass_plain.quantity",
            "glass_conductive_mold.assessed_price","glass_conductive_mold.quantity",
            "fridge.assessed_price","fridge.quantity",
            "tv.assessed_price","tv.quantity",
            "hvac_central.assessed_price","hvac_central.quantity",
            "hvac_non_central.assessed_price","hvac_non_central.quantity",
            "dehumidifier.assessed_price","dehumidifier.quantity",
            "vcr.assessed_price","vcr.quantity",
            "recorder.assessed_price","recorder.quantity",
            "stereo.assessed_price","stereo.quantity",
            "oven.assessed_price","oven.quantity",
            "car_le_2000cc.assessed_price","car_le_2000cc.quantity",
            "car_gt_2000cc.assessed_price","car_gt_2000cc.quantity",
            "truck_bus.assessed_price","truck_bus.quantity",
            "motorcycle.assessed_price","motorcycle.quantity"
        ],
        "optimizable_fields": [
            "cement_white.quantity", "cement_portland_I.quantity", "cement_blast_furnace.quantity", "cement_other.quantity",
            "oil_gasoline.quantity", "oil_diesel.quantity", "oil_kerosene.quantity", "oil_jetfuel.quantity",
            "oil_fueloil.quantity", "oil_solvent.quantity", "lpg.quantity",
            "tire_bus_truck.assessed_price","tire_bus_truck.quantity",
            "tire_other.assessed_price","tire_other.quantity",
            "tire_inner_solid.assessed_price","tire_inner_solid.quantity",
            "drink_diluted_juice.assessed_price","drink_diluted_juice.quantity",
            "drink_other.assessed_price","drink_other.quantity",
            "drink_pure_juice.assessed_price","drink_pure_juice.quantity",
            "glass_plain.assessed_price","glass_plain.quantity",
            "glass_conductive_mold.assessed_price","glass_conductive_mold.quantity",
            "fridge.assessed_price","fridge.quantity",
            "tv.assessed_price","tv.quantity",
            "hvac_central.assessed_price","hvac_central.quantity",
            "hvac_non_central.assessed_price","hvac_non_central.quantity",
            "dehumidifier.assessed_price","dehumidifier.quantity",
            "vcr.assessed_price","vcr.quantity",
            "recorder.assessed_price","recorder.quantity",
            "stereo.assessed_price","stereo.quantity",
            "oven.assessed_price","oven.quantity",
            "car_le_2000cc.assessed_price","car_le_2000cc.quantity",
            "car_gt_2000cc.assessed_price","car_gt_2000cc.quantity",
            "truck_bus.assessed_price","truck_bus.quantity",
            "motorcycle.assessed_price","motorcycle.quantity"
        ],
        "constraint_fields": ["free_vars", "constraints", "budget_tax"],
        "budget_field": "budget_tax",
        "nl_parser": "agents.parsers.cargo_nl_parser:nl_to_payload",
        "report_renderer": "agents.report_renderer:render_report",
        "field_labels": {
            "budget_tax": "稅額上限（元）",
            "cement_white.quantity": "白水泥- 數量",
            "cement_portland_I.quantity": "卜特蘭I型水泥- 數量",
            "cement_blast_furnace.quantity": "卜特蘭高爐水泥- 數量",
            "cement_other.quantity": "代水泥及其他- 數量",
            "oil_gasoline.quantity": "汽油- 數量",
            "oil_diesel.quantity": "柴油- 數量",
            "oil_kerosene.quantity": "煤油- 數量",
            "oil_jetfuel.quantity": "航空燃油- 數量",
            "oil_fueloil.quantity": "燃料油- 數量",
            "oil_solvent.quantity": "溶劑油- 數量",
            "lpg.quantity": "液化石油氣- 數量",
            "tire_bus_truck.assessed_price": "大客/大貨車輪胎- 每單位完稅價格",
            "tire_bus_truck.quantity": "大客/大貨車輪胎- 數量",
            "tire_other.assessed_price": "其他橡膠輪胎- 每單位完稅價格",
            "tire_other.quantity": "其他橡膠輪胎- 數量",
            "tire_inner_solid.assessed_price": "內胎/實心輪胎等- 每單位完稅價格",
            "tire_inner_solid.quantity": "內胎/實心輪胎等- 數量",
            "drink_diluted_juice.assessed_price": "稀釋天然果蔬汁- 每單位完稅價格",
            "drink_diluted_juice.quantity": "稀釋天然果蔬汁- 數量",
            "drink_other.assessed_price": "其他飲料- 每單位完稅價格",
            "drink_other.quantity": "其他飲料- 數量",
            "drink_pure_juice.assessed_price": "天然果汁類- 每單位完稅價格",
            "drink_pure_juice.quantity": "天然果汁類- 數量",
            "glass_plain.assessed_price": "一般平板玻璃- 每單位完稅價格",
            "glass_plain.quantity": "一般平板玻璃- 數量",
            "glass_conductive_mold.assessed_price": "導電/模具用玻璃- 每單位完稅價格",
            "glass_conductive_mold.quantity": "導電/模具用玻璃- 數量",
            "fridge.assessed_price": "冰箱- 每單位完稅價格",
            "fridge.quantity": "冰箱- 數量",
            "tv.assessed_price": "彩色電視機- 每單位完稅價格",
            "tv.quantity": "彩色電視機- 數量",
            "hvac_central.assessed_price": "中央空調- 每單位完稅價格",
            "hvac_central.quantity": "中央空調- 數量",
            "hvac_non_central.assessed_price": "非中央空調- 每單位完稅價格",
            "hvac_non_central.quantity": "非中央空調- 數量",
            "dehumidifier.assessed_price": "除濕機- 每單位完稅價格",
            "dehumidifier.quantity": "除濕機- 數量",
            "vcr.assessed_price": "錄影機- 每單位完稅價格",
            "vcr.quantity": "錄影機- 數量",
            "recorder.assessed_price": "錄音機- 每單位完稅價格",
            "recorder.quantity": "錄音機- 數量",
            "stereo.assessed_price": "音響組合- 每單位完稅價格",
            "stereo.quantity": "音響組合- 數量",
            "oven.assessed_price": "電烤箱- 每單位完稅價格",
            "oven.quantity": "電烤箱- 數量",
            "car_le_2000cc.assessed_price": "小客車 ≤2000cc- 每單位完稅價格",
            "car_le_2000cc.quantity": "小客車 ≤2000cc- 數量",
            "car_gt_2000cc.assessed_price": "小客車 >2000cc- 每單位完稅價格",
            "car_gt_2000cc.quantity": "小客車 >2000cc- 數量",
            "truck_bus.assessed_price": "貨車/大客車等- 每單位完稅價格",
            "truck_bus.quantity": "貨車/大客車等- 數量",
            "motorcycle.assessed_price": "機車- 每單位完稅價格",
            "motorcycle.quantity": "機車- 數量"
        },
        "alias": {
            "free_vars": ["自由變數", "可調變數", "放行變數"],
            "constraints": ["限制條件", "約束條件", "條件式"],
            "budget_tax": ["稅額上限", "目標稅額", "預算上限", "最多稅額"],
            "quantity": ["數量", "出廠數量", "台數", "噸數", "公噸", "公秉"],
            "assessed_price": ["完稅價格", "每單位完稅價格", "單價", "每台完稅價", "每單位價格", "售價"]
        }
    },
    # 營業稅-加值型
    {
        "name": "vat_tax",
        "description": "加值型營業稅",
        "module": "tax_calculators.sale_tax",
        "entry_func": "minimize_vat_tax",
        "keywords": ["加值型營業稅", "VAT", "銷項稅額", "進項稅額"],
        "required_fields": ["output_tax_amt", "input_tax_amt"],
        "optimizable_fields": ["output_tax_amt", "input_tax_amt"],
        "constraint_fields": ["free_vars", "constraints"],
        "nl_parser": "agents.parsers.vat_nl_parser_simple:nl_to_payload",
        "field_labels": {
            "output_tax_amt": "銷項課稅基礎（未乘稅，元）",
            "input_tax_amt":  "進項課稅基礎（未乘稅，元）",
            "free_vars": "可放行變數：output_tax_amt, input_tax_amt（逗號分隔）。",
            "constraints": "可用變數：output_tax_amt, input_tax_amt, out5, in5, total_vat。"
        },
        "alias": {
            "output_tax_amt": ["銷項稅額", "銷項"],
            "input_tax_amt":  ["進項稅額", "進項"],
            "free_vars": ["自由變數", "放行變數"],
            "constraints": ["限制條件", "約束條件"]
        }
    },
    # 營業稅-非加值型
    {
        "name": "nvat_tax",
        "description": "非加值型營業稅",
        "module": "tax_calculators.sale_tax",
        "entry_func": { "minimize": "minimize_nvat_tax", "maximize": "maximize_nvat_under_budget" },
        "keywords": ["非加值型營業稅", "nVAT", "小規模", "夜總會", "酒家等"],
        "required_fields": ["cat1","cat2","cat3","cat4","cat5","cat6","cat7","cat8"],
        "optimizable_fields": ["cat1","cat2","cat3","cat4","cat5","cat6","cat7","cat8"],
        "constraint_fields": ["free_vars", "constraints", "budget_tax"],
        "budget_field": "budget_tax",
        "nl_parser": "agents.parsers.nvat_nl_parser:nl_to_payload",
        "field_labels": {
            "cat1": "小規模營業人（銷售額，元）",
            "cat2": "再保費收入（銷售額，元）",
            "cat3": "金融等業專屬本業（銷售額，元）",
            "cat4": "銀行/保險本業（銷售額，元）",
            "cat5": "金融等非專屬本業（銷售額，元）",
            "cat6": "夜總會（銷售額，元）",
            "cat7": "酒家等（銷售額，元）",
            "cat8": "農產品批發市場（銷售額，元）",
            "budget_tax": "稅額上限（元）",
            "free_vars": "可放行變數（逗號分隔）：cat1..cat8。",
            "constraints": "約束條件（JSON/文字），可跨類別，如 {\"cat7\":{\">=\":\"cat6*1.2\"}}，可用 total_tax。",
        },
        "alias": {
            "cat1": ["小規模營業人", "小規模"],
            "cat2": ["再保費收入", "再保"],
            "cat3": ["金融等業專屬本業", "金融專屬"],
            "cat4": ["銀行/保險本業", "銀行保險本業"],
            "cat5": ["金融等非專屬本業", "金融非專屬"],
            "cat6": ["夜總會"],
            "cat7": ["酒家等"],
            "cat8": ["農產品批發市場", "農批市場"],
            "budget_tax": ["稅額上限", "目標稅額", "預算上限"],
            "free_vars": ["自由變數", "放行變數"],
            "constraints": ["限制條件", "約束條件"]
        }
    },
    # 菸酒稅
    {
        "name": "ta_tax",
        "description": "菸酒稅",
        "module": "tax_calculators.tobacco_alcohol_tax",
        "entry_func": {
            "minimize": "minimize_ta_tax",
            "maximize": "maximize_ta_qty",
        },
        "keywords": ["菸酒稅", "菸稅", "酒稅", "tobacco", "alcohol", "啤酒", "蒸餾酒", "紙菸", "雪茄", "酒精", "ABV"],
        "required_fields": [
            # 菸（新制／舊制）— 只有數量
            "cigarettes_new.quantity",
            "tobacco_cut_new.quantity",
            "cigar_new.quantity",
            "tobacco_other_new.quantity",
            "cigarettes_old.quantity",
            "tobacco_cut_old.quantity",
            "cigar_old.quantity",
            "tobacco_other_old.quantity",

            # 酒 — 全部有 quantity
            "beer.quantity",
            "brew_other.quantity",
            "distilled.quantity",
            "reprocessed_gt20.quantity",
            "reprocessed_le20.quantity",
            "cooking_wine.quantity",
            "cooking_wine_old.quantity",
            "alcohol_other.quantity",
            "ethanol.quantity",
            "ethanol_old.quantity",

            # 需 ABV 的 4 類再多一個 alcohol_content
            "brew_other.alcohol_content",
            "distilled.alcohol_content",
            "reprocessed_le20.alcohol_content",
            "alcohol_other.alcohol_content",
        ],
        "optimizable_fields": [
            # 菸（新制／舊制）— 只有數量
            "cigarettes_new.quantity",
            "tobacco_cut_new.quantity",
            "cigar_new.quantity",
            "tobacco_other_new.quantity",
            "cigarettes_old.quantity",
            "tobacco_cut_old.quantity",
            "cigar_old.quantity",
            "tobacco_other_old.quantity",

            # 酒 — 全部有 quantity
            "beer.quantity",
            "brew_other.quantity",
            "distilled.quantity",
            "reprocessed_gt20.quantity",
            "reprocessed_le20.quantity",
            "cooking_wine.quantity",
            "cooking_wine_old.quantity",
            "alcohol_other.quantity",
            "ethanol.quantity",
            "ethanol_old.quantity",

            # 需 ABV 的 4 類再多一個 alcohol_content
            "brew_other.alcohol_content",
            "distilled.alcohol_content",
            "reprocessed_le20.alcohol_content",
            "alcohol_other.alcohol_content",
        ],
        "constraint_fields": ["free_vars", "constraints", "budget_tax"],
        "budget_field": "budget_tax",
        "nl_parser": "agents.parsers.ta_nl_parser:nl_to_payload",
        "field_labels":TA_FIELD_LABELS_REG,
        "alias": {
            "free_vars": ["自由變數", "放行變數"],
            "constraints": ["限制條件", "約束條件"],
            "budget_tax": ["稅額上限", "目標稅額", "預算上限"],
            "quantity": ["數量", "出廠數量", "條數", "公斤數", "公升數", "瓶數"],
            "alcohol_content": ["酒精度", "ABV", "酒度", "含酒精%"]
        },
        "row_fields": ["quantity", "alcohol_content"],
        # 建議沿用與貨物稅相同的 renderer
        "report_renderer": "agents.report_renderer:render_report",
    },
    # 遺產稅
    {
        "name": "estate_tax",
        "description": "遺產稅",
        "module": "tax_calculators.estate_tax",
        "entry_func": "calculate_estate_tax",
        "keywords": ["遺產稅", "estate tax", "遺產"],
        "required_fields": [
            "death_period",
            "is_military_police",
            "land_value",
            "building_value",
            "house_value",
            "deposit_bonds_value",
            "stock_invest_value",
            "cash_gold_jewelry_value",
            "gift_in_2yrs_value",
            "spouse_count",
            "lineal_descendant_count",
            "father_mother_count",
            "disability_count",
            "dependent_count",
            "farmland_val",
            "inheritance_6to9_val",
            "unpaid_tax_fines_val",
            "unpaid_debts_val",
            "will_management_fee",
            "public_facility_retention_val",
            "spouse_surplus_right_val",
            "gift_tax_offset",
            "foreign_tax_offset",
        ],
        "optimizable_fields": [
            "land_value",
            "building_value",
            "house_value",
            "deposit_bonds_value",
            "stock_invest_value",
            "cash_gold_jewelry_value",
            "gift_in_2yrs_value",
            "spouse_count",
            "lineal_descendant_count",
            "father_mother_count",
            "disability_count",
            "dependent_count",
            "farmland_val",
            "inheritance_6to9_val",
            "unpaid_tax_fines_val",
            "unpaid_debts_val",
            "will_management_fee",
            "public_facility_retention_val",
            "spouse_surplus_right_val",
            "gift_tax_offset",
            "foreign_tax_offset",
        ],
        "constraint_fields": ["free_vars", "constraints"],
        "field_labels": {
            "death_period": (
                "被繼承人死亡日期（民國年/月/日，例如 113/3/15 或 112）"
                "；或直接輸入區間代碼 1-5：\n"
                "  1. 103/01/01 – 106/05/11\n"
                "  2. 106/05/12 – 110/12/31\n"
                "  3. 111/01/01 – 112/12/31\n"
                "  4. 113/01/01 – 113/12/31\n"
                "  5. 114/01/01 以後"
            ),
            "is_military_police":      "是否因公死亡（是/否）",
            "land_value":              "土地價值（元）",
            "building_value":          "地上物價值（元）",
            "house_value":             "房屋價值（元）",
            "deposit_bonds_value":     "存款／債權價值（元）",
            "stock_invest_value":      "股票／投資價值（元）",
            "cash_gold_jewelry_value": "現金／珠寶價值（元）",
            "gift_in_2yrs_value":      "死亡前 2 年贈與總額（元）",
            "spouse_count":            "配偶人數",
            "lineal_descendant_count": "直系卑親屬人數",
            "father_mother_count":     "父母人數",
            "disability_count":        "身心障礙人數",
            "dependent_count":         "扶養其他親屬人數",
            "farmland_val":            "農業用地價值（元）",
            "inheritance_6to9_val":    "6–9 年農業扣除額（元）",
            "unpaid_tax_fines_val":    "未繳稅罰金額（元）",
            "unpaid_debts_val":        "未償債務金額（元）",
            "will_management_fee":     "遺囑管理費（元）",
            "public_facility_retention_val": "公共設施保留價值（元）",
            "spouse_surplus_right_val": "配偶剩餘權益價值（元）",
            "gift_tax_offset":         "贈與稅扣抵（元）",
            "foreign_tax_offset":      "國外已納稅額扣抵（元）",
            "free_vars":               "欲放行的自由變數（逗號分隔）",
            "variable_constraints":    "變數約束條件（JSON／文字，無則留空）",
        },
        "alias": {
            "death_period": ["死亡日期", "過世日期", "死亡時間", "民國年期間代碼"],
            "is_military_police": ["是否因公死亡", "因公殉職"],
            "land_value": ["土地價值", "土地總額"],
            "building_value": ["地上物價值", "建物(地上物)價值"],
            "house_value": ["房屋價值", "房產價值", "房地產價值"],
            "deposit_bonds_value": ["存款債權價值", "存款及債權"],
            "stock_invest_value": ["股票投資價值", "股票/投資價值"],
            "cash_gold_jewelry_value": ["現金珠寶價值", "現金及珠寶"],
            "gift_in_2yrs_value": ["死亡前2年贈與總額", "兩年內贈與總額"],
            "spouse_count": ["配偶人數"],
            "lineal_descendant_count": ["直系卑親屬人數", "子女/孫子女人數"],
            "father_mother_count": ["父母人數"],
            "disability_count": ["身心障礙人數", "身障人數"],
            "dependent_count": ["扶養其他親屬人數"],
            "farmland_val": ["農業用地價值", "農地價值"],
            "inheritance_6to9_val": ["6–9年農業扣除額", "農業扣除額(6-9年)"],
            "unpaid_tax_fines_val": ["未繳稅罰金額", "未繳稅罰"],
            "unpaid_debts_val": ["未償債務金額", "未清償債務"],
            "will_management_fee": ["遺囑管理費"],
            "public_facility_retention_val": ["公共設施保留價值"],
            "spouse_surplus_right_val": ["配偶剩餘權益價值"],
            "gift_tax_offset": ["贈與稅扣抵"],
            "foreign_tax_offset": ["國外已納稅額扣抵"],
            "free_vars": ["自由變數", "放行變數"],
            "constraints": ["限制條件", "約束條件"]
        },
    },
    # 贈與稅
    {
        "name": "gift_tax",
        "description": "贈與稅",
        "module": "tax_calculators.gift_tax",
        "entry_func": "calculate_gift_tax",
        "keywords": ["贈與稅", "gift tax", "贈與"],
        "required_fields": [
            "period_choice",
            "land_value",
            "ground_value",
            "house_value",
            "others_value",
            "not_included_land",
            "not_included_house",
            "not_included_others",
            "remaining_exemption_98",
            "previous_gift_sum_in_this_year",
            "land_increment_tax",
            "deed_tax",
            "other_gift_burdens",
            "previous_gift_tax_or_credit",
            "new_old_system_adjustment",
        ],
        "optimizable_fields": [
            "land_value",
            "ground_value",
            "house_value",
            "others_value",
            "not_included_land",
            "not_included_house",
            "not_included_others",
            "remaining_exemption_98",
            "previous_gift_sum_in_this_year",
            "land_increment_tax",
            "deed_tax",
            "other_gift_burdens",
            "previous_gift_tax_or_credit",
            "new_old_system_adjustment",
        ],
        "constraint_fields": ["free_vars", "constraints"],
        "field_labels": {
            "period_choice": (
                "贈與發生年份（民國年，例如 112），或直接輸入期間代碼 1-4：\n"
                "  1. 98/01/23 - 106/05/11\n"
                "  2. 106/05/12 - 110/12/31\n"
                "  3. 111/01/01 - 113/12/31\n"
                "  4. 114/01/01 以後"
            ),
            "land_value": "土地價值（元）",
            "ground_value": "地上物價值（元）",
            "house_value": "房屋價值（元）",
            "others_value": "動產及權利價值（元）",
            "not_included_land": "不計入土地（元）",
            "not_included_house": "不計入房屋（元）",
            "not_included_others": "不計入其他（元）",
            "remaining_exemption_98": "98 舊制剩餘免稅額（元）",
            "previous_gift_sum_in_this_year": "同年度前次贈與總額（元）",
            "land_increment_tax": "土地增值稅（元）",
            "deed_tax": "契稅（元）",
            "other_gift_burdens": "其他贈與負擔（元）",
            "previous_gift_tax_or_credit": "前次贈與稅／扣抵（元）",
            "new_old_system_adjustment": "新舊制差額調整（元，可負）",
            "free_vars": "欲放行的自由變數（逗號分隔）",
            "variable_constraints": "變數約束條件（JSON／文字，無則留空）",
        },
        "alias": {
            "period_choice": ["贈與發生年份", "贈與年度", "民國年", "期間代碼"],
            "land_value": ["土地價值"],
            "ground_value": ["地上物價值", "建物價值(地上物)"],
            "house_value": ["房屋價值"],
            "others_value": ["動產及權利價值", "其他財產價值"],
            "not_included_land": ["不計入土地"],
            "not_included_house": ["不計入房屋"],
            "not_included_others": ["不計入其他"],
            "remaining_exemption_98": ["98舊制剩餘免稅額", "舊制剩餘免稅額"],
            "previous_gift_sum_in_this_year": ["同年度前次贈與總額", "前次贈與總額"],
            "land_increment_tax": ["土地增值稅"],
            "deed_tax": ["契稅"],
            "other_gift_burdens": ["其他贈與負擔"],
            "previous_gift_tax_or_credit": ["前次贈與稅或扣抵", "前次贈與稅/扣抵"],
            "new_old_system_adjustment": ["新舊制差額調整"],
            "free_vars": ["自由變數", "放行變數"],
            "constraints": ["限制條件", "約束條件"]
        },
    },
    # 證券交易稅
    {
        "name": "securities_tx_tax",
        "description": "證券交易稅",
        "module": "tax_calculators.securities_and_futures_transaction_tax",
        "entry_func": "compute_securities_transaction_tax",
        "keywords": ["證券交易稅", "股票交易稅", "股權證書", "公司債", "債券", "權證", "權證履約", "股票移轉", "現金結算"],
        "required_fields": [
                # 交易金額
                "stock.tp",                 # 公司股票／股權證書- 交易金額
                "bond.tp",                  # 公司債及其他有價證券- 交易金額
                "warrant.tp",               # 認購（售）權證- 交易金額
                # 權證履約（股票移轉／現金結算）
                "warrant_delivery_stock.ep",# 權證履約—股票移轉- 履約價格
                "warrant_delivery_stock.sc",# 權證履約—股票移轉- 股數
                "warrant_delivery_cash.ep", # 權證履約—現金結算- 履約價格
                "warrant_delivery_cash.sc", # 權證履約—現金結算- 股數
            ],
        "optimizable_fields": [
                # 交易金額
                "stock.tp",                 # 公司股票／股權證書- 交易金額
                "bond.tp",                  # 公司債及其他有價證券- 交易金額
                "warrant.tp",               # 認購（售）權證- 交易金額
                # 權證履約（股票移轉／現金結算）
                "warrant_delivery_stock.ep",# 權證履約—股票移轉- 履約價格
                "warrant_delivery_stock.sc",# 權證履約—股票移轉- 股數
                "warrant_delivery_cash.ep", # 權證履約—現金結算- 履約價格
                "warrant_delivery_cash.sc", # 權證履約—現金結算- 股數
            ],              
        "constraint_fields": ["free_vars", "constraints", "target_tax"],
        "budget_field": "target_tax",
        "nl_parser": "agents.parsers.securities_nl_parser:nl_to_payload",
        "field_labels": {
                **__import__("tax_calculators.securities_and_futures_transaction_tax",
                            fromlist=["SEC_FIELD_LABELS"]).SEC_FIELD_LABELS,
                "target_tax": "稅額上限"
            },                
        "report_renderer": "agents.report_renderer:render_report",
        "row_fields": ["tp", "ep", "sc"],
        "alias": {
            "free_vars": ["自由變數", "放行變數"],
            "constraints": ["限制條件", "約束條件", "條件式"],
            "target_tax": ["稅額上限", "目標稅額", "預算上限"],
            "tp": ["交易金額", "成交金額"],
            "ep": ["履約價格", "履約價"],
            "sc": ["股數", "數量"]
        },
    },
    # 期貨交易稅
    {
        "name": "futures_tx_tax",
        "description": "期貨交易稅",
        "module": "tax_calculators.securities_and_futures_transaction_tax",
        "entry_func": "compute_futures_transaction_tax",
        "keywords": ["期貨交易稅", "股價期貨", "股指期貨", "利率期貨", "選擇權", "黃金期貨"],
        "required_fields": [
            # 契約金額（多數品項）
            "stock_index.ca",           # 股價期貨- 契約金額
            "interest_rate_30.ca",      # 利率期貨—30天CP- 契約金額
            "interest_rate_10.ca",      # 利率期貨—10年期債- 契約金額
            "gold.ca",                  # 黃金期貨- 契約金額
            # 選擇權（稅基用 pa；ca 可供約束或參考）
            "option.pa",                # 選擇權及期貨選擇權- 權利金金額
            "option.ca",                # 選擇權及期貨選擇權- 契約金額（非稅基，給 constraints 用）
        ],   
        "optimizable_fields": [
            # 契約金額（多數品項）
            "stock_index.ca",           # 股價期貨- 契約金額
            "interest_rate_30.ca",      # 利率期貨—30天CP- 契約金額
            "interest_rate_10.ca",      # 利率期貨—10年期債- 契約金額
            "gold.ca",                  # 黃金期貨- 契約金額
            # 選擇權（稅基用 pa；ca 可供約束或參考）
            "option.pa",                # 選擇權及期貨選擇權- 權利金金額
            "option.ca",                # 選擇權及期貨選擇權- 契約金額（非稅基，給 constraints 用）
        ],                       
        "constraint_fields": ["free_vars", "constraints", "target_tax"],
        "budget_field": "target_tax",
        "nl_parser": "agents.parsers.futures_nl_parser:nl_to_payload",
        "field_labels": {
            **__import__("tax_calculators.securities_and_futures_transaction_tax",
                        fromlist=["FUT_FIELD_LABELS"]).FUT_FIELD_LABELS,
            "target_tax": "稅額上限"
        },                
        "row_fields": ["ca", "pa"],
        "report_renderer": "agents.report_renderer:render_report",
        "alias": {
            "free_vars": ["自由變數", "放行變數"],
            "constraints": ["限制條件", "約束條件"],
            "target_tax": ["稅額上限", "目標稅額", "預算上限"],
            "ca": ["契約金額", "合約金額", "名目金額"],
            "pa": ["權利金金額", "權利金", "premium"]
        },
    },
   # 特種貨物及勞務稅－高額消費「貨物」
    {
        "name": "special_goods_tax",
        "description": "特種貨物稅",
        "module": "tax_calculators.special_tax",
        "entry_func": {
            "minimize": "minimize_sg_tax",
            "maximize": "maximize_qty_under_budget_sg"
        },
        "keywords": ["特種貨物稅"],
        "required_fields": [
            "car.price",
            "car.quantity",
            "yacht.price",
            "yacht.quantity",
            "aircraft.price",
            "aircraft.quantity",
            "coral_ivory.price",
            "coral_ivory.quantity",
            "furniture.price",
            "furniture.quantity"
        ],
        "optimizable_fields": [
            "car.price",
            "car.quantity",
            "yacht.price",
            "yacht.quantity",
            "aircraft.price",
            "aircraft.quantity",
            "coral_ivory.price",
            "coral_ivory.quantity",
            "furniture.price",
            "furniture.quantity"
        ],
        "constraint_fields": ["free_vars", "constraints", "budget_tax"],
        "budget_field": "budget_tax",
        "nl_parser": "agents.parsers.special_goods_nl_parser:nl_to_payload_fixed",
        "field_labels": {
            "budget_tax": "稅額上限（元）",
            "car.price": "小客車：單價（元）",
            "car.quantity": "小客車：數量",
            "yacht.price": "遊艇：單價（元）",
            "yacht.quantity": "遊艇：數量",
            "aircraft.price": "飛機／直升機：單價（元）",
            "aircraft.quantity": "飛機／直升機：數量",
            "coral_ivory.price": "珊瑚／象牙等：單價（元）",
            "coral_ivory.quantity": "珊瑚／象牙等：數量",
            "furniture.price": "家具：單價（元）",
            "furniture.quantity": "家具：數量"
        },
        "report_renderer": "agents.report_renderer:render_report",
        "alias": {
            "free_vars": ["自由變數", "放行變數"],
            "constraints": ["限制條件", "約束條件"],
            "budget_tax": ["稅額上限", "目標稅額", "預算上限"],
            "car.price": ["小客車單價", "車價"],
            "car.quantity": ["小客車數量", "車輛數量"],
            "yacht.price": ["遊艇單價", "艇價"],
            "yacht.quantity": ["遊艇數量"],
            "aircraft.price": ["飛機單價", "直升機單價"],
            "aircraft.quantity": ["飛機數量", "直升機數量"],
            "coral_ivory.price": ["珊瑚象牙單價"],
            "coral_ivory.quantity": ["珊瑚象牙數量"],
            "furniture.price": ["家具單價"],
            "furniture.quantity": ["家具數量"]
        },
    },
    # 特種貨物及勞務稅－高額消費「勞務」
    {
        "name": "special_tax",
        "description": "勞務稅",
        "module": "tax_calculators.special_tax",
        "entry_func": "minimize_special_services_tax",
        "keywords": ["特種勞務稅", "特銷稅", "高額消費", "勞務稅"],
        "required_fields": ["sales_price"],
        "optimizable_fields": ["sales_price"],
        "constraint_fields": ["free_vars", "constraints"],
        "field_labels": {
            "sales_price": "高額消費勞務入會權利銷售額（元）",
            "free_vars": "欲放行的自由變數（逗號分隔）",
            "constraints": "變數約束條件（JSON／文字，無則留空）",
        },
        "alias": {
            "sales_price": ["銷售額", "入會權利銷售額", "價格", "金額"],
            "free_vars": ["自由變數", "放行變數"],
            "constraints": ["限制條件", "約束條件"]
        },
    }
]

def get_tool(name: str) -> Callable[[dict], dict]:
    """
    單入口工具使用；若該工具是多入口（entry_func 為 dict），請改用 resolve_entry_func(name, op)。
    """
    meta = next(t for t in TOOLS if t["name"] == name)
    entry = meta["entry_func"]
    if isinstance(entry, str):
        return getattr(import_module(meta["module"]), entry)
    raise ValueError(f"Tool '{name}' has multiple entry functions; use resolve_entry_func(name, op).")

def resolve_entry_func(name: str, op: Optional[str] = None) -> Callable[[dict], dict]:
    """
    多入口工具的通用解析：op 可為 'minimize' 或 'maximize'（或你之後擴增的 key）。
    若 entry_func 是字串，亦可直接回該函式，方便統一呼叫。
    """
    meta = next(t for t in TOOLS if t["name"] == name)
    entry = meta["entry_func"]
    mod = import_module(meta["module"])
    if isinstance(entry, dict):
        if not op:
            raise ValueError(f"Tool '{name}' requires an 'op' to select entry function.")
        fn = entry.get(op)
        if not fn:
            raise ValueError(f"Tool '{name}' has no entry for op='{op}'.")
        return getattr(mod, fn)
    return getattr(mod, entry)
