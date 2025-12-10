from flask import Blueprint, request, jsonify
from .income_tax import calculate_comprehensive_income_tax
from .foreigner_income_tax import calculate_foreigner_income_tax
from .business_income_tax import calculate_business_income_tax
from .sale_tax import minimize_vat_tax, minimize_nvat_tax, maximize_nvat_under_budget
from .cargo_tax import minimize_goods_tax, maximize_goods_qty
from .tobacco_alcohol_tax import minimize_ta_tax, maximize_qty_under_budget
from .estate_tax import calculate_estate_tax
from .gift_tax import calculate_gift_tax
from .securities_and_futures_transaction_tax import compute_futures_transaction_tax, compute_securities_transaction_tax
from .special_tax import minimize_sg_tax, minimize_special_services_tax, maximize_qty_under_budget_sg

tax_bp = Blueprint('tax_bp', __name__)


# 1. 綜所稅
@tax_bp.route('/income', methods=['POST'])
def income_api():
    data = request.get_json()
    result = calculate_comprehensive_income_tax(**data)
    return jsonify({'tax': result})


# 2.外僑所得稅
@tax_bp.route('/foreigner', methods=['POST'])
def foreigner_api():
    data = request.get_json()
    result = calculate_foreigner_income_tax(**data)
    return jsonify({'tax': result})


# 3.營利事業所得稅
@tax_bp.route('/businessIncome', methods=['POST'])
def corporate_api():
    data = request.get_json()
    result = calculate_business_income_tax(**data)
    return jsonify({'tax': result})


# 4-1 加值型 - minimize
@tax_bp.route('/value-added/minimize', methods=['POST'])
def value_added_min():
    d = request.get_json()
    res = minimize_vat_tax(
        rows=d['rows'],
        free_vars=d.get('free_vars', []),
        constraints=d.get('constraints_json', {})
    )
    return jsonify({'tax': res})


# 4-2 非加值型 - minimize
@tax_bp.route('/non-value-added/minimize', methods=['POST'])
def nvalue_added_min():
    d = request.get_json()
    res = minimize_nvat_tax(
        rows=d['rows'],
        free_vars=d.get('free_vars', []),
        constraints=d.get('constraints_json', {})
    )
    return jsonify({'tax': res})

# 4-3 非加值型 - maximize
@tax_bp.route('/non-value-added/maximize', methods=['POST'])
def nvalue_added_max():
    d = request.get_json()
    res = maximize_nvat_under_budget(
        rows=d['rows'],
        free_vars=d.get('free_vars', []),
        constraints=d.get('constraints_json', {}),
        budget_tax=d.get('target_tax')
    )
    return jsonify({'tax': res})

# 5-1.貨物稅-最小化總稅額
@tax_bp.route('/cargo/minimize', methods=['POST'])
def cargo_api_minimize():
    data = request.get_json()
    rows        = data.get('rows', [])
    free_vars   = data.get('free_vars', [])
    constraints = data.get('constraints', {})
    result = minimize_goods_tax(
        rows=rows,
        free_vars=free_vars,
        constraints=constraints
    )
    return jsonify({'tax': result})

# 5-2.貨物稅-給定稅額下最大化購買數量
@tax_bp.route('/cargo/maximize', methods=['POST'])
def cargo_api_maximize():
    data = request.get_json()
    rows        = data.get('rows', [])
    free_vars   = data.get('free_vars', [])
    constraints = data.get('constraints', {})
    budget_tax  = data.get('target_tax')
    result = maximize_goods_qty(
        rows=rows,
        free_vars=free_vars,
        constraints=constraints,
        budget_tax=budget_tax
    )
    return jsonify({'tax': result})

# 6-1. 菸酒稅-給定條件下最小化菸酒稅額
@tax_bp.route('/tobacco-alcohol/minimize', methods=['POST'])
def tobacco_alcohol_api_minimize():
    data = request.get_json()
    if 'rows' in data:
        result = minimize_ta_tax(
            rows=data['rows'],
            free_vars=data.get('free_vars', []),
            constraints=data.get('constraints_json', {})
        )
        return jsonify({'tax': result})


# 6-2.  菸酒稅-給定稅額下最大化購買數量
@tax_bp.route('/tobacco-alcohol/maximize', methods=['POST'])
def tobacco_alcohol_api_maximize():
    data = request.get_json()
    if 'rows' in data:
        result = maximize_qty_under_budget(
            rows=data['rows'],
            free_vars=data.get('free_vars', []),
            constraints=data.get('constraints_json', {}),
            budget_tax=data.get('target_tax', None)
        )
        return jsonify({'tax': result})


# 7.遺產稅
@tax_bp.route('/estate', methods=['POST'])
def estate_api():
    data = request.get_json()
    result = calculate_estate_tax(**data)
    return jsonify({'tax': result})


# 8.贈與稅
@tax_bp.route('/gift', methods=['POST'])
def gift_api():
    data = request.get_json()
    result = calculate_gift_tax(**data)
    return jsonify({'tax': result})


# 9-1.證交期交稅-證券交易稅
@tax_bp.route('/securities-transaction', methods=['POST'])
def securities_transaction_api():
    data = request.get_json()
    result = compute_securities_transaction_tax(**data)
    return jsonify({'tax': result})


# 9-2.證交期交稅-期貨交易稅
@tax_bp.route('/futures-transaction', methods=['POST'])
def futures_transaction_api():
    data = request.get_json()
    result = compute_futures_transaction_tax(**data)
    return jsonify({'tax': result})


# 10-1. 特種貨物稅：最小化總稅額
@tax_bp.route('/special-goods/minimize', methods=['POST'])
def special_goods_minimize_api():
    data = request.get_json()
    result = minimize_sg_tax(**data)            # rows, free_vars, constraints
    return jsonify({'tax': result})


# 10-1. 特種貨物稅：在預算內最大化總數量
@tax_bp.route('/special-goods/maximize', methods=['POST'])
def special_goods_maximize_api():
    data = request.get_json()
    result = maximize_qty_under_budget_sg(**data)  # rows, free_vars, constraints, budget_tax
    return jsonify({'tax': result})


# 10-2. 特種勞務稅：單筆最小化
@tax_bp.route('/special-services/minimize', methods=['POST'])
def special_services_minimize_api():
    data = request.get_json()
    result = minimize_special_services_tax(**data)  # sales_price, free_vars, constraints
    return jsonify({'tax': result})