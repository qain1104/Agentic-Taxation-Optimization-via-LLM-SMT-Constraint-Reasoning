# main.py
"""
快速測試：
1. minimize_special_services_tax     ── 針對單筆特種勞務稅，最小化稅額
2. minimize_sg_tax                   ── 多筆特種貨物稅，最小化總稅額
3. maximize_qty_under_budget_sg      ── 預算內最大化總數量
"""

from pprint import pprint

# 按實際檔名修改 ↓↓↓
from special_tax import (
    minimize_special_services_tax,
    minimize_sg_tax,
    maximize_qty_under_budget_sg,
)

def test_special_services():
    print("\n=== 特種勞務稅：單筆最小化 ===")
    res = minimize_special_services_tax(
        sales_price=800_000,
        free_vars=['sales_price'],                 # 讓 Z3 去找更低的價格
        constraints={'sales_price': {'>=': 600_000}}  # 但不能低於 60 萬
    )
    pprint(res)

def test_minimize_sg():
    print("\n=== 特種貨物稅：多筆最小化總稅額 ===")
    rows = [
        {'price': 350_000, 'quantity': 2},
        {'price': 500_000, 'quantity': 1},
    ]
    res = minimize_sg_tax(
        rows=rows,
        free_vars=['row0.quantity', 'row1.price'],   # 調整第 0 列數量、第 1 列價格
        constraints={
            'row0.quantity': {'>=': 1, '<=': 5},
            'row1.price':    {'=': 480_000},
            'row0.quantity + row1.quantity': {'<=': 6},
        }
    )
    pprint(res)

def test_maximize_sg():
    print("\n=== 特種貨物稅：預算內最大化總數量 ===")
    rows = [
        {'price': 300_000, 'quantity': 1},
        {'price': 400_000, 'quantity': 1},
        {'price': 250_000, 'quantity': 1},
    ]
    res = maximize_qty_under_budget_sg(
        rows=rows,
        free_vars=[
            'row0.quantity', 'row1.quantity', 'row2.quantity'
        ],
        constraints={
            'row0.quantity': {'>=': 1},
            'row1.quantity': {'>=': 1},
            'row2.quantity': {'>=': 1},
        },
        budget_tax=300_000   # 稅額上限三十萬
    )
    pprint(res)

if __name__ == "__main__":
    test_special_services()
    test_minimize_sg()
    test_maximize_sg()
