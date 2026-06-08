# -*- coding: utf-8 -*-
from __future__ import annotations
from z3 import Optimize, Real, Int, RealVal, ToInt, Sum
from tracked_tax_core import BaseTrackedTaxCase, cli_main, run_tracked_analysis

ITEMS = {
    "cement_white": {"mode": "fixed", "unit_tax": 600},
    "cement_portland_I": {"mode": "fixed", "unit_tax": 320},
    "cement_blast_furnace": {"mode": "fixed", "unit_tax": 196},
    "cement_other": {"mode": "fixed", "unit_tax": 440},
    "oil_gasoline": {"mode": "fixed", "unit_tax": 6830},
    "oil_diesel": {"mode": "fixed", "unit_tax": 3990},
    "oil_kerosene": {"mode": "fixed", "unit_tax": 4250},
    "oil_jetfuel": {"mode": "fixed", "unit_tax": 610},
    "oil_fueloil": {"mode": "fixed", "unit_tax": 110},
    "oil_solvent": {"mode": "fixed", "unit_tax": 720},
    "lpg": {"mode": "fixed", "unit_tax": 690},
    "tire_bus_truck": {"mode": "adval", "rate": 0.10},
    "tire_other": {"mode": "adval", "rate": 0.15},
    "tire_inner_solid": {"mode": "adval", "rate": 0.00},
    "drink_diluted_juice": {"mode": "adval", "rate": 0.08},
    "drink_other": {"mode": "adval", "rate": 0.15},
    "drink_pure_juice": {"mode": "adval", "rate": 0.00},
    "glass_plain": {"mode": "adval", "rate": 0.10},
    "glass_conductive_mold": {"mode": "adval", "rate": 0.00},
    "fridge": {"mode": "adval", "rate": 0.13},
    "tv": {"mode": "adval", "rate": 0.13},
    "hvac_central": {"mode": "adval", "rate": 0.15},
    "hvac_non_central": {"mode": "adval", "rate": 0.20},
    "dehumidifier": {"mode": "adval", "rate": 0.15},
    "vcr": {"mode": "adval", "rate": 0.13},
    "recorder": {"mode": "adval", "rate": 0.10},
    "stereo": {"mode": "adval", "rate": 0.10},
    "oven": {"mode": "adval", "rate": 0.15},
    "car_le_2000cc": {"mode": "adval", "rate": 0.25},
    "car_gt_2000cc": {"mode": "adval", "rate": 0.30},
    "truck_bus": {"mode": "adval", "rate": 0.15},
    "motorcycle": {"mode": "adval", "rate": 0.17},
}

BUILTIN_PAYLOADS = [
    {
        "case_id": "cargo_minimize_cement_drink_car",
        "description": "白水泥固定 100 噸，飲料與小客車數量合計 50，小客車數量與申報價格受限，最小化貨物稅。",
        "payload": {
            "cement_white.quantity": 100,
            "drink_other.quantity": 0,
            "drink_other.assessed_price": 200,
            "car_le_2000cc.quantity": 0,
            "car_le_2000cc.assessed_price": 600_000,
            "free_vars": ["drink_other.quantity", "car_le_2000cc.quantity", "car_le_2000cc.assessed_price"],
            "constraints": {
                "drink_other.quantity + car_le_2000cc.quantity": {"=": 50},
                "car_le_2000cc.quantity": {"<=": 10},
                "car_le_2000cc.assessed_price": {">=": 400_000, "<=": 600_000},
            },
            "objective": "min_tax",
        },
    },
    {
        "case_id": "cargo_maximize_qty_under_budget",
        "description": "汽油固定 5 公秉，輪胎與飲料可調，在稅額 80000 內最大化總件數。",
        "payload": {
            "oil_gasoline.quantity": 5,
            "tire_bus_truck.quantity": 0,
            "tire_bus_truck.assessed_price": 20_000,
            "drink_diluted_juice.quantity": 0,
            "drink_diluted_juice.assessed_price": 180,
            "free_vars": ["tire_bus_truck.quantity", "drink_diluted_juice.quantity"],
            "constraints": {
                "drink_diluted_juice.quantity - tire_bus_truck.quantity": {">=": 0},
                "tire_bus_truck.quantity": {"<=": 500},
                "drink_diluted_juice.quantity": {"<=": 3000},
            },
            "budget_tax": 80_000,
            "objective": "max_qty",
        },
    },
]
BUILTIN_PAYLOAD = BUILTIN_PAYLOADS[0]["payload"]


class CargoTaxTrackedCase(BaseTrackedTaxCase):
    title = "Cargo Tax Tracked SMT Lab"

    def build(self):
        if self._built:
            return self
        self._built = True
        self.opt = Optimize()
        for slug, spec in ITEMS.items():
            q = Real(f"{slug}_quantity")
            self._add_param(f"{slug}.quantity", q, self._num(f"{slug}.quantity"), [("nonnegative", lambda v: v >= 0)])
            if spec["mode"] == "adval":
                p = Real(f"{slug}_assessed_price")
                self._add_param(f"{slug}.assessed_price", p, self._num(f"{slug}.assessed_price"), [("nonnegative", lambda v: v >= 0)])
        self._bind_params()
        tax_terms = []
        qty_terms = []
        for slug, spec in ITEMS.items():
            q = self.params[f"{slug}.quantity"][0]
            qty_terms.append(q)
            if spec["mode"] == "fixed":
                expr = RealVal(str(spec["unit_tax"])) * q
            else:
                p = self.params[f"{slug}.assessed_price"][0]
                expr = RealVal(str(spec["rate"])) * p * q
            row_tax = Real(f"{slug}_tax")
            self.vars[f"{slug}.tax"] = row_tax
            self.add(row_tax == expr, name=f"law.{slug}.tax")
            tax_terms.append(row_tax)
        total_tax = Real("total_tax")
        final_tax = Int("final_tax_z")
        total_qty = Real("total_qty")
        objective_qty = Int("objective_qty")
        self.vars.update({"total_tax": total_tax, "total_qty": total_qty, "objective_qty": objective_qty, "final_tax_z": final_tax})
        self.add(total_tax == Sum(tax_terms), name="law.total_tax")
        self.add(final_tax == ToInt(total_tax), name="law.final_tax_floor")
        self.add(total_qty == Sum(qty_terms), name="law.total_qty")
        self.add(objective_qty == ToInt(total_qty), name="objective_link.total_qty_int", group="objective_link")
        if self.payload.get("budget_tax") is not None:
            self.add(total_tax <= RealVal(str(float(self.payload.get("budget_tax")))), name="user_constraint.budget_tax", group="user_constraint", releasable=True)
        self.final_tax_z = final_tax
        self.net_taxable_z = total_tax
        if self.payload.get("objective") == "max_qty":
            self.objective_z = objective_qty
            self.objective_sense = "max"
        else:
            self.objective_z = final_tax
            self.objective_sense = "min"
        self._add_user_constraints()
        self._optimize()
        return self


def minimize_cargo_tax(**payload):
    payload = dict(payload); payload["objective"] = "min_tax"
    return run_tracked_analysis(CargoTaxTrackedCase, payload)


def maximize_cargo_qty(**payload):
    payload = dict(payload); payload["objective"] = "max_qty"
    return run_tracked_analysis(CargoTaxTrackedCase, payload)


if __name__ == "__main__":
    raise SystemExit(cli_main(CargoTaxTrackedCase, BUILTIN_PAYLOADS, title=CargoTaxTrackedCase.title, default_json="cargo_tax_unsat_report.json", default_md="cargo_tax_unsat_report.md"))
