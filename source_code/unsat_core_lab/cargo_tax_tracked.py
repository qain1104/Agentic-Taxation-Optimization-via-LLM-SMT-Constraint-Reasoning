# -*- coding: utf-8 -*-
from __future__ import annotations

import re

from z3 import Optimize, Real, Int, RealVal, IntVal, ToReal, Sum

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
            "free_vars": [
                "drink_other.quantity",
                "car_le_2000cc.quantity",
                "car_le_2000cc.assessed_price",
            ],
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

    def _simple_bound(self, field: str, op: str):
        """Read simple payload constraints like {"field": {">=": 400000}}.

        This helper is intentionally conservative. It only reads direct
        single-field bounds. It is used to linearize the minimization objective
        for ad-valorem price * quantity terms when price is free.
        """
        rules = (self.payload.get("constraints") or {}).get(field)
        if not isinstance(rules, dict):
            return None
        value = rules.get(op)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _payload_mentions_field(self, field: str) -> bool:
        """Return True when a field belongs to the actual user scenario.

        For maximize-under-budget cases, absent catalog rows with default zero
        should remain form defaults, not decision variables in the maximized
        quantity. This prevents releasing an unrelated default item from
        changing the objective semantics.
        """
        if field in set(self.payload.get("free_vars") or []):
            return True
        if field in self.payload:
            try:
                return abs(float(self.payload.get(field) or 0)) > 1e-9
            except Exception:
                return True

        constraints = self.payload.get("constraints") or {}
        if isinstance(constraints, dict):
            token = re.escape(field)
            pattern = re.compile(rf"(?<![A-Za-z0-9_.]){token}(?![A-Za-z0-9_.])")
            for lhs, rules in constraints.items():
                if pattern.search(str(lhs)):
                    return True
                if isinstance(rules, dict):
                    for rhs in rules.values():
                        values = rhs if isinstance(rhs, (list, tuple)) else [rhs]
                        for one in values:
                            if isinstance(one, str) and pattern.search(one):
                                return True
        return False

    def _minimizing_unit_price(self, slug: str) -> float:
        """Return the price that minimizes tax for this ad-valorem item.

        The original traced version encoded rate * assessed_price * quantity
        directly. If both price and quantity are free, this is nonlinear, and
        Z3 Optimize may return a satisfiable but non-optimal model. For the
        current cargo cases, assessed prices are only bounded by simple lower
        and upper bounds. Since the rate is non-negative and the objective is
        minimization, the optimal price is the lower bound.

        This keeps the cargo model linear in quantity and makes the
        strictly-better UNSAT probe consistent with the base optimum.
        """
        key = f"{slug}.assessed_price"
        if key not in set(self.payload.get("free_vars") or []):
            return float(self._num(key))

        lb = self._simple_bound(key, ">=")
        if lb is None:
            lb = self._simple_bound(key, ">")
        if lb is None:
            lb = 0.0
        return float(lb)

    def build(self):
        if self._built:
            return self
        self._built = True
        self.opt = Optimize()

        quantity_int_vars = {}

        for slug, spec in ITEMS.items():
            q_int = Int(f"{slug}_quantity_int")
            q = ToReal(q_int)
            quantity_int_vars[slug] = q_int

            self._add_param(
                f"{slug}.quantity",
                q,
                self._num(f"{slug}.quantity"),
                [("nonnegative", lambda v: v >= 0)],
                releasable=(f"{slug}.quantity" in self.explicit_keys),
            )

            if spec["mode"] == "adval":
                p = Real(f"{slug}_assessed_price")
                self._add_param(
                    f"{slug}.assessed_price",
                    p,
                    self._num(f"{slug}.assessed_price"),
                    [("nonnegative", lambda v: v >= 0)],
                    releasable=(f"{slug}.assessed_price" in self.explicit_keys),
                )

        self._bind_params()

        for slug, q_int in quantity_int_vars.items():
            self.add(q_int >= 0, name=f"domain.{slug}.quantity_int_nonnegative", group="domain")

        # Stable model values for minimization when price is free:
        # force the assessed price to the lower bound that is already optimal.
        # This is a modeling shortcut for current cargo experiments; it avoids
        # nonlinear Optimize instability while preserving the minimum tax value.
        if self.payload.get("objective") != "max_qty":
            for slug, spec in ITEMS.items():
                if spec["mode"] != "adval":
                    continue
                pkey = f"{slug}.assessed_price"
                if pkey in set(self.payload.get("free_vars") or []):
                    p = self.params[pkey][0]
                    p_min = self._minimizing_unit_price(slug)
                    self.add(
                        p == RealVal(str(p_min)),
                        name=f"linearization.{slug}.price_at_min_bound",
                        group="tax_law",
                    )

        item_tax_ints = []
        qty_terms = []

        for slug, spec in ITEMS.items():
            q = self.params[f"{slug}.quantity"][0]
            q_field = f"{slug}.quantity"
            if self.payload.get("objective") == "max_qty":
                # Maximize only the quantities that define the user scenario.
                # Other catalog rows are merely absent/default form fields.
                if self._payload_mentions_field(q_field):
                    qty_terms.append(q)
            else:
                qty_terms.append(q)

            if spec["mode"] == "fixed":
                expr_real = RealVal(str(spec["unit_tax"])) * q
            else:
                if self.payload.get("objective") != "max_qty":
                    # Use the minimizing price to keep the objective linear.
                    unit_price = self._minimizing_unit_price(slug)
                    expr_real = RealVal(str(spec["rate"])) * RealVal(str(unit_price)) * q
                else:
                    # Maximize cases in the built-ins have fixed assessed prices.
                    p = self.params[f"{slug}.assessed_price"][0]
                    expr_real = RealVal(str(spec["rate"])) * p * q

            row_tax = Int(f"{slug}_tax")
            self.vars[f"{slug}.tax"] = row_tax

            # row_tax = floor(expr_real), matching cargo_tax.py.
            self.add(ToReal(row_tax) <= expr_real, name=f"law.{slug}.tax_floor_lower")
            self.add(expr_real < ToReal(row_tax) + RealVal("1"), name=f"law.{slug}.tax_floor_upper")

            item_tax_ints.append(row_tax)

        total_tax = Int("total_tax")
        final_tax = Int("final_tax_z")
        total_qty = Real("total_qty")
        objective_qty = Int("objective_qty")

        self.vars.update(
            {
                "total_tax": total_tax,
                "total_qty": total_qty,
                "objective_qty": objective_qty,
                "final_tax_z": final_tax,
            }
        )

        self.add(total_tax == Sum(item_tax_ints), name="law.total_tax_int_sum")
        self.add(final_tax == total_tax, name="law.final_tax_equals_total_tax")
        self.add(total_qty == Sum(qty_terms), name="law.total_qty")
        self.add(ToReal(objective_qty) == total_qty, name="objective_link.total_qty_int", group="objective_link")

        if self.payload.get("budget_tax") is not None:
            self.add(
                total_tax <= IntVal(int(float(self.payload.get("budget_tax")))),
                name="user_constraint.budget_tax",
                group="user_constraint",
                releasable=True,
            )

        self.final_tax_z = final_tax
        self.net_taxable_z = total_tax

        if self.payload.get("objective") == "max_qty":
            self.objective_z = objective_qty
            self.objective_sense = "max"
            self.objective_label = "total_qty"
        else:
            self.objective_z = final_tax
            self.objective_sense = "min"
            self.objective_label = "total_tax"

        self._add_user_constraints()
        self._optimize()
        return self


def minimize_cargo_tax(**payload):
    payload = dict(payload)
    payload["objective"] = "min_tax"
    return run_tracked_analysis(CargoTaxTrackedCase, payload)


def maximize_cargo_qty(**payload):
    payload = dict(payload)
    payload["objective"] = "max_qty"
    return run_tracked_analysis(CargoTaxTrackedCase, payload)


if __name__ == "__main__":
    raise SystemExit(
        cli_main(
            CargoTaxTrackedCase,
            BUILTIN_PAYLOADS,
            title=CargoTaxTrackedCase.title,
            default_json="cargo_tax_unsat_report.json",
            default_md="cargo_tax_unsat_report.md",
        )
    )
