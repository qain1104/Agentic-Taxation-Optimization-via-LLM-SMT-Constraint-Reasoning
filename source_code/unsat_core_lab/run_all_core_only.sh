#!/usr/bin/env bash
set -e

mkdir -p reports

python income_tax_tracked.py --case 0 --no-auto-combinations --core-only \
  --json-out reports/income_tax_unsat_report_case0.json \
  --md-out reports/income_tax_unsat_report_case0.md

python income_tax_tracked.py --case 1 --no-auto-combinations --core-only \
  --json-out reports/income_tax_unsat_report_case1.json \
  --md-out reports/income_tax_unsat_report_case1.md

python foreigner_income_tax_tracked.py --case 0 --no-auto-combinations --core-only \
  --json-out reports/foreigner_income_tax_unsat_report_case0.json \
  --md-out reports/foreigner_income_tax_unsat_report_case0.md

python foreigner_income_tax_tracked.py --case 1 --no-auto-combinations --core-only \
  --json-out reports/foreigner_income_tax_unsat_report_case1.json \
  --md-out reports/foreigner_income_tax_unsat_report_case1.md

python business_income_tax_tracked.py --case 0 --no-auto-combinations --core-only \
  --json-out reports/business_income_tax_unsat_report_case0.json \
  --md-out reports/business_income_tax_unsat_report_case0.md

python business_income_tax_tracked.py --case 1 --no-auto-combinations --core-only \
  --json-out reports/business_income_tax_unsat_report_case1.json \
  --md-out reports/business_income_tax_unsat_report_case1.md

python cargo_tax_tracked.py --case 0 --no-auto-combinations --core-only \
  --json-out reports/cargo_tax_unsat_report_case0.json \
  --md-out reports/cargo_tax_unsat_report_case0.md

python cargo_tax_tracked.py --case 1 --no-auto-combinations --core-only \
  --json-out reports/cargo_tax_unsat_report_case1.json \
  --md-out reports/cargo_tax_unsat_report_case1.md

python tobacco_alcohol_tax_tracked.py --case 0 --no-auto-combinations --core-only \
  --json-out reports/tobacco_alcohol_tax_unsat_report_case0.json \
  --md-out reports/tobacco_alcohol_tax_unsat_report_case0.md

python tobacco_alcohol_tax_tracked.py --case 1 --no-auto-combinations --core-only \
  --json-out reports/tobacco_alcohol_tax_unsat_report_case1.json \
  --md-out reports/tobacco_alcohol_tax_unsat_report_case1.md

python estate_tax_tracked.py --case 0 --no-auto-combinations --core-only \
  --json-out reports/estate_tax_unsat_report_case0.json \
  --md-out reports/estate_tax_unsat_report_case0.md

python estate_tax_tracked.py --case 1 --no-auto-combinations --core-only \
  --json-out reports/estate_tax_unsat_report_case1.json \
  --md-out reports/estate_tax_unsat_report_case1.md

python gift_tax_tracked.py --case 0 --no-auto-combinations --core-only \
  --json-out reports/gift_tax_unsat_report_case0.json \
  --md-out reports/gift_tax_unsat_report_case0.md

python gift_tax_tracked.py --case 1 --no-auto-combinations --core-only \
  --json-out reports/gift_tax_unsat_report_case1.json \
  --md-out reports/gift_tax_unsat_report_case1.md

python sale_tax_tracked.py --case 0 --no-auto-combinations --core-only \
  --json-out reports/sale_tax_unsat_report_case0.json \
  --md-out reports/sale_tax_unsat_report_case0.md

python sale_tax_tracked.py --case 1 --no-auto-combinations --core-only \
  --json-out reports/sale_tax_unsat_report_case1.json \
  --md-out reports/sale_tax_unsat_report_case1.md

python securities_futures_tax_tracked.py --case 0 --no-auto-combinations --core-only \
  --json-out reports/securities_futures_tax_unsat_report_case0.json \
  --md-out reports/securities_futures_tax_unsat_report_case0.md

python securities_futures_tax_tracked.py --case 1 --no-auto-combinations --core-only \
  --json-out reports/securities_futures_tax_unsat_report_case1.json \
  --md-out reports/securities_futures_tax_unsat_report_case1.md

python special_tax_tracked.py --case 0 --no-auto-combinations --core-only \
  --json-out reports/special_tax_unsat_report_case0.json \
  --md-out reports/special_tax_unsat_report_case0.md

python special_tax_tracked.py --case 1 --no-auto-combinations --core-only \
  --json-out reports/special_tax_unsat_report_case1.json \
  --md-out reports/special_tax_unsat_report_case1.md
