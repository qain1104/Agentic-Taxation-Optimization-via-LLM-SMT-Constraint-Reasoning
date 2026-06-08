# Tracked Tax SMT / UNSAT-Core Lab

This directory contains tracked/trace SMT versions of the tax solvers used for
post-optimality diagnosis, constraint-relaxation experiments, and release-advice
generation.

The tracked solvers extend the original tax calculators with named constraints,
UNSAT-core optimality probes, release tests, cone-of-influence (COI) combination
tests, and automatic release classification.

## Included tax solvers

Current tracked solvers include:

- `income_tax_tracked.py`
- `foreigner_income_tax_tracked.py`
- `business_income_tax_tracked.py`
- `cargo_tax_tracked.py`
- `tobacco_alcohol_tax_tracked.py`
- `estate_tax_tracked.py`
- `gift_tax_tracked.py`
- `sale_tax_tracked.py`
- `securities_futures_tax_tracked.py`
- `special_tax_tracked.py`

Shared utilities are implemented in:

- `tracked_tax_core.py`

Additional integration utilities:

- `release_advice_engine.py`
- `multi_agent_tax_system_with_release_advice.py`

## What each tracked solver does

Each tracked solver supports:

1. base optimization,
2. strictly-better UNSAT probe,
3. UNSAT core extraction,
4. automatic single release tests,
5. COI-based combination release tests,
6. automatic domain-bound / variable-type release classification,
7. Markdown and JSON report generation.

For minimization tasks, the strictly-better probe is:

```text
objective <= optimum - 1
```

For maximization tasks, the strictly-better probe is:

```text
objective >= optimum + 1
```

If the probe is UNSAT, the returned UNSAT core identifies a subset of tracked
constraints sufficient to prove that no strictly better solution exists under
the current assumptions.

## Output reports

Generated reports should be written to the `reports/` directory:

```text
reports/
├── income_tax_unsat_report_case0.json
├── income_tax_unsat_report_case0.md
├── ...
└── run_all_core_only.log
```

To move existing reports into `reports/`:

```bash
mkdir -p reports
mv *_unsat_report*.json *_unsat_report*.md reports/ 2>/dev/null
```

## Basic run examples

Run a single case:

```bash
python income_tax_tracked.py --case 0
python estate_tax_tracked.py --case 1
python cargo_tax_tracked.py --case 0
```

Run with full core printed to terminal:

```bash
python income_tax_tracked.py --case 0 --print-core
```

Run and write reports into `reports/`:

```bash
python estate_tax_tracked.py --case 0 \
  --json-out reports/estate_tax_unsat_report_case0.json \
  --md-out reports/estate_tax_unsat_report_case0.md
```

## Fast modes

For quick smoke tests:

```bash
python estate_tax_tracked.py --case 0 --probe-only
```

`--probe-only` runs only:

1. base solve,
2. strictly-better UNSAT probe.

It skips release tests and COI combination tests.

For a faster release-analysis run:

```bash
python estate_tax_tracked.py --case 0 --no-auto-combinations --core-only
```

This runs release tests only for releasable constraints that appear in the
UNSAT core and skips COI combination tests.

Available speed/debug flags:

```text
--probe-only
--no-release-tests
--no-auto-combinations
--core-only
--print-core
```

## Batch execution

Recommended core-only batch run:

```bash
./run_all_core_only.sh > reports/run_all_core_only.log 2>&1
```

To monitor progress:

```bash
tail -f reports/run_all_core_only.log
```

## Custom payloads

Each tracked solver supports custom JSON payloads:

```bash
python gift_tax_tracked.py --payload case.json
```

Payloads may use either direct calculator keyword format or a wrapper format:

```json
{
  "payload": {
    "free_vars": ["rent_deduction"],
    "constraints": {
      "rent_deduction": {"<=": 180000}
    }
  }
}
```

## Release tests

A release test removes one selected fixed input or user constraint and then
re-optimizes the solver model.

For minimization tasks:

```text
delta_vs_base < 0
```

means the release decreases the objective.

For maximization tasks:

```text
delta_vs_base > 0
```

means the release improves the objective.

Release tests are diagnostic. They do not automatically imply legal advice.

## COI-based combination release

COI stands for cone of influence.

The COI procedure follows variable-formula dependencies in the tracked SMT
model. Starting from a releasable fixed input, it finds tax-law formulas that
use the same Z3 variable, then identifies other fixed inputs appearing in the
same local formula layer.

This helps find coupled release candidates, such as:

```text
education_fee + education_count
```

where each variable may have no effect alone, but the pair may matter together.

## Release classification

Each release result is automatically classified as one of:

- `useful_bounded_release`
- `unsafe_needs_domain_bound`
- `semantic_change`
- `no_effect`
- `worse`
- `infeasible_or_unknown`

The classification is based on automatic domain-bound and variable-type
inference. For example, a release is safer when the variable is bounded by a
statutory cap or explicit upper bound. Count-like factual fields and monetary
claim fields without upper bounds are usually marked as requiring additional
domain bounds.

## RQ4 experiment use

These tracked solvers are used for RQ4:

> How efficiently can unsat core-guided constraint relaxation identify feasible
> tax-reduction strategies compared with baseline relaxation methods?

The RQ4 experiment should run at the traced solver level, not through the
full multi-agent UI. This avoids mixing relaxation-search cost with natural
language parsing, UI overhead, and report-generation latency.

Recommended comparison methods:

1. random relaxation over all releasable constraints,
2. field-name heuristic relaxation,
3. exhaustive single-constraint relaxation,
4. unsat-core-guided relaxation,
5. unsat-core-guided relaxation plus COI combination tests.

Recommended metrics:

- solver calls to first improvement,
- time to first improvement,
- Precision@k,
- best objective improvement under a fixed candidate budget,
- COI coverage of coupled improvements,
- release classification distribution.

## Notes

- Z3 UNSAT cores are not unique. Compare group/name patterns rather than exact ordering.
- A release that improves the objective is not necessarily a legally valid recommendation.
- `semantic_change` means the released constraint changes the case definition.
- `unsafe_needs_domain_bound` means the solver found improvement, but the field requires factual evidence or explicit bounds before it can be used as advice.
- For user-facing reports, release candidates should be passed through the release-advice layer before being shown as suggestions.
