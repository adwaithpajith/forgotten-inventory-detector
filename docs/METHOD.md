# Method

## Inputs

A movement ledger `(sku, date, kind, qty)` with `kind ∈ {receipt, issue}`, plus an optional SKU master. Raw files are cleaned by `fid/schema.py` (column mapping, signed or sales-only quantities, same-day aggregation). If the master carries an on-hand quantity, an opening-balance receipt reconciles the ledger to it.

## Features (`fid/features.py`)

All computed from a daily SKU × day matrix over the last 760 days, in vectorised chunks:

- **Stock position per day** from the cumulative ledger. A day counts as *in stock* if the opening balance plus that day's receipts is positive.
- **Idle in-stock days:** in-stock days since the last pick (or since first receipt if never picked). Calendar days since the last pick are kept as well, for the naive baseline.
- **Gaps between pick days**, counted in in-stock days: mean, max, coefficient of variation, ADI and CV² (the Syntetos–Boylan demand pattern).
- **Rolling activity:** pick days and units in 30 / 90 / 180 / 365 days, receipts in the last 90.
- **Stock age and expiry:** FIFO age of the oldest unit, share of shelf life used, days to expiry, quantity already past shelf life, days of cover at the recent rate.
- **Erosion:** pick rate in the 180 days before the last pick relative to the 180 days before that, to see demand fading before it stops.
- **Seasonality:** 28-day blocks over two years. Year-on-year profile correlation, share of demand in the three busiest blocks, how quiet the same period was last year, and the expected return in the next 12 weeks.

## Is this silence unusual? (`fid/survival.py`)

For a SKU with pick rate λ estimated from its own history (in in-stock days), the chance of an idle spell of length *h* is `exp(−λh)` under a constant-rate model. With at least three gaps a log-normal fit of the gap distribution gives a second estimate. The reported `p_idle` is the **larger** of the two, so a silence is only called unusual when both models agree. A SKU that sells once a quarter is therefore not flagged for 40 quiet days.

## Decision order (`fid/classify.py`)

Applied to in-stock SKUs with at least `min_history_days` of history; first match wins.

1. **Obsolete**: stock that cannot be sold before shelf life ends at the current pace; or a successor SKU that is selling while this one is quiet; or a long unusual silence preceded by demand that had already fallen below `erosion_ratio_threshold` of its earlier rate.
2. **Seasonal**: strong yearly profile (`season_corr ≥ 0.5`, `season_top3 ≥ 0.5`), equally quiet at this point last year, and a current trough (last 90 days under half the yearly rate).
3. **Forgotten**: idle ≥ `dead_min_idle_days` in-stock days **and** `p_idle ≤ p_unusual`; or never picked for `never_moved_days` in-stock days.
4. **Temporarily inactive**: idle ≥ `watch_min_idle_days` and `p_idle ≤ p_ok`, but not yet dead.
5. **Slow-moving**: fewer than `active_pick_days_180` pick days in 180 days.
6. **Active**: everything else.

Defaults: `dead_min_idle_days=60`, `p_unusual=0.05`, `p_ok=0.10`, `watch_min_idle_days=14`, `never_moved_days=120`, `active_pick_days_180=12`, `min_history_days=90`. They are starting points, set from first principles; the app's sidebar exposes them and the backtest shows their effect on your own history.

## Risk score

`risk = 100 × (0.45·evidence + 0.25·age + 0.15·cover + 0.15·expiry)` where *evidence* = 1 − `p_idle` (damped ×0.3 for seasonal SKUs, 1 for never-picked), *age* = oldest stock age / 365 (capped), *cover* = days of cover / 730 (capped), *expiry* = share of shelf life used (1 if anything is past it). Forgotten and obsolete SKUs are floored at 60. `value_at_risk = capital_tied × risk / 100` orders the action list.

**Money:** capital tied = on hand × unit cost; holding cost = capital × annual rate; space = on hand × volume per unit. **Replenishment conflict** = a dead or watch SKU that received stock in the last 90 days.

## Simulator (`fid/simulate.py`)

No public dataset says which stock was forgotten, so ground truth is planted.

- **Demand archetypes:** active, seasonal (Gaussian season around a random peak), slow-moving, temporarily inactive (a pause that straddles the as-of date), forgotten (steady then an abrupt, permanent stop, or stock never picked), obsolete (linear fade over 180–300 days, some with a successor or a shelf life), new.
- **Supply side:** a reorder-point policy with random lead times and pack sizes, an exponentially weighted demand forecast (so re-ordering lags when demand dies, as real forecasts do), lost sales on stock-outs, and standing purchase orders that keep delivering to about a third of dead SKUs.
- **Hold-out:** 120 days after the as-of date are simulated and withheld, so flags can be checked against what happened next.
- **Real demand:** `make_warehouse_from_sales` takes real sales history, holds out its last days, plants forgotten / obsolete / paused behaviour on a random subset of well-selling SKUs by editing their demand, and leaves the rest untouched and unlabelled.

The *active* and *slow-moving* archetypes are kept apart (≥ 0.08 vs ≤ 0.035 pick events per day) because the boundary between them is a business definition, not a statistical one.

## Evaluation (`fid/evaluate.py`)

- **Planted truth:** six-class confusion matrix, per-class precision/recall, and dead-vs-alive scores against the naive "no pick in N days" rule at 90 days *and* at the engine's own idle threshold, so any gain from timing is separated from any gain from the statistics.
- **Outcome check:** share of flagged SKUs picked again in the held-out window.
- **Lead time:** days from a planted abrupt stop to the first flag, checked on a 15-day grid.
- **Label-free backtest:** classify as of *h* days ago and measure whether flagged SKUs stayed dormant. Needs no labels, so it runs on any real history, including inside the app.
- **Threshold sweep:** precision/recall as `p_unusual` and `dead_min_idle_days` move. Read with care: the optimum depends on the longest pause the simulator allows.

## Known limits

A long pause is indistinguishable from death until it ends; the engine escalates such SKUs for investigation. Counting pick *days* ignores order size. Seasonality needs ≈ 2 years. The ML second opinion is trained on simulated data only.
