# Offline Evaluation Report

- Project: `retail-churn-90d`
- Run (UTC): 2026-08-07T15:07:58.503544+00:00
- Label: `churn_90d` = no purchase within 90 days, scoped to customers active in the prior 180 days
- Events: 399,654 (2010-12-01 to 2011-12-09)
- Supervised rows: 15,478

## Temporal split

| fold   | snapshots                          |   n_rows |   n_customers |   churn_rate |
|:-------|:-----------------------------------|---------:|--------------:|-------------:|
| train  | 2011-03-31, 2011-04-30, 2011-05-31 |     7222 |          2716 |       0.4754 |
| valid  | 2011-06-30                         |     2722 |          2722 |       0.4838 |
| test   | 2011-08-31                         |     2768 |          2768 |       0.3931 |

## Validation fold

| model              |    n |   base_rate |   roc_auc |   pr_auc |   accuracy |   precision |   recall |     f1 |   brier |   log_loss |   precision_at_top_decile |   recall_at_top_decile |   lift_at_top_decile |   precision_at_top_quintile |   recall_at_top_quintile |   lift_at_top_quintile |
|:-------------------|-----:|------------:|----------:|---------:|-----------:|------------:|---------:|-------:|--------:|-----------:|--------------------------:|-----------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|
| trivial_prior      | 2722 |      0.4838 |    0.5    |   0.4838 |     0.5162 |      0      |   0      | 0      |  0.2498 |     0.6928 |                    0.326  |                 0.0676 |               0.6738 |                      0.345  |                   0.1427 |                 0.713  |
| heuristic_recency  | 2722 |      0.4838 |    0.6654 |   0.6406 |     0.6209 |      0.6843 |   0.4017 | 0.5062 |  0.2652 |     0.8309 |                    0.7473 |                 0.1549 |               1.5444 |                      0.7138 |                   0.2954 |                 1.4752 |
| baseline_logreg    | 2722 |      0.4838 |    0.7531 |   0.6944 |     0.6984 |      0.6782 |   0.7168 | 0.6969 |  0.1998 |     0.5807 |                    0.7399 |                 0.1534 |               1.5293 |                      0.7468 |                   0.309  |                 1.5435 |
| candidate_lightgbm | 2722 |      0.4838 |    0.7915 |   0.7306 |     0.7237 |      0.678  |   0.817  | 0.741  |  0.1835 |     0.5429 |                    0.8059 |                 0.167  |               1.6656 |                      0.778  |                   0.3219 |                 1.6079 |

## Test fold (untouched until now)

| model              |    n |   base_rate |   roc_auc |   pr_auc |   accuracy |   precision |   recall |     f1 |   brier |   log_loss |   precision_at_top_decile |   recall_at_top_decile |   lift_at_top_decile |   precision_at_top_quintile |   recall_at_top_quintile |   lift_at_top_quintile |
|:-------------------|-----:|------------:|----------:|---------:|-----------:|------------:|---------:|-------:|--------:|-----------:|--------------------------:|-----------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------:|
| trivial_prior      | 2768 |      0.3931 |    0.5    |   0.3931 |     0.6069 |      0      |   0      | 0      |  0.2453 |     0.6838 |                    0.2347 |                 0.0597 |               0.597  |                      0.278  |                   0.1415 |                 0.7072 |
| heuristic_recency  | 2768 |      0.3931 |    0.6571 |   0.5413 |     0.6441 |      0.5586 |   0.4513 | 0.4992 |  0.2462 |     0.7777 |                    0.6318 |                 0.1608 |               1.6073 |                      0.6137 |                   0.3125 |                 1.5614 |
| baseline_logreg    | 2768 |      0.3931 |    0.743  |   0.6009 |     0.6734 |      0.5685 |   0.7022 | 0.6283 |  0.203  |     0.5857 |                    0.6498 |                 0.1654 |               1.6532 |                      0.648  |                   0.33   |                 1.6486 |
| candidate_lightgbm | 2768 |      0.3931 |    0.7256 |   0.5623 |     0.6546 |      0.5437 |   0.7546 | 0.632  |  0.2111 |     0.6071 |                    0.6101 |                 0.1553 |               1.5522 |                      0.6065 |                   0.3088 |                 1.543  |

## Promotion guardrail

**Decision: PROMOTE**

| gate                         |   observed | operator   |   threshold | passed   | rationale                                                                                                                                                                                                    |
|:-----------------------------|-----------:|:-----------|------------:|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| roc_auc_floor                |   0.791527 | >=         |        0.72 | True     | Absolute ranking-quality floor; below this the campaign list is not reliably better than intuition.                                                                                                          |
| pr_auc_floor                 |   0.730556 | >=         |        0.55 | True     | Guards precision on the positive class, which drives campaign waste.                                                                                                                                         |
| calibration_brier_ceiling    |   0.183486 | <=         |        0.21 | True     | Probabilities are multiplied by margin to rank; poor calibration corrupts the expected-value ordering.                                                                                                       |
| recall_at_top_decile_floor   |   0.167046 | >=         |        0.15 | True     | The contactable slice must capture a meaningful share of churners.                                                                                                                                           |
| campaign_breakeven_precision |   0.777982 | >=         |        0.6  | True     | THE decisive gate. Break-even precision in the contacted slice is offer_cost / (uplift * margin) = 8 / 13.50 = 0.593. Below this the retention campaign destroys value, so no AUC improvement can rescue it. |
| no_regression_vs_production  |   0.038428 | >=         |       -0.01 | True     | A candidate may not be worse than the live model by more than the tolerance, even if the floors pass.                                                                                                        |

## Top features (permutation importance, validation AUC drop)

| feature                    |   auc_drop_mean |   auc_drop_std |
|:---------------------------|----------------:|---------------:|
| lifetime_revenue           |         0.03953 |        0.0036  |
| avg_interpurchase_gap_days |         0.03189 |        0.00205 |
| avg_unit_price_90d         |         0.01815 |        0.0031  |
| avg_items_per_order_90d    |         0.01579 |        0.00233 |
| product_breadth_ratio      |         0.0137  |        0.00144 |
| distinct_products_90d      |         0.01341 |        0.00134 |
| revenue_180d               |         0.01335 |        0.00217 |
| return_value_ratio_180d    |         0.01332 |        0.00184 |
| interpurchase_gap_cv       |         0.01271 |        0.00267 |
| revenue_90d                |         0.00867 |        0.00228 |
| active_days_ratio          |         0.00859 |        0.00087 |
| aov_90d                    |         0.00805 |        0.00093 |

## Expected campaign value on the test fold

|                   |   capacity_pct |   n_contacted |   true_churners_contacted |   expected_customers_saved |   gross_margin_gbp |   campaign_cost_gbp |   net_value_gbp |   net_value_random_targeting_gbp |   uplift_vs_random_gbp |     roi |
|:------------------|---------------:|--------------:|--------------------------:|---------------------------:|-------------------:|--------------------:|----------------:|---------------------------------:|-----------------------:|--------:|
| trivial_prior     |            0.2 |           554 |                       154 |                       46.2 |             2079   |                4432 |         -2353   |                         -1492.28 |                -860.72 | -0.5309 |
| heuristic_recency |            0.2 |           554 |                       340 |                      102   |             4590   |                4432 |           158   |                         -1492.28 |                1650.28 |  0.0356 |
| baseline_logreg   |            0.2 |           554 |                       359 |                      107.7 |             4846.5 |                4432 |           414.5 |                         -1492.28 |                1906.78 |  0.0935 |
| candidate_gbdt    |            0.2 |           554 |                       336 |                      100.8 |             4536   |                4432 |           104   |                         -1492.28 |                1596.28 |  0.0235 |

Production model: `retail-churn-90d:v6`
