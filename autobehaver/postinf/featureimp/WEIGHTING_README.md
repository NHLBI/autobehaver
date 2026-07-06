# FeatureImp: subset-weighted ensemble + weighted-excess SHAP

This FeatureImp pipeline trains an ensemble of **specialist** XGBoost models on many random feature subsets.

The committee supports a subset-level weighting scheme that is designed to:

1) reduce the influence of “uninformative” random feature subsets (near-random OOF performance)
2) keep the original **equitable** philosophy for feature importance (a feature only “earns” SHAP when it was present)
3) provide an optional **permutation-derived noise floor** so global importance reflects *excess* information above null.

This document explains how those pieces fit together.

---

## Terminology

- **Subset**: one random feature subset within a given *h-hit* round. Identified by `(h_hit, subset_idx)`.
- **Fold-model**: one trained model for a subset and a CV fold. Identified by `(h_hit, subset_idx, fold_idx)`.
- **OOF**: out-of-fold predictions/SHAP for training rows only (leakage-free).

Training convention:
- `training_group_a` is always encoded as **class 0**
- `training_group_b` is always encoded as **class 1**

---

## Step 1: subset OOF “skill” table

After committee training, the pipeline writes per-subset OOF metrics:

- `COMMITTEE_SUBSETS/SUBSET_PERFORMANCE_REAL.csv`

Each row is one subset and includes (among other fields):

- `oof_logloss`
- `baseline_logloss_full` (fold-aware constant baseline)
- `delta_logloss = baseline - oof`
- `delta_norm = delta_logloss / baseline` (normalized improvement)

Permutation runs (if enabled) also write:

- `PERMUTED_RESULTS/COMMITTEE_SUBSETS/SUBSET_PERFORMANCE_PERMUTED_ALL.csv`

This file pools all permutations (and includes `permutation_idx`).

---

## Step 2: subset weights (tau + gamma)

The pipeline computes weights from the *real* subset skill values.

### Tau (dead-zone)
If permutation subset performance exists, a dead-zone threshold is computed from the permuted null:

- `tau = max(0, quantile(permuted_delta_norm, subset_weight_tau_quantile))`

If no permutation performance exists, `tau = 0`.

### Weight function
For each subset:

- `w = max(0, delta_norm - tau)^gamma`

Outputs:
- `COMMITTEE_SUBSETS/SUBSET_WEIGHTS_REAL.csv`
- `COMMITTEE_SUBSETS/WEIGHTING_MANIFEST.json`

Key user parameters:
- `subset_weight_tau_quantile` (default `0.95`)
- `subset_weight_gamma` (default `2.0`)

---

## Step 3: predictions are aggregated in **logit** space

Instead of averaging probabilities across models, FeatureImp aggregates the **raw margins / logits** and then applies the sigmoid.

This is the more principled way to combine probabilistic classifiers.

Predictions workbook:

- `AVERAGED_PREDICTIONS_FULL.xlsx`

Tabs:
- `WeightedPredictions` (canonical)
- `UnweightedPredictions` (comparison)
- `WeightingDiagnostics` (tau, gamma, nonzero weights, etc.)

The canonical score column remains `Prob_<Class>` within each tab.

---

## Step 4: weighted SHAP + weighted-excess global importance

SHAP aggregation is computed using the **same subset weights** as predictions.

### Weighted equitable SHAP (canonical)
A feature is only updated when present in the subset, and denominators are feature-specific:

- Numerator: sum over subsets that contain the feature
- Denominator: sum of weights over those same subsets

This preserves the “equitable assessment” idea while down-weighting weak subsets.

### Global importance variants
FeatureImp outputs multiple global importance columns:

- `Global_Importance_Unweighted`  (unweighted comparison)
- `Global_Importance` (canonical) = `max(0, WeightedRaw - PermBaseline)`

The canonical ranking is written to:

- `GLOBAL_IMPORTANCE_RANKING.csv`

---

## Step 5: output clarity + manifests

The consolidated SHAP workbook:

- `SHAP_CONSOLIDATED_SUMMARY.xlsx`

Tabs:
- `Weighted SHAP` (canonical)
- `Unweighted SHAP` (comparison)

Post-analysis stage manifests record which importance semantics were used.

---

## Built-in consistency checks

The FeatureImp workflow performs core consistency checks while writing the
merged outputs, including:

- subset weights are nonnegative; nonzero weight fraction is recorded
- prediction workbooks contain expected tabs
- global importance invariants are checked where the required columns exist:
  - `Global_Importance >= 0`
  - if baseline columns exist: `Global_Importance == max(0, WeightedRaw - PermBaseline)`

Summary-validation errors can be made strict with `VALIDATION_ERROR_ON_FAIL`.
