# Streamlined MIMIC Synthetic Minority Evaluation Plan

## Objective

Build one reusable experimental framework for evaluating MIMIC-style synthetic minority-class generation on binary tabular classification tasks. The framework should compare direct-space and latent-space generation, and SMOTE-style interpolation and displacement generation, against a balanced real-data reference.

The goal is not a single global leaderboard. The experiment should produce regime-specific recommendations: which method works best under mild versus severe imbalance, small versus larger training sets, simple versus high-dimensional feature spaces, and easy versus difficult classification problems.

## Hypotheses

1. Displacement-based generation outperforms SMOTE-style interpolation in at least some low-data or high-imbalance regimes.
2. Generation in a learned MIMIC latent space outperforms generation directly in the preprocessed input feature space in at least some dataset regimes.
3. The benefit of latent generation and displacement generation is regime-dependent, so early, mid, and full learning-curve summaries should be reported separately.

## Methods

Evaluate five training-data construction methods under the same downstream classifier, preprocessing, split, and metric protocol.

| Method key | Description | Generation space | Generation rule |
|---|---|---|---|
| `real_balanced` | Balanced 50:50 real-data reference using only real examples from both classes. | none | none |
| `direct_smote` | Synthetic minority rows from SMOTE-style interpolation in preprocessed feature space. | input | interpolation |
| `direct_displacement` | Synthetic minority rows from displacement generation in preprocessed feature space. | input | `x_anchor + lambda * (x_to - x_from)` |
| `latent_smote` | Synthetic minority embeddings from SMOTE-style interpolation, decoded back to rows. | MIMIC embedding | interpolation |
| `latent_displacement` | Synthetic minority embeddings from displacement generation, decoded back to rows. | MIMIC embedding | displacement |

MIMIC must be fit jointly on all available training rows for a condition. It should not be trained separately per class. Class awareness happens only during sampling: anchors, neighbours, and displacement endpoints are restricted to the requested class, normally the minority class.

For these generation-focused experiments, configure MIMIC with `bootstrap=False` unless a condition explicitly studies confidence estimates. Generation should use the full-data feature embeddings; bootstrap members are for uncertainty and calibration, not for defining the synthetic-sampling geometry.

## Experimental Factors

Use a full factorial design, with a smaller smoke profile for development and a full profile for manuscript results.

Start with three large mixed-type binary datasets:

| Dataset key | Source | Task | Why include it |
|---|---|---|---|
| `adult` | OpenML `adult`, version 2, or UCI Adult | Predict income `>50K` from census attributes. | Large mixed numerical/categorical benchmark with missing categorical values and moderate class imbalance. |
| `bank_marketing` | UCI Bank Marketing or OpenML full bank-marketing mirror | Predict term-deposit subscription. | Large mixed business dataset with many categorical campaign/client variables and a naturally imbalanced positive class. |
| `default_credit` | UCI Default of Credit Card Clients or OpenML mirror | Predict credit-card default. | Large credit-risk dataset with numerical payment/bill features plus categorical demographic/payment-status fields encoded as integers. |

These should be the initial full-profile datasets. Additional datasets can be added after the runner and analysis pipeline are stable.

| Factor | Smoke profile | Full profile |
|---|---|---|
| Datasets | one fast subset of `adult` and `bank_marketing` | `adult`, `bank_marketing`, `default_credit` |
| Imbalance ratios | `2:1`, `5:1` | `2:1`, `3:1`, `5:1`, `10:1` |
| Training sizes | 2-3 sizes | enough sizes for stable learning curves |
| Seeds | 2 | 10 or more |
| Methods | all five | all five |

Training sizes should be expressed as total pre-balancing training rows where possible. Each condition then creates the requested majority-to-minority ratio from the stratified training pool.

## Data Flow Per Condition

For each dataset, imbalance ratio, training size, and seed:

1. Load the dataset through a registry entry with no dataset-specific logic in the experiment runner.
2. Split into train and test sets using stratification.
3. Fit preprocessing on the training split only.
4. Build a fixed imbalanced training subset from the training split.
5. For `real_balanced`, construct a 50:50 real-only training set.
6. For synthetic methods, keep the real minority examples fixed and generate enough synthetic minority rows to create a 50:50 training set.
7. Fit the same downstream classifier for every method.
8. Evaluate on the unchanged held-out test set.
9. Save raw per-condition metrics and enough metadata to reproduce the run.

The test split must never be used for preprocessing, MIMIC fitting, neighbour search, generation, classifier fitting, or model selection.

## Metrics

Record these per condition:

- ROC-AUC as the primary metric.
- PR-AUC.
- Balanced accuracy.
- F1.
- Calibration metric when classifier probabilities are available, such as Brier score or expected calibration error.

For each dataset, imbalance ratio, method, and seed, compute learning-curve summaries from ROC-AUC:

```text
AULC = trapezoidal_area(training_size, roc_auc)
```

Report:

- `early_aulc`: small-data segment.
- `mid_aulc`: intermediate training-size segment.
- `full_aulc`: full learning curve.

The exact cutoffs should live in YAML so the smoke and full profiles can use different training-size grids without changing code.

## Statistical Analysis

Aggregate over seeds first, then compare methods at dataset and imbalance-ratio level.

Required pairwise comparisons:

- `latent_displacement` vs `latent_smote`.
- `direct_displacement` vs `direct_smote`.
- `latent_smote` vs `direct_smote`.
- `latent_displacement` vs `direct_displacement`.
- Each synthetic method vs `real_balanced`.

Required summaries:

- Pairwise effect sizes and confidence intervals for AULC segments.
- Significance or equivalence decisions for synthetic methods versus `real_balanced`.
- Critical difference diagrams ranking methods by `full_aulc` and `early_aulc`.
- Separate critical difference diagrams by imbalance ratio.

For the real-data comparison, report where each synthetic method is statistically indistinguishable from `real_balanced`, and the most severe imbalance ratio where that remains true.

## Dataset Metadata

Compute metadata for every dataset and attach it to result summaries:

- Number of samples.
- Number of raw features.
- Number of numerical features.
- Number of categorical features.
- Original imbalance ratio.
- Minority-class size.
- Preprocessed feature dimensionality.
- Missingness rate.
- Baseline classifier ROC-AUC as a separability or difficulty estimate.

Use these metadata to produce regime-level conclusions, not just average rankings.

## Outputs

All outputs must be reproducible from saved config and result files.

```text
manuscript/
  experiments/
    plan.md
    configs/
      smoke.yaml
      full.yaml
    src/
      streamlined/
        __init__.py
        config.py
        datasets.py
        preprocessing.py
        sampling.py
        runner.py
        metrics.py
        analysis.py
        plotting.py
    tests/
      test_streamlined_*.py
    notebooks/
      01_run_or_view_results.ipynb
    artifacts/
      raw/
      tables/
      figures/
      reports/
```

The streamlined experiment code is intentionally local to `manuscript/experiments/src/`. It should import the installed project package, but it should not add new experiment-specific modules under the repository-level `src/` tree.

Required generated artifacts:

- Raw per-condition CSV or Parquet results.
- Per-dataset learning curves, with one plot per imbalance ratio and one curve per method.
- Mean learning curves across datasets, grouped by imbalance ratio and method.
- AULC tables for early, mid, and full regimes.
- Critical difference diagrams for full and early AULC, overall and by imbalance ratio.
- Regime summary table with the best imbalance range, best data-size regime, indistinguishability from real data, latent-vs-direct result, and displacement-vs-SMOTE result for each method.
- Final prescriptive conclusions written as actionable rules.

## Implementation Phases

### Phase 1: Skeleton and Config

- Create YAML-driven experiment profiles.
- Define typed config objects and method identifiers.
- Define result table schemas.
- Keep imports local to `manuscript/experiments/src/streamlined`.
- Add tests for config loading and result schema validation.

### Phase 2: Dataset Registry and Preprocessing

- Implement a generic binary dataset registry.
- Return raw dataframe, target column, feature roles, and dataset metadata.
- Fit preprocessing inside each train split only.
- Add tests for registry shape, feature role handling, and leakage boundaries.

### Phase 3: Sampling Backends

- Implement direct-space SMOTE and direct-space displacement over preprocessed training arrays.
- Repair direct-space generated samples by default before classifier fitting: each one-hot categorical block is projected back to a valid one-hot category by argmax. Allow this to be disabled with `repair_direct_samples: false` for raw preprocessed-space comparisons.
- Implement latent-space SMOTE and latent-space displacement through jointly fit MIMIC models.
- Ensure class-aware sampling restricts neighbours at generation time only.
- Add tests using synthetic toy datasets where generated sample counts, class labels, and neighbour restrictions are easy to verify.

### Phase 4: Runner

- Implement the condition runner for dataset, imbalance ratio, training size, method, and seed.
- Save raw condition-level outputs incrementally.
- Add a smoke command that can run in minutes.
- Add tests for fixed test-set reuse, class balance after generation, and reproducibility with fixed seeds.

### Phase 5: Analysis and Reporting

- Compute learning curves and segmented AULC.
- Implement pairwise comparisons, equivalence or indistinguishability summaries, and rank tables.
- Generate plots and report-ready tables from saved results only.
- Add tests for AULC calculations, segment handling, and method comparison table shape.

## Documentation Notes

The docs and notebook must explicitly state:

- MIMIC is trained jointly on all training rows in a condition.
- Class conditioning is applied only during synthetic generation.
- Preprocessing, generation, and classifiers are fit only on training data.
- The test set remains untouched until final evaluation.
- Conclusions should be regime-specific, for example:
  - Use latent displacement when the minority class is small and imbalance is at least `5:1`.
  - Use direct SMOTE when imbalance is mild and the feature space is simple.
  - Expect latent methods to help most in low-data regimes.
  - Expect displacement to help most when the learning curve is still steep.
