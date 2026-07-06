# MIMIC Oversampling Experiment Plan

## Aim

Evaluate whether MIMIC's learned embedding-space generation produces better minority-class augmentation than standard feature-space oversampling methods, especially on mixed tabular data where categorical decoding, local geometry, traceability, and privacy-aware filtering matter.

The main comparison is not only "does MIMIC improve classifier scores?" but also whether MIMIC gives a better utility, fidelity, privacy, and provenance trade-off than conventional synthetic oversampling.

## Core Questions

1. Does MIMIC oversampling improve downstream minority-class performance compared with no resampling, random oversampling, SMOTE, and modern SMOTE variants?
2. Does generating in a learned embedding space improve synthetic sample plausibility compared with direct feature-space interpolation?
3. Does factorised or joint decoding improve mixed-type tabular fidelity compared with deterministic direct decoding?
4. Does local displacement generation outperform simple interpolation in MIMIC embedding space?
5. Does the nearest-neighbor ambiguity privacy filter reduce memorisation risk without destroying downstream utility?
6. Can trace records explain which real instances, neighborhoods, and decoder decisions created each synthetic row?

## Methods to Compare

### MIMIC variants

- `MIMIC-identity`: identity encoder/decoder generation in preprocessed feature space. This is the closest internal baseline to classical SMOTE-style interpolation.
- `MIMIC-direct`: learned encoder with deterministic feature-wise decoding.
- `MIMIC-factorised`: learned encoder with stochastic feature-wise conditional decoding.
- `MIMIC-joint`: learned encoder with joint row decoding, when computationally feasible.
- `MIMIC-factorised-private`: default factorised generation plus nearest-neighbor ambiguity filtering.
- `MIMIC-tangent/LOESS`: proposed extension. Estimate a local tangent plane in MIMIC embedding space, sample in tangent coordinates, optionally lift samples through a LOESS-style local curvature correction, then decode.

### Classical baselines

- No resampling.
- Random oversampling.
- Random undersampling.
- Class-weighted classifier without synthetic generation.
- SMOTE.
- SMOTE-NC for mixed numeric/categorical inputs.
- Borderline-SMOTE.
- ADASYN.

### Geometry and manifold baselines

- Geometric SMOTE.
- LoRAS.
- ProWSyn, if implementation is available.
- Manifold-distance based oversampling, if implementation is available.

### Optional synthetic-data baselines

These are not direct oversampling methods but are useful if the manuscript broadens to synthetic tabular generation:

- CTGAN.
- TVAE.
- Gaussian copula / classical tabular synthesizer.

## Comparator Papers

Use these papers to justify baselines and align the experimental design:

| Method | Paper | Why compare |
|---|---|---|
| SMOTE | Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique", JAIR, 2002. https://www.jair.org/index.php/jair/article/view/10302 | Natural baseline for interpolation-based minority synthesis. |
| Borderline-SMOTE | Han, Wang, and Mao, "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning", 2005. https://link.springer.com/chapter/10.1007/11538059_91 | Tests whether focusing generation near hard boundary examples beats MIMIC's neighborhood policies. |
| ADASYN | He et al., "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning", 2008. https://ieeexplore.ieee.org/document/4633969 | Tests adaptive allocation of synthetic points to hard minority examples. |
| Geometric SMOTE | Douzas and Bacao, "Geometric SMOTE: Effective oversampling for imbalanced learning through a geometric extension of SMOTE", 2017/2019. https://arxiv.org/abs/1709.07377 | Important because it replaces line-segment interpolation with geometric regions; close to the tangent-surface idea. |
| LoRAS | Bej et al., "LoRAS: an oversampling approach for imbalanced datasets", Machine Learning, 2021. https://link.springer.com/article/10.1007/s10994-020-05913-4 | Most relevant manifold-style comparator; it explicitly aims to oversample from an approximated local minority manifold. |
| ProWSyn | Barua, Islam, and Murase, "ProWSyn: Proximity Weighted Synthetic Oversampling Technique for Imbalanced Data Set Learning", 2013. https://link.springer.com/chapter/10.1007/978-3-642-37456-2_27 | Tests proximity-weighted allocation of synthetic points. |
| Manifold-distance oversampling | "Manifold Distance-Based Over-Sampling Technique for Class Imbalance Learning", AAAI, 2019. https://ojs.aaai.org/index.php/AAAI/article/view/5172 | Directly relevant to the claim that local manifold geometry matters. |

## Datasets

### First-pass datasets

Use these for the first reproducible experiment because they fit the current notebooks and package state:

- Adult / Census Income from OpenML.
  - Target: income.
  - Minority class: `>50K`.
  - Why: mixed categorical/numeric, already used in `notebooks/05_simplified_adult_sampling.ipynb`.
- Synthetic mixed tabular benchmark.
  - Target: controlled rare class.
  - Why: allows controlled imbalance ratio, class overlap, categorical cardinality, missingness, and nonlinear feature interactions.

### Paper-aligned tabular datasets

Add datasets used repeatedly in SMOTE/LoRAS/imbalanced-learning papers where available through OpenML, UCI, KEEL, or imbalanced-learn examples:

- Pima Indians Diabetes.
- Breast Cancer Wisconsin.
- Glass.
- Vehicle.
- Ecoli.
- Yeast.
- Abalone.
- Wine Quality.
- Page Blocks.
- Satimage.
- Letter / Optdigits, if a higher-dimensional numeric benchmark is needed.

### Clinical / MIMIC-style datasets

Only add these after the first-pass protocol is stable:

- MIMIC-derived cohort with a binary rare outcome.
- Sepsis, mortality, readmission, or prolonged length-of-stay prediction task.
- A de-identified EHR tabular subset with mixed labs, demographics, diagnoses, interventions, and outcome labels.

These datasets are the strongest motivation for MIMIC, but they add governance, preprocessing, leakage, and cohort-definition work. They should not block the baseline method comparison.

## Experimental Design

For each dataset:

1. Create stratified train/validation/test splits.
2. Keep the test split untouched and imbalanced.
3. Apply oversampling only to the training split.
4. Tune classifier hyperparameters on validation data.
5. Train the final classifier on oversampled training data.
6. Evaluate once on the untouched test split.
7. Repeat across multiple random seeds.

Use several imbalance ratios when possible:

- Natural class imbalance.
- Moderate imbalance, approximately 1:5.
- Severe imbalance, approximately 1:20.
- Extreme imbalance, approximately 1:50, only where enough minority examples remain.

## Classifiers

Use classifier families that expose different sensitivity to synthetic samples:

- Logistic regression.
- Random forest.
- Gradient boosting / HistGradientBoosting.
- XGBoost or LightGBM, if available.
- Shallow MLP for numeric/encoded tabular data.

The main tables should report averages across classifiers and also show per-classifier results in an appendix.

## Metrics

### Downstream utility

- Minority F1.
- Macro-F1.
- Balanced accuracy.
- AUROC.
- AUPRC.
- Sensitivity/recall at fixed specificity.
- Matthews correlation coefficient.

### Synthetic fidelity

- Numeric distribution distance: KS statistic and Wasserstein distance.
- Categorical distribution distance: total variation distance and Jensen-Shannon distance.
- Pairwise dependence preservation: correlation difference for numeric pairs, Cramer's V difference for categorical pairs.
- Class-conditional fidelity: compute the same metrics within the generated minority class.

### Boundary behavior

- Distance from generated samples to nearest minority neighbor.
- Distance from generated samples to nearest majority neighbor.
- Fraction of generated samples whose nearest real neighbor is majority class.
- Classifier disagreement or entropy around generated samples.
- Local density ratio before and after oversampling.

### Privacy and memorisation

- Nearest-neighbor distance to training rows.
- Generated-to-source distance distribution.
- Membership-inference proxy using nearest-neighbor distinguishability.
- Source-pair attribution rate for traceable MIMIC samples.
- Effect of nearest-neighbor ambiguity filtering on utility and fidelity.

### Traceability

- Fraction of generated samples with complete trace records.
- Number of unique anchors and neighbors used.
- Concentration of synthetic samples per anchor.
- Relationship between trace uncertainty and downstream usefulness.
- Examples of rejected or suspicious generations with trace explanations.

## MIMIC Ablations

- Encoder: identity vs neural vs random forest embedding.
- Policy: interpolation vs displacement vs tangent/LOESS.
- Neighbor rule: ordinary kNN vs mutual neighbors.
- Decode mode: direct vs factorised vs joint.
- Capacity: low, medium, high.
- Bootstrap count: no ensemble vs small ensemble.
- Privacy filter: off vs on.
- Conditioning: unconditional generation vs class-conditioned generation.

## Expected Claims to Test

Do not assume these are true; the experiments should be able to falsify them:

- MIMIC improves mixed-type synthetic fidelity because it decodes through learned feature-wise conditionals rather than interpolating encoded categories directly.
- MIMIC's displacement and tangent policies better preserve curved minority manifolds than line-segment SMOTE.
- Factorised stochastic decoding improves categorical diversity compared with deterministic decoding.
- Traceability makes synthetic-data failures easier to diagnose than opaque generative baselines.
- Privacy filtering reduces nearest-neighbor attribution risk with an acceptable utility cost.

## Minimum Viable Result Table

For the first manuscript draft, produce one table per dataset:

| Method | Minority F1 | Macro-F1 | Balanced accuracy | AUROC | AUPRC | NN privacy risk | Fidelity score |
|---|---:|---:|---:|---:|---:|---:|---:|
| No resampling | | | | | | | |
| Random oversampling | | | | | | | |
| SMOTE / SMOTE-NC | | | | | | | |
| Borderline-SMOTE | | | | | | | |
| ADASYN | | | | | | | |
| Geometric SMOTE | | | | | | | |
| LoRAS | | | | | | | |
| MIMIC-identity | | | | | | | |
| MIMIC-direct | | | | | | | |
| MIMIC-factorised | | | | | | | |
| MIMIC-factorised-private | | | | | | | |

## Reproducibility Checklist

- Fix train/test splits and random seeds.
- Log package versions.
- Store raw and processed dataset hashes.
- Store oversampling ratio and number of generated rows.
- Save MIMIC trace records for every generated sample.
- Save classifier predictions and probabilities, not only aggregate scores.
- Use statistical tests or confidence intervals across repeated seeds.

## Implementation Notes

- Keep the first implementation in scripts or thin notebooks that call reusable experiment functions.
- Use `notebooks/05_simplified_adult_sampling.ipynb` as the initial Adult loading/reference workflow.
- Use `notebooks/03_generation_oversampling_traceability.ipynb` as the initial traceability and curved-manifold reference.
- Use `imblearn` for SMOTE, SMOTE-NC, Borderline-SMOTE, ADASYN, random oversampling, and random undersampling.
- Add optional dependencies for Geometric SMOTE and LoRAS only after the core baselines run.
- Store outputs under a separate ignored experiment-output directory, not in `manuscript/`.
