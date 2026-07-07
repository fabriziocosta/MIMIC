# MIMIC

MIMIC stands for **Modular Inference for Missingness, Inconsistency, and
Creation**. It is a modular framework for working with mixed tabular data. It treats
imputation, consistency checking, supervised prediction, uncertainty estimation,
and synthetic data generation as variations of the same problem: predicting one
feature from the rest of the row.

The project is designed around a scikit-learn-style estimator with replaceable
encoders, decoders, and generation policies. The current implementation includes
tree-based, neural, linear, stochastic, and identity-style components so the same
interface can be used for practical modelling, diagnostics, and baselines.

## What MIMIC Does

- Fills missing values in numerical and categorical columns.
- Estimates confidence and uncertainty for feature-wise predictions.
- Treats labels as ordinary columns, allowing supervised prediction through the
  same imputation interface.
- Generates synthetic tabular rows from learned representations or baseline
  feature-space policies.
- Keeps generation traceable by exposing the steps used to create synthetic
  values.

## Core Idea

For each modelled column, MIMIC learns how to predict that column from the other
columns. These feature-wise models produce latent representations, predictions,
and uncertainty diagnostics. Because every column is handled through the same
abstraction, the framework can support missing-value repair, label prediction,
outlier-style inconsistency checks, and synthetic generation without separate
task-specific pipelines.

## Quick Example

```python
from mimic import mimic_data

synthetic = mimic_data(df, mode="factorised", capacity=0.25)
```

`mimic_data(df, mode="factorised", capacity=0.25)` fits a default `MIMIC` model internally and returns the same number
of synthetic rows. It is the simplest interface when you only need a sampled
dataframe.

The manuscript datasets can be loaded with matching MIMIC column roles in one
call:

```python
from mimic import load_paper_dataset, mimic_data

df, columns = load_paper_dataset("adult", n_rows=1000, random_state=0)
synthetic = mimic_data(df, columns=columns)
```

Available keys are `adult`, `bank_marketing`, and `default_credit`. The returned
`columns` mapping contains `ignore`, `regression`, and `classification` lists and
can be passed directly to `MIMIC` or `mimic_data`.

Nearest-neighbor ambiguity filtering is available as an opt-in attribution-risk
reduction step. It rejects generated embeddings that are too easily associated
with one training row or generation pair; it is not a formal differential
privacy guarantee.

```python
synthetic = mimic_data(df, privacy_filter=True)
```

Fitted models can be saved and loaded locally so generation can be repeated
without retraining:

```python
synthetic = mimic_data(df, save_model="adult_mimic.joblib")
more_synthetic = mimic_data(df, load_model="adult_mimic.joblib", n_samples=1000)
```

The estimator API exposes the same persistence path:

```python
model = MIMIC(mode="factorised", random_state=0).fit(df)
model.save("adult_mimic.joblib")

loaded = MIMIC.load("adult_mimic.joblib")
synthetic = loaded.sample(1000)
```

The same one-shot workflow is available from the command line:

```bash
mimic-data data.csv
```

This writes `data_mimic.csv` beside the input file. CSV, Parquet, and Excel
inputs are supported. Optional flags include `--columns`, `--mode`,
`--capacity`, `--save-model`, and `--load-model`; run `mimic-data --help` for
details.

Use the estimator directly when you want imputation, confidence diagnostics,
traceability, or reusable fitted state:

```python
from mimic import MIMIC

model = MIMIC(
    columns="auto",
    random_state=0,
)

model.fit(df)

imputed = model.impute(df)
confidence = model.confidence(df)
synthetic = model.sample(100)
```

`columns="auto"` infers roles using simple heuristics: ID-like or near-unique columns are ignored, small integer-valued numeric columns are classification, other numeric columns are regression, and non-numeric columns are classification. For production use, pass an explicit role mapping such as `columns={"ignore": [...], "regression": [...], "classification": [...]}`.

Embedding plots can focus on selected feature embeddings while colouring by raw
dataframe values:

```python
fig, axes = model.plot(df, embedding_columns=["age", "income"], color_by="segment")
```

Synthetic-data diagnostics include numeric pairwise plots and categorical
distribution comparisons:

```python
from mimic.diagnostics import categorical_feature_report, pairwise_feature_plot

pairwise_feature_plot(
    heldout,
    synthetic,
    features=["age", "hours-per-week", "capital-gain", "capital-loss"],
    log1p_features=["capital-gain", "capital-loss"],
)
category_report, category_summary = categorical_feature_report(
    heldout,
    synthetic,
    features=["workclass", "education", "income"],
)
```

`MIMIC` also provides simplified `mode` and `capacity` presets. The default
`mode="factorised", capacity=0.25` uses neural components, factorised probabilistic decoding, `n_bootstrap=2`, and a
mutual-neighbour displacement generation policy:

```python
GenerationPolicy(
    method="displacement",
    neighbour_mode="mutual",
    n_neighbors=5,
    lambda_range=(0.25, 0.75),
)
```

### `mode="identity"` / `mode=0`

Uses the identity encoder and identity decoder. Generation operates directly in
the preprocessed feature space, so this mode is useful as a transparent baseline
for checking what MIMIC adds beyond SMOTE-style interpolation or displacement.

### `mode="direct"` / `mode=1`

Uses the neural encoder preset and deterministic decoding. Synthetic embeddings
are decoded directly into feature values without stochastic conditional
resampling, making outputs easier to reproduce and traces simpler.

### `mode="factorised"` / `mode=2`

Uses the neural encoder preset with probabilistic factorised decoding. After an
initial deterministic decode, feature values are sampled from feature-wise
conditional decoders given the rest of the embedding. This is the default mode
because it keeps generation stochastic while preserving the modular
feature-wise design.

### `mode="joint"` / `mode=3`

Uses the neural encoder preset with deterministic joint decoding. A shared
neural row decoder predicts all modelled columns from concatenated conditional
evidence. This mode must be requested explicitly and currently requires the
neural conditional sampler.

`capacity` is a number from `0` to `1` that scales preset hyperparameters:
embedding dimension, hidden dimension, layer count, epochs, patience, batch
size, MDN components, bootstrap count, dropout, learning rate, and weight decay.
`capacity=0` is the smallest useful preset; `capacity=1` is the largest preset.
Learning rate is scaled downward in log space as capacity increases.

Explicit `encoder`, `decoder`, `policy`, `n_bootstrap`, or
`generation_decode_mode` arguments override the preset where supplied.

Generation uses one encoder-decoder member per feature trained on all observed
training rows for that feature. Bootstrap members are kept separate and are used
for confidence diagnostics and out-of-bag calibration, not for defining the
neighbour-search geometry used by `sample()`. Set `bootstrap=False` for
generation-only runs to skip fitting bootstrap members while still fitting the
full-data generation member:

```python
model = MIMIC(mode="factorised", capacity=1.0, bootstrap=False, random_state=0)
synthetic = model.fit(df).sample(1000)
```

Parallelism has two layers. `n_jobs` is passed to compatible underlying
estimators such as random forests, while `feature_n_jobs` parallelizes MIMIC's
feature-wise module fitting across target columns. Keep only one outer layer
high at a time: for example, when cross-validation folds are already parallel,
leave `feature_n_jobs=1`; when fitting one model at a time, increase
`feature_n_jobs` to train target features concurrently.

Set `verbose=True` to print constructor hyperparameters immediately and fitted
data/embedding sizes during `fit`.

Calibration is opt-in. When enabled, MIMIC uses out-of-bag bootstrap predictions
from `fit()` to calibrate confidence outputs. Calibration therefore requires
bootstrap members; leave `bootstrap=True` when using these diagnostics:

```python
model = MIMIC(
    classification_calibration="temperature",  # or "isotonic"
    regression_calibration="conformal",
)
model.fit(df)
confidence = model.confidence(df)
calibration = model.calibration_report()
```

Classification calibration adjusts reported probabilities. Regression conformal
calibration adds interval columns such as `lower_90` and `upper_90` to
`confidence()`.

Synthetic generation can choose how generated embeddings are decoded back to
rows with `generation_decode_mode`:

- `"auto"` preserves the default behaviour, using direct deterministic decoding
  for ordinary decoders and factorised stochastic decoding for conditional
  samplers.
- `"direct"` uses feature-wise point predictions from the generated embedding.
- `"factorised"` samples each non-conditioned feature from a fitted conditional
  sampler such as `ForestConditionalSampler` or `NeuralConditionalSampler`.
- `"joint"` uses `NeuralConditionalSampler` evidence from all feature-wise
  conditionals and a deterministic neural row decoder. It must be requested
  explicitly and trains on complete modelled rows.

## Installation

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

For notebook and test dependencies:

```bash
pip install -e ".[dev]"
```

## Project Structure

- `src/mimic/`: package source code.
- `tests/`: regression and behaviour tests.
- `notebooks/`: worked examples and operating-mode demonstrations.
- `WHITEPAPER.md`: conceptual overview and motivation.
- `IMPLEMENTATION.md`: implementation details and API design notes.

## Development

Run the test suite with:

```bash
pytest
```

The notebooks provide examples for basic usage, classification and regression,
generation, oversampling, traceability, and alternative encoder/decoder choices.

## Status

MIMIC is an experimental research-oriented codebase. The public API is compact,
but individual modelling components are still intended to be interchangeable as
the framework evolves.
