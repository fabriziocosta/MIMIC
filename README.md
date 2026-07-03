# MIMIC

MIMIC is a scikit-learn-style framework for feature-wise prediction, missing-value imputation, uncertainty diagnostics, and traceable synthetic tabular generation.

```python
from mimic import MIMIC, RandomForestPathEncoder, LinearMixedFeatureDecoder

model = MIMIC(
    regression_columns=["age", "income"],
    classification_columns=["segment"],
    ignore_columns=["id"],
    encoder=RandomForestPathEncoder(n_estimators=50),
    decoder=LinearMixedFeatureDecoder(),
    n_bootstrap=2,
    random_state=0,
)
model.fit(df)
imputed = model.impute(df)
confidence = model.confidence(df)
synthetic, trace = model.sample(5, return_trace=True)
```

See `WHITEPAPER.md`, `IMPLEMENTATION.md`, and the notebooks in `notebooks/` for details.

Use `MixedFeatureDecoder.random_forest(...)` for random-forest decoding, or
`LinearMixedFeatureDecoder(...)` / `MixedFeatureDecoder.linear()` for scikit-learn
linear regression plus logistic regression decoding.

Use `ForestConditionalSampler(...)` when generation should sample feature values
instead of taking deterministic decoder predictions. It fits random-forest
conditional samplers for each feature using `z_{-j}` embedding context, then
`MIMIC.sample(..., return_trace=True)` returns both embedding-generation trace
rows and per-cell Gibbs sampling trace rows.

Use `IdentityEncoder()` with `IdentityDecoder()` for baseline generation in the
preprocessed original feature space. With `GenerationPolicy(method="smote")`,
this recovers a classical SMOTE-style interpolation baseline; with
`method="displacement"`, it applies the same displacement idea without a learned
embedding.
