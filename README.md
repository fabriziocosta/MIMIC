# MIMIC

MIMIC is a scikit-learn-style framework for feature-wise prediction, missing-value imputation, uncertainty diagnostics, and traceable synthetic tabular generation.

```python
from mimic import MIMIC, RandomForestPathEncoder, MixedFeatureDecoder

model = MIMIC(
    regression_columns=["age", "income"],
    classification_columns=["segment"],
    ignore_columns=["id"],
    encoder=RandomForestPathEncoder(n_estimators=50),
    decoder=MixedFeatureDecoder.random_forest(n_estimators=50),
    n_bootstrap=2,
    random_state=0,
)
model.fit(df)
imputed = model.impute(df)
confidence = model.confidence(df)
synthetic, trace = model.sample(5, return_trace=True)
```

See `WHITEPAPER.md`, `IMPLEMENTATION.md`, and the notebooks in `notebooks/` for details.

