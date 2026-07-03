# MIMIC Implementation Plan

## 1. Goal

The first implementation of MIMIC should be a modular, scikit-learn-style estimator for tabular data. It should expose a small public API while keeping encoders, decoders, uncertainty estimation, and generation policies independently replaceable.

The core object is:

```python
class MIMIC(BaseEstimator, TransformerMixin):
    ...
```

It should support:

* `fit(X, y=None)`: learn feature-wise encoders, decoders, imputers, and generation state;
* `transform(X)`: return the learned embedding representation for each row;
* `impute(X, columns=None, return_confidence=False)`: fill missing values, including target columns treated as ordinary missing columns;
* `sample(n_samples, condition=None, return_trace=False)`: generate synthetic rows in embedding space and decode them;
* `confidence(X, columns=None)`: return uncertainty and bias-variance-style diagnostics for entries;
* `plot(X=None, color_by=None, center=None, random_state=None, ax=None)`: plot two classical-MDS views, one from the preprocessed original data and one from MIMIC embeddings.

The implementation should follow scikit-learn conventions: constructor arguments should be stored without side effects, learned attributes should end with `_`, and fitted state should be checked before inference.

## 2. Public API

### 2.1 Constructor

```python
MIMIC(
    ignore_columns=None,
    regression_columns=None,
    classification_columns=None,
    encoder=None,
    decoder=None,
    policy=None,
    n_bootstrap=2,
    random_state=None,
    n_jobs=None,
)
```

The column sets have explicit meanings:

* `ignore_columns`: columns excluded from modelling, for example IDs, timestamps used only as row identifiers, or free-text fields not yet supported;
* `regression_columns`: continuous columns decoded by regressors;
* `classification_columns`: categorical columns decoded by classifiers.

Every modelled column must belong to exactly one of `regression_columns` or `classification_columns`. Columns in `ignore_columns` are copied through when possible but are not used for fitting encoders, decoders, neighbours, confidence, or sample generation.

Feature-wise embedding dimensionality is specified by the encoder. Encoders with naturally fixed-width outputs, such as `ResNetEncoder`, should expose an `embedding_dim` parameter. Encoders with naturally sparse outputs, such as random-forest path encodings, should expose the native sparse dimensionality by default and may optionally use encoder-level SVD reduction to produce a fixed dense representation.

If column types are omitted, the initial implementation may infer a conservative default:

* numeric columns are treated as regression columns;
* non-numeric columns are treated as classification columns;
* no columns are ignored unless specified.

In production code, explicit column declarations should be preferred.

### 2.2 Encoder and decoder arguments

`encoder` should be a configured encoder object or encoder alias. `decoder` should be a generalized scikit-style estimator that can fit one target-specific model per feature type. It is not a single regressor or classifier. It is a mixed-feature decoder that learns how to map embeddings back to regression and classification columns separately.

Examples:

```python
MIMIC(
    regression_columns=["age", "income"],
    classification_columns=["diagnosis", "outcome"],
    ignore_columns=["patient_id"],
    encoder=RandomForestPathEncoder(n_estimators=200, embedding="path"),
    decoder=MixedFeatureDecoder(
        regression_estimator=RandomForestRegressor(n_estimators=200),
        classification_estimator=RandomForestClassifier(n_estimators=200),
    ),
    policy=GenerationPolicy(method="smote", neighbour_mode="mutual"),
)
```

The initial decoder helper constructors should be:

* `LinearMixedFeatureDecoder()` or `MixedFeatureDecoder.linear()`: linear regression for regression targets and logistic regression for classification targets;
* `MixedFeatureDecoder.random_forest()`: random forest regressor and random forest classifier;
* `ForestConditionalSampler()`: random-forest prediction plus stochastic conditional sampling for generation;
* `MixedFeatureDecoder(regression_estimator=..., classification_estimator=...)`: custom scikit estimators supplied by the user.

The important requirement is that the combined decoder owns the type dispatch. MIMIC should ask the decoder to fit, predict, or sample a named target column with a declared task, rather than choosing a decoder alias internally.

The initial encoder implementations should be:

* `RandomForestPathEncoder`;
* `ResNetEncoder`.

## 3. Fitted Structure

After `fit`, the object should contain:

```python
self.columns_
self.model_columns_
self.ignore_columns_
self.regression_columns_
self.classification_columns_
self.feature_modules_
self.input_preprocessor_
self.embedding_index_
self.train_embeddings_
self.train_index_
self.neighbour_index_
self.policy_
self.random_state_
```

`feature_modules_` maps each modelled column to a fitted module:

```python
{
    "column_name": FeatureModule(
        target_column="column_name",
        task="regression" | "classification",
        encoder_ensemble=[...],
        decoder_ensemble=[...],
        observed_mask=...,
        classes_=...,          # classification only
        target_statistics_=..., # regression and calibration metadata
    )
}
```

Each feature module is trained only on rows where its target column is not missing. For target column `j`, the training input is all modelled columns except `j`.

## 4. Fit Semantics

For each modelled column `j`:

1. Build a row mask selecting rows where `X[j]` is observed.
2. Build the context matrix `X_{-j}` by removing `j` and all ignored columns.
3. Fit the encoder ensemble on `X_{-j}`.
4. Transform `X_{-j}` through each encoder to obtain embeddings.
5. Fit the decoder ensemble to predict `X[j]` from the embeddings.
6. Store metadata needed for decoding, uncertainty, and later generation.

Bootstrapping is used to support uncertainty:

```text
for b in range(n_bootstrap):
    sample observed rows with replacement
    fit encoder_b on X_{-j}^{(b)}
    fit decoder_b on E_b(X_{-j}^{(b)}) -> x_j^{(b)}
```

The initial implementation can fit one encoder-decoder pair per bootstrap member and combine their predictions at inference time.

## 5. Transform Semantics

`transform(X)` returns the global embedding representation `H(X)`.

For each modelled column `j`, MIMIC computes the feature-wise embedding:

$$
h_j = E_j(X_{-j})
$$

The global row representation is the concatenation of the feature-wise embeddings:

$$
H(X) = [h_1, h_2, ..., h_p]
$$

MIMIC uses concatenation because it is simple, inspectable, and compatible with nearest-neighbour search.

For ensemble encoders, `transform` should average the embeddings across bootstrap members:

$$
h_j = \frac{1}{B}\sum_{b=1}^{B}E_j^{(b)}(X_{-j})
$$

This gives one deterministic feature-wise embedding per target column while still using the full encoder ensemble.

## 6. Imputation

`impute(X, columns=None, return_confidence=False)` fills missing values in selected columns. If `columns` is omitted, all missing modelled columns are imputed.

For a target column `j`, MIMIC computes:

$$
\hat{x}_j^{(b)} = D_j^{(b)}(E_j^{(b)}(X_{-j}))
$$

The default behaviour is **single-pass masked imputation**. If several columns are missing in the same row, MIMIC does not first fill the other missing columns before predicting `j`. Instead, each feature module receives the available context values plus an explicit missingness mask for the context columns:

$$
\hat{x}_j^{(b)} = D_j^{(b)}(E_j^{(b)}(X_{-j}, M_{-j}))
$$

where `M_{-j}` indicates which non-target context values are missing. Context preprocessors may use simple placeholder imputation internally so that encoders and decoders receive numeric arrays, but the missingness mask must remain available to the model. This makes each missing entry predictable in one pass without creating circular dependencies between missing columns.

This does not change the fitting rule for target values. The module for column `j` is trained only on rows where `X[j]` is observed. Missing values in context columns `X_{-j}` are allowed during training and inference, but they must be represented through context preprocessing and the missingness mask. In other words:

```text
exclude rows with missing target X[j]
allow rows with missing context X[-j]
encode missing context through placeholder values plus M[-j]
```

Regression output:

* point estimate: ensemble mean;
* uncertainty: ensemble variance plus optional residual variance.

Classification output:

* point estimate: class with highest mean predicted probability;
* uncertainty: entropy, class-probability variance, and ensemble disagreement.

Supervised prediction is represented by passing rows where the target column is missing. For example, to predict a label column named `"outcome"`, the caller supplies `X` with `X["outcome"] = NaN` and calls:

```python
X_hat = mimic.impute(X, columns=["outcome"])
```

This keeps target prediction and missing-value imputation as the same operation.

## 7. Confidence

`confidence(X, columns=None)` returns entry-level diagnostics. The intended return value is a tidy dataframe with one row per requested row-column pair:

```text
row_index
column
task
prediction
observed
bias
variance
residual
uncertainty
discrepancy
entropy
confidence
probabilities
predicted_probability
probability_variance
probability_std
probability_min
probability_max
probability_margin
vote_counts
vote_fraction
disagreement
top_class
second_class
noise
```

For regression columns:

$$
\mu_j = \frac{1}{B}\sum_{b=1}^{B}\hat{x}_j^{(b)}
$$

$$
\mathrm{variance}_j = \frac{1}{B-1}\sum_{b=1}^{B}(\hat{x}_j^{(b)} - \mu_j)^2
$$

If the observed value is available:

$$
\mathrm{residual}_j = x_j - \mu_j
$$

The practical bias term should be estimated from validation or out-of-bag residuals during fitting. The first implementation should store per-column out-of-bag mean error:

$$
\widehat{\mathrm{bias}}_j = \mathbb{E}_{\mathrm{oob}}[\hat{x}_j - x_j]
$$

The confidence method can then expose an approximate decomposition:

$$
\mathrm{error}_j^2 \approx \widehat{\mathrm{bias}}_j^2 + \mathrm{variance}_j + \widehat{\sigma}_{\epsilon,j}^2
$$

where `variance` comes from ensemble disagreement and `noise` is estimated from residual variance. This should be documented as an empirical diagnostic rather than a strict theoretical guarantee.

For classification columns:

MIMIC should call `predict_proba` on each bootstrap decoder:

$$
p_j^{(b)}(c) = D_j^{(b)}(H_{-j})_c
$$

Then it should aggregate the class-probability distributions:

$$
\bar{p}_j(c) = \frac{1}{B}\sum_{b=1}^{B}p_j^{(b)}(c)
$$

The predicted class is:

$$
\hat{x}_j = \arg\max_c \bar{p}_j(c)
$$

The primary confidence score is the mean probability assigned to the predicted class:

$$
\mathrm{confidence}_j = \max_c \bar{p}_j(c)
$$

The classification diagnostic suite should include:

* `probabilities`: the full mean class-probability dictionary `{class_label: mean_probability}`;
* `predicted_probability`: `max_c mean_probability`;
* `probability_variance`: bootstrap variance of `p_j^{(b)}(\hat{x}_j)`;
* `probability_std`: bootstrap standard deviation of `p_j^{(b)}(\hat{x}_j)`;
* `probability_min`: minimum bootstrap probability assigned to the predicted class;
* `probability_max`: maximum bootstrap probability assigned to the predicted class;
* `entropy`: entropy of the mean class-probability distribution;
* `probability_margin`: difference between the highest and second-highest mean class probabilities;
* `vote_counts`: hard-vote counts from the bootstrap decoders;
* `vote_fraction`: fraction of bootstrap decoders voting for the final predicted class;
* `disagreement`: `1 - vote_fraction`;
* `top_class`: class with highest mean probability;
* `second_class`: class with second-highest mean probability.

Entropy is computed from the mean probability distribution:

$$
\mathrm{entropy}_j = -\sum_c \bar{p}_j(c)\log \bar{p}_j(c)
$$

Probability variance measures bootstrap uncertainty for the final predicted class:

$$
\mathrm{probability\_variance}_j =
\frac{1}{B-1}\sum_{b=1}^{B}
\left(p_j^{(b)}(\hat{x}_j) - \bar{p}_j(\hat{x}_j)\right)^2
$$

Vote disagreement should be derived from hard predictions:

$$
\mathrm{disagreement}_j =
1 -
\frac{1}{B}
\sum_{b=1}^{B}
\mathbf{1}
\left[
\arg\max_c p_j^{(b)}(c) = \hat{x}_j
\right]
$$

These diagnostics separate three different ideas: average class confidence, probability uncertainty across bootstrap models, and hard-vote instability. Bootstrap does not replace `predict_proba`; it gives an additional uncertainty layer around the probability estimates.

## 8. Sampling

`sample(n_samples, condition=None, return_trace=False)` generates synthetic rows from the training embedding space.

The method should:

1. select anchor rows from the fitted training data;
2. select neighbours according to the configured policy;
3. generate synthetic embeddings;
4. decode embeddings back into feature values;
5. if the decoder supports stochastic sampling, run Gibbs-style conditional sampling over decoded rows;
6. optionally return trace records for embedding generation and per-cell sampling.

### 8.1 Generation policy

The policy object should be explicit:

```python
GenerationPolicy(
    method="smote" | "displacement",
    neighbour_mode="normal" | "mutual",
    n_neighbors=5,
    lambda_range=(0.0, 1.0),
    class_conditioned=False,
    cluster_conditioned=False,
)
```

`lambda_range` controls the sampled generation scale for both policies. For SMOTE-style interpolation, `lambda` determines the position between anchor `A` and neighbour `B`. For displacement generation, `lambda` determines how much of the local displacement from `B` to `C` is applied to anchor `A`.

`method="smote"` performs interpolation:

$$
H_{\mathrm{new}} = (1 - \lambda)H(A) + \lambda H(B)
$$

`method="displacement"` transfers a local displacement:

$$
H_{\mathrm{new}} = H(A) + \lambda(H(C) - H(B))
$$

`neighbour_mode="normal"` uses ordinary nearest neighbours.

`neighbour_mode="mutual"` requires the neighbour relation to hold in both directions:

$$
B \in \mathcal{N}_k(A)
\quad \text{and} \quad
A \in \mathcal{N}_k(B)
$$

For displacement, the minimal mutual-neighbour check should apply to `B` and `C`. A stricter later mode may also require `A` and `B` to be mutual neighbours.

### 8.2 Trace records

When `return_trace=True`, `sample` should return:

```python
synthetic_X, trace
```

`trace` should be a dataframe or list of dictionaries with one row per generated sample.

For SMOTE-style interpolation:

```python
{
    "sample_index": 0,
    "method": "smote",
    "anchor_index": A,
    "neighbour_index": B,
    "lambda": 0.37,
    "neighbour_mode": "mutual",
    "decoder": "MixedFeatureDecoder.random_forest",
    "random_state": 123,
}
```

For displacement:

```python
{
    "sample_index": 0,
    "method": "displacement",
    "anchor_index": A,
    "displacement_from_index": B,
    "displacement_to_index": C,
    "lambda": 0.5,
    "neighbour_mode": "normal",
    "restriction": "basic",
    "decoder": "MixedFeatureDecoder.linear",
    "random_state": 123,
}
```

Trace records are required for auditability. They do not by themselves guarantee privacy. Privacy checks should be a separate validation layer.

### 8.3 Forest conditional sampling

`ForestConditionalSampler` is an opt-in decoder for stochastic generation. It keeps the ordinary deterministic decoder interface, but also fits feature-wise conditional samplers.

For each target feature `j`, the sampler conditions on:

$$
z_{-j}
$$

which is the concatenated MIMIC embedding with the target feature block removed. This avoids conditioning a feature sampler on its own direct embedding block.

For categorical targets, the sampler uses a random-forest classifier:

$$
x_j^\star \sim \operatorname{Categorical}(\hat p_j(\cdot \mid z_{-j}))
$$

For continuous targets, the sampler uses a random-forest regressor as an adaptive neighbourhood model. If a query context reaches leaf `L_t(z_{-j})` in tree `t`, each training row sharing that leaf receives weight:

$$
w_{ij}(z_{-j})
=
\frac{1}{T}
\sum_{t=1}^{T}
\frac{\mathbf{1}\{z_{-j}^{(i)} \in L_t(z_{-j})\}}{|L_t(z_{-j})|}
$$

The sampled value is an exact observed training target:

$$
x_j^\star = x_j^{(i)}
$$

with probability:

$$
w_{ij}(z_{-j})
$$

Generation with this decoder proceeds in two stages:

1. create an initial synthetic embedding with SMOTE or displacement and decode it deterministically;
2. run three Gibbs refinement sweeps, sampling every non-conditioned feature from its `z_{-j}` forest conditional sampler.

If `condition={"label": "minority"}` is passed, matching training rows are used as anchors, neighbours are preferentially condition-matching, and conditioned output columns remain fixed during Gibbs sweeps.

When `return_trace=True`, the trace contains both `trace_type="embedding"` rows and `trace_type="cell"` rows. Cell trace rows include the sweep, target column, sampled value, conditioning type, target embedding slice, source row and weight for continuous features, or class probabilities for categorical features.

## 9. Plotting

`plot(X=None, color_by=None, center=None, random_state=None, ax=None)` should provide a diagnostic visualization of the learned representation.

The method should produce two side-by-side plots:

1. classical MDS in 2D on the original data after numeric preprocessing;
2. classical MDS in 2D on the MIMIC embedding returned by `transform(X)`.

The original-data panel must use a numeric representation before MDS. A pragmatic default is the same preprocessing discipline used internally:

* scale numeric columns, for example with `StandardScaler`;
* one-hot encode categorical columns, for example with `OneHotEncoder(handle_unknown="ignore")`;
* apply simple missing-value handling plus missingness indicators where needed;
* exclude `ignore_columns`.

The embedding panel should use the concatenated MIMIC embedding:

$$
H(X) = [h_1, h_2, ..., h_p]
$$

Both panels should use classical MDS from a distance matrix. The default distance is Euclidean distance on the numeric matrix used for that panel.

The `center` argument controls how the data are centred before MDS:

* `center=None`: centre using the data mean;
* `center="random"`: choose a random observed instance as the centre;
* `center=<row index>`: use a specific instance as the centre;
* `center=<array-like>`: use a supplied point in the relevant feature space.

Centres are useful because high-dimensional projections can produce fish-eye effects. Allowing a specific instance as the centre lets the user inspect local geometry around a row of interest instead of always centring on the global mean.

`color_by` may name any column in the input dataframe. If provided, both panels should use the same colour mapping. Numeric colour columns should use a continuous scale. Classification columns should use discrete colours. Missing colour values should use a distinct fallback colour.

The method should return matplotlib objects rather than only displaying the plot:

```python
fig, axes = mimic.plot(X, color_by="outcome", center="random")
```

If `X` is omitted, `plot` should use the fitted training data when available. If `ax` is supplied, it should accept either two axes or a container from which two axes can be derived.

## 10. Encoders

All encoders should follow a scikit-learn-style interface:

```python
encoder.fit(X, y=None)
H = encoder.transform(X)
```

For feature-wise modules, the encoder is trained on `X_{-j}` and may receive `y=X[j]` if it is supervised.

### 10.1 RandomForestPathEncoder

The random forest encoder should fit a random forest to predict the target column from the context columns.

For classification targets, the initial embedding should be a sparse encoding of node IDs traversed along each tree path:

```text
tree_0: node_0 -> node_4 -> node_9 -> leaf_12
tree_1: node_0 -> node_2 -> node_6 -> leaf_7
...
```

The transformed representation is a sparse binary vector where active entries indicate visited tree nodes. This captures more structure than leaf-only encodings because internal decision-path information is retained.

For regression targets, the initial embedding may use leaf IDs only:

```text
tree_0: leaf_12
tree_1: leaf_7
...
```

This gives a compact representation and can be extended later to path encodings if useful.

Suggested constructor:

```python
RandomForestPathEncoder(
    task="regression" | "classification",
    embedding_dim=None,
    n_estimators=100,
    max_depth=None,
    embedding="auto" | "path" | "leaf",
    svd_random_state=None,
    sparse=True,
    random_state=None,
    n_jobs=None,
)
```

`embedding_dim=None` means no dimensionality reduction is applied. The encoder returns the full native CSR sparse representation of the forest path or leaf encoding.

If `embedding_dim` is an integer, the encoder should fit a dedicated `TruncatedSVD` reducer on the sparse forest embedding and return a dense fixed-width representation with that many dimensions. This is useful when downstream decoders or neighbour search require dense vectors with stable dimensionality.

Implementation notes:

* use `RandomForestClassifier` for classification targets;
* use `RandomForestRegressor` for regression targets;
* use `decision_path(X)` for sparse path encodings;
* use `apply(X)` for leaf encodings;
* one-hot encode leaf IDs if a sparse numeric matrix is required downstream.
* if `embedding_dim=None`, return the full CSR sparse encoding.
* if `embedding_dim` is set, fit `TruncatedSVD(n_components=embedding_dim)` on the sparse encoding and return the dense SVD embedding.

### 10.2 ResNetEncoder

The PyTorch encoder should be a tabular residual network built from repeated blocks:

```text
Linear -> BatchNorm -> ReLU -> Dropout
```

Every two linear layers should have a residual connection:

$$
z_{l+2} = F_{l+2}(F_{l+1}(z_l)) + z_l
$$

If dimensions change, the residual branch should use a projection:

$$
z_{l+2} = F_{l+2}(F_{l+1}(z_l)) + P(z_l)
$$

Suggested constructor:

```python
ResNetEncoder(
    task="regression" | "classification",
    embedding_dim=64,
    hidden_dim=128,
    n_layers=4,
    dropout=0.1,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    max_epochs=100,
    patience=10,
    validation_fraction=0.1,
    restore_best_checkpoint=True,
    random_state=None,
    device="auto",
)
```

The encoder should train with a temporary prediction head for the target column. After fitting, `transform(X)` should return the penultimate embedding of dimension `embedding_dim`.

Training should use a validation split when `validation_fraction > 0`. The encoder should track validation loss, stop early after `patience` epochs without improvement, and restore the best checkpoint when `restore_best_checkpoint=True`. `weight_decay` should be passed to the optimizer to regularise the linear layers. Batch normalization should be part of each residual block unless a later implementation exposes it as an explicit option.

The first implementation should keep the PyTorch encoder optional so the package can run with scikit-learn-only dependencies.

## 11. Decoders

The decoder should be a generalized mixed-output scikit-style estimator. It receives embeddings and target metadata, then fits the correct target-specific model for each feature. This keeps classification and regression decoding behind a single interface while still allowing different estimator families for different feature types.

The initial implementation should provide:

```python
MixedFeatureDecoder(
    regression_estimator=None,
    classification_estimator=None,
)
ForestConditionalSampler(
    n_estimators=100,
    random_state=None,
    n_jobs=None,
    min_samples_leaf=1,
)
```

and convenience constructors:

```python
LinearMixedFeatureDecoder()
MixedFeatureDecoder.linear()
MixedFeatureDecoder.random_forest()
ForestConditionalSampler()
```

Internally, the combined decoder should clone the appropriate base estimator for each fitted target column:

```python
self.models_[column] = clone(self.regression_estimator)
self.models_[column] = clone(self.classification_estimator)
```

The public decoder contract should be:

```python
decoder.fit_target(column, task, H, y)
decoder.predict_target(column, H)
decoder.predict_proba_target(column, H)  # classification only
decoder.decode(H, columns=None)
decoder.can_sample_target(column)
decoder.fit_sampler_target(column, task, H_context, y, train_indices=None)
decoder.sample_target(column, task, H_context, rng, return_trace=False)
```

`fit_target` trains one target-specific model from embeddings to one feature. `decode` applies all fitted target models and returns a dataframe with one decoded column per target.

The stochastic sampler methods are optional. Deterministic decoders should return `False` from `can_sample_target`. `ForestConditionalSampler` should fit sampler targets on `z_{-j}` contexts after the full training embedding is available.

Regression target models must implement:

```python
fit(H, y)
predict(H)
```

Classification target models must implement:

```python
fit(H, y)
predict(H)
predict_proba(H)
```

The first implementation should require `predict_proba` for classification confidence. If a classifier lacks `predict_proba`, `MixedFeatureDecoder` should either wrap it with calibration or reject it with a clear error.

## 12. Data Handling

The implementation should accept pandas dataframes as the primary interface because column identity is central to MIMIC.

Minimum behaviour:

* preserve input column order;
* preserve ignored columns in `impute`;
* return dataframes from `impute` and `sample`;
* return sparse matrices from `transform` when encoders are sparse;
* store original training indices for trace records.

Internal preprocessing should handle:

* numeric scaling for linear and neural components;
* categorical encoding for context columns;
* missing context values during feature-wise training and inference.

A pragmatic first version can use:

* `SimpleImputer` for context missingness;
* `OneHotEncoder(handle_unknown="ignore")` for categorical context columns;
* `StandardScaler` for numeric context columns used by linear or neural models.

Important caveat: MIMIC may need both preprocessing and postprocessing layers around the encoder and decoder estimators. Many scikit-learn estimators require numeric arrays, scaled continuous inputs, or integer-coded targets. For example, regression features may need `StandardScaler`, categorical context variables may need one-hot encoding, and classification targets may need a `LabelEncoder` or equivalent class-index mapping. These transformations must be stored as fitted state and inverted where appropriate so that `impute` and `sample` return values in the original dataframe schema, not encoded internal representations.

This means the fitted feature module should own target-side preprocessing as well as context-side preprocessing. For a classification target, the module should store the mapping between original labels and internal class IDs. For a scaled regression target, it should inverse-transform predictions before returning them to the user. Confidence calculations may use internal numeric representations, but reported predictions, observed values, residuals, and trace metadata should refer back to original column names and user-facing values whenever possible.

The target column itself should not be imputed before fitting its module. Rows with missing target values are excluded for that module.

## 13. Error Handling and Validation

The estimator should validate:

* no overlap between ignored, regression, and classification columns;
* every declared column exists in `X`;
* every non-ignored column has a task;
* each modelled column has enough observed rows to fit;
* classification targets have at least two observed classes;
* `policy.method` and `policy.neighbour_mode` are valid;
* `sample` is called only after fitting and only when a neighbour index exists.

Clear errors matter because the framework has several moving parts. Validation should fail early during `fit`, not halfway through sample generation.

## 14. Initial Milestones

1. Implement data schema validation and column bookkeeping.
2. Implement `MixedFeatureDecoder` with linear, random forest, and custom estimator configurations.
3. Implement `RandomForestPathEncoder` with leaf and path embeddings.
4. Implement feature-wise module fitting with bootstrap ensembles.
5. Implement `impute` for regression and classification columns.
6. Implement `confidence` with ensemble variance, entropy, residuals, and out-of-bag bias estimates.
7. Implement global concatenated `transform`.
8. Implement neighbour indexing and SMOTE-style sampling.
9. Add displacement sampling and trace records.
10. Implement `plot` with two classical-MDS panels for preprocessed original data and MIMIC embeddings.
11. Add optional `ResNetEncoder`.

The implementation should start with correctness and inspectability. Performance optimisations, richer embedding combiners, calibration layers, and privacy filters can be added once the basic estimator contract is stable.
