# MIMIC — Modular Inference for Missingness, Inconsistency and Creation 

## A white paper for a general framework for missing-value prediction, data repair, uncertainty estimation, and synthetic instance generation

### 1. Executive summary

**MIMIC** is a modular machine-learning framework for treating several common tabular-data problems as variations of the same operation: predicting unknown feature values from known feature values.

The central idea is simple. For each feature column, MIMIC learns how that feature can be predicted from all the other features. This creates a family of feature-wise predictive models. Each model contains an encoder, which maps the available information into a latent representation, and a decoder, which predicts the target feature from the latent representations of the other features.

Once trained, the same framework can be used for four related tasks.

First, it can **impute missing values**. A missing entry is treated as an unknown target feature, and the model predicts it from the rest of the row.

Second, it can **detect suspicious values**. If the observed value of an entry disagrees strongly with the value predicted from the rest of the row, or if the ensemble variance around that prediction is high, the entry can be flagged for inspection.

Third, it can treat ordinary **supervised learning** as a special case of imputation. A target label is simply another column. Training rows contain the label; test rows have the label missing. Prediction is therefore equivalent to imputing the missing target column.

Fourth, MIMIC can act as a **synthetic data generator**. By moving through the learned embedding space, either by interpolation between neighbours or by local displacement vectors, the model can generate new plausible instances and decode them back into feature space.

MIMIC is not a single algorithm in the narrow sense. It is a framework. The encoders may be random forests, gradient-boosted trees, neural networks, ResNets, transformers, or other models. The decoders may be classifiers, regressors, density estimators, calibrated probabilistic predictors, or task-specific heads. The defining feature is the organisation of these components into a universal column-wise predictive system with uncertainty-aware imputation and manifold-aware generation.

### 2. Background and motivation

Missing data, noisy data, supervised prediction, class imbalance, and synthetic data generation are usually treated as separate machine-learning problems. In practice, however, they often reduce to the same question: given part of an instance, what should the remaining part be?

Classical imputation methods already exploit this idea. Multivariate Imputation by Chained Equations models each incomplete variable conditional on the other variables, using a sequence of regression models. Random-forest-based imputation methods such as missForest use non-parametric models to predict missing entries and can handle mixed-type data. Synthetic oversampling methods such as SMOTE generate new minority-class examples by interpolating between existing minority-class examples.

MIMIC extends this family of ideas in three directions.

First, it separates the **representation problem** from the **prediction problem**. Instead of only predicting each feature directly from raw feature values, MIMIC learns reusable latent representations for each feature-wise prediction task.

Second, it makes **uncertainty** a core output. By training bootstrapped ensembles of encoders and decoders, MIMIC can attach an uncertainty estimate to each imputed, corrected, or generated value.

Third, it treats **generation** as movement on a learned data manifold. Rather than generating synthetic instances only in raw feature space, MIMIC can generate in embedding space and decode the resulting latent point back into observable feature values.

### 3. Core intuition

Consider a dataset with rows as instances and columns as features. Some features may be numerical, some categorical, some ordinal, some binary, and some may be structured outputs.

For each feature column (x_j), MIMIC asks:

$$
x_j \approx f_j(x_{-j})
$$

where (x_{-j}) denotes all features except (x_j).

The model for (x_j) is trained using rows where (x_j) is observed. At prediction time, if (x_j) is missing, the model predicts it from the available values of the other columns.

The important addition is that MIMIC does not only produce a prediction. It also produces a representation:

$$
h_j = E_j(x_{-j})
$$

where (E_j) is the encoder associated with predicting feature (j). These representations can be concatenated, pooled, attended over, or otherwise combined into a global representation of the instance:

$$
H(x) = \mathrm{Combine}(h_1, h_2, ..., h_p)
$$

When predicting feature (j), the decoder should not use the representation of (j) itself. It should use the representations induced by the other features:

$$
\hat{x}_j = D_j(H_{-j})
$$

This exclusion is central. It prevents the decoder from trivially copying the feature being predicted, and it makes the framework suitable for missing-value prediction, label prediction, and consistency checking.

### 4. Architecture

MIMIC has four conceptual layers.

#### 4.1 Feature-wise predictive modules

For every feature (x_j), MIMIC learns a module that predicts (x_j) from all other features.

Each module contains:

1. an encoder (E_j), which maps the available context into an embedding;
2. a decoder (D_j), which predicts the value of (x_j);
3. an uncertainty estimator, usually obtained through an ensemble.

The encoder can be any model that exposes a useful internal representation. For a neural network, this may be the penultimate-layer embedding. For a tree ensemble, this may be the vector of leaf indices, path activations, node statistics, or learned proximity representation. For a transformer-style tabular model, it may be a contextual token representation.

#### 4.2 Global instance representation

The feature-wise embeddings are combined into a global representation of the row. The simplest version concatenates all embeddings. More sophisticated versions may use pooling, attention, masking, graph aggregation, or learned feature interaction modules.

The global representation should preserve two kinds of information:

* the value-level information needed to predict missing entries;
* the relational information needed to understand how features constrain each other.

This makes the representation useful not only for imputation, but also for inconsistency detection and generation.

#### 4.3 Target-specific decoders

For each feature (x_j), a target-specific decoder predicts (x_j) from the global representation excluding direct information from (x_j).

The decoder type depends on the target feature:

$$
D_j =
\begin{cases}
\text{regressor}, & \text{if } x_j \text{ is continuous} \\
\text{classifier}, & \text{if } x_j \text{ is categorical} \\
\text{ordinal model}, & \text{if } x_j \text{ is ordinal} \\
\text{structured decoder}, & \text{if } x_j \text{ is structured}
\end{cases}
$$

This allows MIMIC to handle heterogeneous tabular data without forcing every feature into the same output model.

#### 4.4 Bootstrap ensemble

MIMIC can train multiple versions of each feature-wise module on bootstrapped samples of the data. The ensemble produces a distribution of predictions:

$$
\hat{x}_{j}^{(1)}, \hat{x}_{j}^{(2)}, ..., \hat{x}_{j}^{(B)}
$$

From this distribution, MIMIC can compute uncertainty measures such as variance, entropy, disagreement, prediction intervals, or calibrated confidence scores.

This is particularly important because the model’s most useful output may not be the imputed value itself, but the confidence attached to that value.

### 5. Operating modes

MIMIC supports four main operating modes.

#### 5.1 Missing-value imputation

Given a row with a missing value in column (j), MIMIC predicts that value using all available non-(j) information.

The output is:

$$
(\hat{x}_j, u_j)
$$

where (\hat{x}_j) is the imputed value and (u_j) is an uncertainty score.

This allows the downstream user to decide whether to accept the imputation automatically, mark it as provisional, or route it to human review.

The method can also support iterative imputation. When multiple values are missing in the same row, MIMIC can impute the most confident missing entries first, update the row, and then proceed to less certain entries. This should be treated as a design option rather than a default guarantee, because iterative imputation can also propagate earlier mistakes if uncertainty is poorly calibrated.

#### 5.2 Data repair and inconsistency checking

MIMIC can also be used when values are present but potentially wrong.

For an observed entry (x_j), the model predicts what (x_j) should be from the other features:

$$
\hat{x}_j = D_j(H_{-j})
$$

The observed value can then be compared with the predicted distribution.

A suspicious entry is one where:

$$
\mathrm{Discrepancy}(x_j, \hat{x}_j) \text{ is high}
$$

or where the model assigns low probability to the observed value.

This turns MIMIC into a prioritisation tool for data cleaning. It does not prove that an entry is wrong. It identifies entries that are inconsistent with the rest of the dataset under the learned model.

This distinction is important. A high-variance or low-probability entry may be a data error, a rare but valid case, a distribution shift, or a limitation of the model. The correct use of MIMIC is therefore to rank entries for inspection, not to automatically overwrite unusual values.

#### 5.3 Supervised learning as target-column imputation

A supervised prediction problem can be represented as a missing-value problem.

Suppose the dataset contains a target column (y). For the training set, (y) is observed. For the test set, (y) is missing. MIMIC treats the target as another feature column:

$$
y = x_j
$$

Training is performed on rows where (y) is observed. Prediction is imputation on rows where (y) is missing.

This gives a unified view of supervised learning, semi-supervised learning, and missing-value prediction. It also allows the same uncertainty mechanism used for imputation to be used for label prediction.

#### 5.4 Synthetic data generation

MIMIC can generate new instances by operating in the learned embedding space.

There are two proposed policies.

#### Policy A: neighbour interpolation

Given an instance (A), compute its embedding (H(A)). Find a neighbour (B) in embedding space. Generate a new latent point by interpolation:

$$
H_{\mathrm{new}} = (1 - \lambda)H(A) + \lambda H(B)
$$

where (\lambda \in [0,1]).

The new embedding is then decoded into feature space:

$$
x_{\mathrm{new}} = D(H_{\mathrm{new}})
$$

This is analogous in spirit to SMOTE, which generates synthetic minority-class examples by interpolation, but MIMIC performs the operation in a learned representation space rather than necessarily in raw feature space.

#### Policy B: local displacement transfer

Given an instance (A), find a nearby instance (B). Then find a neighbour (C) of (B). Compute the local displacement:

$$
\Delta = H(C) - H(B)
$$

Apply this displacement to (A):

$$
H_{\mathrm{new}} = H(A) + \Delta
$$

Then decode:

$$
x_{\mathrm{new}} = D(H_{\mathrm{new}})
$$

The intuition is that the displacement (C - B) captures a locally valid direction of variation on the data manifold. Applying that displacement to (A) may generate a new point that remains close to the local structure of the data.

This is a hypothesis to be tested empirically. The method is attractive because it tries to generate variation by borrowing local transformations rather than by adding arbitrary noise.

#### 5.5 Addressable and traceable generation

A further design principle in MIMIC is that synthetic generation should be **addressable**. A generated instance should not appear as an anonymous output of the model. It should carry a provenance record that identifies the source training instances, the generation policy, the embedding space, the decoder, and the parameters that produced it.

This makes generation auditable, controllable, and scientifically interpretable. If a generated row looks unrealistic, the source instances can be inspected. If a user wants only conservative synthetic data, generated rows can be filtered according to source quality, class membership, local density, uncertainty, or neighbour relation. Rather than treating synthetic data as an opaque cloud of samples, MIMIC can expose the local transformation that produced each sample.

For neighbour interpolation, the provenance record is:

$$
\mathrm{Prov}(x_{\mathrm{new}}) =
(\text{mode}=\text{interpolation}, A, B, \lambda)
$$

where (A) and (B) are the source instances and (\lambda) gives the position of the generated point between them in embedding space. For example, (\lambda = 0.25) produces a point closer to (A), while (\lambda = 0.5) produces a midpoint between (A) and (B). This allows the decoded row to be inspected as a plausible intermediate case between two observed rows.

For local displacement, the provenance record is:

$$
\mathrm{Prov}(x_{\mathrm{new}}) =
(\text{mode}=\text{displacement}, A, B, C, \alpha)
$$

where (B) and (C) define the displacement direction and (A) is the anchor to which that displacement is applied:

$$
H_{\mathrm{new}} = H(A) + \alpha(H(C) - H(B))
$$

The parameter (\alpha) controls how much of the displacement is applied. A conservative implementation may restrict (\alpha \in [0,1]), while exploratory variants may allow a wider real-valued range. In either case, MIMIC is not simply sampling arbitrary noise around (A). It is applying a controlled fraction of a locally observed transformation.

Addressability also supports stricter neighbour policies. Interpolation and displacement may use ordinary nearest neighbours, or a stronger **mutual-neighbour** criterion:

$$
B \in \mathcal{N}_k(A)
\quad \text{and} \quad
A \in \mathcal{N}_k(B)
$$

For interpolation, this requires (A) and (B) to be mutual neighbours before MIMIC generates between them. For displacement, the minimal mutual-neighbour restriction requires (B) and (C) to be mutual neighbours because their difference vector defines the displacement. Stricter versions can also require compatibility between (A) and the region from which the displacement is borrowed, for example by requiring that (A), (B), and (C) share a class label, cluster assignment, density region, or local chart.

The generation record can therefore include fields such as:

$$
g =
{
\text{mode},
\text{source instances},
\text{generation parameters},
\text{neighbour type},
\text{restriction},
\text{embedding space},
\text{decoder}
}
$$

This record makes each generated row reproducible, auditable, and filterable. It should be distinguished from privacy protection. Provenance records explain how a synthetic point was generated; privacy checks determine whether the synthetic point is too close to a real point. High-stakes or scientific deployments should therefore combine generation provenance with nearest-neighbour distance checks, membership-inference tests, or minimum-distance filtering.

### 6. Minority-class generation

If one column represents a class label, MIMIC can condition generation on a class value.

For example, if class (c) is underrepresented, MIMIC can:

1. select instances from class (c);
2. generate new embeddings by interpolation or local displacement;
3. decode them into new synthetic feature vectors;
4. retain only synthetic instances predicted to belong to class (c) with sufficient confidence.

This gives MIMIC a role as a class-balancing generator. The generated examples should be evaluated carefully, because synthetic oversampling can improve apparent classifier performance while also introducing artefacts if the generator does not preserve the true data distribution.

### 7. Uncertainty model

Uncertainty is a first-class output of MIMIC.

For each prediction, the ensemble gives a set of candidate values. For regression, MIMIC can estimate predictive mean and variance:

$$
\mu_j = \frac{1}{B} \sum_{b=1}^{B} \hat{x}_j^{(b)}
$$

$$
\sigma_j^2 = \frac{1}{B-1} \sum_{b=1}^{B} (\hat{x}_j^{(b)} - \mu_j)^2
$$

For classification, MIMIC can aggregate predicted class probabilities:

$$
p_j(c) = \frac{1}{B} \sum_{b=1}^{B} p_j^{(b)}(c)
$$

and compute entropy:

$$
U_j = - \sum_c p_j(c) \log p_j(c)
$$

High variance or high entropy indicates that the model is uncertain. This uncertainty may arise because the row is underdetermined, because similar rows have different target values, because the instance is outside the training distribution, or because the model class is inadequate.

A useful distinction is between uncertainty caused by noise in the observations and uncertainty caused by lack of model knowledge. This distinction is often described as aleatoric versus epistemic uncertainty in Bayesian deep learning. In MIMIC, bootstrap variation mainly targets epistemic uncertainty, while residual error and calibrated predictive distributions may be needed to capture aleatoric uncertainty.

### 8. What makes MIMIC different?

MIMIC is best understood as a framework that unifies several operations that are often implemented separately.

Its distinctive features are:

* every feature can become a prediction target;
* every prediction target has its own decoder;
* the decoder for a target uses the embeddings of the other features, not the target itself;
* uncertainty is attached to each predicted entry;
* supervised prediction is treated as target-column imputation;
* synthetic data generation is performed in learned embedding space;
* local manifold displacement can be used as a generation policy;
* generated rows can carry explicit provenance records;
* data repair is framed as inconsistency between observed entries and model-implied entries.

MIMIC is therefore not merely an imputer. It is a general-purpose system for learning the conditional structure of a dataset.

### 9. Evaluation plan

MIMIC should be evaluated across four tasks.

#### 9.1 Imputation accuracy

Artificially mask known entries and measure how accurately MIMIC reconstructs them.

For numerical features, use RMSE, MAE, normalised RMSE, and calibration of prediction intervals.

For categorical features, use accuracy, macro-F1, log loss, and calibration error.

Baselines should include simple statistical imputation, chained-equation models, random-forest imputation, and task-specific neural imputation models. MICE and missForest are important reference points because they already implement feature-conditional imputation in established ways.

#### 9.2 Error-detection ability

Inject controlled corruptions into known entries and test whether MIMIC ranks corrupted entries above clean entries.

Relevant metrics include precision at (k), average precision, AUROC, and workload reduction. Workload reduction measures how many human checks are avoided by prioritising the most suspicious entries first.

#### 9.3 Supervised prediction

Treat the target label as a missing column and compare MIMIC against conventional supervised models trained directly on the same labelled rows.

The key question is not whether MIMIC always beats specialised supervised models. The key question is whether it performs competitively while also providing imputation, uncertainty, and data-repair functionality in the same framework.

#### 9.4 Synthetic data quality

Evaluate synthetic data using downstream task performance, distributional similarity, nearest-neighbour distance, privacy leakage tests, and class-balance utility.

For minority-class augmentation, compare against SMOTE and related oversampling methods. SMOTE is the natural baseline because it explicitly constructs synthetic minority-class examples through interpolation.

Synthetic data should also be evaluated through its provenance records. Useful diagnostics include the plausibility of decoded interpolations between source rows, the failure rate of displacement transfers, the effect of ordinary versus mutual-neighbour restrictions, and the relationship between generation uncertainty and downstream utility.

### 10. Key research questions

MIMIC raises several important research questions.

First, what is the best representation for tree-based encoders? Neural networks expose natural hidden-layer embeddings, but tree ensembles require a principled representation, such as leaf-index vectors, path indicators, proximity embeddings, or learned embeddings over tree paths.

Second, should the global representation be a simple concatenation, or should it use attention over feature embeddings? Concatenation is simple and transparent, but attention may better model conditional dependence between features.

Third, how should MIMIC handle rows with many missing values? The model may need masking-aware encoders, iterative refinement, or confidence-ordered imputation.

Fourth, how should uncertainty be calibrated? Raw ensemble variance is useful, but it may not be sufficient. Calibration methods may be needed before uncertainty scores are used for high-stakes decisions.

Fifth, how should generated instances be validated? A decoded synthetic row may look plausible locally but violate global constraints. MIMIC may therefore need constraint checking, density filtering, or human-in-the-loop validation.

Sixth, what provenance schema is sufficient for addressable generation? The record should be detailed enough to reproduce and audit each synthetic row, but compact enough to be practical for large synthetic datasets.

### 11. Conclusion

MIMIC proposes a unified view of tabular learning. Instead of separating missing-value imputation, supervised prediction, data cleaning, uncertainty estimation, class balancing, and synthetic data generation into unrelated pipelines, MIMIC treats them as different uses of the same learned conditional structure.

Each feature can be predicted from the rest. Each prediction can be accompanied by uncertainty. Each row can be embedded into a space where interpolation and local displacement generate new candidate instances. Each generated instance can carry an address that records where it came from and how it was produced. Each observed value can be checked against the value implied by the rest of the row.

The result is a flexible framework rather than a fixed model. Its practical value will depend on the quality of the encoders, the calibration of uncertainty, the reliability of decoding, and the validity of the embedding-space generation policies. But the conceptual advantage is clear: MIMIC turns imputation from a preprocessing step into a general modelling principle.
