import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy import sparse

from mimic import (
    ForestConditionalSampler,
    GenerationPolicy,
    IdentityDecoder,
    IdentityEncoder,
    LinearMixedFeatureDecoder,
    MIMIC,
    MixedFeatureDecoder,
    NeuralConditionalSampler,
    RandomForestPathEncoder,
    ResNetEncoder,
)


def make_frame(n=80, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 10, n)
    income = 25_000 + age * 900 + rng.normal(0, 5_000, n)
    segment = np.where(age + rng.normal(0, 4, n) > 50, "older", "younger")
    outcome = np.where(income + rng.normal(0, 3_000, n) > np.median(income), "yes", "no")
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "age": age,
            "income": income,
            "segment": segment,
            "outcome": outcome,
        }
    )


def test_random_forest_encoder_sparse_and_svd():
    X = np.arange(40).reshape(20, 2)
    y = np.array(["a", "b"] * 10)
    enc = RandomForestPathEncoder(task="classification", n_estimators=5, random_state=0)
    Z = enc.fit(X, y).transform(X)
    assert sparse.isspmatrix_csr(Z)
    assert Z.shape[0] == 20

    dense = RandomForestPathEncoder(
        task="classification",
        n_estimators=5,
        embedding_dim=3,
        random_state=0,
    )
    Zd = dense.fit(X, y).transform(X)
    assert isinstance(Zd, np.ndarray)
    assert Zd.shape == (20, 3)


def test_mixed_feature_decoder_regression_and_classification():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(30, 4))
    decoder = MixedFeatureDecoder.random_forest(n_estimators=5, random_state=0)
    decoder.fit_target("x", "regression", H, H[:, 0])
    decoder.fit_target("y", "classification", H, np.array([0, 1] * 15))
    assert decoder.predict_target("x", H).shape == (30,)
    assert decoder.predict_target("y", H).shape == (30,)
    assert decoder.predict_proba_target("y", H).shape[0] == 30


def test_linear_mixed_feature_decoder_regression_and_classification():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(40, 5))
    y_reg = H[:, 0] - 2 * H[:, 1]
    y_cls = np.where(H[:, 2] > 0, "high", "low")
    decoder = LinearMixedFeatureDecoder(logistic_max_iter=500)
    decoder.fit_target("x", "regression", H, y_reg)
    decoder.fit_target("y", "classification", H, y_cls)

    assert decoder.predict_target("x", H).shape == (40,)
    assert set(decoder.predict_target("y", H)).issubset({"high", "low"})
    assert decoder.predict_proba_target("y", H).shape == (40, 2)
    assert isinstance(MixedFeatureDecoder.linear(), LinearMixedFeatureDecoder)


def test_identity_encoder_and_decoder_recover_direct_coordinates():
    rng = np.random.default_rng(12)
    H = rng.normal(size=(40, 5))
    y_reg = H[:, 2]
    y_cls = np.where(H[:, 4] > H[:, 3], 1, 0)

    encoder = IdentityEncoder()
    Z = encoder.fit(sparse.csr_matrix(H), y_reg).transform(sparse.csr_matrix(H))
    assert np.allclose(Z, H)

    decoder = IdentityDecoder()
    decoder.fit_target("x", "regression", H, y_reg)
    decoder.fit_target("label", "classification", H, y_cls)
    assert np.allclose(decoder.predict_target("x", H), y_reg)
    assert set(decoder.predict_target("label", H)).issubset({0, 1})
    assert decoder.predict_proba_target("label", H).shape == (40, 2)


def test_forest_conditional_sampler_samples_with_trace():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(50, 3))
    y_reg = np.linspace(-2.0, 2.0, 50)
    y_cls = np.array([0, 1] * 25)
    sampler = ForestConditionalSampler(n_estimators=10, random_state=0)
    sampler.fit_target("x", "regression", H, y_reg)
    sampler.fit_sampler_target("x", "regression", H, y_reg, train_indices=np.arange(100, 150))
    sampler.fit_target("c", "classification", H, y_cls)
    sampler.fit_sampler_target("c", "classification", H, y_cls)

    reg_values, reg_trace = sampler.sample_target("x", "regression", H[:5], rng, return_trace=True)
    cls_values, cls_trace = sampler.sample_target("c", "classification", H[:5], rng, return_trace=True)

    assert set(reg_values).issubset(set(y_reg))
    assert {"source_index", "source_weight", "leaf_support_size"}.issubset(reg_trace[0])
    assert set(cls_values).issubset({0, 1})
    assert "class_probabilities" in cls_trace[0]


def test_mdn_negative_log_likelihood_gradients_and_matching_component():
    import torch

    from mimic.decoders import mdn_negative_log_likelihood

    y = torch.tensor([0.0, 2.0], dtype=torch.float32)
    logits = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
    matching_mu = torch.tensor([[0.0, 5.0], [5.0, 2.0]], dtype=torch.float32, requires_grad=True)
    poor_mu = torch.tensor([[5.0, 6.0], [5.0, 6.0]], dtype=torch.float32, requires_grad=True)
    raw_sigma = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)

    matching_loss = mdn_negative_log_likelihood(logits, matching_mu, raw_sigma, y)
    poor_loss = mdn_negative_log_likelihood(logits.detach(), poor_mu, raw_sigma.detach(), y)
    matching_loss.backward()

    assert torch.isfinite(matching_loss)
    assert matching_loss < poor_loss
    assert logits.grad is not None
    assert matching_mu.grad is not None
    assert raw_sigma.grad is not None


def test_neural_conditional_sampler_regression_and_classification_trace():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(50, 3))
    y_reg = H[:, 0] - 0.5 * H[:, 1]
    y_cls = np.where(H[:, 2] > 0, 1, 0)
    sampler = NeuralConditionalSampler(
        n_components=3,
        hidden_dim=12,
        n_layers=1,
        max_epochs=6,
        patience=3,
        batch_size=16,
        random_state=0,
        device="cpu",
    )
    sampler.fit_target("x", "regression", H, y_reg)
    sampler.fit_sampler_target("x", "regression", H, y_reg)
    sampler.fit_target("c", "classification", H, y_cls)
    sampler.fit_sampler_target("c", "classification", H, y_cls)

    pred = sampler.predict_target("x", H[:5])
    reg_values, reg_trace = sampler.sample_target("x", "regression", H[:5], rng, return_trace=True)
    cls_values, cls_trace = sampler.sample_target("c", "classification", H[:5], rng, return_trace=True)

    assert pred.shape == (5,)
    assert reg_values.shape == (5,)
    assert {"component_index", "component_probability", "component_mean", "component_sigma", "sample_log_probability"}.issubset(reg_trace[0])
    assert set(cls_values).issubset({0, 1})
    assert "class_probabilities" in cls_trace[0]


def test_neural_conditional_sampler_mdn_samples_both_numeric_modes():
    rng = np.random.default_rng(1)
    H = np.zeros((80, 2), dtype=float)
    y = np.r_[np.full(40, -2.0), np.full(40, 2.0)]
    sampler = NeuralConditionalSampler(
        n_components=2,
        hidden_dim=16,
        n_layers=1,
        max_epochs=40,
        patience=8,
        batch_size=20,
        min_sigma=0.05,
        max_sigma=1.0,
        noise_scale=0.25,
        random_state=1,
        device="cpu",
    )
    sampler.fit_target("x", "regression", H, y)
    sampler.fit_sampler_target("x", "regression", H, y)

    samples = sampler.sample_target("x", "regression", np.zeros((120, 2)), rng)

    assert (samples < -0.75).sum() >= 10
    assert (samples > 0.75).sum() >= 10


def test_mimic_fit_transform_impute_confidence_sample_plot():
    df = make_frame()
    df_missing = df.copy()
    df_missing.loc[[0, 1, 2], "age"] = np.nan
    df_missing.loc[[3, 4], "outcome"] = np.nan
    df_missing.loc[[5], "segment"] = np.nan

    model = MIMIC(
        ignore_columns=["id"],
        regression_columns=["age", "income"],
        classification_columns=["segment", "outcome"],
        encoder=RandomForestPathEncoder(n_estimators=6, embedding_dim=4, random_state=0),
        decoder=MixedFeatureDecoder.random_forest(n_estimators=6, random_state=0),
        policy=GenerationPolicy(method="smote", n_neighbors=3),
        n_bootstrap=2,
        random_state=0,
    )
    model.fit(df_missing)

    H = model.transform(df_missing)
    assert H.shape[0] == len(df_missing)
    assert H.shape[1] > 0

    imputed = model.impute(df_missing)
    assert not imputed[["age", "outcome", "segment"]].isna().any().any()

    conf = model.confidence(df_missing, columns=["age", "outcome"])
    assert {"row_index", "column", "prediction", "variance", "confidence"}.issubset(conf.columns)
    assert len(conf) == len(df_missing) * 2

    synthetic, trace = model.sample(5, return_trace=True)
    assert synthetic.shape == (5, 4)
    assert {"sample_index", "method", "anchor_index", "lambda"}.issubset(trace.columns)

    fig, axes = model.plot(df_missing, color_by="outcome", center="random")
    assert len(axes) == 2
    fig.canvas.draw()


def test_mimic_with_linear_mixed_feature_decoder():
    df = make_frame(n=50)
    df.loc[[0, 1], "age"] = np.nan
    df.loc[[2, 3], "outcome"] = np.nan
    model = MIMIC(
        ignore_columns=["id"],
        regression_columns=["age", "income"],
        classification_columns=["segment", "outcome"],
        encoder=RandomForestPathEncoder(n_estimators=5, embedding_dim=4, random_state=2),
        decoder=LinearMixedFeatureDecoder(logistic_max_iter=500),
        n_bootstrap=1,
        random_state=2,
    ).fit(df)

    imputed = model.impute(df)
    assert not imputed[["age", "outcome"]].isna().any().any()


def test_regression_targets_are_decoded_on_original_scale():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "x": rng.normal(1000.0, 50.0, 80),
            "y": rng.normal(-500.0, 25.0, 80),
            "label": np.where(np.arange(80) % 2 == 0, "a", "b"),
        }
    )
    model = MIMIC(
        regression_columns=["x", "y"],
        classification_columns=["label"],
        encoder=RandomForestPathEncoder(n_estimators=5, embedding_dim=3, random_state=4),
        decoder=MixedFeatureDecoder.linear(),
        policy=GenerationPolicy(method="displacement", n_neighbors=3, lambda_range=(0.0, 0.1)),
        n_bootstrap=1,
        random_state=4,
    ).fit(df)

    generated = model.sample(20)
    assert generated["x"].between(df["x"].min() - 200, df["x"].max() + 200).all()
    assert generated["y"].between(df["y"].min() - 100, df["y"].max() + 100).all()


def test_identity_encoder_decoder_generation_uses_full_row_context():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=60),
            "y": rng.normal(size=60),
            "label": np.where(np.arange(60) % 2 == 0, "a", "b"),
        }
    )
    model = MIMIC(
        regression_columns=["x", "y"],
        classification_columns=["label"],
        encoder=IdentityEncoder(),
        decoder=IdentityDecoder(),
        policy=GenerationPolicy(method="smote", n_neighbors=4),
        n_bootstrap=1,
        random_state=7,
    ).fit(df)

    assert model.feature_modules_["x"].context_columns == ["x", "y", "label"]
    samples, trace = model.sample(8, condition={"label": "a"}, return_trace=True)
    assert samples.shape == (8, 3)
    assert samples["label"].eq("a").all()
    assert trace["method"].eq("smote").all()


def test_sample_restores_original_model_column_order_and_integer_dtype():
    rng = np.random.default_rng(9)
    df = pd.DataFrame(
        {
            "id": np.arange(70),
            "label": np.where(np.arange(70) % 2 == 0, "a", "b"),
            "count": rng.integers(0, 20, size=70),
            "score": rng.normal(size=70),
        }
    )
    model = MIMIC(
        ignore_columns=["id"],
        regression_columns=["count", "score"],
        classification_columns=["label"],
        encoder=IdentityEncoder(),
        decoder=IdentityDecoder(),
        policy=GenerationPolicy(method="smote", n_neighbors=4),
        n_bootstrap=1,
        random_state=9,
    ).fit(df)

    samples = model.sample(6)
    assert list(samples.columns) == ["label", "count", "score"]
    assert pd.api.types.is_integer_dtype(samples["count"].dtype)
    assert pd.api.types.is_float_dtype(samples["score"].dtype)


def test_sample_condition_restricts_anchor_rows():
    df = make_frame(n=50)
    model = MIMIC(
        ignore_columns=["id"],
        regression_columns=["age", "income"],
        classification_columns=["segment", "outcome"],
        encoder=RandomForestPathEncoder(n_estimators=5, embedding_dim=4, random_state=5),
        decoder=MixedFeatureDecoder.random_forest(n_estimators=5, random_state=5),
        policy=GenerationPolicy(method="displacement", n_neighbors=3),
        n_bootstrap=1,
        random_state=5,
    ).fit(df)

    samples, trace = model.sample(10, condition={"segment": "older"}, return_trace=True)
    anchor_segments = df.loc[trace["anchor_index"], "segment"].to_numpy()
    assert set(anchor_segments) == {"older"}
    assert samples["segment"].eq("older").all()
    assert trace["condition"].eq('{"segment": "older"}').all()


def test_mimic_sample_with_forest_conditional_sampler_has_cell_trace():
    df = make_frame(n=45)
    model = MIMIC(
        ignore_columns=["id"],
        regression_columns=["age", "income"],
        classification_columns=["segment", "outcome"],
        encoder=RandomForestPathEncoder(n_estimators=5, embedding_dim=4, random_state=6),
        decoder=ForestConditionalSampler(n_estimators=8, random_state=6),
        policy=GenerationPolicy(method="displacement", n_neighbors=3),
        n_bootstrap=1,
        random_state=6,
    ).fit(df)

    samples, trace = model.sample(4, condition={"segment": "older"}, return_trace=True)
    cell_trace = trace[trace["trace_type"] == "cell"]
    assert samples["segment"].eq("older").all()
    assert not cell_trace.empty
    assert {"sweep", "column", "sampled_value", "conditioning"}.issubset(cell_trace.columns)
    assert cell_trace["conditioning"].eq("z_minus_j").all()


def test_mimic_sample_with_neural_conditional_sampler_has_mdn_cell_trace():
    df = make_frame(n=42)
    model = MIMIC(
        ignore_columns=["id"],
        regression_columns=["age", "income"],
        classification_columns=["segment", "outcome"],
        encoder=RandomForestPathEncoder(n_estimators=4, embedding_dim=3, random_state=8),
        decoder=NeuralConditionalSampler(
            n_components=2,
            hidden_dim=10,
            n_layers=1,
            max_epochs=4,
            patience=2,
            batch_size=16,
            random_state=8,
            device="cpu",
        ),
        policy=GenerationPolicy(method="displacement", n_neighbors=3),
        n_bootstrap=1,
        random_state=8,
    ).fit(df)

    samples, trace = model.sample(3, condition={"segment": "older"}, return_trace=True)
    cell_trace = trace[trace["trace_type"] == "cell"]
    numeric_trace = cell_trace[cell_trace["task"] == "regression"]

    assert samples.shape == (3, 4)
    assert samples["segment"].eq("older").all()
    assert not cell_trace.empty
    assert {"component_index", "component_probability", "component_mean", "component_sigma", "sample_log_probability"}.issubset(numeric_trace.columns)
    assert numeric_trace["conditioning"].eq("z_minus_j").all()


def test_classification_probability_alignment_when_bootstrap_misses_class():
    df = make_frame(n=60)
    df["rare"] = "common"
    df.loc[0, "rare"] = "rare"
    model = MIMIC(
        ignore_columns=["id"],
        regression_columns=["age", "income"],
        classification_columns=["segment", "outcome", "rare"],
        encoder=RandomForestPathEncoder(n_estimators=4, embedding_dim=3, random_state=3),
        decoder=MixedFeatureDecoder.random_forest(n_estimators=4, random_state=3),
        n_bootstrap=1,
        random_state=3,
    ).fit(df)

    conf = model.confidence(df.head(5), columns=["rare"])
    assert len(conf) == 5
    assert conf["probabilities"].map(lambda x: set(x) == {"common", "rare"}).all()


def test_missing_target_excluded_but_missing_context_allowed():
    df = make_frame(n=40)
    df.loc[0, "age"] = np.nan
    df.loc[1, "income"] = np.nan
    model = MIMIC(
        ignore_columns=["id"],
        regression_columns=["age", "income"],
        classification_columns=["segment", "outcome"],
        encoder=RandomForestPathEncoder(n_estimators=4, embedding_dim=3, random_state=1),
        decoder=MixedFeatureDecoder.random_forest(n_estimators=4, random_state=1),
        n_bootstrap=1,
        random_state=1,
    ).fit(df)
    assert model.feature_modules_["age"].observed_mask.sum() == len(df) - 1
    assert model.feature_modules_["income"].observed_mask.sum() == len(df) - 1


def test_resnet_encoder_smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    enc = ResNetEncoder(
        task="classification",
        embedding_dim=4,
        hidden_dim=8,
        n_layers=2,
        max_epochs=3,
        patience=2,
        batch_size=8,
        random_state=0,
    )
    Z = enc.fit(sparse.csr_matrix(X), y).transform(sparse.csr_matrix(X))
    assert Z.shape == (24, 4)
