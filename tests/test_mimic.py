import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
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
    sample,
    sample_dataframe,
)
from mimic.diagnostics import categorical_feature_plot, categorical_feature_report, pairwise_feature_plot
from mimic.mimic import GlobalContextPreprocessor


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


def make_mixed_preprocessing_frame():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "age": [20.0, np.nan, 40.0, 50.0],
            "income": [100.0, 120.0, np.nan, 180.0],
            "segment": ["a", "b", np.nan, "a"],
            "outcome": ["yes", "no", "yes", "no"],
        }
    )


def test_global_context_preprocessor_maps_encoded_indices():
    df = make_mixed_preprocessing_frame()
    preprocessor = GlobalContextPreprocessor().fit(
        df,
        numeric_columns=["age", "income"],
        categorical_columns=["segment", "outcome"],
    )

    Xp = preprocessor.transform_all(df)

    assert Xp.shape == (4, 11)
    assert preprocessor.output_dim_ == 11
    assert preprocessor.numeric_value_indices_["age"].tolist() == [0]
    assert preprocessor.numeric_value_indices_["income"].tolist() == [1]
    assert preprocessor.categorical_onehot_indices_["segment"].tolist() == [2, 3, 4]
    assert preprocessor.categorical_onehot_indices_["outcome"].tolist() == [5, 6]
    assert preprocessor.missing_indicator_indices_["age"].tolist() == [7]
    assert preprocessor.missing_indicator_indices_["income"].tolist() == [8]
    assert preprocessor.missing_indicator_indices_["segment"].tolist() == [9]
    assert preprocessor.missing_indicator_indices_["outcome"].tolist() == [10]
    assert preprocessor.column_indices_["age"].tolist() == [0, 7]
    assert preprocessor.column_indices_["segment"].tolist() == [2, 3, 4, 9]


def test_global_context_preprocessor_context_slicing_and_unknown_categories():
    df = make_mixed_preprocessing_frame()
    preprocessor = GlobalContextPreprocessor().fit(
        df,
        numeric_columns=["age", "income"],
        categorical_columns=["segment", "outcome"],
    )
    incoming = df.copy()
    incoming.loc[0, "segment"] = "new"

    Xp = preprocessor.transform_all(incoming)
    without_age = preprocessor.context_matrix(Xp, ["income", "segment", "outcome"])
    without_segment = preprocessor.context_matrix(Xp, ["age", "income", "outcome"])

    assert Xp.shape[1] == preprocessor.output_dim_
    assert without_age.shape[1] == 9
    assert without_segment.shape[1] == 7
    assert not set(preprocessor.column_indices_["age"]) & set(preprocessor.encoded_indices_for(["income", "segment", "outcome"]))
    assert not set(preprocessor.column_indices_["segment"]) & set(preprocessor.encoded_indices_for(["age", "income", "outcome"]))


def test_pairwise_feature_plot_diagonal_histograms_do_not_share_feature_y_axis():
    original = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "capital-gain": [0, 0, 1000, 100000],
        }
    )
    generated = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "capital-gain": [0, 0, 500, 1200],
        }
    )

    grid = pairwise_feature_plot(
        original,
        generated,
        features=["age", "capital-gain"],
        max_rows_per_source=None,
    )

    twin_axes = [ax for ax in grid.fig.axes if ax not in grid.axes.flat]
    assert len(twin_axes) == 2
    assert max(ax.get_ylim()[1] for ax in twin_axes) < 10


def test_pairwise_feature_plot_caps_histogram_bins():
    original = pd.DataFrame({"x": np.linspace(0, 100, 200), "y": np.linspace(0, 10, 200)})
    generated = pd.DataFrame({"x": np.linspace(5, 95, 200), "y": np.linspace(1, 9, 200)})

    grid = pairwise_feature_plot(
        original,
        generated,
        features=["x", "y"],
        max_rows_per_source=None,
        max_hist_bins=20,
    )

    twin_axes = [ax for ax in grid.fig.axes if ax not in grid.axes.flat]
    for ax in twin_axes:
        assert len(ax.patches) <= 40


def test_pairwise_feature_plot_can_log1p_selected_features():
    original = pd.DataFrame({"age": [20, 30, 40], "capital-gain": [0, 1000, 100000]})
    generated = pd.DataFrame({"age": [25, 35, 45], "capital-gain": [0, 500, 1200]})

    grid = pairwise_feature_plot(
        original,
        generated,
        features=["age", "capital-gain"],
        log1p_features=["capital-gain"],
        max_rows_per_source=None,
    )

    assert grid.axes[1, 0].get_ylabel() == "log1p(capital-gain)"
    assert grid.axes[1, 1].get_xlabel() == "log1p(capital-gain)"


def test_pairwise_feature_plot_log1p_ignores_nonfinite_values():
    original = pd.DataFrame({"age": [20, 30, 40], "capital-loss": [0, -1, 2000]})
    generated = pd.DataFrame({"age": [25, 35, 45], "capital-loss": [0, -5, 1200]})

    grid = pairwise_feature_plot(
        original,
        generated,
        features=["age", "capital-loss"],
        log1p_features=["capital-loss"],
        max_rows_per_source=None,
    )

    assert grid.axes[1, 1].get_xlabel() == "log1p(capital-loss)"


def test_categorical_feature_report_and_plot_compare_proportions():
    original = pd.DataFrame({"segment": ["a", "a", "b", "c"], "label": ["yes", "no", "yes", "yes"]})
    generated = pd.DataFrame({"segment": ["a", "b", "b", "b"], "label": ["yes", "no", "no", "yes"]})

    report, summary = categorical_feature_report(
        original,
        generated,
        features=["segment", "label"],
        original_label="heldout",
        generated_label="synthetic",
    )
    fig, axes, plot_report, plot_summary = categorical_feature_plot(
        original,
        generated,
        features=["segment", "label"],
        original_label="heldout",
        generated_label="synthetic",
    )

    assert {"feature", "category", "heldout_proportion", "synthetic_proportion", "absolute_difference"}.issubset(report.columns)
    assert {"feature", "total_variation_distance", "n_categories"}.issubset(summary.columns)
    assert summary.loc[summary["feature"] == "segment", "total_variation_distance"].iat[0] > 0
    assert len(axes) == 2
    assert not plot_report.empty
    assert not plot_summary.empty


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


def test_neural_conditional_sampler_conditional_evidence_shapes():
    rng = np.random.default_rng(14)
    H = rng.normal(size=(36, 3))
    y_reg = H[:, 0] - H[:, 1]
    y_cls = np.where(H[:, 2] > 0, 1, 0)
    sampler = NeuralConditionalSampler(
        n_components=3,
        hidden_dim=10,
        n_layers=1,
        max_epochs=4,
        patience=2,
        batch_size=12,
        random_state=14,
        device="cpu",
    )
    sampler.fit_sampler_target("x", "regression", H, y_reg)
    sampler.fit_sampler_target("c", "classification", H, y_cls)

    reg_evidence = sampler.conditional_evidence_target("x", "regression", H[:5])
    cls_evidence = sampler.conditional_evidence_target("c", "classification", H[:5])

    assert reg_evidence.shape == (5, 9)
    assert cls_evidence.shape == (5, 2)
    assert np.allclose(cls_evidence.sum(axis=1), 1.0)


def test_neural_conditional_sampler_joint_decoder_predicts_all_targets():
    rng = np.random.default_rng(15)
    evidence = rng.normal(size=(40, 7))
    targets = {
        "x": evidence[:, 0] - 0.25 * evidence[:, 1],
        "label": np.where(evidence[:, 2] > 0, 1, 0),
    }
    target_specs = [
        {"column": "x", "task": "regression"},
        {"column": "label", "task": "classification", "n_classes": 2},
    ]
    sampler = NeuralConditionalSampler(
        hidden_dim=12,
        n_layers=1,
        max_epochs=5,
        patience=2,
        batch_size=16,
        random_state=15,
        device="cpu",
    )

    sampler.fit_joint_decoder(evidence, targets, target_specs)
    predictions = sampler.predict_joint(evidence[:6])

    assert set(predictions) == {"x", "label"}
    assert predictions["x"].shape == (6,)
    assert predictions["label"].shape == (6,)
    assert set(predictions["label"]).issubset({0, 1})


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
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
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
    assert model.generation_decode_mode_ == "direct"
    assert trace["trace_type"].eq("embedding").all()
    assert trace["resolved_generation_decode_mode"].eq("direct").all()

    fig, axes = model.plot(df_missing, color_by="outcome", center="random")
    assert len(axes) == 2
    fig.canvas.draw()


def test_plot_can_select_embedding_columns_and_color_from_values():
    df = make_frame(n=45)
    model = MIMIC(
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
        encoder=RandomForestPathEncoder(n_estimators=4, embedding_dim=3, random_state=19),
        decoder=MixedFeatureDecoder.random_forest(n_estimators=4, random_state=19),
        n_bootstrap=1,
        random_state=19,
    ).fit(df)

    fig, axes = model.plot(df, embedding_columns=["age", "segment"], color_by="income")
    fig.canvas.draw()

    assert len(axes) == 2
    assert len(fig.axes) == 4

    with pytest.raises(ValueError, match="Unknown embedding columns"):
        model.plot(df, embedding_columns=["not_a_column"])


def test_sample_function_fits_and_generates_matching_dataframe():
    df = pd.DataFrame(
        {
            "age": [30.5, 30.5, 42.0, 42.0, 51.5, 51.5, 63.0, 63.0],
            "segment": ["younger", "younger", "younger", "older", "older", "older", "older", "younger"],
        }
    )

    synthetic = sample(
        df,
        mode="identity",
        random_state=0,
    )
    alias_synthetic = sample_dataframe(
        df,
        n_samples=5,
        mode="identity",
        random_state=0,
    )

    assert isinstance(synthetic, pd.DataFrame)
    assert synthetic.shape == df.shape
    assert list(synthetic.columns) == list(df.columns)
    assert alias_synthetic.shape == (5, df.shape[1])


def test_verbose_false_is_silent_by_default(capsys):
    df = pd.DataFrame(
        {
            "x": [1.0, 1.0, 2.0, 2.0],
            "label": ["a", "b", "a", "b"],
        }
    )

    MIMIC(mode="identity", random_state=0).fit(df)

    assert capsys.readouterr().out == ""


def test_verbose_prints_hyperparameters_and_fitted_sizes(capsys):
    df = pd.DataFrame(
        {
            "x": [1.0, 1.0, 2.0, 2.0],
            "label": ["a", "b", "a", "b"],
        }
    )

    model = MIMIC(
        columns={
            "regression": ["x"],
            "classification": ["label"],
        },
        mode="direct",
        capacity=0.25,
        random_state=0,
        verbose=True,
    )
    init_out = capsys.readouterr().out
    model.fit(df)
    fit_out = capsys.readouterr().out

    assert "MIMIC init hyperparameters:" in init_out
    assert "verbose: True" in init_out
    assert "columns:\n" in init_out
    assert "  regression:\n" in init_out
    assert "    - x\n" in init_out
    assert "mode: direct" in init_out
    assert "capacity: 0.25" in init_out
    assert "encoder: ResNetEncoder" in init_out
    assert "decoder: NeuralConditionalSampler" in init_out
    assert "policy: GenerationPolicy" in init_out
    assert "generation_decode_mode: direct" in init_out
    assert "MIMIC resolved fit configuration:" in fit_out
    assert "capacity_parameters_:\n" in fit_out
    assert "  embedding_dim:" in fit_out
    assert "n_bootstrap_:" in fit_out
    assert "MIMIC fitted data sizes:" in fit_out
    assert "input_rows: 4" in fit_out
    assert "model_columns: 2" in fit_out
    assert "encoded_input_dim:" in fit_out
    assert "train_embeddings_shape: (4," in fit_out
    assert "embedding[x]: rows=4" in fit_out


def test_columns_dict_role_parsing_and_missing_keys():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )

    model = MIMIC(
        columns={
            "ignore": ["id"],
            "regression": ["x"],
        },
        mode="identity",
    ).fit(df)

    assert model.columns_ == ["id", "x"]
    assert model.ignore_columns_ == ["id"]
    assert model.regression_columns_ == ["x"]
    assert model.classification_columns_ == []
    assert model.model_columns_ == ["x"]


def test_columns_auto_infers_ignore_regression_and_classification_roles():
    n = 20
    df = pd.DataFrame(
        {
            "id": np.arange(100, 100 + n),
            "row_code": [f"r{i}" for i in range(n)],
            "x": np.repeat(np.linspace(1.25, 5.75, 5), 4),
            "flag": [0, 1] * 10,
            "rating": [1, 2, 3, 4] * 5,
            "label": ["a", "b"] * 10,
        }
    )

    model = MIMIC(columns="auto", mode="identity").fit(df)

    assert model.ignore_columns_ == ["id", "row_code"]
    assert model.regression_columns_ == ["x"]
    assert model.classification_columns_ == ["flag", "rating", "label"]


def test_columns_none_is_auto_alias():
    df = pd.DataFrame(
        {
            "id": [101, 102, 103, 104],
            "x": [1.0, 2.1, 3.2, 4.3],
            "flag": [0, 1, 0, 1],
            "label": ["a", "b", "a", "b"],
        }
    )

    auto = MIMIC(columns="auto", mode="identity").fit(df)
    model = MIMIC(mode="identity").fit(df)
    none = MIMIC(columns=None, mode="identity").fit(df)

    assert model.ignore_columns_ == auto.ignore_columns_ == none.ignore_columns_
    assert model.regression_columns_ == auto.regression_columns_ == none.regression_columns_
    assert model.classification_columns_ == auto.classification_columns_ == none.classification_columns_


def test_columns_dict_validation_errors():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "x": [1.0, 2.0, 3.0, 4.0],
            "label": ["a", "b", "a", "b"],
        }
    )

    with pytest.raises(ValueError, match="columns must only contain"):
        MIMIC(columns={"regression": ["x"], "target": ["label"]}).fit(df)

    with pytest.raises(ValueError, match="columns must be 'auto' or a mapping"):
        MIMIC(columns="infer").fit(df)

    with pytest.raises(ValueError, match="Columns cannot appear in multiple roles"):
        MIMIC(columns={"regression": ["x"], "classification": ["x", "label"], "ignore": ["id"]}).fit(df)

    with pytest.raises(ValueError, match="Every non-ignored column needs a task"):
        MIMIC(columns={"regression": ["x"]}).fit(df)


def test_mode_identity_uses_identity_components_and_default_generation_policy():
    df = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, 12),
            "label": np.array(["a", "b"] * 6),
        }
    )
    model = MIMIC(
        columns={
            "regression": ["x"],
            "classification": ["label"],
        },
        mode="identity",
        random_state=20,
    ).fit(df)

    samples, trace = model.sample(3, return_trace=True)

    assert isinstance(model.encoder_, IdentityEncoder)
    assert isinstance(model.decoder_, IdentityDecoder)
    assert model.n_bootstrap_ == 3
    assert model.policy_.method == "displacement"
    assert model.policy_.neighbour_mode == "mutual"
    assert model.policy_.n_neighbors == 5
    assert model.policy_.lambda_range == (0.25, 0.75)
    assert model.generation_decode_mode_ == "direct"
    assert samples.shape == (3, 2)
    assert trace["resolved_generation_decode_mode"].eq("direct").all()


def test_numeric_mode_and_legacy_level_aliases_resolve_to_modes():
    df = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, 12),
            "label": np.array(["a", "b"] * 6),
        }
    )

    numeric = MIMIC(
        columns={
            "regression": ["x"],
            "classification": ["label"],
        },
        mode=0,
        random_state=24,
    ).fit(df)
    legacy = MIMIC(
        columns={
            "regression": ["x"],
            "classification": ["label"],
        },
        level=0,
        random_state=24,
    ).fit(df)

    assert numeric.mode_ == "identity"
    assert numeric.level_ == 0
    assert legacy.mode_ == "identity"
    assert legacy.level_ == 0


def test_capacity_scales_preset_hyperparameters():
    low = MIMIC._capacity_parameters(0.0)
    mid = MIMIC._capacity_parameters(0.5)
    high = MIMIC._capacity_parameters(1.0)

    assert low["embedding_dim"] == 1
    assert low["hidden_dim"] == 8
    assert low["n_layers"] == 1
    assert low["max_epochs"] == 10
    assert low["patience"] == 2
    assert low["batch_size"] == 32
    assert low["n_components"] == 1
    assert low["n_bootstrap"] == 1

    assert mid["n_bootstrap"] == 3
    assert low["learning_rate"] > mid["learning_rate"] > high["learning_rate"]
    assert low["weight_decay"] < mid["weight_decay"] < high["weight_decay"]

    assert high["embedding_dim"] == 128
    assert high["hidden_dim"] == 128
    assert high["n_layers"] == 8
    assert high["max_epochs"] == 300
    assert high["patience"] == 30
    assert high["batch_size"] == 256
    assert high["n_components"] == 8
    assert high["n_bootstrap"] == 5


def test_mode_factorised_uses_neural_decoder_preset():
    df = make_frame(n=36)
    model = MIMIC(
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
        encoder=RandomForestPathEncoder(n_estimators=4, embedding_dim=3, random_state=21),
        mode="factorised",
        capacity=0.0,
        n_bootstrap=1,
        random_state=21,
    ).fit(df)

    samples, trace = model.sample(3, condition={"segment": "older"}, return_trace=True)

    assert isinstance(model.decoder_, NeuralConditionalSampler)
    assert model.capacity_ == 0.0
    assert model.decoder_.n_components == 1
    assert model.decoder_.max_epochs == 10
    assert model.n_bootstrap_ == 1
    assert model.generation_decode_mode_ == "factorised"
    assert samples["segment"].eq("older").all()
    assert not trace[trace["trace_type"] == "cell"].empty


def test_mode_joint_uses_neural_decoder_preset():
    df = make_frame(n=36)
    model = MIMIC(
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
        encoder=RandomForestPathEncoder(n_estimators=4, embedding_dim=3, random_state=22),
        mode="joint",
        capacity=0.0,
        n_bootstrap=1,
        random_state=22,
    ).fit(df)

    samples, trace = model.sample(3, condition={"segment": "older"}, return_trace=True)

    assert isinstance(model.decoder_, NeuralConditionalSampler)
    assert model.generation_decode_mode_ == "joint"
    assert samples["segment"].eq("older").all()
    assert trace["trace_type"].eq("embedding").all()


def test_custom_decoder_with_default_mode_keeps_auto_decode_resolution():
    df = make_frame(n=35)
    model = MIMIC(
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
        encoder=RandomForestPathEncoder(n_estimators=4, embedding_dim=3, random_state=23),
        decoder=LinearMixedFeatureDecoder(logistic_max_iter=500),
        n_bootstrap=1,
        random_state=23,
    ).fit(df)

    assert model.mode == "joint"
    assert model.mode_ == "joint"
    assert isinstance(model.decoder_, LinearMixedFeatureDecoder)
    assert model.generation_decode_mode_ == "direct"


def test_mimic_with_linear_mixed_feature_decoder():
    df = make_frame(n=50)
    df.loc[[0, 1], "age"] = np.nan
    df.loc[[2, 3], "outcome"] = np.nan
    model = MIMIC(
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
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
        columns={
            "regression": ["x", "y"],
            "classification": ["label"],
        },
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
        columns={
            "regression": ["x", "y"],
            "classification": ["label"],
        },
        encoder=IdentityEncoder(),
        decoder=IdentityDecoder(),
        policy=GenerationPolicy(method="smote", n_neighbors=4),
        n_bootstrap=1,
        random_state=7,
    ).fit(df)

    assert model.feature_modules_["x"].context_columns == ["x", "y", "label"]
    assert model.feature_modules_["x"].context_indices.tolist() == list(
        range(model.global_preprocessor_.output_dim_)
    )
    samples, trace = model.sample(8, condition={"label": "a"}, return_trace=True)
    assert samples.shape == (8, 3)
    assert samples["label"].eq("a").all()
    assert trace["method"].eq("smote").all()


def test_feature_module_context_indices_match_encoder_input_dimensions():
    df = make_frame(n=50)
    model = MIMIC(
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
        encoder=RandomForestPathEncoder(n_estimators=4, embedding_dim=3, random_state=18),
        decoder=MixedFeatureDecoder.random_forest(n_estimators=4, random_state=18),
        n_bootstrap=2,
        random_state=18,
    ).fit(df)

    for module in model.feature_modules_.values():
        assert len(module.context_indices) < model.global_preprocessor_.output_dim_
        for member in module.members:
            assert member.encoder.forest_.n_features_in_ == len(module.context_indices)


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
        columns={
            "ignore": ["id"],
            "regression": ["count", "score"],
            "classification": ["label"],
        },
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
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
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
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
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
    assert model.generation_decode_mode_ == "factorised"
    assert trace["resolved_generation_decode_mode"].eq("factorised").all()


def test_generation_decode_mode_direct_suppresses_sampler_trace():
    df = make_frame(n=45)
    model = MIMIC(
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
        encoder=RandomForestPathEncoder(n_estimators=5, embedding_dim=4, random_state=10),
        decoder=ForestConditionalSampler(n_estimators=8, random_state=10),
        policy=GenerationPolicy(method="displacement", n_neighbors=3),
        generation_decode_mode="direct",
        n_bootstrap=1,
        random_state=10,
    ).fit(df)

    samples, trace = model.sample(4, condition={"segment": "older"}, return_trace=True)

    assert samples["segment"].eq("older").all()
    assert model.generation_decode_mode_ == "direct"
    assert trace["trace_type"].eq("embedding").all()
    assert trace["resolved_generation_decode_mode"].eq("direct").all()


def test_generation_decode_mode_factorised_produces_sampler_trace():
    df = make_frame(n=45)
    model = MIMIC(
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
        encoder=RandomForestPathEncoder(n_estimators=5, embedding_dim=4, random_state=13),
        decoder=ForestConditionalSampler(n_estimators=8, random_state=13),
        policy=GenerationPolicy(method="displacement", n_neighbors=3),
        generation_decode_mode="factorised",
        n_bootstrap=1,
        random_state=13,
    ).fit(df)

    samples, trace = model.sample(4, condition={"segment": "older"}, return_trace=True)
    cell_trace = trace[trace["trace_type"] == "cell"]

    assert samples["segment"].eq("older").all()
    assert model.generation_decode_mode_ == "factorised"
    assert not cell_trace.empty
    assert trace["resolved_generation_decode_mode"].eq("factorised").all()


def test_generation_decode_mode_factorised_requires_sampler_capable_decoder():
    df = make_frame(n=35)
    model = MIMIC(
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
        encoder=RandomForestPathEncoder(n_estimators=4, embedding_dim=3, random_state=11),
        decoder=LinearMixedFeatureDecoder(logistic_max_iter=500),
        generation_decode_mode="factorised",
        n_bootstrap=1,
        random_state=11,
    )

    with pytest.raises(
        ValueError,
        match="generation_decode_mode='factorised' requires a decoder that supports conditional sampling",
    ):
        model.fit(df)


def test_generation_decode_mode_invalid_error():
    df = make_frame(n=35)
    base_kwargs = {
        "columns": {
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
        "encoder": RandomForestPathEncoder(n_estimators=4, embedding_dim=3, random_state=12),
        "decoder": MixedFeatureDecoder.random_forest(n_estimators=4, random_state=12),
        "n_bootstrap": 1,
        "random_state": 12,
    }

    with pytest.raises(ValueError, match="generation_decode_mode must be one of"):
        MIMIC(**base_kwargs, generation_decode_mode="unknown").fit(df)

    with pytest.raises(ValueError, match="mode must be one of"):
        MIMIC(**base_kwargs, mode=9).fit(df)

    with pytest.raises(ValueError, match="mode and level specify different presets"):
        MIMIC(**base_kwargs, mode="direct", level=0).fit(df)

    with pytest.raises(ValueError, match="capacity must be between 0 and 1"):
        MIMIC(**base_kwargs, capacity=1.5).fit(df)


def test_generation_decode_mode_joint_requires_neural_decoder():
    df = make_frame(n=35)
    base_kwargs = {
        "columns": {
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
        "encoder": RandomForestPathEncoder(n_estimators=4, embedding_dim=3, random_state=12),
        "n_bootstrap": 1,
        "random_state": 12,
        "generation_decode_mode": "joint",
    }

    with pytest.raises(ValueError, match="requires NeuralConditionalSampler-style conditional evidence"):
        MIMIC(**base_kwargs, decoder=MixedFeatureDecoder.random_forest(n_estimators=4, random_state=12)).fit(df)

    with pytest.raises(ValueError, match="requires NeuralConditionalSampler-style conditional evidence"):
        MIMIC(**base_kwargs, decoder=ForestConditionalSampler(n_estimators=4, random_state=12)).fit(df)


def test_generation_decode_mode_joint_requires_complete_rows():
    df = make_frame(n=40)
    df.loc[1, "age"] = np.nan
    df.loc[2:, "income"] = np.nan
    model = MIMIC(
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
        encoder=RandomForestPathEncoder(n_estimators=4, embedding_dim=3, random_state=16),
        decoder=NeuralConditionalSampler(
            n_components=2,
            hidden_dim=8,
            n_layers=1,
            max_epochs=3,
            patience=1,
            batch_size=16,
            random_state=16,
            device="cpu",
        ),
        generation_decode_mode="joint",
        n_bootstrap=1,
        random_state=16,
    )

    with pytest.raises(ValueError, match="requires at least two complete modelled training rows"):
        model.fit(df)


def test_mimic_sample_with_neural_joint_decode_mode():
    df = make_frame(n=42)
    model = MIMIC(
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
        encoder=RandomForestPathEncoder(n_estimators=4, embedding_dim=3, random_state=17),
        decoder=NeuralConditionalSampler(
            n_components=2,
            hidden_dim=10,
            n_layers=1,
            max_epochs=4,
            patience=2,
            batch_size=16,
            random_state=17,
            device="cpu",
        ),
        policy=GenerationPolicy(method="displacement", n_neighbors=3),
        generation_decode_mode="joint",
        n_bootstrap=1,
        random_state=17,
    ).fit(df)

    samples, trace = model.sample(3, condition={"segment": "older"}, return_trace=True)

    assert model.generation_decode_mode_ == "joint"
    assert samples.shape == (3, 4)
    assert samples["segment"].eq("older").all()
    assert trace["trace_type"].eq("embedding").all()
    assert trace["resolved_generation_decode_mode"].eq("joint").all()


def test_mimic_sample_with_neural_conditional_sampler_has_mdn_cell_trace():
    df = make_frame(n=42)
    model = MIMIC(
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
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
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome", "rare"],
        },
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
        columns={
            "ignore": ["id"],
            "regression": ["age", "income"],
            "classification": ["segment", "outcome"],
        },
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
