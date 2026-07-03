import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy import sparse

from mimic import (
    GenerationPolicy,
    MIMIC,
    MixedFeatureDecoder,
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
    Z = enc.fit(X, y).transform(X)
    assert Z.shape == (24, 4)

