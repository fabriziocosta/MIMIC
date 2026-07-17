import pandas as pd

from streamlined.config import ExperimentConfig, ProfileConfig
from streamlined.embedding_visualization import (
    LatentDisplacementRun,
    fit_latent_displacement_embedding,
    latent_displacement_embedding_path,
    load_latest_latent_displacement_run,
    load_latent_displacement_embedding,
    save_latest_latent_displacement_run,
    save_latent_displacement_embedding,
)
from streamlined import embedding_visualization


class FakeMIMIC:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, frame):
        self.fitted_frame = frame.copy()
        return self


def test_fit_latent_displacement_embedding_prepares_each_feature(monkeypatch):
    frame = pd.DataFrame(
        {
            "age": range(40),
            "workclass": ["a", "b"] * 20,
            "label": ["minority" if i % 4 == 0 else "majority" for i in range(40)],
        }
    )
    monkeypatch.setattr(
        embedding_visualization,
        "load_dataset",
        lambda *args, n_rows=None, **kwargs: frame.head(n_rows),
    )
    monkeypatch.setattr(embedding_visualization, "MIMIC", FakeMIMIC)
    config = ExperimentConfig(
        ProfileConfig(
            name="tiny",
            datasets=("adult",),
            imbalance_ratios=(2.0,),
            training_sizes=(20,),
            seeds=(0,),
        )
    )

    view = fit_latent_displacement_embedding(
        config,
        dataset_key="adult",
        sample_size=20,
        max_plot_rows=8,
    )

    assert view.feature_columns == ["age", "workclass"]
    assert len(view.plot_frame) == 8
    assert view.fitted_rows == 20
    assert view.fitted_rows == len(view.model.fitted_frame)
    assert view.model.kwargs["policy"].method == "displacement"


def test_latent_displacement_embedding_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_visualization, "MIMIC", FakeMIMIC)
    config = ExperimentConfig(
        ProfileConfig(
            name="tiny",
            datasets=("adult",),
            imbalance_ratios=(2.0,),
            training_sizes=(20,),
            seeds=(0,),
        ),
        artifact_dir=tmp_path,
    )
    view = embedding_visualization.LatentDisplacementEmbedding(
        dataset_key="adult",
        model=FakeMIMIC(),
        plot_frame=pd.DataFrame({"age": [1]}),
        feature_columns=["age"],
        fitted_rows=20,
    )
    path = latent_displacement_embedding_path(
        config,
        dataset_key="adult",
        sample_size=20,
        seed=0,
    )

    save_latent_displacement_embedding(view, path)
    loaded = load_latent_displacement_embedding(
        config,
        dataset_key="adult",
        sample_size=20,
        seed=0,
    )

    assert loaded.dataset_key == "adult"
    assert loaded.plot_frame.equals(view.plot_frame)


def test_latest_run_round_trip_uses_saved_config_and_models(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_visualization, "MIMIC", FakeMIMIC)
    config = ExperimentConfig(
        ProfileConfig(
            name="latest",
            datasets=("adult",),
            imbalance_ratios=(2.0,),
            training_sizes=(20,),
            seeds=(7,),
        ),
        artifact_dir=tmp_path,
    )
    model_path = latent_displacement_embedding_path(
        config, dataset_key="adult", sample_size=20, seed=7
    )
    save_latent_displacement_embedding(
        embedding_visualization.LatentDisplacementEmbedding(
            dataset_key="adult",
            model=FakeMIMIC(),
            plot_frame=pd.DataFrame({"age": [1]}),
            feature_columns=["age"],
            fitted_rows=20,
        ),
        model_path,
    )
    saved = LatentDisplacementRun(config, sample_size=20, seed=7, max_plot_rows=10)

    save_latest_latent_displacement_run(saved)
    loaded = load_latest_latent_displacement_run(tmp_path)

    assert loaded == saved


def test_latest_run_rejects_missing_referenced_model(tmp_path):
    config = ExperimentConfig(
        ProfileConfig(
            name="latest",
            datasets=("adult",),
            imbalance_ratios=(2.0,),
            training_sizes=(20,),
            seeds=(0,),
        ),
        artifact_dir=tmp_path,
    )
    save_latest_latent_displacement_run(
        LatentDisplacementRun(config, sample_size=20, seed=0, max_plot_rows=10)
    )

    try:
        load_latest_latent_displacement_run(tmp_path)
    except FileNotFoundError as error:
        assert "references missing model artifacts" in str(error)
    else:
        raise AssertionError("Expected a missing-model error")
