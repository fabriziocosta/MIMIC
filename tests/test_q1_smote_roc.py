import numpy as np
import pandas as pd

from mimic_experiments.q1_smote_roc import (
    Q1Config,
    _emit_progress,
    fold_indices,
    format_seconds,
    generated_count_for_fraction,
    infer_mimic_columns,
    manuscript_result_row,
    maybe_subsample,
    model_cache_path,
    q1_dataset_registry,
    roc_curve_points,
    standardize_binary_frame,
    _summarize_sampling_point,
)


def test_q1_dataset_registry_contains_ready_keys():
    registry = q1_dataset_registry()

    assert {"key", "dataset", "majority", "minority", "status"}.issubset(registry.columns)
    assert {"adult_mixed", "adult_numeric", "pima", "phoneme", "satimage", "forest_cover", "mammography"}.issubset(
        set(registry["key"])
    )


def test_standardize_binary_frame_uses_minority_class_by_default():
    frame = standardize_binary_frame(pd.DataFrame({"x": [1, 2, 3, 4]}), ["a", "a", "a", "b"])

    assert frame["label"].tolist() == ["majority", "majority", "majority", "minority"]


def test_generated_count_for_fraction_uses_training_deficit():
    train = pd.DataFrame({"label": ["majority"] * 7 + ["minority"] * 3})

    assert generated_count_for_fraction(train, 0.0) == 0
    assert generated_count_for_fraction(train, 0.5) == 2
    assert generated_count_for_fraction(train, 1.0) == 4


def test_infer_mimic_columns_keeps_label_as_classification():
    frame = pd.DataFrame({"age": [1.0, 2.0], "segment": ["a", "b"], "label": ["majority", "minority"]})

    columns = infer_mimic_columns(frame)

    assert columns == {"regression": ["age"], "classification": ["segment", "label"]}


def test_roc_curve_points_adds_endpoints():
    points = pd.DataFrame({"mean_fpr": [0.2], "mean_tpr": [0.8], "deficit_fraction": [1.0]})

    curve = roc_curve_points(points)

    assert curve[["mean_fpr", "mean_tpr"]].to_numpy().tolist() == [[0.0, 0.0], [0.2, 0.8], [1.0, 1.0]]


def test_q1_config_profile_properties():
    smoke = Q1Config(run_profile="smoke", n_rows_smoke=12)
    paper = Q1Config(run_profile="paper", n_rows_smoke=12)

    assert smoke.n_splits == 3
    assert smoke.n_rows == 12
    assert smoke.mimic_capacity == smoke.mimic_capacity_smoke
    assert paper.n_splits == 10
    assert paper.n_rows is None
    assert paper.mimic_capacity == paper.mimic_capacity_paper


def test_maybe_subsample_is_reproducible():
    frame = pd.DataFrame({"x": np.arange(10)})

    left = maybe_subsample(frame, n_rows=4, random_state=3)
    right = maybe_subsample(frame, n_rows=4, random_state=3)

    assert left.equals(right)
    assert len(left) == 4


def test_manuscript_result_row_shape():
    summary = pd.DataFrame({"roc_curve_auc": [0.75]})

    row = manuscript_result_row(Q1Config(dataset_key="adult_mixed"), summary)

    assert row.loc[0, "dataset"] == "adult_mixed"
    assert row.loc[0, "mimic_auc"] == 0.75


def test_format_seconds_uses_compact_units():
    assert format_seconds(4.2) == "4s"
    assert format_seconds(65) == "1m 05s"
    assert format_seconds(3661) == "1h 01m 01s"


def test_fold_indices_are_stratified_and_counted():
    frame = pd.DataFrame({"x": range(12), "label": ["majority"] * 6 + ["minority"] * 6})
    config = Q1Config(run_profile="smoke")

    folds = fold_indices(frame, config)

    assert len(folds) == 3
    assert [fold for fold, _train, _test in folds] == [1, 2, 3]
    for _fold, train_idx, test_idx in folds:
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(test_idx) == 4


def test_model_cache_path_uses_artifact_dir_and_fold(tmp_path):
    config = Q1Config(dataset_key="adult_mixed", artifact_dir=str(tmp_path))

    path = model_cache_path(config, fold=2, train_idx=np.array([1, 3, 5]))

    assert path is not None
    assert path.parent == tmp_path / "models"
    assert "adult_mixed" in path.name
    assert "fold-2" in path.name
    assert path.suffix == ".joblib"


def test_model_cache_path_disabled_without_artifact_dir_or_cache(tmp_path):
    assert model_cache_path(Q1Config(), fold=1, train_idx=np.array([1])) is None
    assert model_cache_path(Q1Config(artifact_dir=str(tmp_path), cache_models=False), fold=1, train_idx=np.array([1])) is None


def test_summarize_sampling_point_aggregates_fold_results():
    fold_results = pd.DataFrame(
        {
            "deficit_fraction": [0.5, 0.5, 1.0],
            "fpr": [0.1, 0.3, 0.4],
            "tpr": [0.6, 0.8, 0.9],
            "roc_auc_score": [0.7, 0.9, 0.8],
            "n_generated": [4, 6, 10],
            "model_cache_hit": [False, True, True],
        }
    )

    summary = _summarize_sampling_point(fold_results, 0.5)

    assert summary["mean_fpr"] == 0.2
    assert summary["mean_tpr"] == 0.7
    assert summary["mean_fold_auc"] == 0.8
    assert summary["mean_generated"] == 5
    assert summary["folds"] == 2
    assert summary["model_cache_hits"] == 1


def test_emit_progress_builds_eta_event():
    events = []

    _emit_progress(events.append, completed=2, total=5, started_at=0.0, fold=2, now=10.0)

    assert events[0]["completed"] == 2
    assert events[0]["total"] == 5
    assert events[0]["fold"] == 2
    assert events[0]["elapsed_seconds"] == 10.0
    assert events[0]["eta_seconds"] == 15.0
