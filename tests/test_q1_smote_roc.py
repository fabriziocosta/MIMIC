import numpy as np
import pandas as pd

from mimic_experiments.q1_smote_roc import (
    Q1Config,
    _emit_progress,
    _effective_worker_count,
    _load_satimage,
    _summarize_paper_sampling_point,
    apply_majority_under_sampling,
    evaluate_paper_fold,
    fit_mimic_model,
    fold_indices,
    format_seconds,
    generated_count_for_smote_percent,
    generated_count_for_fraction,
    infer_mimic_columns,
    load_q1_result_tables,
    majority_count_for_under_sampling,
    manuscript_result_row,
    maybe_subsample,
    model_cache_path,
    plot_q1_roc_sweep,
    q1_dataset_registry,
    q1_result_table_manifest,
    q1_result_table_paths,
    roc_curve_points,
    save_q1_result_tables,
    standardize_binary_frame,
    _summarize_sampling_point,
)


def test_q1_dataset_registry_contains_ready_keys():
    registry = q1_dataset_registry()

    assert {"key", "dataset", "majority", "minority", "status", "experiment"}.issubset(registry.columns)
    assert {"adult_mixed", "adult_numeric", "pima", "phoneme", "satimage", "forest_cover", "mammography"}.issubset(
        set(registry["key"])
    )


def test_q1_dataset_registry_reports_cached_experiment_status(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for fold in range(1, 11):
        path = model_dir / f"pima__run_full__factorised__cap-0.25__smote-normal-k5__seed-0__fold-{fold}__train-test.joblib"
        path.write_text("cached")

    registry = q1_dataset_registry(artifact_dir=tmp_path, run_profile="view", mimic_capacity=0.25)

    experiment = registry.set_index("key").loc["pima", "experiment"]
    assert experiment == "complete: 10/10 folds"


def test_q1_dataset_registry_reports_saved_summary_status(tmp_path):
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    summary = table_dir / "pima__run_full__factorised__cap-0.25__smote-normal-k5__seed-0__summary.csv"
    summary.write_text("dataset_key,roc_curve_auc\npima,0.7\n")

    registry = q1_dataset_registry(artifact_dir=tmp_path, run_profile="view", mimic_capacity=0.25)

    experiment = registry.set_index("key").loc["pima", "experiment"]
    assert experiment == "complete: current summary csv"


def test_q1_dataset_registry_reports_existing_other_config_summaries(tmp_path):
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    summary = table_dir / "pima__run_full__factorised__cap-0.25__smote-normal-k5__seed-0__summary.csv"
    summary.write_text("dataset_key,roc_curve_auc\npima,0.7\n")

    registry = q1_dataset_registry(
        artifact_dir=tmp_path,
        run_profile="run_full",
        mimic_mode="joint",
        mimic_capacity=0.35,
    )

    experiment = registry.set_index("key").loc["pima", "experiment"]
    assert experiment == "existing: factorised cap-0.25 smote-normal-k5 seed-0"


def test_load_satimage_pins_openml_data_id(monkeypatch):
    calls = []

    def fake_fetch_openml(*, data_id=None, name=None, as_frame=True):
        calls.append({"data_id": data_id, "name": name, "as_frame": as_frame})
        return type(
            "Dataset",
            (),
            {
                "data": pd.DataFrame({"x": [1, 2, 3, 4]}),
                "target": pd.Series(["a", "a", "a", "b"]),
            },
        )()

    monkeypatch.setattr("mimic_experiments.q1_smote_roc.fetch_openml", fake_fetch_openml)

    frame = _load_satimage(n_rows=None, random_state=0)

    assert calls == [{"data_id": 182, "name": None, "as_frame": True}]
    assert frame["label"].tolist() == ["majority", "majority", "majority", "minority"]


def test_standardize_binary_frame_uses_minority_class_by_default():
    frame = standardize_binary_frame(pd.DataFrame({"x": [1, 2, 3, 4]}), ["a", "a", "a", "b"])

    assert frame["label"].tolist() == ["majority", "majority", "majority", "minority"]


def test_generated_count_for_fraction_uses_training_deficit():
    train = pd.DataFrame({"label": ["majority"] * 7 + ["minority"] * 3})

    assert generated_count_for_fraction(train, 0.0) == 0
    assert generated_count_for_fraction(train, 0.5) == 2
    assert generated_count_for_fraction(train, 1.0) == 4


def test_generated_count_for_smote_percent_uses_minority_count():
    train = pd.DataFrame({"label": ["majority"] * 7 + ["minority"] * 3})

    assert generated_count_for_smote_percent(train, 50) == 2
    assert generated_count_for_smote_percent(train, 100) == 3
    assert generated_count_for_smote_percent(train, 200) == 6


def test_majority_count_for_under_sampling_matches_paper_definition():
    train = pd.DataFrame({"label": ["majority"] * 20 + ["minority"] * 10})

    assert majority_count_for_under_sampling(train, 50) == 20
    assert majority_count_for_under_sampling(train, 100) == 10
    assert majority_count_for_under_sampling(train, 200) == 5


def test_apply_majority_under_sampling_keeps_all_minority_and_samples_majority():
    train = pd.DataFrame({"x": range(30), "label": ["majority"] * 20 + ["minority"] * 10})

    sampled = apply_majority_under_sampling(train, under_sampling_percent=200, random_state=0)

    assert sampled["label"].value_counts().to_dict() == {"minority": 10, "majority": 5}


def test_infer_mimic_columns_keeps_label_as_classification():
    frame = pd.DataFrame({"age": [1.0, 2.0], "segment": ["a", "b"], "label": ["majority", "minority"]})

    columns = infer_mimic_columns(frame)

    assert columns == {"regression": ["age"], "classification": ["segment", "label"]}


def test_roc_curve_points_adds_endpoints():
    points = pd.DataFrame({"mean_fpr": [0.2], "mean_tpr": [0.8], "deficit_fraction": [1.0]})

    curve = roc_curve_points(points)

    assert curve[["mean_fpr", "mean_tpr"]].to_numpy().tolist() == [[0.0, 0.0], [0.2, 0.8], [1.0, 1.0]]


def test_roc_curve_points_supports_paper_under_sampling_points():
    points = pd.DataFrame({"mean_fpr": [0.2], "mean_tpr": [0.8], "under_sampling_percent": [100]})

    curve = roc_curve_points(points)

    assert curve[["mean_fpr", "mean_tpr"]].to_numpy().tolist() == [[0.0, 0.0], [0.2, 0.8], [1.0, 1.0]]
    assert "under_sampling_percent" in curve.columns


def test_plot_q1_roc_sweep_returns_configured_figure():
    points = pd.DataFrame(
        {
            "mean_fpr": [0.2, 0.4],
            "mean_tpr": [0.7, 0.9],
            "deficit_fraction": [0.5, 1.0],
        }
    )

    fig, ax = plot_q1_roc_sweep(points, Q1Config(dataset_key="pima"))

    assert ax.get_title() == "Q1 MIMIC ROC sweep: pima"
    assert ax.get_xlabel() == "False positive rate"
    assert ax.get_ylabel() == "True positive rate"
    assert len(ax.lines) == 2
    assert len(ax.texts) == 2
    plt = fig.canvas.figure
    plt.clear()


def test_q1_config_profile_properties():
    run_full = Q1Config(run_profile="run_full", mimic_feature_n_jobs=2)
    view = Q1Config(run_profile="view")

    assert run_full.n_splits == 10
    assert run_full.n_rows is None
    assert run_full.mimic_capacity == run_full.mimic_capacity_run_full
    assert run_full.mimic_feature_n_jobs == 2
    assert run_full.saves_as_profile == "run_full"
    assert run_full.should_run_experiment is True
    assert view.n_splits == 10
    assert view.saves_as_profile == "run_full"
    assert view.should_run_experiment is False


def test_fit_mimic_model_passes_feature_n_jobs(monkeypatch):
    calls = []

    class DummyMIMIC:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def fit(self, train):
            self.train = train
            return self

    monkeypatch.setattr("mimic_experiments.q1_smote_roc.MIMIC", DummyMIMIC)
    train = pd.DataFrame({"x": [1.0, 2.0], "label": ["majority", "minority"]})
    config = Q1Config(mimic_feature_n_jobs=3)

    fit_mimic_model(train, config=config, random_state=0)

    assert calls[0]["feature_n_jobs"] == 3


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


def test_q1_result_tables_round_trip_to_artifact_dir(tmp_path):
    config = Q1Config(dataset_key="pima", run_profile="run_full", artifact_dir=str(tmp_path))
    roc_points = pd.DataFrame({"deficit_fraction": [0.0, 1.0], "mean_fpr": [0.1, 0.2], "mean_tpr": [0.6, 0.8]})
    summary = pd.DataFrame({"dataset_key": ["pima"], "roc_curve_auc": [0.72]})

    paths = save_q1_result_tables(config, roc_points, summary)
    loaded_roc_points, loaded_summary = load_q1_result_tables(config)

    assert paths == q1_result_table_paths(config)
    assert paths["roc_points"].name == "pima__run_full__factorised__cap-0.25__smote-normal-k5__seed-0__roc_points.csv"
    assert paths["summary"].exists()
    pd.testing.assert_frame_equal(loaded_roc_points, roc_points)
    pd.testing.assert_frame_equal(loaded_summary, summary)


def test_q1_result_table_manifest_lists_concrete_paths(tmp_path):
    config = Q1Config(dataset_key="pima", run_profile="view", artifact_dir=str(tmp_path))

    manifest = q1_result_table_manifest(config)

    assert manifest["table"].tolist() == ["roc_points", "summary"]
    assert manifest["path"].str.endswith(".csv").all()
    assert str(tmp_path / "tables") in manifest.loc[0, "path"]


def test_format_seconds_uses_compact_units():
    assert format_seconds(4.2) == "4s"
    assert format_seconds(65) == "1m 05s"
    assert format_seconds(3661) == "1h 01m 01s"


def test_fold_indices_are_stratified_and_counted():
    frame = pd.DataFrame({"x": range(20), "label": ["majority"] * 10 + ["minority"] * 10})
    config = Q1Config(run_profile="run_full")

    folds = fold_indices(frame, config)

    assert len(folds) == 10
    assert [fold for fold, _train, _test in folds] == list(range(1, 11))
    for _fold, train_idx, test_idx in folds:
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(test_idx) == 2


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


def test_summarize_paper_sampling_point_aggregates_fold_results():
    fold_results = pd.DataFrame(
        {
            "smote_percent": [100, 100, 100],
            "under_sampling_percent": [100, 100, 200],
            "fpr": [0.1, 0.3, 0.4],
            "tpr": [0.6, 0.8, 0.9],
            "roc_auc_score": [0.7, 0.9, 0.8],
            "n_generated": [4, 6, 10],
            "n_sampled_majority": [8, 10, 5],
            "n_sampled_minority": [8, 10, 10],
            "model_cache_hit": [False, True, True],
        }
    )

    summary = _summarize_paper_sampling_point(fold_results, 100)

    assert summary["smote_percent"] == 100
    assert summary["mean_fpr"] == 0.2
    assert summary["mean_tpr"] == 0.7
    assert summary["mean_fold_auc"] == 0.8
    assert summary["mean_generated"] == 5
    assert summary["mean_sampled_majority"] == 9
    assert summary["mean_sampled_minority"] == 9
    assert summary["folds"] == 2
    assert summary["model_cache_hits"] == 1


def test_evaluate_paper_fold_uses_fixed_smote_and_under_sampling(monkeypatch):
    class DummyMIMIC:
        def sample(self, n, condition=None, return_trace=False):
            synthetic = pd.DataFrame({"x": np.linspace(0.2, 0.8, n), "label": ["minority"] * n})
            trace = pd.DataFrame({"row": range(n)})
            return synthetic, trace

    monkeypatch.setattr(
        "mimic_experiments.q1_smote_roc.load_or_fit_mimic_model",
        lambda *args, **kwargs: (DummyMIMIC(), "cache.joblib", False),
    )
    frame = pd.DataFrame(
        {
            "x": np.r_[np.linspace(0, 1, 20), np.linspace(1, 2, 10)],
            "label": ["majority"] * 20 + ["minority"] * 10,
        }
    )
    config = Q1Config(
        protocol="paper_under_sampling",
        smote_percent=100,
        under_sampling_percentages=(100, 200),
        cache_models=False,
    )

    result = evaluate_paper_fold(
        frame,
        train_idx=np.array([*range(16), *range(20, 28)]),
        test_idx=np.array([16, 17, 18, 19, 28, 29]),
        fold=1,
        config=config,
    )

    assert result["under_sampling_percent"].tolist() == [100, 200]
    assert result["n_generated"].tolist() == [8, 8]
    assert result["n_sampled_minority"].tolist() == [16, 16]
    assert result["n_sampled_majority"].tolist() == [16, 8]


def test_emit_progress_builds_eta_event():
    events = []

    _emit_progress(events.append, completed=2, total=5, started_at=0.0, fold=2, now=10.0, parallel_workers=3)

    assert events[0]["completed"] == 2
    assert events[0]["total"] == 5
    assert events[0]["fold"] == 2
    assert events[0]["parallel_workers"] == 3
    assert events[0]["elapsed_seconds"] == 10.0
    assert events[0]["eta_seconds"] == 5.0


def test_effective_worker_count_is_capped_by_total():
    assert _effective_worker_count(1, 10) == 1
    assert _effective_worker_count(999, 3) <= 3
