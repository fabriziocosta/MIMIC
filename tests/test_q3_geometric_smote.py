import numpy as np
import pandas as pd

from mimic_experiments.q3_geometric_smote import (
    Q3Config,
    binary_metrics,
    classifier_grid,
    generated_count_to_balance,
    infer_mimic_columns,
    load_q3_result_tables,
    manuscript_result_table,
    plot_q3_metric_comparison,
    published_reference_table,
    q3_dataset_registry,
    q3_jobs,
    q3_result_table_manifest,
    q3_result_table_paths,
    save_q3_result_tables,
    standardize_binary_frame,
    summarize_q3_results,
)


def test_q3_dataset_registry_contains_geometric_smote_keys():
    registry = q3_dataset_registry()

    assert {"key", "dataset", "features", "instances", "minority", "majority", "ir", "status", "experiment"}.issubset(
        registry.columns
    )
    assert {"pima", "iris", "wine", "vehicle", "segment"}.issubset(set(registry["key"]))


def test_q3_config_profile_properties():
    run_full = Q3Config(run_profile="run_full", mimic_feature_n_jobs=2)
    view = Q3Config(run_profile="view")

    assert run_full.should_run_experiment is True
    assert run_full.saves_as_profile == "run_full"
    assert run_full.mimic_feature_n_jobs == 2
    assert view.should_run_experiment is False
    assert view.saves_as_profile == "run_full"


def test_published_reference_table_contains_all_table2_rows():
    table = published_reference_table()

    assert {
        "dataset",
        "classifier",
        "metric",
        "no_oversampling",
        "smote",
        "borderline_smote1",
        "borderline_smote2",
        "adasyn",
        "geometric_smote",
        "published_best_method",
        "published_best_value",
    }.issubset(table.columns)
    assert len(table) == 13 * 2 * 3
    assert table.loc[table["dataset"].eq("Pima") & table["classifier"].eq("GBC") & table["metric"].eq("AUC"), "published_best_value"].iloc[0] == 0.822


def test_standardize_binary_frame_uses_minority_by_default():
    frame = standardize_binary_frame(pd.DataFrame({"x": [1, 2, 3, 4]}), ["a", "a", "a", "b"])

    assert frame["label"].tolist() == ["majority", "majority", "majority", "minority"]


def test_generated_count_to_balance_uses_training_deficit():
    train = pd.DataFrame({"label": ["majority"] * 7 + ["minority"] * 3})

    assert generated_count_to_balance(train) == 4


def test_infer_mimic_columns_keeps_label_as_classification():
    frame = pd.DataFrame({"x": [1.0, 2.0], "group": ["a", "b"], "label": ["majority", "minority"]})

    assert infer_mimic_columns(frame) == {"regression": ["x"], "classification": ["group", "label"]}


def test_classifier_grid_matches_paper_gbc_grid():
    assert classifier_grid("GBC") == [
        {"max_depth": 5, "n_estimators": 50},
        {"max_depth": 5, "n_estimators": 100},
        {"max_depth": 8, "n_estimators": 50},
        {"max_depth": 8, "n_estimators": 100},
    ]
    assert classifier_grid("LR") == [{"max_iter": 1000, "solver": "lbfgs"}]


def test_q3_jobs_crosses_repeats_folds_modes_classifiers_and_grid():
    frame = pd.DataFrame({"x": range(20), "label": ["majority"] * 10 + ["minority"] * 10})
    config = Q3Config(n_repeats=2, n_splits=5, mimic_modes=("identity",), classifiers=("LR",))

    jobs = q3_jobs(frame, config)

    assert len(jobs) == 2 * 5 * 1 * 1 * 1
    assert {job["repeat"] for job in jobs} == {1, 2}
    assert {job["fold"] for job in jobs} == {1, 2, 3, 4, 5}


def test_binary_metrics_match_confusion_counts_and_auc():
    metrics = binary_metrics(
        ["majority", "majority", "minority", "minority"],
        ["majority", "minority", "minority", "majority"],
        [0.1, 0.8, 0.7, 0.2],
    )

    assert metrics["f_measure"] == 0.5
    assert metrics["g_mean"] == 0.5
    assert metrics["auc"] == 0.5
    assert {key: metrics[key] for key in ["tp", "fp", "tn", "fn"]} == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_summarize_q3_results_selects_best_classifier_params_per_metric():
    config = Q3Config(dataset_key="pima")
    fold_results = pd.DataFrame(
        {
            "mimic_mode": ["factorised", "factorised"],
            "classifier": ["GBC", "GBC"],
            "classifier_params": ["a", "b"],
            "f_measure": [0.5, 0.6],
            "g_mean": [0.7, 0.4],
            "auc": [0.8, 0.9],
        }
    )

    summary = summarize_q3_results(fold_results, config)

    selected = summary.set_index("metric")
    assert selected.loc["F", "classifier_params"] == "b"
    assert selected.loc["G", "classifier_params"] == "a"
    assert selected.loc["AUC", "classifier_params"] == "b"


def test_manuscript_result_table_adds_mimic_columns():
    summary = pd.DataFrame(
        {
            "mimic_mode": ["identity", "factorised"],
            "classifier": ["GBC", "GBC"],
            "metric": ["AUC", "AUC"],
            "value": [0.81, 0.83],
        }
    )

    table = manuscript_result_table(summary, dataset_key="pima")
    row = table.loc[table["classifier"].eq("GBC") & table["metric"].eq("AUC")].iloc[0]

    assert row["mimic_identity"] == 0.81
    assert row["mimic_factorised"] == 0.83
    assert np.isclose(row["mimic_factorised_delta_vs_best"], 0.008)


def test_q3_result_tables_round_trip_to_artifact_dir(tmp_path):
    config = Q3Config(dataset_key="pima", run_profile="view", artifact_dir=str(tmp_path))
    fold_results = pd.DataFrame({"fold": [1], "auc": [0.7]})
    summary = pd.DataFrame({"dataset": ["Pima"], "mimic_mode": ["factorised"], "auc": [0.7]})
    manuscript = pd.DataFrame({"dataset": ["Pima"], "mimic_factorised": [0.7]})

    paths = save_q3_result_tables(config, fold_results, summary, manuscript)
    loaded_fold_results, loaded_summary, loaded_manuscript = load_q3_result_tables(config)

    assert paths == q3_result_table_paths(config)
    assert paths["summary"].name == "pima__run_full__identity-factorised__cap-0.25__smote-normal-k5__seed-0__summary.csv"
    pd.testing.assert_frame_equal(loaded_fold_results, fold_results)
    pd.testing.assert_frame_equal(loaded_summary, summary)
    pd.testing.assert_frame_equal(loaded_manuscript, manuscript)


def test_q3_result_table_manifest_lists_concrete_paths(tmp_path):
    config = Q3Config(dataset_key="pima", artifact_dir=str(tmp_path))

    manifest = q3_result_table_manifest(config)

    assert manifest["table"].tolist() == ["fold_results", "summary", "manuscript"]
    assert manifest["path"].str.endswith(".csv").all()


def test_q3_dataset_registry_reports_completed_summary(tmp_path):
    config = Q3Config(dataset_key="pima", artifact_dir=str(tmp_path))
    paths = q3_result_table_paths(config)
    paths["summary"].parent.mkdir(parents=True)
    paths["summary"].write_text("dataset,method\nPima,MIMIC\n")

    registry = q3_dataset_registry(artifact_dir=tmp_path, run_profile="view")

    assert registry.set_index("key").loc["pima", "experiment"] == "complete: summary csv"


def test_plot_q3_metric_comparison_returns_axis():
    table = pd.DataFrame(
        {
            "dataset": ["Pima"],
            "classifier": ["GBC"],
            "metric": ["AUC"],
            "no_oversampling": [0.8],
            "smote": [0.81],
            "borderline_smote1": [0.82],
            "borderline_smote2": [0.83],
            "adasyn": [0.79],
            "geometric_smote": [0.84],
            "mimic_identity": [0.82],
            "mimic_factorised": [0.85],
        }
    )

    fig, ax = plot_q3_metric_comparison(table, classifier="GBC", metric="AUC")

    assert ax.get_title() == "Q3 AUC comparison: Pima / GBC"
    assert ax.get_ylabel() == "AUC"
    assert len(ax.patches) == 8
    fig.clear()


def test_plot_q3_metric_comparison_autoscales_y_axis():
    table = pd.DataFrame(
        {
            "dataset": ["Pima"],
            "classifier": ["GBC"],
            "metric": ["AUC"],
            "no_oversampling": [0.8],
            "smote": [0.81],
            "borderline_smote1": [0.82],
            "borderline_smote2": [0.83],
            "adasyn": [0.79],
            "geometric_smote": [1.2],
            "mimic_identity": [0.82],
            "mimic_factorised": [0.85],
        }
    )

    fig, ax = plot_q3_metric_comparison(table, classifier="GBC", metric="AUC")

    assert ax.get_ylim()[1] > 1.2
    fig.clear()
