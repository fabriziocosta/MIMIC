import numpy as np
import pandas as pd

from mimic_experiments.q2_adasyn import (
    Q2Config,
    _load_abalone,
    binary_metrics,
    generated_count_to_balance,
    half_split_indices,
    infer_mimic_columns,
    load_q2_result_tables,
    manuscript_result_table,
    plot_q2_metric_comparison,
    published_reference_table,
    q2_dataset_registry,
    q2_result_table_manifest,
    q2_result_table_paths,
    save_q2_result_tables,
    standardize_binary_frame,
    summarize_q2_results,
)


def test_q2_dataset_registry_contains_plan_keys():
    registry = q2_dataset_registry()

    assert {"key", "dataset", "status", "experiment"}.issubset(registry.columns)
    assert {"vehicle", "pima", "vowel", "ionosphere", "abalone"}.issubset(set(registry["key"]))


def test_q2_config_view_reuses_run_full_artifact_stem():
    run_full = Q2Config(dataset_key="pima", run_profile="run_full", mimic_feature_n_jobs=2)
    view = Q2Config(dataset_key="pima", run_profile="view")

    assert run_full.should_run_experiment is True
    assert run_full.saves_as_profile == "run_full"
    assert run_full.mimic_feature_n_jobs == 2
    assert view.should_run_experiment is False
    assert view.saves_as_profile == "run_full"


def test_standardize_binary_frame_uses_minority_by_default():
    frame = standardize_binary_frame(pd.DataFrame({"x": [1, 2, 3, 4]}), ["a", "a", "a", "b"])

    assert frame["label"].tolist() == ["majority", "majority", "majority", "minority"]


def test_load_abalone_pins_openml_data_id_and_classes(monkeypatch):
    calls = []

    def fake_fetch_openml(*, data_id=None, as_frame=True):
        calls.append({"data_id": data_id, "as_frame": as_frame})
        frame = pd.DataFrame(
            {
                "Sex": ["M", "F", "I", "M"],
                "Length": [1.0, 2.0, 3.0, 4.0],
                "Rings": ["9", "18", "10", "9"],
            }
        )
        target = pd.Series(frame["Rings"], name="Rings")
        return type("Dataset", (), {"frame": frame, "target": target})()

    monkeypatch.setattr("mimic_experiments.q2_adasyn.fetch_openml", fake_fetch_openml)

    frame = _load_abalone()

    assert calls == [{"data_id": 183, "as_frame": True}]
    assert "Sex" not in frame.columns
    assert frame["label"].tolist() == ["majority", "minority", "majority"]


def test_half_split_indices_split_each_class_in_half():
    frame = pd.DataFrame({"x": range(12), "label": ["majority"] * 8 + ["minority"] * 4})

    train_idx, test_idx = half_split_indices(frame, random_state=0)

    assert len(set(train_idx) & set(test_idx)) == 0
    assert frame.loc[train_idx, "label"].value_counts().to_dict() == {"majority": 4, "minority": 2}
    assert frame.loc[test_idx, "label"].value_counts().to_dict() == {"majority": 4, "minority": 2}


def test_generated_count_to_balance_uses_training_deficit():
    train = pd.DataFrame({"label": ["majority"] * 7 + ["minority"] * 3})

    assert generated_count_to_balance(train) == 4


def test_infer_mimic_columns_keeps_label_as_classification():
    frame = pd.DataFrame({"age": [1.0, 2.0], "segment": ["a", "b"], "label": ["majority", "minority"]})

    assert infer_mimic_columns(frame) == {"regression": ["age"], "classification": ["segment", "label"]}


def test_binary_metrics_match_confusion_counts():
    metrics = binary_metrics(
        ["majority", "majority", "minority", "minority"],
        ["majority", "minority", "minority", "majority"],
    )

    assert metrics["oa"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f_measure"] == 0.5
    assert metrics["g_mean"] == 0.5
    assert {key: metrics[key] for key in ["tp", "fp", "tn", "fn"]} == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_summarize_q2_results_shape():
    config = Q2Config(dataset_key="pima", artifact_dir="/tmp/q2")
    run_results = pd.DataFrame(
        {
            "oa": [0.6, 0.8],
            "precision": [0.5, 0.7],
            "recall": [0.4, 0.6],
            "f_measure": [0.45, 0.65],
            "g_mean": [0.5, 0.7],
        }
    )

    summary = summarize_q2_results(run_results, config)

    assert summary.loc[0, "dataset"] == "Pima Indian Diabetes"
    assert summary.loc[0, "method"] == "MIMIC-factorised"
    assert summary.loc[0, "n_runs"] == 2
    assert summary.loc[0, "oa"] == 0.7


def test_published_reference_table_contains_adasyn_rows():
    table = published_reference_table()

    assert {"dataset", "method", "oa", "precision", "recall", "f_measure", "g_mean"}.issubset(table.columns)
    assert table.loc[table["method"].eq("ADASYN"), "dataset"].nunique() == 5


def test_manuscript_result_table_appends_mimic_row():
    summary = pd.DataFrame(
        {
            "dataset": ["Pima Indian Diabetes"],
            "method": ["MIMIC-factorised"],
            "oa": [0.7],
            "precision": [0.6],
            "recall": [0.5],
            "f_measure": [0.55],
            "g_mean": [0.58],
        }
    )

    table = manuscript_result_table(summary)

    assert table["method"].tolist() == ["Decision tree", "SMOTE", "ADASYN", "MIMIC-factorised"]


def test_q2_result_tables_round_trip_to_artifact_dir(tmp_path):
    config = Q2Config(dataset_key="pima", run_profile="view", artifact_dir=str(tmp_path))
    run_results = pd.DataFrame({"run": [1], "oa": [0.7]})
    summary = pd.DataFrame({"dataset": ["Pima Indian Diabetes"], "method": ["MIMIC-factorised"], "oa": [0.7]})
    manuscript = pd.DataFrame({"dataset": ["Pima Indian Diabetes"], "method": ["MIMIC-factorised"], "oa": [0.7]})

    paths = save_q2_result_tables(config, run_results, summary, manuscript)
    loaded_run_results, loaded_summary, loaded_manuscript = load_q2_result_tables(config)

    assert paths == q2_result_table_paths(config)
    assert paths["summary"].name == "pima__run_full__factorised__cap-0.25__smote-normal-k5__seed-0__summary.csv"
    pd.testing.assert_frame_equal(loaded_run_results, run_results)
    pd.testing.assert_frame_equal(loaded_summary, summary)
    pd.testing.assert_frame_equal(loaded_manuscript, manuscript)


def test_q2_result_table_manifest_lists_concrete_paths(tmp_path):
    config = Q2Config(dataset_key="pima", artifact_dir=str(tmp_path))

    manifest = q2_result_table_manifest(config)

    assert manifest["table"].tolist() == ["run_results", "summary", "manuscript"]
    assert manifest["path"].str.endswith(".csv").all()


def test_q2_dataset_registry_reports_completed_summary(tmp_path):
    config = Q2Config(dataset_key="pima", artifact_dir=str(tmp_path))
    paths = q2_result_table_paths(config)
    paths["summary"].parent.mkdir(parents=True)
    paths["summary"].write_text("dataset,method\nPima,MIMIC\n")

    registry = q2_dataset_registry(artifact_dir=tmp_path, run_profile="view")

    assert registry.set_index("key").loc["pima", "experiment"] == "complete: summary csv"


def test_plot_q2_metric_comparison_returns_axis():
    table = pd.DataFrame(
        {
            "dataset": ["Pima Indian Diabetes", "Pima Indian Diabetes"],
            "method": ["ADASYN", "MIMIC-factorised"],
            "g_mean": [0.66, 0.7],
        }
    )

    fig, ax = plot_q2_metric_comparison(table)

    assert ax.get_title() == "Q2 g-mean comparison: Pima Indian Diabetes"
    assert ax.get_ylabel() == "g-mean"
    assert len(ax.patches) == 2
    fig.clear()
