import numpy as np
import pandas as pd

from streamlined.analysis import aulc_table, pairwise_comparisons
from streamlined.config import ExperimentConfig, ProfileConfig
from streamlined.plotting import critical_difference_inputs, generate_critical_difference_diagram, plot_learning_curves, save_all_figures
from streamlined.preprocessing import fit_preprocess_train_test
from streamlined.runner import run_condition
from streamlined.runner import _emit_progress
from streamlined.runner import run_profile
from streamlined import runner
from streamlined import sampling
from streamlined.sampling import build_balanced_training_set, generated_count_for_balance, make_imbalanced_subset, repair_preprocessed_samples


def make_frame(n=80):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    segment = np.where(x > 0, "high", "low")
    label = np.where(np.arange(n) % 4 == 0, "minority", "majority")
    return pd.DataFrame({"x": x, "segment": segment, "label": label})


def test_preprocessing_fits_train_and_transforms_test_with_stable_width():
    train = make_frame(40)
    test = make_frame(20)

    prepared = fit_preprocess_train_test(train, test, dataset_key="adult")

    assert prepared.X_train.shape[0] == 40
    assert prepared.X_test.shape[0] == 20
    assert prepared.X_train.shape[1] == prepared.X_test.shape[1]


def test_direct_sampling_balances_to_majority_count():
    frame = make_frame(80)
    imbalanced = make_imbalanced_subset(frame, ratio=3.0, training_size=40, random_state=0)
    prepared = fit_preprocess_train_test(frame, frame, dataset_key="adult")

    balanced = build_balanced_training_set(
        "direct_smote",
        imbalanced_raw=imbalanced,
        prepared=prepared,
        dataset_key="adult",
        random_state=0,
        n_neighbors=3,
        lambda_range=(0.0, 1.0),
        mimic_mode="identity",
        mimic_capacity=0.0,
    )

    assert generated_count_for_balance(imbalanced["label"]) == balanced.generated_count
    assert balanced.y.sum() * 2 == len(balanced.y)
    for sl in prepared.categorical_slices.values():
        generated = balanced.X[-balanced.generated_count :, sl]
        assert np.allclose(generated.sum(axis=1), 1.0)
        assert set(np.unique(generated)).issubset({0.0, 1.0})


def test_real_balanced_tops_up_minority_from_training_pool():
    frame = pd.DataFrame(
        {
            "x": np.linspace(-1.0, 1.0, 120),
            "segment": ["low", "high"] * 60,
            "label": ["minority"] * 60 + ["majority"] * 60,
        }
    )
    imbalanced = make_imbalanced_subset(frame, ratio=3.0, training_size=40, random_state=0)
    prepared = fit_preprocess_train_test(frame, frame, dataset_key="adult")

    balanced = build_balanced_training_set(
        "real_balanced",
        imbalanced_raw=imbalanced,
        train_pool_raw=frame,
        prepared=prepared,
        dataset_key="adult",
        random_state=0,
        n_neighbors=3,
        lambda_range=(0.0, 1.0),
        mimic_mode="identity",
        mimic_capacity=0.0,
    )

    majority = int(np.sum(balanced.y == 0))
    minority = int(np.sum(balanced.y == 1))
    assert majority == minority
    assert majority == int(imbalanced["label"].eq("majority").sum())
    assert len(balanced.y) == 2 * majority


def test_direct_sampling_repair_switch_controls_repair_call(monkeypatch):
    frame = make_frame(80)
    imbalanced = make_imbalanced_subset(frame, ratio=3.0, training_size=40, random_state=0)
    prepared = fit_preprocess_train_test(frame, frame, dataset_key="adult")
    calls = []

    def fake_repair(X, prepared_data):
        calls.append(X.shape)
        return X

    monkeypatch.setattr(sampling, "repair_preprocessed_samples", fake_repair)

    build_balanced_training_set(
        "direct_smote",
        imbalanced_raw=imbalanced,
        prepared=prepared,
        dataset_key="adult",
        random_state=2,
        n_neighbors=3,
        lambda_range=(0.25, 0.75),
        mimic_mode="identity",
        mimic_capacity=0.0,
        repair_direct_samples=False,
    )
    assert calls == []

    build_balanced_training_set(
        "direct_smote",
        imbalanced_raw=imbalanced,
        prepared=prepared,
        dataset_key="adult",
        random_state=2,
        n_neighbors=3,
        lambda_range=(0.25, 0.75),
        mimic_mode="identity",
        mimic_capacity=0.0,
        repair_direct_samples=True,
    )
    assert calls


def test_repair_preprocessed_samples_projects_onehot_blocks():
    frame = make_frame(20)
    prepared = fit_preprocess_train_test(frame, frame, dataset_key="adult")
    sample = np.zeros((2, prepared.X_train.shape[1]))
    for sl in prepared.categorical_slices.values():
        sample[:, sl] = [[0.2, 0.8], [0.6, 0.4]]

    repaired = repair_preprocessed_samples(sample, prepared)

    for sl in prepared.categorical_slices.values():
        assert repaired[:, sl].tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_runner_condition_returns_metric_schema():
    frame = make_frame(80)
    train = frame.iloc[:60].reset_index(drop=True)
    test = frame.iloc[60:].reset_index(drop=True)
    prepared = fit_preprocess_train_test(train, test, dataset_key="adult")
    imbalanced = make_imbalanced_subset(train, ratio=2.0, training_size=24, random_state=1)
    config = ExperimentConfig(ProfileConfig(name="tiny", datasets=("adult",), imbalance_ratios=(2.0,), training_sizes=(24,), seeds=(0,), methods=("real_balanced",)))

    row = run_condition(
        config,
        dataset_key="adult",
        imbalanced_raw=imbalanced,
        prepared=prepared,
        method="real_balanced",
        seed=0,
        ratio=2.0,
        training_size=24,
        metadata={},
    )

    assert {"roc_auc", "pr_auc", "balanced_accuracy", "f1", "brier"}.issubset(row)


def test_aulc_pairwise_and_plotting():
    results = pd.DataFrame(
        {
            "dataset_key": ["d"] * 8,
            "imbalance_ratio": [2.0] * 8,
            "training_size": [10, 20, 10, 20, 10, 20, 10, 20],
            "seed": [0, 0, 0, 0, 1, 1, 1, 1],
            "method": ["real_balanced", "real_balanced", "direct_smote", "direct_smote"] * 2,
            "roc_auc": [0.8, 0.9, 0.7, 0.8, 0.82, 0.91, 0.71, 0.81],
        }
    )
    config = ExperimentConfig(ProfileConfig(name="tiny", datasets=("d",), imbalance_ratios=(2.0,), training_sizes=(10, 20), seeds=(0, 1)))

    aulc = aulc_table(results, config)
    pairwise = pairwise_comparisons(aulc, config)

    assert {"segment", "aulc"}.issubset(aulc.columns)
    assert {"left_method", "right_method", "mean_delta"}.issubset(pairwise.columns)
    fig, ax = plot_learning_curves(results.rename(columns={"roc_auc": "roc_auc"}), dataset_key="d", imbalance_ratio=2.0)
    assert ax.get_xlabel() == "Training size"
    fig.clear()


def test_critical_difference_inputs_and_plot():
    aulc = pd.DataFrame(
        {
            "dataset_key": ["d1", "d1", "d1", "d2", "d2", "d2"],
            "imbalance_ratio": [2.0] * 6,
            "method": ["a", "b", "c"] * 2,
            "seed": [0] * 6,
            "segment": ["full"] * 6,
            "aulc": [0.9, 0.8, 0.7, 0.85, 0.75, 0.65],
        }
    )

    ranks, sig = critical_difference_inputs(aulc)
    fig, ax = generate_critical_difference_diagram(aulc, segment="full")

    assert ranks.index.tolist() == ["a", "b", "c"]
    assert sig.shape == (3, 3)
    assert "Critical difference" in ax.get_title()
    fig.clear()


def test_save_all_figures_writes_png_and_svg(tmp_path):
    learning = pd.DataFrame(
        {
            "dataset_key": ["d", "d"],
            "imbalance_ratio": [2.0, 2.0],
            "training_size": [10, 20],
            "method": ["a", "a"],
            "roc_auc": [0.7, 0.8],
        }
    )
    rank = pd.DataFrame({"segment": ["full"], "imbalance_ratio": [2.0], "method": ["a"], "mean_rank": [1.0]})

    paths = save_all_figures(learning, rank, tmp_path)

    suffixes = {path.suffix for path in paths}
    assert {".png", ".svg"}.issubset(suffixes)
    assert all(path.exists() for path in paths)


def test_progress_emitter_prints_text_bar(capsys):
    _emit_progress(
        1,
        4,
        show_progress=True,
        dataset_key="adult",
        ratio=2.0,
        training_size=128,
        seed=0,
        method="direct_smote",
        started_at=0.0,
    )

    out = capsys.readouterr().out
    assert "1/4" in out
    assert "ETA=" in out
    assert "adult" in out
    assert "direct_smote" in out


def test_run_profile_resumes_completed_raw_results(tmp_path, monkeypatch):
    frame = make_frame(40)
    calls = []

    monkeypatch.setattr(runner, "load_dataset", lambda dataset_key, n_rows=None, random_state=0: frame)

    def fake_run_condition(config, *, dataset_key, imbalanced_raw, prepared, method, seed, ratio, training_size, metadata, train_pool_raw=None):
        calls.append((dataset_key, ratio, training_size, seed, method))
        return {
            "dataset_key": dataset_key,
            "imbalance_ratio": ratio,
            "training_size": training_size,
            "seed": seed,
            "method": method,
            "generated_count": 0,
            "real_minority_count": 2,
            "real_majority_count": 2,
            "roc_auc": 0.5 + 0.01 * len(calls),
            "pr_auc": 0.5,
            "balanced_accuracy": 0.5,
            "f1": 0.5,
            "brier": 0.25,
        }

    monkeypatch.setattr(runner, "run_condition", fake_run_condition)
    config = ExperimentConfig(
        ProfileConfig(
            name="tiny",
            datasets=("adult",),
            imbalance_ratios=(2.0,),
            training_sizes=(16,),
            seeds=(0,),
            methods=("real_balanced", "direct_smote"),
        ),
        artifact_dir=tmp_path,
    )

    first = run_profile(config, run_experiment=True, restart=False)
    second = run_profile(config, run_experiment=True, restart=False)

    assert len(calls) == 2
    assert len(first["raw_results"]) == 2
    assert len(second["raw_results"]) == 2


def test_run_profile_restart_reruns_completed_conditions(tmp_path, monkeypatch):
    frame = make_frame(40)
    calls = []

    monkeypatch.setattr(runner, "load_dataset", lambda dataset_key, n_rows=None, random_state=0: frame)
    def fake_run_condition(config, **kwargs):
        calls.append(kwargs["method"])
        return {
        "dataset_key": kwargs["dataset_key"],
        "imbalance_ratio": kwargs["ratio"],
        "training_size": kwargs["training_size"],
        "seed": kwargs["seed"],
        "method": kwargs["method"],
        "generated_count": 0,
        "real_minority_count": 2,
        "real_majority_count": 2,
        "roc_auc": 0.5,
        "pr_auc": 0.5,
        "balanced_accuracy": 0.5,
        "f1": 0.5,
        "brier": 0.25,
    }

    monkeypatch.setattr(runner, "run_condition", fake_run_condition)
    config = ExperimentConfig(
        ProfileConfig(
            name="tiny",
            datasets=("adult",),
            imbalance_ratios=(2.0,),
            training_sizes=(16,),
            seeds=(0,),
            methods=("real_balanced",),
        ),
        artifact_dir=tmp_path,
    )

    run_profile(config, run_experiment=True, restart=False)
    run_profile(config, run_experiment=True, restart=True)

    assert calls == ["real_balanced", "real_balanced"]
