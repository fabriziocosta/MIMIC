import numpy as np
import pandas as pd

from streamlined.analysis import aulc_table, pairwise_comparisons
from streamlined.config import ExperimentConfig, ProfileConfig
from streamlined.plotting import plot_learning_curves
from streamlined.preprocessing import fit_preprocess_train_test
from streamlined.runner import run_condition
from streamlined.sampling import build_balanced_training_set, generated_count_for_balance, make_imbalanced_subset


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
