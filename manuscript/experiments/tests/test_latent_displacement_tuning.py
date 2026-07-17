import pandas as pd

from streamlined.latent_displacement_tuning import candidate_grid, load_tuning_result, select_candidate, summarize_scores, _persist_partial_tuning_results


def test_candidate_grid_is_cartesian_and_ids_are_unique():
    candidates = candidate_grid(
        capacities=(0.1, 0.4),
        neighbor_counts=(3, 5),
        lambda_ranges=((0.0, 1.0),),
    )

    assert len(candidates) == 4
    assert len({candidate.candidate_id for candidate in candidates}) == 4


def test_select_candidate_uses_mean_validation_auc():
    results = pd.DataFrame(
        {
            "method": ["latent_displacement"] * 4 + ["direct_displacement"],
            "candidate_id": ["a", "a", "b", "b", "direct_displacement"],
            "mimic_capacity": [0.1, 0.1, 0.4, 0.4, None],
            "n_neighbors": [3, 3, 5, 5, None],
            "lambda_low": [0.0, 0.0, 0.25, 0.25, None],
            "lambda_high": [1.0, 1.0, 1.25, 1.25, None],
            "mimic_mode": ["factorised"] * 4 + [None],
            "roc_auc": [0.70, 0.72, 0.76, 0.74, 0.71],
        }
    )

    selected = select_candidate(results)

    assert selected.mimic_capacity == 0.4
    assert selected.n_neighbors == 5
    assert selected.lambda_range == (0.25, 1.25)


def test_summary_reports_paired_delta_and_wins():
    results = pd.DataFrame(
        {
            "seed": [0, 1, 0, 1],
            "imbalance_ratio": [3.0] * 4,
            "training_size": [256] * 4,
            "method": ["direct_displacement", "direct_displacement", "latent_displacement", "latent_displacement"],
            "candidate_id": ["direct_displacement", "direct_displacement", "tuned", "tuned"],
            "roc_auc": [0.70, 0.74, 0.73, 0.72],
        }
    )

    summary = summarize_scores(results)
    tuned = summary.loc[summary["candidate_id"].eq("tuned")].iloc[0]

    assert abs(tuned["mean_delta_vs_direct"] - 0.005) < 1e-12
    assert tuned["wins_vs_direct"] == 1
    assert tuned["comparisons"] == 2


def test_partial_results_can_be_loaded_for_plotting(tmp_path):
    root = tmp_path / "tuning" / "latent_displacement_default_credit"
    validation_rows = [
        {"seed": 0, "imbalance_ratio": 5.0, "training_size": 50, "method": "direct_displacement", "candidate_id": "direct_displacement", "roc_auc": 0.70},
        {"seed": 0, "imbalance_ratio": 5.0, "training_size": 50, "method": "latent_displacement", "candidate_id": "best", "roc_auc": 0.72},
    ]
    history_rows = [{
        "trial": 0, "objective_mean_delta_vs_direct": 0.02, "mimic_capacity": 0.8,
        "n_neighbors": 13, "lambda_low": 0.1, "lambda_high": 0.8, "mimic_mode": "factorised",
    }]

    _persist_partial_tuning_results(root, validation_rows=validation_rows, history_rows=history_rows)
    loaded = load_tuning_result(tmp_path)

    assert len(loaded.optimization_history) == 1
    assert loaded.selected_candidate.n_neighbors == 13
    assert loaded.heldout_results.empty
