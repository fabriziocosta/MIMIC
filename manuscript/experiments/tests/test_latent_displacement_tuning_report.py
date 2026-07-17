import matplotlib.pyplot as plt
import pandas as pd

from streamlined.latent_displacement_tuning import LatentDisplacementCandidate, LatentDisplacementTuningResult
from streamlined.latent_displacement_tuning_report import heldout_conclusion, plot_hyperparameter_performance, plot_hyperparameter_quantile_bands, plot_optimization_progress, save_optimization_report, top_trials_table


def make_result():
    history = pd.DataFrame({
        "trial": [1, 0], "mean_direct_roc_auc": [0.70, 0.70], "mean_latent_roc_auc": [0.74, 0.71],
        "objective_mean_delta_vs_direct": [0.04, 0.01], "wins_vs_direct": [3, 2], "comparisons": [3, 3],
        "mimic_capacity": [0.4, 0.1], "n_neighbors": [5, 3], "lambda_low": [0.2, 0.0], "lambda_high": [1.2, 0.8],
    })
    heldout = pd.DataFrame({
        "method": ["direct_displacement", "latent_displacement"], "candidate_id": ["direct", "best"],
        "mean_roc_auc": [0.71, 0.73], "std_roc_auc": [0.01, 0.02], "mean_delta_vs_direct": [0.0, 0.02],
        "wins_vs_direct": [0, 2], "comparisons": [3, 3],
    })
    return LatentDisplacementTuningResult(pd.DataFrame(), pd.DataFrame(), LatentDisplacementCandidate(0.4, 5, (0.2, 1.2)), pd.DataFrame(), heldout, history)


def test_report_tables_and_plots_are_readable():
    result = make_result()
    top = top_trials_table(result)
    conclusion = heldout_conclusion(result)
    progress, _ = plot_optimization_progress(result)
    parameters, axes = plot_hyperparameter_performance(result)
    quantiles, quantile_axes = plot_hyperparameter_quantile_bands(result, n_bins=2)

    assert top.loc[0, "trial"] == 2
    assert conclusion.loc[0, "mean_roc_auc_advantage"] == 0.02
    assert len(axes.flat) == 6
    assert len(quantile_axes.flat) == 4
    plt.close(progress)
    plt.close(parameters)
    plt.close(quantiles)


def test_partial_report_without_heldout_results_can_be_saved(tmp_path):
    result = make_result()
    result.heldout_summary = pd.DataFrame()
    result.heldout_results = pd.DataFrame()

    manifest = save_optimization_report(result, tmp_path)

    assert len(manifest) == 5
    conclusion = pd.read_csv(tmp_path / "tuning" / "latent_displacement_default_credit" / "report" / "heldout_conclusion.csv")
    assert conclusion.empty
    assert "verdict" in conclusion.columns
