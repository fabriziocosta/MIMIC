from pathlib import Path
import os

import pytest

from streamlined.config import (
    METHOD_KEYS,
    artifact_paths,
    experiment_artifact_dir,
    load_config,
    resolve_experiment_artifact_dir,
    with_dataset_n_rows,
)


def test_load_smoke_and_full_configs():
    smoke = load_config(Path("manuscript/experiments/configs/smoke.yaml"))
    full = load_config(Path("manuscript/experiments/configs/full.yaml"))

    assert smoke.profile.name == "smoke"
    assert full.profile.name == "full"
    assert set(smoke.profile.methods) == set(METHOD_KEYS)
    assert full.profile.imbalance_ratios == (2.0, 3.0, 5.0, 10.0)
    assert smoke.aulc_segments["early"] == (None, 128)
    assert smoke.repair_direct_samples is True


def test_artifact_paths_are_under_configured_root(tmp_path):
    config = load_config(Path("manuscript/experiments/configs/smoke.yaml"), artifact_dir=tmp_path)
    paths = artifact_paths(config)

    assert paths["raw_results"] == tmp_path / "raw" / "condition_results.csv"
    assert paths["figures"] == tmp_path / "figures"


def test_experiment_artifact_dir_isolates_named_runs(tmp_path):
    assert experiment_artifact_dir(tmp_path, "latent-displacement-best") == (
        tmp_path / "latent-displacement-best"
    )


@pytest.mark.parametrize("name", ["", "../other-run", "run/name", "run name"])
def test_experiment_artifact_dir_rejects_unsafe_names(tmp_path, name):
    with pytest.raises(ValueError, match="experiment_name"):
        experiment_artifact_dir(tmp_path, name)


def test_resolve_experiment_artifact_dir_uses_named_run(tmp_path):
    assert resolve_experiment_artifact_dir(tmp_path, "chosen-run") == (
        tmp_path / "chosen-run"
    )


def test_resolve_experiment_artifact_dir_finds_latest_results(tmp_path):
    older = tmp_path / "older" / "raw" / "condition_results.csv"
    latest = tmp_path / "latest" / "raw" / "condition_results.csv"
    older.parent.mkdir(parents=True)
    latest.parent.mkdir(parents=True)
    older.touch()
    latest.touch()
    os.utime(older, ns=(1, 1))
    os.utime(latest, ns=(2, 2))

    assert resolve_experiment_artifact_dir(tmp_path, None) == tmp_path / "latest"


def test_resolve_experiment_artifact_dir_requires_available_results(tmp_path):
    with pytest.raises(FileNotFoundError, match="No experiment results"):
        resolve_experiment_artifact_dir(tmp_path, None)


def test_with_dataset_n_rows_overrides_every_profile_dataset():
    config = load_config(Path("manuscript/experiments/configs/full.yaml"))

    updated = with_dataset_n_rows(config, 10_000)

    assert updated.profile.dataset_n_rows == {
        dataset_key: 10_000 for dataset_key in config.profile.datasets
    }
    assert config.profile.dataset_n_rows == {}


def test_with_dataset_n_rows_rejects_non_positive_values():
    config = load_config(Path("manuscript/experiments/configs/full.yaml"))

    with pytest.raises(ValueError, match="at least 1"):
        with_dataset_n_rows(config, 0)
