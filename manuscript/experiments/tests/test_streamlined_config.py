from pathlib import Path

from streamlined.config import METHOD_KEYS, artifact_paths, load_config


def test_load_smoke_and_full_configs():
    smoke = load_config(Path("manuscript/experiments/configs/smoke.yaml"))
    full = load_config(Path("manuscript/experiments/configs/full.yaml"))

    assert smoke.profile.name == "smoke"
    assert full.profile.name == "full"
    assert set(smoke.profile.methods) == set(METHOD_KEYS)
    assert full.profile.imbalance_ratios == (2.0, 3.0, 5.0, 10.0)
    assert smoke.aulc_segments["early"] == (None, 128)


def test_artifact_paths_are_under_configured_root(tmp_path):
    config = load_config(Path("manuscript/experiments/configs/smoke.yaml"), artifact_dir=tmp_path)
    paths = artifact_paths(config)

    assert paths["raw_results"] == tmp_path / "raw" / "condition_results.csv"
    assert paths["figures"] == tmp_path / "figures"
