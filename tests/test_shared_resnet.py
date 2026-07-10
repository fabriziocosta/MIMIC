import numpy as np
import pandas as pd
import pytest

from mimic import MIMIC, SharedFeatureGroup, SharedResNetEncoder
from mimic_vision import VisionDataset, vision_feature_group


def _encoder(**kwargs):
    params = {
        "embedding_dim": 3,
        "feature_embedding_dim": 2,
        "coordinate_frequencies": 2,
        "hidden_dim": 8,
        "n_layers": 2,
        "max_epochs": 2,
        "patience": 1,
        "batch_size": 6,
        "validation_fraction": 0.2,
        "random_state": 0,
    }
    params.update(kwargs)
    return SharedResNetEncoder(**params)


def _frame(n=14):
    rng = np.random.default_rng(4)
    a = rng.normal(size=n)
    b = 0.5 * a + rng.normal(scale=0.2, size=n)
    c = a - b + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c})


def _model(frame, *, bootstrap=False):
    group = SharedFeatureGroup(
        columns=list(frame.columns),
        coordinates=np.asarray([[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]]),
        encoder=_encoder(max_epochs=1 if bootstrap else 2),
        target_chunk_size=2,
    )
    return MIMIC(
        columns={"regression": list(frame.columns), "classification": [], "ignore": []},
        mode="direct",
        bootstrap=bootstrap,
        n_bootstrap=1,
        random_state=0,
        shared_feature_groups={"pixels": group},
        show_progress=False,
    ).fit(frame)


def test_shared_resnet_masks_target_and_preserves_embedding_blocks(tmp_path):
    frame = _frame()
    model = _model(frame)
    embedding = model.transform(frame)

    assert embedding.shape == (len(frame), 3 * 3)
    assert np.isfinite(embedding).all()
    assert list(model.embedding_slices_) == list(frame.columns)
    shared = model.shared_group_modules_["pixels"].full_encoder
    assert all(
        module.full_member.encoder.shared_encoder is shared
        for module in model.feature_modules_.values()
    )

    changed_target = frame.copy()
    changed_target["a"] += 100.0
    a_slice = model.embedding_slices_["a"]
    np.testing.assert_allclose(
        embedding[:, a_slice], model.transform(changed_target)[:, a_slice], atol=1e-6
    )

    changed_context = frame.copy()
    changed_context["b"] += 100.0
    assert not np.allclose(
        embedding[:, a_slice], model.transform(changed_context)[:, a_slice]
    )

    decoded = model.decode(embedding)
    assert decoded.shape == frame.shape
    generated = model.sample(2)
    assert generated.shape == (2, 3)
    assert np.isfinite(generated.to_numpy(dtype=float)).all()
    confidence = model.confidence(frame.iloc[:2], columns=["a"])
    assert len(confidence) == 2
    missing = frame.copy()
    missing.loc[0, "a"] = np.nan
    assert np.isfinite(model.impute(missing).loc[0, "a"])

    path = tmp_path / "shared.joblib"
    model.save(path)
    restored = MIMIC.load(path)
    np.testing.assert_allclose(restored.transform(frame), embedding, atol=1e-6)


def test_shared_resnet_bootstrap_is_shared_across_targets():
    frame = _frame(12)
    model = _model(frame, bootstrap=True)
    group = model.shared_group_modules_["pixels"]
    assert len(group.encoders) == 1
    assert all(len(module.members) == 1 for module in model.feature_modules_.values())
    assert all(
        module.members[0].encoder.shared_encoder is group.encoders[0]
        for module in model.feature_modules_.values()
    )
    encoder = group.encoders[0]
    assert not set(encoder.training_row_ids_) & set(encoder.validation_row_ids_)


@pytest.mark.parametrize(
    "group, message",
    [
        (SharedFeatureGroup(["a", "a"], _encoder()), "duplicate"),
        (SharedFeatureGroup(["missing"], _encoder()), "unknown"),
        (
            SharedFeatureGroup(["a", "b"], _encoder(), coordinates=np.asarray([[0.0, 0.0]])),
            "one row per column",
        ),
    ],
)
def test_shared_group_schema_validation(group, message):
    frame = _frame()
    model = MIMIC(
        columns={"regression": list(frame.columns), "classification": [], "ignore": []},
        mode="direct",
        bootstrap=False,
        shared_feature_groups={"bad": group},
        show_progress=False,
    )
    with pytest.raises(ValueError, match=message):
        model.fit(frame)


def test_fourier_descriptors_and_vision_coordinates_are_deterministic():
    coordinates = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    first = SharedResNetEncoder.fourier_descriptors(coordinates, frequencies=2)
    second = SharedResNetEncoder.fourier_descriptors(coordinates, frequencies=2)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 10)

    images = np.zeros((2, 2, 3), dtype=np.float32)
    dataset = VisionDataset(
        name="tiny",
        X=pd.DataFrame(images.reshape(2, -1), columns=[f"px_{i}" for i in range(6)]),
        y=pd.Series([0, 1]),
        images=images,
        image_shape=(2, 3),
        target_names={0: "zero", 1: "one"},
    )
    group = vision_feature_group(dataset, encoder=_encoder(), target_chunk_size=3)
    np.testing.assert_array_equal(
        group.coordinates,
        np.asarray(
            [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]],
            dtype=np.float32,
        ),
    )

    rgb_images = np.zeros((1, 2, 2, 3), dtype=np.float32)
    rgb_dataset = VisionDataset(
        name="tiny_rgb",
        X=pd.DataFrame(rgb_images.reshape(1, -1), columns=[f"px_{i}" for i in range(12)]),
        y=pd.Series([0]),
        images=rgb_images,
        image_shape=(2, 2, 3),
        target_names={0: "zero"},
    )
    rgb_group = vision_feature_group(rgb_dataset, encoder=_encoder())
    assert rgb_group.coordinates.shape == (12, 3)
    np.testing.assert_array_equal(
        rgb_group.coordinates[:3, 2], np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    )
