"""Shared feature-group construction for vectorized images."""

from __future__ import annotations

import numpy as np

from mimic import SharedFeatureGroup, SharedResNetEncoder

from .datasets import VisionDataset


def vision_feature_group(
    dataset: VisionDataset,
    *,
    encoder: SharedResNetEncoder,
    target_chunk_size: int = 64,
) -> SharedFeatureGroup:
    """Build a shared numerical group with normalized pixel coordinates."""

    shape = tuple(int(size) for size in dataset.image_shape)
    if len(shape) not in {2, 3}:
        raise ValueError("Vision feature groups require a 2D grayscale or 3D channel-last image shape")
    columns = list(dataset.X.columns)
    if int(np.prod(shape)) != len(columns):
        raise ValueError("The dataset image shape does not match its vectorized feature columns")
    grid = np.asarray(list(np.ndindex(shape)), dtype=np.float32)
    denominators = np.maximum(np.asarray(shape, dtype=np.float32) - 1.0, 1.0)
    coordinates = grid / denominators
    return SharedFeatureGroup(
        columns=columns,
        coordinates=coordinates,
        encoder=encoder,
        target_chunk_size=target_chunk_size,
    )
