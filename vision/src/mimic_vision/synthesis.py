"""Helpers for synthesizing images from MIMIC vision embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class SmoteSynthesis:
    """Result of interpolating two image embeddings and decoding the midpoint."""

    source_index: int
    neighbor_index: int
    lambda_value: float
    embedding: np.ndarray
    vector: np.ndarray
    image: np.ndarray


def select_embedding_neighbors(
    embeddings: np.ndarray,
    *,
    source_index: int | None = None,
    random_state: int = 0,
) -> tuple[int, int]:
    """Select one row and its nearest non-self neighbor in embedding space."""

    values = np.asarray(embeddings, dtype="float32")
    if values.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix.")
    if len(values) < 2:
        raise ValueError("At least two embeddings are required.")

    if source_index is None:
        rng = np.random.default_rng(random_state)
        source_index = int(rng.integers(0, len(values)))
    if not 0 <= int(source_index) < len(values):
        raise IndexError(f"source_index={source_index} is outside the range [0, {len(values)}).")

    neighbors = NearestNeighbors(n_neighbors=min(2, len(values)))
    neighbors.fit(values)
    indices = neighbors.kneighbors(values[[source_index]], return_distance=False)[0]
    neighbor_candidates = [int(index) for index in indices if int(index) != int(source_index)]
    if not neighbor_candidates:
        raise ValueError("Could not find a non-self neighbor.")
    return int(source_index), neighbor_candidates[0]


def synthesize_smote_image(
    model,
    embeddings: np.ndarray,
    *,
    source_index: int,
    neighbor_index: int,
    image_shape: tuple[int, ...],
    lambda_value: float = 0.5,
) -> SmoteSynthesis:
    """Interpolate two embeddings, decode the result, and reshape it as an image."""

    values = np.asarray(embeddings, dtype="float32")
    if values.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix.")
    for name, index in {"source_index": source_index, "neighbor_index": neighbor_index}.items():
        if not 0 <= int(index) < len(values):
            raise IndexError(f"{name}={index} is outside the range [0, {len(values)}).")
    if not 0.0 <= float(lambda_value) <= 1.0:
        raise ValueError("lambda_value must be between 0 and 1.")

    embedding = (1.0 - lambda_value) * values[source_index] + lambda_value * values[neighbor_index]
    if hasattr(model, "decode_embedding"):
        decoded = model.decode_embedding(embedding.reshape(1, -1))
    else:
        decoded = model.decode(embedding.reshape(1, -1))
    vector = decoded.iloc[0].to_numpy(dtype="float32")
    image = np.clip(vector.reshape(image_shape), 0.0, 1.0)
    return SmoteSynthesis(
        source_index=int(source_index),
        neighbor_index=int(neighbor_index),
        lambda_value=float(lambda_value),
        embedding=embedding,
        vector=vector,
        image=image,
    )


def plot_smote_synthesis(
    images: np.ndarray,
    synthesis: SmoteSynthesis,
    *,
    labels=None,
    target_names: dict[int, str] | None = None,
    figsize: tuple[float, float] = (7, 2.4),
):
    """Display source, neighbor, and synthesized SMOTE image side by side."""

    label_array = None if labels is None else np.asarray(labels)
    panels = [
        ("source", synthesis.source_index, images[synthesis.source_index]),
        ("neighbor", synthesis.neighbor_index, images[synthesis.neighbor_index]),
        (f"SMOTE lambda={synthesis.lambda_value:g}", None, synthesis.image),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (name, index, image) in zip(axes, panels):
        display_image = _display_image(image)
        ax.imshow(display_image, cmap="gray" if display_image.ndim == 2 else None)
        title = name if index is None else f"{name}: {index}"
        if index is not None and label_array is not None:
            label = int(label_array[index])
            label_name = target_names.get(label, str(label)) if target_names is not None else str(label)
            title = f"{title}\nlabel: {label_name}"
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


def _display_image(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(image), 0.0, 1.0)
    if clipped.ndim == 3 and clipped.shape[-1] == 1:
        return clipped[..., 0]
    return clipped
