"""Helpers for synthesizing images from MIMIC vision embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
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
    k: int = 1,
    random_state: int = 0,
) -> tuple[int, int]:
    """Select one row and one neighbor from its nearest non-self neighbors."""

    values = np.asarray(embeddings, dtype="float32")
    if values.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix.")
    if len(values) < 2:
        raise ValueError("At least two embeddings are required.")
    if int(k) < 1:
        raise ValueError("k must be at least 1.")

    rng = np.random.default_rng(random_state)
    if source_index is None:
        source_index = int(rng.integers(0, len(values)))
    if not 0 <= int(source_index) < len(values):
        raise IndexError(f"source_index={source_index} is outside the range [0, {len(values)}).")

    neighbors = NearestNeighbors(n_neighbors=min(int(k) + 1, len(values)))
    neighbors.fit(values)
    indices = neighbors.kneighbors(values[[source_index]], return_distance=False)[0]
    neighbor_candidates = [int(index) for index in indices if int(index) != int(source_index)][: int(k)]
    if not neighbor_candidates:
        raise ValueError("Could not find a non-self neighbor.")
    neighbor_index = neighbor_candidates[0]
    if len(neighbor_candidates) > 1:
        neighbor_index = int(rng.choice(neighbor_candidates))
    return int(source_index), neighbor_index


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
    size: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (7, 2.4),
):
    """Display source, synthesized SMOTE image, and neighbor side by side."""

    label_array = None if labels is None else np.asarray(labels)
    panels = [
        ("source", synthesis.source_index, images[synthesis.source_index]),
        (f"SMOTE lambda={synthesis.lambda_value:g}", None, synthesis.image),
        ("neighbor", synthesis.neighbor_index, images[synthesis.neighbor_index]),
    ]

    if size is not None:
        figsize = (float(size[0]) * len(panels), float(size[1]))
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


def plot_smote_differences(
    images: np.ndarray,
    synthesis: SmoteSynthesis,
    *,
    mode: str = "signed",
    threshold: float = 0.5,
    size: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (5, 2.4),
):
    """Display how the synthesized image differs from both endpoints."""

    source = _display_image(images[synthesis.source_index])
    neighbor = _display_image(images[synthesis.neighbor_index])
    smote = _display_image(synthesis.image)
    panels = [
        ("source vs SMOTE", _image_difference(source, smote, mode=mode, threshold=threshold)),
        ("neighbor vs SMOTE", _image_difference(neighbor, smote, mode=mode, threshold=threshold)),
    ]

    if size is not None:
        figsize = (float(size[0]) * len(panels), float(size[1]))
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (title, image) in zip(axes, panels):
        if mode == "signed":
            magnitude = float(np.max(np.abs(image)))
            norm = TwoSlopeNorm(vmin=-magnitude, vcenter=0.0, vmax=magnitude) if magnitude > 0 else None
            ax.imshow(image, cmap="bwr_r", norm=norm)
        else:
            ax.imshow(image, cmap="magma" if mode == "absolute" else "gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


def _display_image(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(image), 0.0, 1.0)
    if clipped.ndim == 3 and clipped.shape[-1] == 1:
        return clipped[..., 0]
    return clipped


def _image_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    mode: str,
    threshold: float,
) -> np.ndarray:
    if mode == "signed":
        return np.asarray(right, dtype="float32") - np.asarray(left, dtype="float32")
    if mode == "absolute":
        return np.abs(np.asarray(left, dtype="float32") - np.asarray(right, dtype="float32"))
    if mode == "xor":
        left_binary = np.asarray(left) >= threshold
        right_binary = np.asarray(right) >= threshold
        return np.logical_xor(left_binary, right_binary).astype("float32")
    raise ValueError("mode must be 'signed', 'absolute', or 'xor'.")
