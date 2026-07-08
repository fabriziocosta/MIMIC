"""Visualization helpers for vectorized image embeddings."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def compute_2d_embedding(
    X,
    *,
    method: str = "pca",
    random_state: int = 0,
    max_points: int | None = None,
) -> np.ndarray:
    """Compute a two-dimensional layout for image rows."""

    values = np.asarray(X, dtype="float32")
    if max_points is not None:
        values = values[:max_points]
    if values.ndim != 2:
        raise ValueError("X must be a 2D array or dataframe of flattened images.")
    if len(values) < 2:
        raise ValueError("At least two rows are required for a 2D embedding.")

    method = method.lower()
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(values)
    if method == "tsne":
        perplexity = min(30, max(2, (len(values) - 1) // 3))
        return TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=random_state,
        ).fit_transform(values)
    raise ValueError("method must be 'pca' or 'tsne'")


def plot_image_embedding(
    coordinates: np.ndarray,
    images: np.ndarray,
    *,
    labels=None,
    target_names: dict[int, str] | None = None,
    title: str | None = None,
    max_images: int = 250,
    min_distance_px: float = 24.0,
    zoom: float = 0.55,
    point_size: float = 8,
    alpha: float = 0.25,
    figsize: tuple[float, float] = (8, 7),
):
    """Plot a 2D image layout with thumbnails only where they do not overlap.

    A faint scatter plot shows all points. Thumbnails are then placed greedily in
    display coordinates. If a candidate thumbnail would be too close to an
    already placed thumbnail, it is skipped.
    """

    coords = np.asarray(coordinates)
    image_array = np.asarray(images)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_samples, 2).")
    if len(coords) != len(image_array):
        raise ValueError("coordinates and images must have the same length.")

    labels_array = None if labels is None else np.asarray(labels)
    if labels_array is not None and len(labels_array) != len(coords):
        raise ValueError("labels must have the same length as coordinates.")

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels_array,
        s=point_size,
        cmap="tab10" if labels_array is not None else None,
        color=None if labels_array is not None else "#9aa0a6",
        alpha=alpha,
        linewidths=0,
    )

    ax.set_title(title or "Image layout")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.margins(0.08)
    fig.canvas.draw()

    selected = _non_overlapping_indices(
        ax,
        coords,
        max_images=max_images,
        min_distance_px=min_distance_px,
    )
    for index in selected:
        image = _display_image(image_array[index])
        thumbnail = OffsetImage(image, cmap="gray" if image.ndim == 2 else None, zoom=zoom)
        box = AnnotationBbox(
            thumbnail,
            coords[index],
            frameon=True,
            pad=0.12,
            bboxprops={"edgecolor": "white", "linewidth": 0.8, "alpha": 0.9},
        )
        ax.add_artist(box)

    if labels_array is not None:
        _add_label_legend(ax, scatter, labels_array, target_names)

    fig.tight_layout()
    return fig, ax, selected


def _non_overlapping_indices(
    ax,
    coords: np.ndarray,
    *,
    max_images: int,
    min_distance_px: float,
) -> list[int]:
    display_coords = ax.transData.transform(coords)
    center = np.median(display_coords, axis=0)
    order = np.argsort(np.linalg.norm(display_coords - center, axis=1))
    selected: list[int] = []
    selected_points: list[np.ndarray] = []

    for index in order:
        candidate = display_coords[index]
        if all(np.linalg.norm(candidate - point) >= min_distance_px for point in selected_points):
            selected.append(int(index))
            selected_points.append(candidate)
        if len(selected) >= max_images:
            break
    return selected


def _display_image(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(image, 0.0, 1.0)
    if clipped.ndim == 1:
        side = int(np.sqrt(len(clipped)))
        if side * side != len(clipped):
            raise ValueError("Flat images must have a square number of pixels.")
        return clipped.reshape(side, side)
    if clipped.ndim == 3 and clipped.shape[0] in (1, 3) and clipped.shape[-1] not in (1, 3):
        clipped = np.moveaxis(clipped, 0, -1)
    if clipped.ndim == 3 and clipped.shape[-1] == 1:
        return clipped[..., 0]
    return clipped


def _add_label_legend(ax, scatter, labels: np.ndarray, target_names: dict[int, str] | None):
    handles, legend_labels = scatter.legend_elements()
    unique_labels = sorted(np.unique(labels).tolist())
    if target_names is not None and len(handles) == len(unique_labels):
        legend_labels = [target_names.get(int(label), str(label)) for label in unique_labels]
    ax.legend(handles, legend_labels, title="target", frameon=False, loc="best")
