"""Visualization helpers for vectorized image embeddings."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances


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
    if method in {"canonical_mds", "classical_mds"}:
        return _classical_mds_2d(values, random_state=random_state)
    if method == "tsne":
        perplexity = min(30, max(2, (len(values) - 1) // 3))
        return TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=random_state,
        ).fit_transform(values)
    raise ValueError("method must be 'pca', 'tsne', or 'canonical_mds'")


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


def reference_axis_coordinates(
    X,
    *,
    origin_index: int,
    x_index: int,
    y_index: int,
    vector_space: str = "raw",
    embedding_vectors=None,
) -> np.ndarray:
    """Project rows onto plot axes defined by reference rows O, X, Y.

    By default, projection happens in the original flattened image vectors. Set
    ``vector_space="embedding"`` and pass ``embedding_vectors`` to use a future
    MIMIC image embedding instead. The plotted x-axis is the ``X - O`` direction
    and the plotted y-axis is the ``Y - O`` direction, so ``O`` maps to the plot
    origin, ``X`` maps to the x-axis, and ``Y`` maps to the y-axis.
    """

    values = _resolve_projection_vectors(X, vector_space=vector_space, embedding_vectors=embedding_vectors)
    if values.ndim != 2:
        raise ValueError("X must be a 2D array or dataframe of flattened images.")
    _validate_reference_indices(len(values), origin_index=origin_index, x_index=x_index, y_index=y_index)

    centered = values - values[origin_index]
    x_direction = values[x_index] - values[origin_index]
    y_direction = values[y_index] - values[origin_index]

    basis = np.column_stack([x_direction, y_direction])
    if np.linalg.matrix_rank(basis) < 2:
        raise ValueError("X - O and Y - O must define two independent directions.")

    coefficients, *_ = np.linalg.lstsq(basis, centered.T, rcond=None)
    axis_lengths = np.array([np.linalg.norm(x_direction), np.linalg.norm(y_direction)])
    return (coefficients.T * axis_lengths).astype("float32", copy=False)


def plot_reference_axis_embedding(
    X,
    images: np.ndarray,
    *,
    references: dict[str, int] | None = None,
    origin_index: int | None = None,
    x_index: int | None = None,
    y_index: int | None = None,
    vector_space: str = "raw",
    embedding_vectors=None,
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
    """Plot images in the coordinate system defined by reference images O, X, Y."""

    origin_index, x_index, y_index = _resolve_reference_indices(
        references=references,
        origin_index=origin_index,
        x_index=x_index,
        y_index=y_index,
    )
    coordinates = reference_axis_coordinates(
        X,
        origin_index=origin_index,
        x_index=x_index,
        y_index=y_index,
        vector_space=vector_space,
        embedding_vectors=embedding_vectors,
    )
    fig, ax, selected = plot_image_embedding(
        coordinates,
        images,
        labels=labels,
        target_names=target_names,
        title=title or f"Reference projection: O={origin_index}, X={x_index}, Y={y_index}",
        max_images=max_images,
        min_distance_px=min_distance_px,
        zoom=zoom,
        point_size=point_size,
        alpha=alpha,
        figsize=figsize,
    )
    ax.scatter(
        coordinates[[origin_index, x_index, y_index], 0],
        coordinates[[origin_index, x_index, y_index], 1],
        color="black",
        marker="x",
        s=80,
        linewidths=2,
        label="reference",
    )
    for name, index in {"O": origin_index, "X": x_index, "Y": y_index}.items():
        ax.annotate(name, coordinates[index], xytext=(6, 6), textcoords="offset points", weight="bold")
    ax.set_xlabel(f"{vector_space} O-X axis")
    ax.set_ylabel(f"{vector_space} O-Y axis")
    return fig, ax, selected, coordinates


def plot_reference_images(
    images: np.ndarray,
    references: dict[str, int],
    *,
    labels=None,
    target_names: dict[int, str] | None = None,
    figsize: tuple[float, float] = (6, 2.2),
):
    """Display the O, X, and Y reference images before plotting their projection."""

    image_array = np.asarray(images)
    label_array = None if labels is None else np.asarray(labels)
    expected = ["O", "X", "Y"]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, name in zip(axes, expected):
        index = int(references[name])
        image = _display_image(image_array[index])
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        title = f"{name}: {index}"
        if label_array is not None:
            label = int(label_array[index])
            label_name = target_names.get(label, str(label)) if target_names is not None else str(label)
            title = f"{title}\nlabel: {label_name}"
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


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


def _classical_mds_2d(X: np.ndarray, *, random_state: int | None = None) -> np.ndarray:
    del random_state
    D2 = pairwise_distances(X, metric="euclidean", squared=True)
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:2]
    vals = np.maximum(vals[order], 0)
    coords = vecs[:, order] * np.sqrt(vals)
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    return coords


def _validate_reference_indices(
    n_rows: int,
    *,
    origin_index: int,
    x_index: int,
    y_index: int,
) -> None:
    indices = {"origin_index": origin_index, "x_index": x_index, "y_index": y_index}
    for name, index in indices.items():
        if not 0 <= int(index) < n_rows:
            raise IndexError(f"{name}={index} is outside the range [0, {n_rows}).")
    if len({int(origin_index), int(x_index), int(y_index)}) != 3:
        raise ValueError("origin_index, x_index, and y_index must identify three distinct rows.")


def _resolve_projection_vectors(X, *, vector_space: str, embedding_vectors):
    vector_space = vector_space.lower()
    if vector_space == "raw":
        return np.asarray(X, dtype="float32")
    if vector_space == "embedding":
        if embedding_vectors is None:
            raise ValueError("embedding_vectors must be provided when vector_space='embedding'.")
        return np.asarray(embedding_vectors, dtype="float32")
    raise ValueError("vector_space must be 'raw' or 'embedding'.")


def _resolve_reference_indices(
    *,
    references: dict[str, int] | None,
    origin_index: int | None,
    x_index: int | None,
    y_index: int | None,
) -> tuple[int, int, int]:
    if references is not None:
        missing = {"O", "X", "Y"} - set(references)
        if missing:
            raise ValueError(f"references is missing keys: {sorted(missing)}")
        return int(references["O"]), int(references["X"]), int(references["Y"])
    if origin_index is None or x_index is None or y_index is None:
        raise ValueError("Provide either references={'O': ..., 'X': ..., 'Y': ...} or all three index arguments.")
    return int(origin_index), int(x_index), int(y_index)


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
