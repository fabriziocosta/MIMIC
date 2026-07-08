"""Dataset loading helpers for MIMIC vision experiments."""

from __future__ import annotations

import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


@dataclass(frozen=True)
class VisionDataset:
    """Container for vectorized image data and its display metadata."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    images: np.ndarray
    image_shape: tuple[int, ...]
    target_names: dict[int, str]


@dataclass(frozen=True)
class VisionEmbedding:
    """Container for MIMIC image embeddings and their source metadata."""

    dataset_file: str
    embeddings: np.ndarray
    mode: str
    capacity: float
    random_state: int


SERIALIZED_DATASET_DIR = Path("vision/data/serialized")
SERIALIZED_EMBEDDING_DIR = Path("vision/data/embeddings")


OPENML_DATASETS = {
    "mnist": {
        "openml_name": "mnist_784",
        "version": 1,
        "image_shape": (28, 28),
        "target_names": {i: str(i) for i in range(10)},
    },
    "fashion_mnist": {
        "openml_name": "Fashion-MNIST",
        "version": 1,
        "image_shape": (28, 28),
        "target_names": {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot",
        },
    },
}

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_TARGET_NAMES = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def load_vision_dataset(
    name: str,
    *,
    targets: Iterable[int] | None = None,
    n_per_target: int | None = None,
    data_dir: str | Path = "vision/data",
    split: str = "train",
    random_state: int = 0,
    as_frame: bool = True,
) -> VisionDataset:
    """Download, filter, normalize, and vectorize a supported vision dataset.

    Parameters
    ----------
    name:
        One of ``"mnist"``, ``"fashion_mnist"``, or ``"cifar10"``.
    targets:
        Optional class ids to keep. For example, ``[3, 8]`` keeps only images
        whose target is 3 or 8.
    n_per_target:
        Optional maximum number of images to keep per target after filtering.
    data_dir:
        Local cache directory for downloaded data.
    split:
        ``"train"``, ``"test"``, or ``"all"``. MNIST-style datasets use the
        standard first 60,000 examples as train and the remaining 10,000 as test.
    random_state:
        Seed used when sampling ``n_per_target``.
    as_frame:
        If true, return ``X`` as a pandas DataFrame. This is convenient for the
        tabular MIMIC estimator.
    """

    normalized_name = _normalize_dataset_name(name)
    if normalized_name == "cifar10":
        images, y, image_shape, target_names = _load_cifar10(Path(data_dir), split=split)
    else:
        images, y, image_shape, target_names = _load_openml_dataset(normalized_name, Path(data_dir), split=split)

    keep = _filtered_indices(y, targets=targets, n_per_target=n_per_target, random_state=random_state)
    images = images[keep]
    y = y[keep]

    flat = images.reshape(len(images), -1)
    X = _as_feature_frame(flat, normalized_name) if as_frame else flat
    y_series = pd.Series(y, name="target")
    return VisionDataset(
        name=normalized_name,
        X=X,
        y=y_series,
        images=images,
        image_shape=image_shape,
        target_names=target_names,
    )


def vision_dataset_filename(dataset: VisionDataset, *, split: str | None = None) -> str:
    """Return a readable filename for a filtered, vectorized vision dataset."""

    classes = "-".join(str(int(value)) for value in sorted(dataset.y.unique()))
    shape = "x".join(str(part) for part in dataset.image_shape)
    split_part = f"_{_clean_filename_part(split)}" if split else ""
    return f"{dataset.name}{split_part}_n{len(dataset.y)}_classes-{classes}_{shape}.pkl"


def save_vision_dataset(
    dataset: VisionDataset,
    *,
    output_dir: str | Path = SERIALIZED_DATASET_DIR,
    split: str | None = None,
    filename: str | None = None,
) -> Path:
    """Serialize a prepared vision dataset and return the written path."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / (filename or vision_dataset_filename(dataset, split=split))
    with file_path.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return file_path


def load_serialized_vision_dataset(
    filename: str | Path,
    *,
    input_dir: str | Path = SERIALIZED_DATASET_DIR,
) -> VisionDataset:
    """Load a dataset previously written by ``save_vision_dataset``."""

    file_path = Path(filename)
    if not file_path.is_absolute():
        file_path = Path(input_dir) / file_path
    with file_path.open("rb") as handle:
        dataset = pickle.load(handle)
    if not isinstance(dataset, VisionDataset):
        raise TypeError(f"{file_path} does not contain a VisionDataset object.")
    return dataset


def vision_embedding_filename(
    *,
    dataset_file: str | Path,
    embeddings: np.ndarray,
    mode: str,
    capacity: float,
) -> str:
    """Return a readable filename for a MIMIC image embedding artifact."""

    dataset_stem = Path(dataset_file).stem
    mode_part = _clean_filename_part(str(mode))
    capacity_part = str(capacity).replace(".", "p")
    return f"{dataset_stem}_mimic-{mode_part}_cap{capacity_part}_emb{embeddings.shape[1]}.pkl"


def save_vision_embedding(
    embedding: VisionEmbedding,
    *,
    output_dir: str | Path = SERIALIZED_EMBEDDING_DIR,
    filename: str | None = None,
) -> Path:
    """Serialize a prepared MIMIC image embedding artifact."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / (
        filename
        or vision_embedding_filename(
            dataset_file=embedding.dataset_file,
            embeddings=embedding.embeddings,
            mode=embedding.mode,
            capacity=embedding.capacity,
        )
    )
    with file_path.open("wb") as handle:
        pickle.dump(embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return file_path


def load_serialized_vision_embedding(
    filename: str | Path,
    *,
    input_dir: str | Path = SERIALIZED_EMBEDDING_DIR,
) -> VisionEmbedding:
    """Load a MIMIC image embedding artifact."""

    file_path = Path(filename)
    if not file_path.is_absolute():
        file_path = Path(input_dir) / file_path
    with file_path.open("rb") as handle:
        embedding = pickle.load(handle)
    if not isinstance(embedding, VisionEmbedding):
        raise TypeError(f"{file_path} does not contain a VisionEmbedding object.")
    return embedding


def _normalize_dataset_name(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    aliases = {
        "fashion": "fashion_mnist",
        "fashionmnist": "fashion_mnist",
        "fmnist": "fashion_mnist",
        "cifar_10": "cifar10",
    }
    normalized = aliases.get(normalized, normalized)
    supported = {"mnist", "fashion_mnist", "cifar10"}
    if normalized not in supported:
        raise ValueError(f"Unsupported dataset {name!r}. Expected one of {sorted(supported)}.")
    return normalized


def _load_openml_dataset(name: str, data_dir: Path, split: str):
    spec = OPENML_DATASETS[name]
    bunch = fetch_openml(
        spec["openml_name"],
        version=spec["version"],
        data_home=str(data_dir / "openml"),
        as_frame=False,
        parser="auto",
    )
    X = bunch.data.astype("float32") / 255.0
    y = bunch.target.astype(int)
    images = X.reshape((-1, *spec["image_shape"]))
    images, y = _apply_split(images, y, split=split, train_size=60_000)
    return images, y, spec["image_shape"], spec["target_names"]


def _load_cifar10(data_dir: Path, split: str):
    archive_path = data_dir / "cifar-10-python.tar.gz"
    extracted_dir = data_dir / "cifar-10-batches-py"
    if not extracted_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        if not archive_path.exists():
            urllib.request.urlretrieve(CIFAR10_URL, archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(data_dir)

    if split == "train":
        batch_names = [f"data_batch_{i}" for i in range(1, 6)]
    elif split == "test":
        batch_names = ["test_batch"]
    elif split == "all":
        batch_names = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    else:
        raise ValueError("split must be 'train', 'test', or 'all'")

    arrays = []
    labels = []
    for batch_name in batch_names:
        with (extracted_dir / batch_name).open("rb") as handle:
            batch = pickle.load(handle, encoding="latin1")
        arrays.append(batch["data"])
        labels.extend(batch["labels"])

    flat = np.vstack(arrays).astype("float32") / 255.0
    images = flat.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return images, np.asarray(labels, dtype=int), (32, 32, 3), CIFAR10_TARGET_NAMES


def _apply_split(images: np.ndarray, y: np.ndarray, *, split: str, train_size: int):
    if split == "train":
        return images[:train_size], y[:train_size]
    if split == "test":
        return images[train_size:], y[train_size:]
    if split == "all":
        return images, y
    raise ValueError("split must be 'train', 'test', or 'all'")


def _filtered_indices(
    y: np.ndarray,
    *,
    targets: Iterable[int] | None,
    n_per_target: int | None,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    target_values = sorted(np.unique(y).tolist()) if targets is None else list(targets)
    selected = []
    for target in target_values:
        positions = np.flatnonzero(y == int(target))
        if len(positions) == 0:
            raise ValueError(f"Target {target!r} is not present in the selected split.")
        if n_per_target is not None:
            if n_per_target <= 0:
                raise ValueError("n_per_target must be positive when supplied.")
            size = min(n_per_target, len(positions))
            positions = np.sort(rng.choice(positions, size=size, replace=False))
        selected.append(positions)
    if not selected:
        return np.array([], dtype=int)
    return np.concatenate(selected)


def _as_feature_frame(flat: np.ndarray, dataset_name: str) -> pd.DataFrame:
    prefix = "px"
    columns = [f"{prefix}_{i:04d}" for i in range(flat.shape[1])]
    return pd.DataFrame(flat, columns=columns)


def _clean_filename_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value.lower())
