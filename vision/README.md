# MIMIC Vision

MIMIC Vision applies the same MIMIC idea to images. Instead of treating a table
row as a set of mixed features, it treats each image as one row and each pixel
channel as a feature. The image is flattened into a vector, MIMIC learns the
feature-wise relationships between pixels, and the learned representation can be
used both as an image embedding and as a basis for generating new images.

The core abstraction stays unchanged:

- A row is one observation.
- A feature is one value in that row.
- Missingness, prediction, embedding, and generation are handled by modelling
  each feature from the rest of the row.

For vision data, the row is an image vector. For a grayscale `28 x 28` image,
the row has `784` pixel features. For an RGB `32 x 32` image, the row has
`3,072` pixel-channel features.

## Core Idea

An image dataset can be written as a matrix:

```text
n_images x n_pixel_features
```

For example:

```text
MNIST          60,000 x 784
Fashion-MNIST 60,000 x 784
CIFAR-10      50,000 x 3,072
```

MIMIC then applies its usual process to this matrix. Each pixel feature can be
predicted from the other pixel features, and the collection of feature-wise
models defines a latent representation of the image. That representation can be
used for:

- image embeddings for visualization, clustering, retrieval, or downstream
  classifiers;
- reconstruction by decoding an embedding back into pixel space;
- conditional repair, such as filling masked or corrupted image regions;
- image generation by sampling or perturbing learned image embeddings and
  decoding them back into images.

This is deliberately simple: MIMIC Vision does not start from convolutional
assumptions. It first asks how far the row-wise MIMIC framework can go when the
features happen to be pixels.

The shared vision workflow remains a pure tabular residual MLP. All pixels use
one `SharedResNetEncoder`; learned pixel identities and normalized Fourier
row/column coordinates condition its residual blocks through FiLM. RGB data
also includes a normalized channel coordinate. The target pixel is masked
before its embedding is computed.

## Datasets

### MNIST

MNIST contains grayscale handwritten digits. Each image is `28 x 28`, so each
image becomes a row with `784` pixel features.

MNIST is the smallest useful starting point because the images are low
resolution, single-channel, and visually simple. It is a good dataset for
checking whether MIMIC can learn meaningful image embeddings and generate
recognizable digit-like samples.

### Fashion-MNIST

Fashion-MNIST has the same shape as MNIST: grayscale `28 x 28` images flattened
to `784` pixel features. The classes are clothing categories rather than
digits.

Fashion-MNIST is useful because it keeps the same input size as MNIST while
making the visual structure more varied. It is a stronger test of whether the
embedding captures shape, texture, and class-level structure rather than only
simple digit strokes.

### CIFAR-10

CIFAR-10 contains RGB natural images. Each image is `32 x 32 x 3`, so each image
becomes a row with `3,072` pixel-channel features.

CIFAR-10 is a higher-dimensional and harder benchmark. It tests whether the same
vectorized-image approach can scale beyond small grayscale images into color
images with more complex object and background variation.

## Workflow

The expected workflow is:

1. Load an image dataset.
2. Normalize pixel values, typically to `[0, 1]`.
3. Flatten each image into a row vector.
4. Fit MIMIC on the resulting matrix.
5. Use the fitted model to extract embeddings, reconstruct images, fill masked
   pixels, or generate new samples.
6. Reshape generated vectors back to image tensors for visualization.

Conceptually:

```python
from mimic import MIMIC, SharedResNetEncoder
from mimic_vision import load_vision_dataset, vision_feature_group

dataset = load_vision_dataset("mnist", targets=[3, 8], n_per_target=200)
pixel_group = vision_feature_group(
    dataset,
    encoder=SharedResNetEncoder(
        embedding_dim=32,
        feature_embedding_dim=8,
        coordinate_frequencies=4,
    ),
    target_chunk_size=64,
)
model = MIMIC(
    columns={
        "regression": list(dataset.X.columns),
        "classification": [],
        "ignore": [],
    },
    mode="direct",
    bootstrap=False,
    shared_feature_groups={"pixels": pixel_group},
    random_state=0,
).fit(dataset.X)

embeddings = model.transform(dataset.X)
generated = model.sample(64)
generated_images = generated.to_numpy().reshape(64, *dataset.image_shape)
```

`transform()` retains one conditioned embedding block per pixel and concatenates
the blocks in dataframe column order. With `p` pixels and embedding width `d`,
the returned matrix has `p * d` columns. The feature identity and coordinates
condition the shared network; they do not replace the per-pixel blocks with one
pooled image vector.

## Generation

Generation follows the same logic as tabular MIMIC. The model learns an
embedding space from the training images, creates new points in or near that
space, and decodes those points back into pixel features. The generated vector
is then reshaped into an image.

For MNIST and Fashion-MNIST, generated samples are reshaped to:

```text
28 x 28
```

For CIFAR-10, generated samples are reshaped to:

```text
32 x 32 x 3
```

This makes image generation a direct extension of synthetic row generation:
instead of producing a synthetic table row, MIMIC produces a synthetic pixel row.

## Why This Matters

MIMIC Vision provides a bridge between tabular generative modelling and image
modelling. It keeps the same estimator interface and the same feature-wise
reasoning, but applies them to image vectors. That makes it possible to compare
image embedding, reconstruction, masking, and generation behavior using the same
machinery already used for tabular data.

The first practical targets are MNIST, Fashion-MNIST, and CIFAR-10 because they
cover a useful progression:

- simple grayscale digits;
- more varied grayscale objects;
- small color natural images.

Together, they give a compact benchmark path for developing the vision version
of MIMIC.

## Repository Layout

- `../src/mimic_vision/datasets.py` downloads MNIST, Fashion-MNIST, and CIFAR-10,
  normalizes pixels, vectorizes images, and filters targets. For example,
  `targets=[3, 8]` keeps only images labelled 3 or 8, and `n_per_target=200`
  keeps at most 200 images for each selected target. `resize_scale` can reduce
  image side lengths before vectorization; for example, `resize_scale=0.5`
  turns `28 x 28` into `14 x 14`. It can also serialize the prepared dataset
  with a readable filename such as `mnist_train_n400_classes-3-8_28x28.pkl`.
- `../src/mimic_vision/groups.py` converts a vectorized `VisionDataset` into a
  shared numerical feature group with coordinates aligned to flattened pixels.
- `../src/mimic_vision/visualization.py` computes simple 2D layouts and plots image
  thumbnails without overlap by skipping thumbnails that would collide with
  already placed images. It also supports reference-axis plots: choose three
  images named `O`, `X`, and `Y`, center vectors at `O`, and project each image
  onto plot axes where `O-X` is the x-axis and `O-Y` is the y-axis. Projection
  uses raw flattened pixel vectors by default, with an `embedding` option ready
  for a future MIMIC image embedding matrix.
- `notebooks/01_download_and_filter.ipynb` demonstrates dataset download,
  target filtering, per-target sampling, and serialization.
- `notebooks/02_fit_mimic_embeddings.ipynb` loads a serialized dataset by
  filename, fits MIMIC on the flattened image rows, computes embeddings with
  `transform`, and serializes the embedding matrix.
- `notebooks/03_visualise_2d_images.ipynb` loads a serialized dataset by
  filename, optionally loads a serialized embedding artifact, and demonstrates
  PCA plus reference-axis layouts with non-overlapping image thumbnails.
- `notebooks/04_smote_embedding_synthesis.ipynb` loads a serialized dataset,
  saved MIMIC model, and embedding artifact, selects two neighboring embeddings,
  interpolates between them, decodes the intermediate embedding, and displays
  the resynthesized image.
- `notebooks/05_iterated_mimic_embeddings.ipynb` fits an `IteratedMIMIC` stack,
  saves its top-level embeddings, and persists the fitted iterated model.
- `notebooks/06_iterated_smote_synthesis.ipynb` interpolates between top-level
  iterated embeddings and decodes through all levels back to an image.

`IteratedMIMIC` preserves each pixel or feature embedding as a vector block at
higher levels. A later level predicts the whole previous-level block for a
pixel or feature at once, rather than treating each embedding dimension as an
independent scalar target.

Notebook filename parameters accept either an explicit filename or `"last"`.
The default `"last"` selects the most recently generated matching dataset,
embedding, or model artifact from `data/vision/`.

Vision data is organized by lifecycle under `data/vision/`: downloaded source
data in `raw/`, prepared datasets in `serialized/`, learned representations in
`embeddings/`, and fitted estimators in `models/`.
