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
images = load_images()

X = images.reshape(len(images), -1)
X = X.astype("float32") / 255.0

model = MIMIC(mode="factorised", capacity=0.25, random_state=0)
model.fit(X)

# Use the fitted vision interface to inspect image embeddings.
generated = model.sample(64)

generated_images = generated.reshape(64, height, width, channels)
```

The exact embedding API may differ depending on the vision implementation, but
the data shape is the important part: MIMIC sees images as rows and pixels as
features.

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

- `src/mimic_vision/datasets.py` downloads MNIST, Fashion-MNIST, and CIFAR-10,
  normalizes pixels, vectorizes images, and filters targets. For example,
  `targets=[3, 8]` keeps only images labelled 3 or 8, and `n_per_target=200`
  keeps at most 200 images for each selected target.
- `src/mimic_vision/visualization.py` computes simple 2D layouts and plots image
  thumbnails without overlap by skipping thumbnails that would collide with
  already placed images.
- `notebooks/01_download_and_filter.ipynb` demonstrates dataset download,
  target filtering, and per-target sampling.
- `notebooks/02_visualise_2d_images.ipynb` demonstrates PCA or t-SNE layouts
  with non-overlapping image thumbnails.
