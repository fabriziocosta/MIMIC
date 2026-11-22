# Development guide

This guide covers setting up a MIMIC checkout, making changes, and preparing a
release. For user-facing examples, see [Getting started](GETTING_STARTED.md).

## Set up a working environment

From the repository root, create and activate a virtual environment, then
install the package with development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
Editable installation means changes under `src/` are available without
reinstalling the package.

## Find the relevant code

| Path | Contents |
| --- | --- |
| `src/mimic/` | Public tabular estimator, encoders, decoders, generation policies, datasets, diagnostics, CLI |
| `src/mimic_vision/` | Image and vision-oriented synthesis helpers |
| `src/mimic_experiments/` | Experiment entry points |
| `tests/` | Main package regression and behaviour tests |
| `manuscript/experiments/` | Reproducible manuscript analysis package and its tests |
| `notebooks/` | Tutorials, demos, and dataset workflows |
| `docs/` | Onboarding and project design documentation |

The main estimator is `MIMIC` in `src/mimic/mimic.py`. `fit` builds the
feature-wise modules and embedding state; `transform`, `impute`, `confidence`,
and `sample` use that fitted state. Encoders, decoders, and generation policies
are defined separately so they can be replaced independently.

## Make and check a change

1. Find the closest existing implementation and test before editing.
2. Keep estimator constructor arguments as configuration and store learned
   values in attributes ending with `_`, following scikit-learn conventions.
3. Use explicit `random_state` values in examples and tests that depend on
   stochastic behavior.
4. Add or update focused tests for behavior changes.
5. Run the relevant tests, then the complete suite when the change affects
   shared model behavior:

   ```bash
   python -m pytest tests/test_mimic.py
   python -m pytest
   ```

The dataset examples may download data from OpenML the first time they run and
therefore need network access. The unit tests use small local data fixtures.
For Parquet and Excel file handling, install a pandas I/O engine such as
`pyarrow` or `openpyxl` in the environment used for those formats.

## Work with notebooks

Open notebooks from the repository root so relative paths resolve as expected.
The notebooks add `src/` and `notebooks/` to the import path. Notebook 05 loads
one of the documented OpenML datasets; the first data load may require network
access. Keep notebook outputs focused on useful results and clear stale error
outputs after fixing a failed cell.

## Version and release

The canonical package version is `src/mimic/_version.py`. Packaging metadata,
`mimic.__version__`, saved model metadata, and `mimic-data --version` use this
value. Before a release, update the version and changelog, then tag the release
commit as `v<version>`.

Follow [Versioning and releases](VERSIONING.md) for the version bump rules and
release steps. Do not add a second version literal to `pyproject.toml`.
