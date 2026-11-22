# Getting started with MIMIC

This guide takes you from an installed checkout to your first generated table.
For contributors, see [Development](DEVELOPMENT.md). For deeper API and design
notes, see [Implementation](IMPLEMENTATION.md).

## 1. Install MIMIC

MIMIC requires Python 3.10 or newer. From the repository root, install the
package in editable mode:

```bash
python -m pip install -e .
```

The default neural modes use PyTorch, which is installed as a package dependency.
To check the installed package version:

```bash
mimic-data --version
```

## 2. Generate a small example

Start with a dataframe and declare each column's role. This example uses the
identity mode as a lightweight baseline:

```python
import pandas as pd
from mimic import MIMIC

df = pd.DataFrame(
    {
        "age": [24, 31, 37, 42, 51, 28, 46, 34],
        "monthly_spend": [35.0, 62.0, 48.0, 91.0, 74.0, 39.0, 83.0, 57.0],
        "plan": ["basic", "plus", "basic", "plus", "plus", "basic", "plus", "basic"],
    }
)

columns = {
    "regression": ["age", "monthly_spend"],
    "classification": ["plan"],
    "ignore": [],
}

model = MIMIC(
    columns=columns,
    mode="identity",
    random_state=0,
    show_progress=False,
)
synthetic = model.fit(df).sample(5)
print(synthetic)
```

`sample(5)` returns five generated rows. The `identity` preset is useful as a
transparent baseline. For neural feature models and factorised decoding, use
`mode="factorised"` and set `capacity` between `0` and `1`:

```python
model = MIMIC(
    columns=columns,
    mode="factorised",
    capacity=0.25,
    random_state=0,
    show_progress=False,
).fit(df)
synthetic = model.sample(100)
```

Higher capacity uses larger neural models and more training work. Start small,
then increase it if model quality warrants the additional compute.

## 3. Choose column roles

Every input column must be assigned to exactly one role:

- `regression`: numerical values treated as continuous;
- `classification`: labels and categories;
- `ignore`: columns excluded from modelling, such as row IDs.

For a first run, explicit roles make the modelling choices clear. You can also
use `columns="auto"` for heuristic role detection, then inspect the inferred
roles before relying on them. See the README's mode guide for preset details.

## 4. Use the fitted model

The fitted estimator supports generation and other feature-wise tasks:

```python
embedding = model.transform(df)
imputed = model.impute(df)
confidence = model.confidence(df)
```

`impute` fills missing modelled values. `confidence` returns per-row,
per-column prediction and uncertainty details. A classification target can be
left as a normal column and predicted by masking the values you want to fill:

```python
incomplete = df.copy()
incomplete.loc[0, "plan"] = pd.NA
imputed = model.impute(incomplete, columns=["plan"])
```

To generate conditioned rows, require values seen in the training data:

```python
minority_rows = model.sample(100, condition={"plan": "basic"})
```

Generated rows contain modelled columns. Ignored columns do not participate in
generation and are not included in the sampled output.

## 5. Save and reload a fitted model

```python
model.save("mimic-model.joblib")
loaded = MIMIC.load("mimic-model.joblib")
more_rows = loaded.sample(100)
```

Only load model files from a trusted source. Joblib model files use Python
serialization and are not safe to load when untrusted.

## Privacy filter note

The optional nearest-neighbor privacy filter rejects some candidates that are
too close to training examples. It can reduce attribution risk, but it is not a
differential privacy guarantee. Strict settings may need more candidate
attempts or may reject too many candidates to fill the requested sample count.

## Use the command line

For a CSV file, the command-line interface can fit and generate in one step:

```bash
mimic-data data.csv --mode identity --random-state 0
```

By default it writes `data_mimic.csv` next to the input. Use
`mimic-data --help` to see column-role, output-path, and saved-model options.
Parquet and Excel support may require installing a pandas I/O engine such as
`pyarrow` or `openpyxl`.

## Continue learning

- [Notebook 01: basic usage](https://github.com/fabriziocosta/MIMIC/blob/master/notebooks/01_basic_usage.ipynb)
- [Notebook 02: classification and regression](https://github.com/fabriziocosta/MIMIC/blob/master/notebooks/02_classification_and_regression.ipynb)
- [Notebook 03: generation and traceability](https://github.com/fabriziocosta/MIMIC/blob/master/notebooks/03_generation_oversampling_traceability.ipynb)
- [Notebook 04: operating modes and shared ResNet](https://github.com/fabriziocosta/MIMIC/blob/master/notebooks/04_dataset_operating_modes_resnet.ipynb)
- [Notebook 05: dataset sampling](https://github.com/fabriziocosta/MIMIC/blob/master/notebooks/05_simplified_dataset_sampling.ipynb)
- [White paper](WHITEPAPER.md) for the research motivation and [implementation notes](IMPLEMENTATION.md) for the API and internal design.
