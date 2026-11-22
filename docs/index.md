# MIMIC Tabular

<p align="center">
  <img src="assets/mimic.png" alt="MIMIC logo" width="64">
</p>

**Modular Inference for Missingness, Inconsistency, and Creation** is a
scikit-learn-style framework for mixed tabular data. It learns feature-wise
models that support imputation, prediction, confidence diagnostics, and
synthetic data generation.

## Choose a starting point

- **Use MIMIC:** [Getting started](GETTING_STARTED.md) walks through
  installation, a first generation run, column roles, imputation, and sampling.
- **Contribute:** [Development](DEVELOPMENT.md) covers the environment, source
  layout, tests, notebooks, and release process.
- **Understand the API:** [Implementation](IMPLEMENTATION.md) describes the
  estimator and its components.
- **Understand the mode presets mathematically:**
  [Mathematical mode reference](MODES.md) defines the identity, direct,
  factorised, and joint presets.
- **Read the research context:** [White paper](WHITEPAPER.md) explains the
  motivation and design.
- **Explore image workflows:** [MIMIC for Images](IMAGE_WORKFLOWS.md) describes
  applying MIMIC to image data and links to its notebooks.

## Quick example

```python
from mimic import MIMIC

model = MIMIC(mode="factorised", capacity=0.25, random_state=0)
model.fit(df)
synthetic = model.sample(100)
```

Declare column roles explicitly for reliable modelling. See the
[getting started guide](GETTING_STARTED.md#3-choose-column-roles) for a complete
example with a dataframe and role mapping.

## Project links

- [GitHub repository](https://github.com/fabriziocosta/MIMIC)
- [README and feature overview](https://github.com/fabriziocosta/MIMIC/blob/master/README.md)
- [Changelog](https://github.com/fabriziocosta/MIMIC/blob/master/CHANGELOG.md)
