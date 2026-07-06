# Manuscript Experiment Notebook Pattern

This captures the working pattern established while building the Q1 notebook. Use it as the starting point for Q2 and later manuscript experiment notebooks.

## Structure

Each experiment should have three parts:

- A thin notebook in `manuscript/experiments/`.
- A reusable implementation module in `src/mimic_experiments/`.
- Focused tests in `tests/`.

For Q2, that likely means:

```text
manuscript/experiments/02_q2_adasyn_benchmark.ipynb
src/mimic_experiments/q2_adasyn.py
tests/test_q2_adasyn.py
```

## Notebook Responsibilities

The notebook should contain only:

- Project-root setup and imports.
- User-editable constants such as `DATASET`, `RUN_PROFILE`, `RANDOM_STATE`, `ARTIFACT_DIR`, and policy knobs.
- Config construction.
- Calls into `mimic_experiments.*`.
- `display(...)` calls for registry, saved table manifests, loaded summaries, and plots.

Do not put loaders, metrics, CV loops, sampling logic, table builders, or plotting internals in the notebook.

## Profile Pattern

Use two profiles:

- `run_full`: run the full experiment protocol and save result CSV tables.
- `view`: skip fitting and sampling, reconstruct the same artifact filenames, load saved CSV tables, and regenerate summaries and plots.

The config object should expose:

```python
config.should_run_experiment
config.saves_as_profile
```

`view` should save and load under the same artifact stem as `run_full`, so switching from `run_full` to `view` does not change filenames.

## Artifact Pattern

CSV result tables should be the source of truth for summaries and plots after a run.

Use deterministic filenames derived from config:

```text
{dataset_key}__{saved_profile}__{model_mode}__cap-{capacity}__{policy}__seed-{random_state}__{table}.csv
```

Save them under:

```text
manuscript/artifacts/{experiment_key}/tables/
```

Cache fitted models separately under:

```text
manuscript/artifacts/{experiment_key}/models/
```

The module should provide helpers equivalent to:

```python
result_table_paths(config)
result_table_manifest(config)
save_result_tables(config, ...)
load_result_tables(config)
```

The notebook should print the concrete filenames and display the manifest before loading tables.

## Execution Flow

Use this notebook flow:

1. Build `config`.
2. Display the dataset registry with experiment status.
3. If `config.should_run_experiment`, load data and run the experiment.
4. Print and display the result table manifest.
5. Load result tables from CSV.
6. Display performance summaries from loaded CSVs.
7. Build plots from loaded CSVs.
8. Display manuscript-ready rows from loaded CSVs.

## Module Responsibilities

The experiment module should own:

- Dataset registry and loader functions.
- Label normalization.
- Train/test split or CV fold generation.
- MIMIC fit/sample orchestration.
- Metric calculations.
- Aggregated summary tables.
- CSV artifact save/load helpers.
- Plot construction helpers that return `(fig, ax)`.
- Manuscript-ready table builders.

Keep module functions typed enough that tests can exercise them without notebook execution.

## Testing Checklist

Add focused tests for:

- Dataset registry columns and required dataset keys.
- Any OpenML or external dataset IDs being pinned explicitly.
- Label normalization to `"majority"` and `"minority"`.
- Split/fold generation matching the paper protocol.
- Generated sample count calculation.
- Metric table shape and column names.
- Result table path construction, including `view` mapping to `run_full`.
- CSV save/load round trip.
- Result table manifest paths.
- Plot helper returning an axis with expected labels/title/lines.
- Notebook JSON validity with `python -m json.tool`.

## Q2-Specific Notes

For Q2, mirror the ADASYN half-split protocol:

- Repeat 100 runs for `run_full`.
- In each run, sample half of each class for training and use the rest for testing.
- Train and sample only inside the training split.
- Save row-level run results and aggregated metric summaries as CSV.
- Compute OA, precision, recall, F-measure, and G-mean in the module.
- Plot and manuscript summaries should read from the saved CSVs, not from in-memory run outputs.
