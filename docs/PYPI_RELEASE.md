# Preparing a PyPI release

The distribution name is `mimic-tabular`; Python code imports it as `mimic`.
The shorter `mimic` name and `mimic-framework` are already registered on PyPI.

## Check a release locally

From the repository root, build and validate both distribution formats:

```bash
python -m pip install --upgrade build twine
rm -rf dist build
python -m build
python -m twine check dist/*
```

Before publishing, inspect the wheel and source archive to confirm they contain
the expected packages and metadata. Install the wheel in a clean environment
and run a small import and CLI check:

```bash
python -m pip install dist/mimic_tabular-*.whl
python -c "import mimic; print(mimic.__version__)"
mimic-data --version
```

## Configure Trusted Publishing

The repository includes `.github/workflows/publish-pypi.yml`. It only runs when
manually dispatched, checks out the requested release tag, verifies it matches
the package version, builds the distributions, and publishes them with PyPI
Trusted Publishing. It does not publish on every tag push.

Before using it:

1. Create the GitHub Actions environment named `pypi` and require a reviewer
   for deployments.
2. Register a Trusted Publisher in PyPI's publishing settings with project
   name `mimic-tabular`, owner `fabriziocosta`, repository `MIMIC`, workflow
   `.github/workflows/publish-pypi.yml`, and environment `pypi`.
3. Dispatch **Publish to PyPI** from the default branch and enter the exact
   version tag, such as `v0.1.2`. Review the environment deployment before it
   publishes.

The `mimic-tabular` name was unregistered when checked, but PyPI names are
claimed by the first successful upload. Confirm availability immediately
before the initial publication.
