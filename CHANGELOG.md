# Changelog

Notable user-facing changes are recorded here. GitHub Releases are generated
from the section matching the package version and release tag.

## [Unreleased]

## [0.1.4] - 2026-11-22

### Added

- Use IMSTE by default for 2D layouts in `MIMIC.plot`, retaining classical MDS
  through `method="mds"`.

### Changed

- Require IMSTE 0.8.10 or later for the default plotting method.

## [0.1.3] - 2026-11-22

### Changed

- Prepare the `mimic-tabular` PyPI distribution while keeping the Python import name `mimic`.
- Add package metadata and a manually dispatched Trusted Publishing workflow.
- Rename the documentation branding and canonical Read the Docs URL to MIMIC Tabular.

## [0.1.2] - 2026-11-22

### Changed

- Keep manuscript materials local to the maintainer and exclude them from the repository.
- Clarify the repository layout, notebook locations, and generated-data conventions.

## [0.1.1] - 2026-11-22

### Added

- Add an MIT license and configure software archiving metadata for Zenodo.

## [0.1.0] - 2026-11-22

Initial public release of MIMIC.

### Added

- Feature-wise mixed-tabular modelling for embeddings, imputation, confidence
  diagnostics, and synthetic data generation.
- Identity, direct, factorised, and joint generation modes with configurable
  encoders, decoders, and generation policies.
- Shared numerical feature groups, traceable sampling, optional nearest-neighbor
  privacy filtering, dataset helpers, and a command-line interface.
- User and developer onboarding guides, version metadata, and Read the Docs
  build configuration.

### Fixed

- Reuse the fitted neighbor index during privacy filtering.
- Validate sample counts and return a typed empty frame for zero requested rows.
- Preserve PyTorch RNG state during seeded neural model fitting.
