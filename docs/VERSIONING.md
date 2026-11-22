# Versioning and releases

MIMIC uses [Semantic Versioning](https://semver.org/) with a `0.y.z` version
while its public API is still evolving. The canonical version lives in
`src/mimic/_version.py`; packaging metadata, `mimic.__version__`, model
artifacts, and the command-line `--version` output all read from that value.

## Preparing a release

1. Add the user-visible changes to the matching section in `CHANGELOG.md`.
2. Update `__version__` in `src/mimic/_version.py`:
   - increment `z` for backward-compatible fixes;
   - increment `y` for backward-compatible features, resetting `z` to zero;
   - increment `y` for breaking changes before `1.0.0`, and describe the break
     prominently in the changelog.
3. Move the released notes from `Unreleased` into a section named for the new
   version, then start a fresh `Unreleased` section.
4. Tag the release commit `v<version>` (for example, `v0.2.0`).
5. Build and publish the package from that tagged commit.

Do not edit the version separately in `pyproject.toml`; it is derived from the
canonical Python version module.
