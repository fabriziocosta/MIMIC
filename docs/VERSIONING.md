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
4. Tag the release commit `v<version>` and push the tag:

   ```bash
   git tag -a v0.2.0 -m "MIMIC 0.2.0"
   git push origin v0.2.0
   ```

5. The GitHub Actions release workflow verifies that the tag matches
   `__version__` and that the corresponding changelog section exists, then
   creates or updates the GitHub Release from those notes.
6. Build and publish the Python package from that tagged commit when a package
   release is intended.

The PyPI distribution is named `mimic-tabular` (the import remains `mimic`).
See [Preparing a PyPI release](PYPI_RELEASE.md) for distribution checks and the
manual Trusted Publishing workflow. That workflow does not run on tag pushes.

Do not edit the version separately in `pyproject.toml`; it is derived from the
canonical Python version module.
