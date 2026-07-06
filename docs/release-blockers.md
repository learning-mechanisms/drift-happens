# Release Blockers

These items cannot be completed safely by local code changes alone.

## Git History

Old notebook paths in the Git history contain Hugging Face token-shaped strings.
GitHub secret scanning may report these reviewed historical findings. The strings
nonetheless remain in the history.

Before publishing the repository, choose one of:

- publish a clean initial import without prior history, or
- rewrite/purge the affected history after explicit owner approval.

## External Research Artifacts

Generated experiments, plots, robustness outputs, runs, sweeps, trained models,
and raw datasets have been removed from the public source boundary. Publish
reproducibility bundles through a release asset, Zenodo record, or dedicated
artifact repository, then add stable links to `docs/artifacts.md`.

## License Confirmation

Resolved: code is distributed under `Apache-2.0` via `LICENSE`. The paper,
figures, website, and documentation are distributed under `CC-BY-4.0` via
`LICENSE-paper` where the project owners own the rights. Third-party references
and dependencies remain under their respective licenses.
