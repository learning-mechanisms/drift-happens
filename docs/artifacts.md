# Artifact Policy

The public source repository should contain code, tests, configs, curated
snapshots, small fixtures, paper assets, and website assets required to render
the site.

The repository should not contain generated experiment outputs by default:

- `artifacts/experiments/`
- `artifacts/plots/`
- `artifacts/reports/`
- `artifacts/robustness_results/`
- `artifacts/runs/`
- `artifacts/sweeps/`
- `artifacts/bundles/`
- `artifacts/remote/`
- trained model checkpoints
- raw datasets and downloaded archives

Curated experiment plan YAML files under `artifacts/experiment_plans/` are kept
because they are small, reviewable launch manifests.

Large reproducibility assets should be published separately through a release
artifact, Zenodo record, or dedicated artifact repository. Add the final links to
this file and the README before publishing the source repository.

Local data and generated output locations can be moved with:

```bash
export DRIFT_DATA_DIR=/path/to/data
export DRIFT_ARTIFACTS_DIR=/path/to/artifacts
export DRIFT_PLOTS_DIR=/path/to/plots
```

Runtime plots are regenerated with `pixi run plots-results` from
`artifacts/runs/**/results/drift_matrix.json`; see `docs/results-plotting.md`.

## Large Artifact Sync

Large local run artifacts can be synchronized to a personal pCloud remote through
the rclone-backed artifact CLI:

```bash
pixi run artifacts-remote-setup
pixi run artifacts-push-dry-run
pixi run artifacts-push
pixi run artifacts-pull-dry-run
pixi run artifacts-pull
```

`artifacts-remote-setup` writes a small local profile under
`artifacts/remote/pcloud.json` and asks rclone to create the pCloud remote when it
is missing. The profile stores only the remote name, remote path, and artifact
roots; credentials remain in rclone's own config.

Push and pull use `rclone copy` by default, so they do not delete files on the
destination. Use `drift artifacts remote push --mirror` or
`drift artifacts remote pull --mirror` only when destructive rclone sync semantics
are intended.

The default private sync includes curated reproducibility artifacts from
`artifacts/runs/` and `artifacts/sweeps/`, plus packed public bundle files under
`artifacts/bundles/`. Run artifacts include metadata, logs, metrics, results,
completion markers, training histories, and
`stages/train/**/trained_model.*` model files. Attempt directories and generic
checkpoint folders are skipped unless `--with-attempts` or
`--with-all-checkpoints` is passed. The default private sync does not upload
`artifacts/cache/`.

## Public Reproducibility Bundles

Public bundles are generated tar archives under `artifacts/bundles/`. They are
separate from the private rclone sync: build them locally, let the private sync
upload the packed `.tar.gz` and `.tar.gz.sha256` files, then publish pCloud
links for the public download commands.

Build the public full-runs bundle:

```bash
pixi run artifacts-bundle-full-runs --overwrite
```

Build the minimal Yearbook saliency bundle:

```bash
pixi run artifacts-bundle-saliency-rebuild
```

The lower-level commands are:

```bash
pixi run drift artifacts bundle stage public-full-runs
pixi run drift artifacts bundle pack public-full-runs
pixi run drift artifacts bundle build public-full-runs
pixi run drift artifacts bundle download public-full-runs --download-link URL
```

The public full-runs bundle stages files from `artifacts/runs/` only. It
includes conference run metadata, configs, manifests, snapshots, logs, metrics,
results, stage completion markers, training histories, and final
`stages/train/**/trained_model.*` files. It excludes sweeps, cache, attempt
directories, generic `checkpoints/` directories, locks, temporary files, smoke
runs, synthetic runs, and W&B internal run directories.

The Yearbook saliency bundle is intentionally minimal. It includes only seed-0
Yearbook checkpoints for `cnn_l`, `resnet_s`, and `mlp_l` at cutoffs 1950 and
1970, in a layout accepted directly by `drift analysis saliency`:

```text
cnn_l/train_slice_1950/trained_model.pt
cnn_l/train_slice_1970/trained_model.pt
mlp_l/train_slice_1950/trained_model.pt
mlp_l/train_slice_1970/trained_model.pt
resnet_s/train_slice_1950/trained_model.pt
resnet_s/train_slice_1970/trained_model.pt
manifest.json
```

After downloading or staging the saliency bundle, regenerate the paper saliency
figure with:

```bash
pixi run analysis-saliency
```

Public bundle archive download links are configured in
`drift_happens/utils/artifact_bundles.py`, along with the expected size and
SHA-256 digest used by the downloader. The public sidecar links are:

| Bundle              | File                              | Public link                                                                   |
| ------------------- | --------------------------------- | ----------------------------------------------------------------------------- |
| `public-full-runs`  | `manifest.json`                   | <https://e.pcloud.link/publink/show?code=XZThvcZHOB9o6nBuF0MqJ58oiEpwSOz3zNX> |
| `public-full-runs`  | `public-full-runs.tar.gz`         | <https://e.pcloud.link/publink/show?code=XZghvcZHNVVTmTJLXm7eYWNk6hiSbgtvviX> |
| `public-full-runs`  | `public-full-runs.tar.gz.sha256`  | <https://e.pcloud.link/publink/show?code=XZPhvcZybuYghiuOsmBHaBbMoOmcyfmNFi7> |
| `yearbook-saliency` | `manifest.json`                   | <https://e.pcloud.link/publink/show?code=XZi3icZDNXtWgamP6kkPxChxKSL4yT5PSYV> |
| `yearbook-saliency` | `yearbook-saliency.tar.gz`        | <https://e.pcloud.link/publink/show?code=XZc3icZyr0QCT2QmfB3gy9qRyiVvFJUyThX> |
| `yearbook-saliency` | `yearbook-saliency.tar.gz.sha256` | <https://e.pcloud.link/publink/show?code=XZo3icZdbd8CmUeHqXGNNYbmgCdTbLIwvr7> |

## Dataset Archive Publishing

Dataset archives under `data/` are separate from the run/sweep artifact sync
above. They are public download inputs for the dataset setup CLIs, so publish
them deliberately and update the hardcoded pCloud links after upload.

Set up rclone once if the `pcloud` remote is not configured:

```bash
pixi run artifacts-remote-setup
pixi run artifacts-remote-status
```

Build or verify the local archives before publishing:

```bash
pixi run datasets-setup yearbook full --yes
pixi run datasets-setup imdb-faces full --yes
pixi run datasets-setup amazon-reviews-23 full --no-from-cache --yes

tar --exclude="._*" --exclude=".DS_Store" \
  -czf data/amazon_reviews_23/processed/reviews_cache.tar.gz \
  -C data/amazon_reviews_23/processed reviews

tar -tzf data/yearbook/yearbook.tar.gz >/dev/null
tar -tzf data/imdb_faces/imdb.tar.gz >/dev/null
tar -tzf data/amazon_reviews_23/processed/reviews_cache.tar.gz >/dev/null
```

Upload with `rclone copyto`, not the filtered artifact push command:

```bash
pixi run rclone copyto data/yearbook/yearbook.tar.gz \
  pcloud:/drift-happens/datasets/yearbook/yearbook.tar.gz --progress

pixi run rclone copyto data/imdb_faces/imdb.tar.gz \
  pcloud:/drift-happens/datasets/imdb_faces/imdb.tar.gz --progress

pixi run rclone copyto data/amazon_reviews_23/processed/reviews_cache.tar.gz \
  pcloud:/drift-happens/datasets/amazon_reviews_23/reviews_cache.tar.gz \
  --progress
```

Generate public links and replace the corresponding `DOWNLOAD_LINK` values:

```bash
pixi run rclone link \
  pcloud:/drift-happens/datasets/yearbook/yearbook.tar.gz

pixi run rclone link \
  pcloud:/drift-happens/datasets/imdb_faces/imdb.tar.gz

pixi run rclone link \
  pcloud:/drift-happens/datasets/amazon_reviews_23/reviews_cache.tar.gz
```

The links must be pCloud public links containing `code=...`, because
`download_pcloud_file` extracts that public file id before downloading. Update:

- `drift_happens/dataset/yearbook/const.py`
- `drift_happens/dataset/imdb_faces/const.py`
- `drift_happens/dataset/amazon_reviews_23/cli.py`

Before committing changed links, run the relevant setup command from a clean
cache path to make sure the public link downloads and extracts the archive.

## W&B Artifact Contract

When W&B artifact upload is enabled, run artifacts include small metadata and
evaluation outputs such as `results/drift_matrix.json` and
`results/drift_matrix.csv`. Checkpoints are excluded unless
`WANDB_UPLOAD_CHECKPOINTS=true` or the corresponding CLI override is used.

Use W&B for metrics, metadata, and matrix artifacts. Use the pCloud/rclone path
above for large checkpoint archives and other heavyweight local artifacts.
