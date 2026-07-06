# Yearbook Dataset

[Original Source](https://shiry.eecs.berkeley.edu/yearbooks/)

## Description

The Yearbook Dataset contains high school yearbook portraits spanning multiple decades, labeled with gender (F/M) and yearbook year. This repo uses it for binary gender classification with year as the drift time axis, using 32×32 downscaled face tensors.

## Provenance

The original dataset is hosted at the Berkeley link above. The archive is fetched from a pcloud mirror (see `const.py`). Use the CLI to download and prepare the data:

```bash
pixi run datasets-setup yearbook full
```

Alternatively, run the individual steps: `download`, `unpack`, `prepare`.
