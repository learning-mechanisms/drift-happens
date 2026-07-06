# Contributing

This project is Pixi-first. Use the Pixi environment for development, tests, and
CLI work.

## Setup

```bash
pixi install
pixi run postinstall
pixi run pre-commit-install
```

## Quality Gates

Run these before opening a pull request:

```bash
pixi run test
pixi run typecheck
pixi run lint
pixi run format-check
pixi run materialize-check
pixi run experiment-plans-check
pixi run typos --force-exclude
pixi run metadata-check
pixi run artifact-policy-check --base-ref origin/main  # bare form only checks staged additions, not committed ones
```

## Pull Requests

Keep changes easy to review:

- Keep behavior changes focused and covered by tests.
- Do not commit generated experiment outputs, trained models, raw datasets, or
  plot dumps.
- Regenerate curated experiment plans with `pixi run experiment-plans` when
  preset or sweep logic changes.
- Document any intentional artifact exception in the pull request.

## Artifacts

Curated source files belong in the repository. Generated research outputs belong
outside the source repository unless explicitly reviewed as public assets.
