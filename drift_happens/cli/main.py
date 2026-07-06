from __future__ import annotations

from pathlib import Path

import typer

from drift_happens.cli.analysis import app as analysis_app
from drift_happens.cli.artifacts import app as artifacts_app
from drift_happens.cli.datasets_setup import create_app as create_datasets_setup_app
from drift_happens.cli.experiment import app as experiment_app
from drift_happens.utils.env import load_envfile
from drift_happens.utils.log import configure_logging
from drift_happens.utils.paths import (
    RUNS_DIR,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(
    name="drift",
    help="Dataset, evaluation, experiment, and artifact tooling for drift-happens.",
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
)
datasets_setup_app = create_datasets_setup_app()
dataset_alias_app = create_datasets_setup_app(
    help_text="Compatibility alias for `datasets-setup`."
)
eval_app = typer.Typer(
    help="Run evaluation and reporting tools.",
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
)


@eval_app.command("robustness")
def robustness(
    runs_root: Path = typer.Option(
        RUNS_DIR,
        "--runs-root",
        help="Runtime runs root scanned for results/drift_matrix.json files.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail instead of finishing with skipped models or runs.",
    ),
) -> None:
    """Compute seed-aggregated drift robustness tables per dataset."""
    from drift_happens.evaluation.robustness.compute_robustness import (
        write_robustness_reports,
    )

    write_robustness_reports(runs_root=runs_root, strict=strict)


app.add_typer(datasets_setup_app, name="datasets-setup")
app.add_typer(dataset_alias_app, name="dataset")
app.add_typer(eval_app, name="eval")
app.add_typer(experiment_app, name="experiment")
app.add_typer(artifacts_app, name="artifacts")
app.add_typer(analysis_app, name="analysis")


@app.callback()
def root_callback() -> None:
    """Load command-boundary environment before dispatching subcommands."""
    load_envfile()


def main() -> None:
    configure_logging()
    app()


if __name__ == "__main__":
    main()
