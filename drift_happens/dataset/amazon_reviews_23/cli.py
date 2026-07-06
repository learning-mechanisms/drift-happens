import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from drift_happens.dataset.amazon_reviews_23.const import (
    AR23_GROUPS_SELECTED,
    AR23_PREPROCESSED_REVIEWS_CACHE_FILE,
    AR23_PREPROCESSED_REVIEWS_DIR,
)
from drift_happens.dataset.amazon_reviews_23.preprocess import (
    merge_review_dfs,
    prepare_reviews_df,
)
from drift_happens.dataset.utils import download_pcloud_file, safe_extract_tar
from drift_happens.tools.cli import prompt_confirmation
from drift_happens.utils.log import configure_logging, get_logger

DOWNLOAD_LINK = (
    "https://e.pcloud.link/publink/show?code=XZwlzcZOmG5zAWHzwJqzYkPYd6kwQXTg7g7"
)

app = typer.Typer(
    name="amazon reviews 2023 dataset manager",
    help="Manage the Amazon Reviews 2023 dataset",
)

logger = get_logger()


@app.command()
def build_from_cache(yes: Annotated[bool, typer.Option("--yes")] = False):
    if AR23_PREPROCESSED_REVIEWS_DIR.exists() and not prompt_confirmation(
        f"Preprocessed reviews directory {AR23_PREPROCESSED_REVIEWS_DIR} already exists. Overwrite?",
        assume_yes=yes,
    ):
        logger.info("Aborting extraction from cache.")
        return

    if not AR23_PREPROCESSED_REVIEWS_CACHE_FILE.exists():
        logger.info(
            f"Downloading preprocessed reviews cache to {AR23_PREPROCESSED_REVIEWS_CACHE_FILE}"
        )
        AR23_PREPROCESSED_REVIEWS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        download_pcloud_file(
            destination=AR23_PREPROCESSED_REVIEWS_CACHE_FILE,
            download_link=DOWNLOAD_LINK,
        )

    # Extract into a staging directory and swap it in only after a clean
    # extraction, so a corrupt cache never destroys existing data.
    AR23_PREPROCESSED_REVIEWS_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=AR23_PREPROCESSED_REVIEWS_DIR.parent, prefix=".reviews-cache-"
        )
    )
    try:
        with tarfile.open(AR23_PREPROCESSED_REVIEWS_CACHE_FILE, "r:gz") as tar:
            safe_extract_tar(tar, staging)
        extracted = staging / AR23_PREPROCESSED_REVIEWS_DIR.name
        if not extracted.is_dir():
            raise ValueError(
                f"cache archive did not contain a {AR23_PREPROCESSED_REVIEWS_DIR.name}/ directory"
            )
        if AR23_PREPROCESSED_REVIEWS_DIR.exists():
            shutil.rmtree(AR23_PREPROCESSED_REVIEWS_DIR)
        extracted.replace(AR23_PREPROCESSED_REVIEWS_DIR)
    except Exception:
        AR23_PREPROCESSED_REVIEWS_CACHE_FILE.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    AR23_PREPROCESSED_REVIEWS_CACHE_FILE.unlink(missing_ok=True)


@app.command()
def download_reviews(skip_existing: bool = True):
    prepare_reviews_df(AR23_GROUPS_SELECTED, skip_existing=skip_existing)


@app.command()
def merge_review_categories(yes: Annotated[bool, typer.Option("--yes")] = False):
    merge_review_dfs(AR23_GROUPS_SELECTED, assume_yes=yes)


@app.command("full")
def full_pipeline(
    from_cache: bool = True, yes: Annotated[bool, typer.Option("--yes")] = False
):
    if from_cache:
        build_from_cache(yes=yes)
    else:
        download_reviews()
    merge_review_categories(yes=yes)


if __name__ == "__main__":
    configure_logging()
    app()
