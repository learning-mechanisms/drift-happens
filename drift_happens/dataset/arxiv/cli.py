import json
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import kagglehub
import pandas as pd
import typer

from drift_happens.dataset.arxiv.const import (
    ARXIV_CACHE_DIR,
    ARXIV_CACHE_FILE,
    ARXIV_PREPROCESSED_DF,
)
from drift_happens.tools.cli import prompt_confirmation
from drift_happens.utils.log import configure_logging, get_logger

logger = get_logger()
app = typer.Typer(name="arxiv dataset manager", help="Manage the arxiv dataset")


@app.command()
def download(yes: Annotated[bool, typer.Option("--yes")] = False):
    if ARXIV_CACHE_DIR.exists() and len(list(ARXIV_CACHE_DIR.iterdir())) > 0:
        logger.info(f"Arxiv cache already exists at {ARXIV_CACHE_DIR}")
        if not prompt_confirmation(f"Overwrite {ARXIV_CACHE_DIR}?", assume_yes=yes):
            return

    logger.info(f"Downloading arxiv cache to {ARXIV_CACHE_DIR}")
    # Copy (not move) kagglehub's files into a staging directory and swap it in
    # atomically: moving them would empty kagglehub's version directory while
    # its completion marker survives, faking a cache hit on the next run.
    downloaded_path = kagglehub.dataset_download("Cornell-University/arxiv")
    ARXIV_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=ARXIV_CACHE_DIR.parent, prefix=".arxiv-cache-"))
    try:
        shutil.copytree(downloaded_path, staging, dirs_exist_ok=True)
        if ARXIV_CACHE_DIR.exists():
            shutil.rmtree(ARXIV_CACHE_DIR)
        staging.replace(ARXIV_CACHE_DIR)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


@app.command()
def prepare():
    logger.info("Preparing arxiv preprocessed dataframe...")

    def extract_first_version_date(sample):
        versions = sample["versions"]
        if versions and isinstance(versions, list):
            first_version = versions[0]
            assert first_version["version"] == "v1"
            return first_version["created"]
        raise ValueError("No versions found in sample")

    rows = []
    with open(ARXIV_CACHE_FILE) as f:
        for line in f:
            sample = json.loads(line)
            rows.append(
                {
                    "created": extract_first_version_date(sample),
                    "title": sample["title"],
                    "abstract": sample.get("abstract", ""),
                    "categories": sample["categories"],
                }
            )

    df = pd.DataFrame(rows)
    assert df["created"].isnull().sum() == 0, "Some created dates are null"
    df["created"] = pd.to_datetime(df["created"])

    # Split categories by space into list of categories
    df["categories"] = df["categories"].str.split(" ")

    ARXIV_PREPROCESSED_DF.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ARXIV_PREPROCESSED_DF, compression="zstd")


@app.command("full")
def full_pipeline(yes: Annotated[bool, typer.Option("--yes")] = False):
    download(yes=yes)
    prepare()


if __name__ == "__main__":
    configure_logging()
    app()
