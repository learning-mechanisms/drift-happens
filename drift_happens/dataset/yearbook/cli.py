import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from drift_happens.dataset.utils import download_pcloud_file, safe_extract_tar
from drift_happens.dataset.yearbook.const import (
    DOWNLOAD_LINK,
    YB_TAR_FILE,
    YB_UNPACK_DIR,
)
from drift_happens.dataset.yearbook.transform import load_downscaled_images_into_df
from drift_happens.tools.cli import prompt_confirmation
from drift_happens.utils.log import configure_logging, get_logger

logger = get_logger()
app = typer.Typer(name="yearbook dataset manager", help="Manage the yearbook dataset")


@app.command()
def download(yes: Annotated[bool, typer.Option("--yes")] = False):
    if YB_TAR_FILE.exists():
        logger.info(f"Yearbook archive already exists at {YB_TAR_FILE}")
        if not prompt_confirmation(f"Overwrite {YB_TAR_FILE}?", assume_yes=yes):
            return

    logger.info(f"Downloading yearbook archive to {YB_TAR_FILE}")
    YB_TAR_FILE.parent.mkdir(parents=True, exist_ok=True)
    download_pcloud_file(YB_TAR_FILE, download_link=DOWNLOAD_LINK)


@app.command()
def unpack(yes: Annotated[bool, typer.Option("--yes")] = False):
    if YB_UNPACK_DIR.exists():
        logger.info(f"Yearbook archive already unpacked to {YB_UNPACK_DIR}")
        if not prompt_confirmation(f"Overwrite {YB_UNPACK_DIR}?", assume_yes=yes):
            return
    if not YB_TAR_FILE.exists():
        raise FileNotFoundError(
            f"Yearbook archive not found at {YB_TAR_FILE}; run `download` first"
        )

    logger.info(f"Unpacking yearbook archive to {YB_UNPACK_DIR}")
    # Extract into a staging directory and swap it in atomically, so a corrupt
    # archive never leaves a stale directory a later run treats as unpacked.
    YB_UNPACK_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=YB_UNPACK_DIR.parent, prefix=".yb-unpack-"))
    try:
        with tarfile.open(YB_TAR_FILE, "r:gz") as tar:
            safe_extract_tar(tar, staging)
        if YB_UNPACK_DIR.exists():
            shutil.rmtree(YB_UNPACK_DIR)
        staging.replace(YB_UNPACK_DIR)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@app.command()
def prepare():
    load_downscaled_images_into_df()


@app.command("full")
def full_pipeline(yes: Annotated[bool, typer.Option("--yes")] = False):
    download(yes=yes)
    unpack(yes=yes)
    prepare()


if __name__ == "__main__":
    configure_logging()
    app()
