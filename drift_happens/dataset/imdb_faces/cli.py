import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from drift_happens.dataset.imdb_faces.const import (
    DOWNLOAD_LINK,
    IMDB_TAR_FILE,
    IMDB_UNPACK_DIR,
)
from drift_happens.dataset.imdb_faces.load import (
    get_cache_path,
    write_preprocessed_df_to_cache,
)
from drift_happens.dataset.imdb_faces.transform import (
    load_and_preprocess_images_into_df,
)
from drift_happens.dataset.utils import download_pcloud_file, safe_extract_tar
from drift_happens.tools.cli import prompt_confirmation
from drift_happens.utils.log import configure_logging, get_logger

logger = get_logger()
app = typer.Typer(
    name="imdb faces dataset manager", help="Manage the imdb faces dataset"
)


@app.command()
def download(yes: Annotated[bool, typer.Option("--yes")] = False) -> None:
    if IMDB_TAR_FILE.exists():
        logger.info(f"IMDB archive already exists at {IMDB_TAR_FILE}")
        if not prompt_confirmation(f"Overwrite {IMDB_TAR_FILE}?", assume_yes=yes):
            return

    logger.info(f"Downloading IMDB archive to {IMDB_TAR_FILE}")
    IMDB_TAR_FILE.parent.mkdir(parents=True, exist_ok=True)
    download_pcloud_file(IMDB_TAR_FILE, download_link=DOWNLOAD_LINK)


@app.command()
def unpack(yes: Annotated[bool, typer.Option("--yes")] = False) -> None:
    if IMDB_UNPACK_DIR.exists():
        logger.info(f"IMDB archive already unpacked to {IMDB_UNPACK_DIR}")
        if not prompt_confirmation(f"Overwrite {IMDB_UNPACK_DIR}?", assume_yes=yes):
            return
    if not IMDB_TAR_FILE.exists():
        raise FileNotFoundError(
            f"IMDB archive not found at {IMDB_TAR_FILE}; run `download` first"
        )

    logger.info(f"Unpacking IMDB archive to {IMDB_UNPACK_DIR}")
    # Extract into a staging directory and swap it in atomically, so a corrupt
    # archive never destroys an already-unpacked dataset.
    IMDB_UNPACK_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=IMDB_UNPACK_DIR.parent, prefix=".imdb-unpack-"))
    try:
        with tarfile.open(IMDB_TAR_FILE, "r:gz") as tar:
            safe_extract_tar(tar, staging)  # archive contains a `raw` folder
        extracted = staging / IMDB_UNPACK_DIR.name
        if not extracted.is_dir():
            raise ValueError(
                f"archive did not contain a {IMDB_UNPACK_DIR.name}/ directory"
            )
        if IMDB_UNPACK_DIR.exists():
            shutil.rmtree(IMDB_UNPACK_DIR)
        extracted.replace(IMDB_UNPACK_DIR)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@app.command()
def prepare(yes: Annotated[bool, typer.Option("--yes")] = False) -> None:
    target_size = 32
    cache_path = get_cache_path(target_size=target_size)

    if cache_path.exists():
        logger.info(f"Preprocessed IMDB dataset already exists at {cache_path}")
        if not prompt_confirmation(f"Overwrite {cache_path}?", assume_yes=yes):
            return

    logger.info(f"Preparing preprocessed IMDB dataset at {cache_path}")
    df = load_and_preprocess_images_into_df(
        raw_dataset_path=IMDB_UNPACK_DIR, target_size=target_size
    )

    logger.info(f"Saving preprocessed IMDB dataset to {cache_path}")
    write_preprocessed_df_to_cache(df, target_size=target_size)


@app.command("full")
def full_pipeline(yes: Annotated[bool, typer.Option("--yes")] = False):
    download(yes=yes)
    unpack(yes=yes)
    prepare(yes=yes)


if __name__ == "__main__":
    configure_logging()
    app()
