import hashlib
import json
import re
import tarfile
from pathlib import Path

import requests


def download_pcloud_file(
    destination: Path,
    *,
    download_link: str | None = None,
    file_id: str | None = None,
    timeout: float = 30.0,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> None:
    """Download a pCloud public file with bounded requests and optional validation."""
    if file_id is None:
        if download_link is None:
            raise ValueError("provide download_link or file_id")
        # https://e.pcloud.link/publink/show?code=XZHLbIZOqqMWk9LE3HKQ6ieRd2EWuddyulV
        parsing_pattern = r"code=([^&]+)"
        file_ids = re.findall(parsing_pattern, download_link)
        if len(file_ids) != 1:
            raise ValueError(f"Expected 1 file ID, got {len(file_ids)}")
        file_id = file_ids[0]

    download_link = f"https://e.pcloud.link/publink/show?code={file_id}"
    response = requests.get(download_link, timeout=timeout)
    response.raise_for_status()

    # extract the download link from the response
    extraction_pattern = r"\"downloadlink\": \"([^\"]+)\""
    match = re.search(extraction_pattern, response.text)
    if match is None:
        raise ValueError("pCloud response did not contain a downloadlink")
    download_link = json.loads(f'"{match.group(1)}"')

    # Stream to file
    destination.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    size = 0
    tmp_destination = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with requests.get(download_link, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(tmp_destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size += len(chunk)
                        hasher.update(chunk)
                        f.write(chunk)
        if expected_size is not None and size != expected_size:
            raise ValueError(
                f"downloaded size mismatch for {destination}: "
                f"expected {expected_size}, got {size}"
            )
        if expected_sha256 is not None and hasher.hexdigest() != expected_sha256:
            raise ValueError(f"sha256 mismatch for {destination}")
    except Exception:
        tmp_destination.unlink(missing_ok=True)
        raise
    tmp_destination.replace(destination)


def safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract a tar archive only if every member stays below ``destination``."""
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != destination and destination not in target.parents:
            raise ValueError(f"unsafe tar member path: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"tar links are not allowed: {member.name}")
    # filter="data" rejects special files (devices, FIFOs) and strips
    # setuid/setgid bits from extracted members.
    archive.extractall(destination, filter="data")
