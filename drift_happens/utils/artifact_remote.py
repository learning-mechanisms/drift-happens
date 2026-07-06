"""Small rclone command builder for external artifact storage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from drift_happens.utils.paths import ARTIFACTS_DIR

DEFAULT_REMOTE_NAME = "pcloud"
DEFAULT_REMOTE_PATH = "/drift-happens/artifacts"
DEFAULT_PROFILE_PATH = ARTIFACTS_DIR / "remote" / "pcloud.json"
DEFAULT_ROOTS: tuple[str, ...] = ("runs", "sweeps", "bundles")
PROFILE_VERSION = 2

_DEFAULT_FILE_PATTERNS: tuple[str, ...] = (
    "**/snapshot.json",
    "**/metadata.json",
    "**/run_manifest.json",
    "**/config.input.json",
    "**/config.input.yaml",
    "**/config.input.yml",
    "**/logs/**",
    "**/metrics/**",
    "**/results/**",
    "**/stages/**/completion.json",
    "**/stages/train/**/training_history.json",
    "**/stages/train/**/trained_model.pt",
    "**/stages/train/**/trained_model.pth",
    "**/stages/train/**/trained_model.model",
)

_BUNDLE_FILE_PATTERNS: tuple[str, ...] = (
    "**/manifest.json",
    "**/*.sha256",
    "**/*.tar.gz",
)


@dataclass(frozen=True, slots=True)
class ArtifactRemoteProfile:
    """Local, non-secret profile for one artifact remote."""

    remote: str = DEFAULT_REMOTE_NAME
    path: str = DEFAULT_REMOTE_PATH
    backend: str = "pcloud"
    roots: tuple[str, ...] = DEFAULT_ROOTS
    version: int = PROFILE_VERSION


def normalize_remote_name(remote: str) -> str:
    """Return an rclone remote name without its trailing colon."""
    value = remote.strip()
    if not value:
        raise ValueError("remote name must not be empty")
    return value.removesuffix(":")


def normalize_remote_path(path: str) -> str:
    """Return a stable absolute-looking remote path for rclone targets."""
    value = path.strip()
    if not value or value == "/":
        return ""
    return "/" + value.strip("/")


def normalize_root(root: str) -> str:
    """Return a relative artifact root name."""
    value = root.strip().strip("/")
    if not value or ".." in Path(value).parts:
        raise ValueError(f"invalid artifact root: {root!r}")
    return value


def make_remote_profile(
    *,
    remote: str = DEFAULT_REMOTE_NAME,
    path: str = DEFAULT_REMOTE_PATH,
    roots: tuple[str, ...] = DEFAULT_ROOTS,
    version: int = PROFILE_VERSION,
) -> ArtifactRemoteProfile:
    """Build a normalized profile from CLI values."""
    return ArtifactRemoteProfile(
        remote=normalize_remote_name(remote),
        path=normalize_remote_path(path),
        roots=tuple(normalize_root(root) for root in roots),
        version=version,
    )


def _migrate_roots(roots: tuple[str, ...]) -> tuple[str, ...]:
    """Append managed roots introduced after a profile was first written."""
    migrated = list(roots)
    for root in DEFAULT_ROOTS:
        if root not in migrated:
            migrated.append(root)
    return tuple(migrated)


def read_remote_profile(path: Path = DEFAULT_PROFILE_PATH) -> ArtifactRemoteProfile:
    """Read a local artifact remote profile, migrating older roots forward."""
    data = json.loads(Path(path).read_text())
    version = int(data.get("version", 1))
    roots = tuple(str(root) for root in data.get("roots", DEFAULT_ROOTS))
    if version < PROFILE_VERSION:
        roots = _migrate_roots(roots)
    return make_remote_profile(
        remote=str(data.get("remote", DEFAULT_REMOTE_NAME)),
        path=str(data.get("path", DEFAULT_REMOTE_PATH)),
        roots=roots,
    )


def write_remote_profile(
    profile: ArtifactRemoteProfile,
    path: Path = DEFAULT_PROFILE_PATH,
) -> Path:
    """Write a local artifact remote profile without credentials."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(profile)
    payload["roots"] = list(profile.roots)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def remote_target(profile: ArtifactRemoteProfile) -> str:
    """Return the rclone target string for a profile."""
    return f"{profile.remote}:{profile.path}"


def build_filter_rules(
    profile: ArtifactRemoteProfile,
    *,
    with_attempts: bool = False,
    with_all_checkpoints: bool = False,
) -> tuple[str, ...]:
    """Return ordered rclone filter rules for curated run/sweep artifacts."""
    rules: list[str] = []
    for root in profile.roots:
        if not with_attempts:
            rules.append(f"- /{root}/**/attempts/**")
        if root == "bundles":
            rules.append(f"- /{root}/**/staged/**")
        rules.extend(
            (
                f"+ /{root}/",
                f"+ /{root}/**/",
            )
        )
        patterns = (
            _BUNDLE_FILE_PATTERNS if root == "bundles" else _DEFAULT_FILE_PATTERNS
        )
        rules.extend(f"+ /{root}/{pattern}" for pattern in patterns)
        if with_all_checkpoints and root != "bundles":
            rules.append(f"+ /{root}/**/checkpoints/**")
    rules.append("- **")
    return tuple(rules)


def build_rclone_transfer_command(
    *,
    direction: Literal["push", "pull"],
    profile: ArtifactRemoteProfile,
    artifacts_root: Path = ARTIFACTS_DIR,
    mirror: bool = False,
    dry_run: bool = False,
    progress: bool = True,
    with_attempts: bool = False,
    with_all_checkpoints: bool = False,
    transfers: int = 4,
    rclone_bin: str = "rclone",
) -> list[str]:
    """Build an rclone copy/sync command without executing it."""
    operation = "sync" if mirror else "copy"
    local_root = str(Path(artifacts_root))
    target = remote_target(profile)
    source, destination = (
        (local_root, target) if direction == "push" else (target, local_root)
    )
    command = [
        rclone_bin,
        operation,
        source,
        destination,
        "--transfers",
        str(transfers),
    ]
    for rule in build_filter_rules(
        profile,
        with_attempts=with_attempts,
        with_all_checkpoints=with_all_checkpoints,
    ):
        command.extend(["--filter", rule])
    if dry_run:
        command.append("--dry-run")
    if progress:
        command.append("--progress")
    return command


def build_rclone_lsf_command(
    *,
    profile: ArtifactRemoteProfile,
    max_depth: int = 2,
    rclone_bin: str = "rclone",
) -> list[str]:
    """Build an rclone listing command for the artifact remote."""
    return [
        rclone_bin,
        "lsf",
        remote_target(profile),
        "--max-depth",
        str(max_depth),
    ]


def build_rclone_config_create_command(
    *,
    profile: ArtifactRemoteProfile,
    rclone_bin: str = "rclone",
) -> list[str]:
    """Build the short pCloud remote creation command."""
    return [rclone_bin, "config", "create", profile.remote, profile.backend]


def remote_exists(remote: str, listremotes_output: str) -> bool:
    """Return whether an rclone `listremotes` output contains the remote."""
    target = f"{normalize_remote_name(remote)}:"
    return target in {line.strip() for line in listremotes_output.splitlines()}
