"""CLI for inspecting, syncing, and safely pruning local artifacts."""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import typer

from drift_happens.dataset.embedding_cache import (
    delete_cache_files,
    delete_feature_cache_dirs,
    discover_feature_caches,
    select_caches,
)
from drift_happens.utils.artifact_bundles import (
    BUNDLE_NAMES,
    BundleResult,
    build_bundle,
    download_bundle,
    pack_bundle,
    stage_bundle,
)
from drift_happens.utils.artifact_remote import (
    DEFAULT_PROFILE_PATH,
    DEFAULT_REMOTE_NAME,
    DEFAULT_REMOTE_PATH,
    ArtifactRemoteProfile,
    build_rclone_config_create_command,
    build_rclone_lsf_command,
    build_rclone_transfer_command,
    make_remote_profile,
    read_remote_profile,
    remote_exists,
    remote_target,
    write_remote_profile,
)
from drift_happens.utils.artifacts import (
    apply_gc,
    list_artifacts,
    plan_gc,
    plan_wandb_deleted_run_gc,
)
from drift_happens.utils.paths import (
    ARTIFACTS_DIR,
    BUNDLES_DIR,
    RUNS_DIR,
    relative_to_project,
)

app = typer.Typer(
    help="Inspect, sync, and safely prune local run and sweep artifacts.",
    no_args_is_help=True,
)

cache_app = typer.Typer(
    help="Inspect and clear on-disk embedding caches.",
    no_args_is_help=True,
)
remote_app = typer.Typer(
    help="Sync local artifacts to an rclone-backed remote such as pCloud.",
    no_args_is_help=True,
)
bundle_app = typer.Typer(
    help="Build and download public reproducibility artifact bundles.",
    no_args_is_help=True,
)
app.add_typer(cache_app, name="cache")
app.add_typer(remote_app, name="remote")
app.add_typer(bundle_app, name="bundle")


@app.command("ls")
def list_local_artifacts(
    kind: Literal["runs", "sweeps", "all"] = typer.Option("all", "--kind"),
    status: Literal["ok", "failed", "running", "partial", "missing"] | None = (
        typer.Option(None, "--status")
    ),
    root: Path = typer.Option(ARTIFACTS_DIR, "--root"),
) -> None:
    """List canonical local artifacts."""
    typer.echo(
        "kind\tstatus\tdataset\ttrainer\texperiment\tseed\ttrain\teval\t"
        "source_identity\tconfig_hash\tpath"
    )
    for row in list_artifacts(root=root, kind=kind, status=status):
        typer.echo(
            "\t".join(
                [
                    row.kind,
                    row.status,
                    row.dataset or "",
                    row.trainer or "",
                    row.experiment or "",
                    "" if row.seed is None else str(row.seed),
                    row.train or "",
                    row.eval or "",
                    row.source_identity or "",
                    row.config_hash or "",
                    str(relative_to_project(row.path)),
                ]
            )
        )


@app.command("gc")
def gc_artifacts(
    keep_attempts: int = typer.Option(
        3,
        "--keep-attempts",
        min=0,
        help="Newest attempt dirs to keep per run stage and per sweep name.",
    ),
    dry_run: bool = typer.Option(True, "--dry-run/--apply"),
    root: Path = typer.Option(ARTIFACTS_DIR, "--root"),
) -> None:
    """Plan or apply safe local artifact cleanup."""
    items = plan_gc(
        root=root,
        keep_attempts=keep_attempts,
    )
    for item in items:
        typer.echo(f"{item.reason}\t{relative_to_project(item.path)}")
    if dry_run:
        typer.echo(f"dry-run: {len(items)} path(s) would be deleted")
        return
    deleted = apply_gc(items, root=root)
    typer.echo(f"deleted {len(deleted)} path(s)")


@app.command("wandb-gc")
def gc_deleted_wandb_runs(
    wandb_project: str | None = typer.Option(
        None,
        "--wandb-project",
        help="W&B project. Defaults to WANDB_PROJECT.",
    ),
    wandb_entity: str | None = typer.Option(
        None,
        "--wandb-entity",
        help="W&B entity. Defaults to WANDB_ENTITY.",
    ),
    dry_run: bool = typer.Option(True, "--dry-run/--apply"),
    root: Path = typer.Option(ARTIFACTS_DIR, "--root"),
) -> None:
    """Delete local run dirs whose recorded W&B runs no longer exist remotely."""
    project = wandb_project or os.environ.get("WANDB_PROJECT")
    if project is None or not project.strip():
        raise typer.BadParameter("W&B GC requires --wandb-project or WANDB_PROJECT")
    entity = (
        wandb_entity if wandb_entity is not None else os.environ.get("WANDB_ENTITY")
    )
    entity = entity or None
    items = plan_wandb_deleted_run_gc(
        project=project,
        entity=entity,
        root=root,
    )
    for item in items:
        typer.echo(f"{item.reason}\t{relative_to_project(item.path)}")
    if dry_run:
        typer.echo(f"dry-run: {len(items)} local run dir(s) would be deleted")
        return
    deleted = apply_gc(items, root=root)
    typer.echo(f"deleted {len(deleted)} local run dir(s)")


@remote_app.command("setup")
def setup_remote(
    remote: str = typer.Option(DEFAULT_REMOTE_NAME, "--remote"),
    path: str = typer.Option(DEFAULT_REMOTE_PATH, "--path"),
    profile_path: Path = typer.Option(DEFAULT_PROFILE_PATH, "--profile"),
    configure_rclone: bool = typer.Option(
        True,
        "--configure-rclone/--skip-rclone-config",
        help="Create the rclone pCloud remote when it is not already configured.",
    ),
    rclone_bin: str = typer.Option("rclone", "--rclone-bin"),
) -> None:
    """Write the local pCloud profile and optionally start rclone setup."""
    try:
        profile = make_remote_profile(remote=remote, path=path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    written = write_remote_profile(profile, profile_path)
    typer.echo(f"wrote {relative_to_project(written)}")
    typer.echo(f"remote target: {remote_target(profile)}")
    if not configure_rclone:
        return

    if _rclone_remote_exists(profile, rclone_bin=rclone_bin):
        typer.echo(f"rclone remote already exists: {profile.remote}:")
        return

    command = build_rclone_config_create_command(profile=profile, rclone_bin=rclone_bin)
    typer.echo(f"running {shlex.join(command)}")
    typer.echo("rclone will handle the pCloud OAuth flow.")
    _run_command(command)


@remote_app.command("status")
def remote_status(
    profile_path: Path = typer.Option(DEFAULT_PROFILE_PATH, "--profile"),
    rclone_bin: str = typer.Option("rclone", "--rclone-bin"),
) -> None:
    """Show the configured artifact remote and basic rclone reachability."""
    profile = _load_profile(profile_path)
    typer.echo(f"profile: {relative_to_project(profile_path)}")
    typer.echo(f"remote target: {remote_target(profile)}")
    _run_command([rclone_bin, "version"])
    _run_command([rclone_bin, "lsd", f"{profile.remote}:"])


@remote_app.command("ls")
def list_remote(
    profile_path: Path = typer.Option(DEFAULT_PROFILE_PATH, "--profile"),
    max_depth: int = typer.Option(2, "--max-depth", min=1),
    rclone_bin: str = typer.Option("rclone", "--rclone-bin"),
) -> None:
    """List files and directories currently visible on the artifact remote."""
    profile = _load_profile(profile_path)
    _run_command(
        build_rclone_lsf_command(
            profile=profile,
            max_depth=max_depth,
            rclone_bin=rclone_bin,
        )
    )


@remote_app.command("push")
def push_remote(
    profile_path: Path = typer.Option(DEFAULT_PROFILE_PATH, "--profile"),
    root: Path = typer.Option(ARTIFACTS_DIR, "--root"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    mirror: bool = typer.Option(
        False,
        "--mirror",
        help="Use rclone sync instead of copy; this can delete remote files.",
    ),
    with_attempts: bool = typer.Option(False, "--with-attempts"),
    with_all_checkpoints: bool = typer.Option(False, "--with-all-checkpoints"),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
    transfers: int = typer.Option(4, "--transfers", min=1),
    rclone_bin: str = typer.Option("rclone", "--rclone-bin"),
) -> None:
    """Copy curated local artifacts to the configured remote."""
    profile = _load_profile(profile_path)
    command = build_rclone_transfer_command(
        direction="push",
        profile=profile,
        artifacts_root=root,
        mirror=mirror,
        dry_run=dry_run,
        progress=progress,
        with_attempts=with_attempts,
        with_all_checkpoints=with_all_checkpoints,
        transfers=transfers,
        rclone_bin=rclone_bin,
    )
    typer.echo(shlex.join(command))
    _run_command(command)


@remote_app.command("pull")
def pull_remote(
    profile_path: Path = typer.Option(DEFAULT_PROFILE_PATH, "--profile"),
    root: Path = typer.Option(ARTIFACTS_DIR, "--root"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    mirror: bool = typer.Option(
        False,
        "--mirror",
        help="Use rclone sync instead of copy; this can delete local files.",
    ),
    with_attempts: bool = typer.Option(False, "--with-attempts"),
    with_all_checkpoints: bool = typer.Option(False, "--with-all-checkpoints"),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
    transfers: int = typer.Option(4, "--transfers", min=1),
    rclone_bin: str = typer.Option("rclone", "--rclone-bin"),
) -> None:
    """Copy curated artifacts from the configured remote into the local artifact
    root."""
    profile = _load_profile(profile_path)
    command = build_rclone_transfer_command(
        direction="pull",
        profile=profile,
        artifacts_root=root,
        mirror=mirror,
        dry_run=dry_run,
        progress=progress,
        with_attempts=with_attempts,
        with_all_checkpoints=with_all_checkpoints,
        transfers=transfers,
        rclone_bin=rclone_bin,
    )
    typer.echo(shlex.join(command))
    _run_command(command)


@bundle_app.command("stage")
def stage_artifact_bundle(
    name: str = typer.Argument(..., help=f"Bundle name: {', '.join(BUNDLE_NAMES)}"),
    runs_root: Path = typer.Option(RUNS_DIR, "--runs-root"),
    bundle_root: Path = typer.Option(BUNDLES_DIR, "--bundle-root"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Create the unpacked staging directory for a public artifact bundle."""
    result = _run_bundle_command(
        lambda: stage_bundle(
            name,
            runs_root=runs_root,
            bundle_root=bundle_root,
            overwrite=overwrite,
        )
    )
    _echo_bundle_result("staged", result)


@bundle_app.command("pack")
def pack_artifact_bundle(
    name: str = typer.Argument(..., help=f"Bundle name: {', '.join(BUNDLE_NAMES)}"),
    bundle_root: Path = typer.Option(BUNDLES_DIR, "--bundle-root"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Pack a staged public artifact bundle into a tar archive."""
    result = _run_bundle_command(
        lambda: pack_bundle(name, bundle_root=bundle_root, overwrite=overwrite)
    )
    _echo_bundle_result("packed", result)


@bundle_app.command("build")
def build_artifact_bundle(
    name: str = typer.Argument(..., help=f"Bundle name: {', '.join(BUNDLE_NAMES)}"),
    runs_root: Path = typer.Option(RUNS_DIR, "--runs-root"),
    bundle_root: Path = typer.Option(BUNDLES_DIR, "--bundle-root"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Stage and pack a public artifact bundle."""
    result = _run_bundle_command(
        lambda: build_bundle(
            name,
            runs_root=runs_root,
            bundle_root=bundle_root,
            overwrite=overwrite,
        )
    )
    _echo_bundle_result("built", result)


@bundle_app.command("download")
def download_artifact_bundle(
    name: str = typer.Argument(..., help=f"Bundle name: {', '.join(BUNDLE_NAMES)}"),
    bundle_root: Path = typer.Option(BUNDLES_DIR, "--bundle-root"),
    download_link: str | None = typer.Option(None, "--download-link"),
    expected_sha256: str | None = typer.Option(
        None,
        "--expected-sha256",
        help="Expected SHA-256 for an override bundle archive.",
    ),
    expected_size: int | None = typer.Option(
        None,
        "--expected-size",
        min=1,
        help="Expected byte size for an override bundle archive.",
    ),
    skip_integrity_check: bool = typer.Option(
        False,
        "--skip-integrity-check",
        help="Download without validating the archive SHA-256 or byte size.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Download and safely extract a public artifact bundle."""
    result = _run_bundle_command(
        lambda: download_bundle(
            name,
            bundle_root=bundle_root,
            download_link=download_link,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            skip_integrity_check=skip_integrity_check,
            overwrite=overwrite,
        )
    )
    _echo_bundle_result("downloaded", result)


@cache_app.command("ls")
def list_embedding_caches() -> None:
    """List on-disk embedding caches by dataset."""
    typer.echo("kind\tdataset\tproducer\toutput\tfiles\tbytes\tpath")
    for name, cache in select_caches(None).items():
        files = cache.cached_files()
        total = sum(file.stat().st_size for file in files)
        typer.echo(
            f"image\t{name}\t-\t-\t{len(files)}\t{total}\t"
            f"{relative_to_project(cache.cache_dir)}"
        )
    for entry in discover_feature_caches():
        typer.echo(
            f"text\t{entry.dataset}\t{entry.producer}\t{entry.kind}\t"
            f"{len(entry.files)}\t{entry.size_bytes}\t{relative_to_project(entry.path)}"
        )


@cache_app.command("clear")
def clear_embedding_cache(
    dataset: str | None = typer.Option(None, "--dataset"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply"),
) -> None:
    """Delete on-disk embedding caches so the next run recomputes them."""
    feature_entries = tuple(discover_feature_caches())
    # image caches: only registered datasets; text-only datasets produce no entry here
    caches = {
        k: v for k, v in select_caches(None).items() if dataset is None or k == dataset
    }
    dirs = tuple(
        entry.path
        for entry in feature_entries
        if dataset is None or entry.dataset == dataset
    )
    if dataset is not None and not caches and not dirs:
        known = ", ".join(
            sorted(
                {*select_caches(None), *(entry.dataset for entry in feature_entries)}
            )
        )
        raise typer.BadParameter(f"unknown dataset {dataset!r}; choose one of: {known}")
    files = tuple(file for cache in caches.values() for file in cache.cached_files())
    for file in files:
        typer.echo(str(relative_to_project(file)))
    for directory in dirs:
        typer.echo(str(relative_to_project(directory)))
    if dry_run:
        typer.echo(
            f"dry-run: {len(files)} file(s) and {len(dirs)} dir(s) would be deleted"
        )
        return
    deleted_files = delete_cache_files(files)
    deleted_dirs = delete_feature_cache_dirs(dirs)
    typer.echo(f"deleted {len(deleted_files)} file(s) and {len(deleted_dirs)} dir(s)")


def _run_bundle_command(action: Callable[[], BundleResult]) -> BundleResult:
    try:
        return action()
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


def _echo_bundle_result(action: str, result: BundleResult) -> None:
    typer.echo(f"{action}: {relative_to_project(result.bundle_dir)}")
    typer.echo(f"staged: {relative_to_project(result.staged_dir)}")
    typer.echo(f"archive: {relative_to_project(result.archive_path)}")
    typer.echo(f"manifest: {relative_to_project(result.manifest_path)}")
    typer.echo(f"sha256: {relative_to_project(result.sha256_path)}")
    typer.echo(f"files: {result.file_count}")
    typer.echo(f"bytes: {result.size_bytes}")


def _load_profile(path: Path) -> ArtifactRemoteProfile:
    try:
        return read_remote_profile(path)
    except FileNotFoundError as exc:
        typer.echo(
            f"missing artifact remote profile: {relative_to_project(path)}; "
            "run `drift artifacts remote setup` first",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _rclone_remote_exists(profile: ArtifactRemoteProfile, *, rclone_bin: str) -> bool:
    result = _run_command(
        [rclone_bin, "listremotes"],
        capture_output=True,
        exit_on_failure=False,
    )
    return result.returncode == 0 and remote_exists(profile.remote, result.stdout)


def _run_command(
    command: list[str],
    *,
    capture_output: bool = False,
    exit_on_failure: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=capture_output,
            text=True,
        )
    except FileNotFoundError as exc:
        typer.echo(f"command not found: {command[0]}", err=True)
        raise typer.Exit(code=1) from exc
    if result.returncode != 0 and exit_on_failure:
        raise typer.Exit(code=result.returncode)
    return result
