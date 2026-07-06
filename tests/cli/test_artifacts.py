from click.utils import strip_ansi
from typer.testing import CliRunner

from drift_happens.cli.artifacts import app

runner = CliRunner()


def test_gc_rejects_negative_keep_attempts_as_usage_error() -> None:
    result = runner.invoke(app, ["gc", "--keep-attempts", "-1"])

    assert result.exit_code == 2
    assert isinstance(result.exception, SystemExit)
    assert "--keep-attempts" in strip_ansi(result.output)


def test_ls_rejects_unknown_status_as_usage_error() -> None:
    result = runner.invoke(app, ["ls", "--status", "okk"])

    assert result.exit_code == 2
    assert isinstance(result.exception, SystemExit)
    assert "--status" in strip_ansi(result.output)


def test_ls_accepts_known_status() -> None:
    result = runner.invoke(app, ["ls", "--status", "failed", "--root", "/tmp/nope"])

    assert result.exit_code == 0


def test_remote_setup_writes_profile_without_configuring_rclone(tmp_path) -> None:
    profile_path = tmp_path / "pcloud.json"

    result = runner.invoke(
        app,
        [
            "remote",
            "setup",
            "--profile",
            str(profile_path),
            "--remote",
            "test-pcloud",
            "--path",
            "/drift/artifacts",
            "--skip-rclone-config",
        ],
    )

    assert result.exit_code == 0, result.output
    assert profile_path.exists()
    assert "test-pcloud:/drift/artifacts" in result.output
