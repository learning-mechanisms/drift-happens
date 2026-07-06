"""
Guard the shared SLURM env-file bootstrap block against silent drift.

The DRIFT_ENV_FILE discovery block is intentionally duplicated verbatim across the three
sbatch scripts. It is bootstrap code: it must resolve the env file *before* the repo
location is known, so it cannot source a repo-relative helper -- SLURM relocates the
submitted script to a spool copy and runs it from the submit directory, so neither the
script's own path nor a repo path is reliably available at that point. Keeping the block
self-contained is what makes it work from any submit directory. This test keeps the
three copies identical so an edit to one is mirrored to all, which is the guarantee the
extraction would have provided without the bootstrap-robustness cost.
"""

from __future__ import annotations

from pathlib import Path

SLURM_DIR = Path(__file__).resolve().parents[2] / "scripts" / "slurm"

_START_MARKER = "env_candidates=()"
_END_MARKER = 'source "$DRIFT_ENV_FILE"'


def _env_file_block(script: Path) -> str:
    lines = script.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line == _START_MARKER), None)
    end = next((i for i, line in enumerate(lines) if line == _END_MARKER), None)
    assert start is not None, f"{script.name}: missing '{_START_MARKER}' marker"
    assert end is not None, f"{script.name}: missing '{_END_MARKER}' marker"
    assert start < end, f"{script.name}: env-file block markers out of order"
    return "\n".join(lines[start : end + 1])


def _sbatch_scripts_with_env_block() -> list[Path]:
    """
    Every sbatch script that carries the bootstrap block, auto-discovered.

    Auto-discovery (rather than a hardcoded list) means a future sbatch script that
    copies the block is guarded too, instead of silently escaping the check.
    """
    return sorted(
        path
        for path in SLURM_DIR.glob("*.sbatch")
        if "DRIFT_ENV_FILE" in path.read_text()
    )


def test_slurm_env_file_block_is_identical_across_sbatch_scripts() -> None:
    scripts = _sbatch_scripts_with_env_block()
    assert len(scripts) >= 3, (
        "expected at least the three known sbatch scripts to carry the env-file block; "
        f"found {[p.name for p in scripts]}"
    )

    blocks = {path.name: _env_file_block(path) for path in scripts}
    reference_name, reference = next(iter(blocks.items()))
    # Block is non-degenerate and includes the failure-exit path.
    assert "exit 1" in reference
    assert reference.count("\n") > 20

    for name, block in blocks.items():
        assert block == reference, (
            f"{name} DRIFT_ENV_FILE block drifted from {reference_name}; "
            "the sbatch bootstrap blocks must stay identical"
        )
