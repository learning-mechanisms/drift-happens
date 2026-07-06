from __future__ import annotations

import argparse
import subprocess
import sys

DISALLOWED_PREFIXES = (
    "artifacts/experiments/",
    "artifacts/plots/",
    "artifacts/robustness_results/",
    "artifacts/runs/",
    "artifacts/sweeps/",
    "artifacts/bundles/",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reject newly added generated experiment artifacts."
    )
    parser.add_argument(
        "--base-ref",
        help="Git ref to diff against. Defaults to staged additions.",
    )
    args = parser.parse_args()

    paths = _added_paths(args.base_ref)
    blocked = [
        path
        for path in paths
        if any(path.startswith(prefix) for prefix in DISALLOWED_PREFIXES)
    ]
    if not blocked:
        return 0

    print("Generated experiment artifacts must not be committed:", file=sys.stderr)
    for path in blocked:
        print(f"  - {path}", file=sys.stderr)
    print(
        "Keep curated files under artifacts/experiment_plans/ or document an explicit "
        "exception in the pull request.",
        file=sys.stderr,
    )
    return 1


def _added_paths(base_ref: str | None) -> list[str]:
    # ACR: Added, Copied, Renamed — catches files landing under blocked prefixes via any git operation.
    if base_ref:
        cmd = [
            "git",
            "diff",
            "-z",
            "--name-only",
            "--diff-filter=ACR",
            f"{base_ref}...HEAD",
        ]
    else:
        cmd = ["git", "diff", "--cached", "-z", "--name-only", "--diff-filter=ACR"]
    result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    return [part for part in result.stdout.split("\0") if part]


if __name__ == "__main__":
    raise SystemExit(main())
