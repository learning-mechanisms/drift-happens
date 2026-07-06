from __future__ import annotations


def prompt_confirmation(message: str, *, assume_yes: bool = False) -> bool:
    """
    Prompt user for yes/no confirmation.

    Returns True for yes, False for no. With ``assume_yes`` the prompt is skipped and
    True is returned, for non-interactive runs.
    """
    if assume_yes:
        return True
    while True:
        try:
            response = input(f"{message} [y/n]: ").strip().lower()
        except EOFError:
            import sys

            print(
                "\nNon-interactive stdin; pass --yes / assume_yes=True to skip prompt.",
                file=sys.stderr,
            )
            return False
        if response in {"y", "yes"}:
            return True
        elif response in {"n", "no"}:
            return False
        else:
            print("Please enter 'y' or 'n'.")
