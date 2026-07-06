"""LaTeX escaping for label text used in the paper's tables and appendix."""

from __future__ import annotations

from drift_happens.analysis.plots.names import get_display_name

_TEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex(text: str) -> str:
    """Escape LaTeX special characters in arbitrary label text."""
    return "".join(_TEX_SPECIALS.get(char, char) for char in text)


def tex_name(model: str) -> str:
    """LaTeX-safe display label for a trainer key."""
    return tex(get_display_name(model))
