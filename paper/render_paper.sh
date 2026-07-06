#!/usr/bin/env sh
set -eu

# Builds both paper versions from the shared source:
#   main-lncs.tex      Official Springer LNCS proceedings version (QCDS 2026).
#   main-preprint.tex  Neutral compact two-column extended version (web / arXiv).
#
# Both roots \input the same pages/, tables/, and figures/; only the layout
# differs. The extended two-column version is published as the website's
# primary PDF; the LNCS version is offered as the secondary (proceedings) link.

PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$PAPER_DIR/.." && pwd)"
WEBSITE_DIR="$ROOT_DIR/website"

cd "$PAPER_DIR"

build_pdf() {
  jobname="$1"
  tex_input="$2"

  rm -f "$jobname.aux" "$jobname.bbl" "$jobname.blg" "$jobname.out"

  pdflatex -interaction=nonstopmode -halt-on-error -jobname="$jobname" "$tex_input"
  bibtex "$jobname" || true
  pdflatex -interaction=nonstopmode -halt-on-error -jobname="$jobname" "$tex_input"
  pdflatex -interaction=nonstopmode -halt-on-error -jobname="$jobname" "$tex_input"
  pdflatex -interaction=nonstopmode -halt-on-error -jobname="$jobname" "$tex_input"
}

build_pdf "main-lncs" "main-lncs.tex"
build_pdf "main-preprint" "main-preprint.tex"

mkdir -p "$WEBSITE_DIR"
cp "$PAPER_DIR/main-preprint.pdf" "$WEBSITE_DIR/drift-happens.pdf"        # primary (extended)
cp "$PAPER_DIR/main-lncs.pdf"     "$WEBSITE_DIR/drift-happens-lncs.pdf"   # secondary (proceedings)

printf "Wrote %s\n" "$PAPER_DIR/main-lncs.pdf"
printf "Wrote %s\n" "$PAPER_DIR/main-preprint.pdf"
printf "Synced %s\n" "$WEBSITE_DIR/drift-happens.pdf"
printf "Synced %s\n" "$WEBSITE_DIR/drift-happens-lncs.pdf"
