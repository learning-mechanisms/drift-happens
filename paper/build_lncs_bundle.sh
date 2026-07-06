#!/usr/bin/env sh
set -eu

# Packages the QCDS 2026 proceedings source bundle: every file needed to
# compile main-lncs.pdf from scratch, plus the generated .bbl and the
# reference PDF, zipped for submission to the proceedings chairs.
#
# The bundle is staged into a scratch directory and compiled there, so a
# missing source file fails loudly here instead of at the publisher.

PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
BUNDLE_NAME="qcds2026-drift-happens-sources"
DIST_DIR="$PAPER_DIR/dist"
ZIP_PATH="$DIST_DIR/$BUNDLE_NAME.zip"

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT
SRC_DIR="$STAGE_DIR/$BUNDLE_NAME"
mkdir -p "$SRC_DIR"

cd "$PAPER_DIR"
cp main-lncs.tex preamble-common.tex bibliography.bib llncs.cls \
  splncs04.bst splncs04short.bst \
  "$SRC_DIR/"
cp -R pages figures generated img plots_experiments tables "$SRC_DIR/"

cd "$SRC_DIR"
pdflatex -interaction=nonstopmode -halt-on-error main-lncs.tex >/dev/null
bibtex main-lncs >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main-lncs.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main-lncs.tex >/dev/null

pages="$(sed -n 's/.*Output written on main-lncs\.pdf (\([0-9]*\) pages.*/\1/p' main-lncs.log)"

# Keep sources, main-lncs.bbl, and main-lncs.pdf; drop other build byproducts.
rm -f main-lncs.aux main-lncs.log main-lncs.out main-lncs.blg

mkdir -p "$DIST_DIR"
rm -f "$ZIP_PATH"
cd "$STAGE_DIR"
zip -qr "$ZIP_PATH" "$BUNDLE_NAME"

printf "Wrote %s (%s pages incl. appendix)\n" "$ZIP_PATH" "${pages:-unknown}"
