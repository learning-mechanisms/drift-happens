#!/usr/bin/env sh
set -eu

# Packages the arXiv preprint source bundle: every file needed to compile
# main-preprint.pdf from scratch, plus the generated .bbl. The zip contents
# are rooted at main-preprint.tex because arXiv compiles from the upload root.

PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
BUNDLE_NAME="drift-happens-arxiv-preprint-sources"
DIST_DIR="$PAPER_DIR/dist"
ZIP_PATH="$DIST_DIR/$BUNDLE_NAME.zip"

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT
SRC_DIR="$STAGE_DIR/source"
mkdir -p "$SRC_DIR"

cd "$PAPER_DIR"
cp main-preprint.tex preamble-common.tex bibliography.bib neutral2col.sty \
  algorithm.sty algorithmic.sty fancyhdr.sty \
  "$SRC_DIR/"
cp -R pages figures generated img plots_experiments tables "$SRC_DIR/"

# The source tree contains a few LNCS-only or site-only figure exports. Leave
# them out of the arXiv upload so the submission contains only build inputs.
find "$SRC_DIR/plots_experiments" -name "*_lncs.pdf" -delete
find "$SRC_DIR/plots_experiments" -type d -name overview -prune -exec rm -rf {} +

cd "$SRC_DIR"
pdflatex -interaction=nonstopmode -halt-on-error main-preprint.tex >/dev/null
bibtex main-preprint >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main-preprint.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main-preprint.tex >/dev/null

pages="$(sed -n 's/.*Output written on main-preprint\.pdf (\([0-9]*\) pages.*/\1/p' main-preprint.log)"

# Keep sources and main-preprint.bbl; drop build byproducts and local metadata.
rm -f main-preprint.aux main-preprint.log main-preprint.out main-preprint.blg \
  main-preprint.pdf main-preprint.fdb_latexmk main-preprint.fls \
  main-preprint.synctex.gz
find . -name .DS_Store -delete

mkdir -p "$DIST_DIR"
rm -f "$ZIP_PATH"
zip -qr "$ZIP_PATH" .

printf "Wrote %s (%s pages incl. appendix)\n" "$ZIP_PATH" "${pages:-unknown}"
