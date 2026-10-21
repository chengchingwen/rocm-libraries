#!/bin/bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Assemble deck.tex from preamble.tex + the frames named in order.txt, then run pdflatex twice
# (the second pass resolves the counters metropolis puts in the progress bar).
#
# Frame order comes from order.txt, NOT from the filename sort, so a frame can move between
# sections or drop to the appendix without being renamed or deleted.
set -u
cd "$(dirname "$0")"

OUT=${1:-/tmp/deck}
mkdir -p "$OUT"
cp -f preamble.tex order.txt "$OUT"/
rm -rf "$OUT/frames"; cp -r frames "$OUT/frames"
cd "$OUT"

{
  cat preamble.tex
  while read -r line; do
    line="${line%%#*}"
    line="$(echo "$line" | sed 's/[[:space:]]*$//;s/^[[:space:]]*//')"
    [ -z "$line" ] && continue
    case "$line" in
      @appendix)  echo '\appendix' ;;
      @section*)  echo "\\section{${line#@section }}" ;;
      *)
        if [ -f "frames/$line.tex" ]; then
          echo "% ==== frames/$line.tex"; cat "frames/$line.tex"; echo
        else
          echo "MISSING frame: $line" >&2
        fi ;;
    esac
  done < order.txt
  echo '\end{document}'
} > deck.tex

# A frame on disk but absent from order.txt would silently vanish from the deck -- the exact
# failure the manifest exists to prevent, so name it rather than let it pass.
for f in frames/*.tex; do
  b=$(basename "$f" .tex)
  grep -qE "^[[:space:]]*$b([[:space:]]|#|$)" order.txt || echo "NOT IN order.txt: $b" >&2
done

pdflatex -interaction=nonstopmode deck.tex >b1.log 2>&1
pdflatex -interaction=nonstopmode deck.tex >b2.log 2>&1
if [ -f deck.pdf ]; then
  echo "OK pages=$(pdfinfo deck.pdf | awk '/^Pages/{print $2}')  ($OUT/deck.pdf)"
else
  echo "FAIL"; grep -m5 '^!' b2.log
fi
