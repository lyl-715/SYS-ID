#!/bin/bash
f=${1%.md}
pandoc "$1" -s -t latex -V papersize=a4 -V mainfont="Liberation Serif" -V geometry:margin=2.2cm -V fontsize=11pt -H <(printf '\\usepackage{amsmath,amssymb,booktabs,longtable,array}\n\\providecommand{\\tightlist}{}\n\\setlength{\\parskip}{4pt}\n') -o "$f.tex" && sed -i '/\\usepackage{lmodern}/d' "$f.tex" && xelatex -interaction=nonstopmode "$f.tex" >/dev/null && xelatex -interaction=nonstopmode "$f.tex" >/dev/null
grep -E "^!|Undefined control|Missing" "$f.log" | head -5; pdfinfo "$f.pdf" | grep Pages
