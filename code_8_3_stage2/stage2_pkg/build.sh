#!/bin/bash
# usage: ./build.sh file.md  -> file.pdf (CN template)
f=${1%.md}
pandoc "$1" -s -t latex -H tmpl_head.tex -V papersize=a4 -V mainfont="Noto Serif CJK SC" -o "$f.tex" \
 && sed -i '/\\usepackage{lmodern}/d' "$f.tex" \
 && xelatex -interaction=nonstopmode "$f.tex" >/dev/null && xelatex -interaction=nonstopmode "$f.tex" >/dev/null
grep -E "^!|Undefined control|Missing" "$f.log" | head
pdfinfo "$f.pdf" | grep Pages
