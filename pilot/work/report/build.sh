#!/bin/sh
# build.sh -> report.pdf from report.html (headless Chrome), plus page images for checking
cd "$(dirname "$(readlink -f "$0")")" || exit 1
google-chrome --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf=report.pdf "file://$PWD/report.html" 2>/dev/null
pdfinfo report.pdf | grep -E "Pages|Page size"
rm -f page-*.png && pdftoppm -r 110 -png report.pdf page
