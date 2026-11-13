#!/bin/bash
# Build the final report PDF on the GPU box with the cached Tectonic (no LaTeX on the PC).
# Run:  ssh haiderserver@192.168.1.55 'bash "/srv/pool/uni/COMP8240/A3 Final Report/build-on-gpu-box.sh"'
R="/srv/pool/uni/COMP8240/A3 Final Report"
C=/tmp/c8240tex-cache
B=/tmp/c8240tex-build
rm -rf "$B"; mkdir -p "$B"
if [ ! -d "$C/Tectonic" ]; then
  mkdir -p "$C"
  cp -r /srv/pool/uni/_uni-bot/cache/Tectonic "$C/" 2>/dev/null || true
fi
cp "$R/acl.sty" "$R/acl_natbib.bst" "$R/references.bib" "$B/"
cp "$R/Haider Jalal - COMP8240 Final Report.tex" "$B/report.tex"
cd "$B" || exit 1
XDG_CACHE_HOME="$C" /srv/pool/uni/_uni-bot/render/tex/tectonic --only-cached --keep-logs report.tex > build.out 2>&1
st=$?
grep -E "error|Overfull|Underfull|warning: .*undefined|Citation|Reference" build.out | grep -v "^note" | head -40
echo "tectonic exit $st"
if [ $st -eq 0 ] && [ -f report.pdf ]; then
  cp report.pdf "$R/Haider Jalal - COMP8240 Final Report.pdf"
  echo "copied PDF"
else
  tail -30 build.out
fi
