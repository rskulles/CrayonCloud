#!/usr/bin/env bash
# Produce CrayonCloud.icns and the menu bar template icon from packaging/icon.png and packaging/macos/MenuIcon.svg.
#   tools/make-icons.sh <out-dir>
set -euo pipefail
OUT=${1:?out dir}
HERE=$(cd "$(dirname "$0")/.." && pwd)
SET=$(mktemp -d)/CrayonCloud.iconset
mkdir -p "$SET"
for size in 16 32 128 256 512; do
    sips -z $size $size "$HERE/packaging/icon.png" --out "$SET/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -z $double $double "$HERE/packaging/icon.png" --out "$SET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$SET" -o "$OUT/CrayonCloud.icns"
cp "$HERE/packaging/macos/MenuIcon.png" "$HERE/packaging/macos/MenuIcon@2x.png" "$OUT/"
