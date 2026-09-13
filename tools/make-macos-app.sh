#!/usr/bin/env bash
# Build Crayon Cloud.app: a menu bar helper (Swift, compiled here) that sets up a Python environment on first launch
# from the package source shipped inside the bundle, then runs `crayoncloud serve`.
#
#   tools/make-macos-app.sh <version> <out-dir> [--dmg]
#
# Needs the Xcode Command Line Tools (swiftc, codesign, iconutil). The app is ad-hoc signed, which is enough to run
# on the machine that built it; distributing it to other Macs would need a Developer ID signature and notarization.
set -euo pipefail

VERSION=${1:?version, e.g. 0.1.0}; OUT=${2:?out dir}; DMG=${3:-}
HERE=$(cd "$(dirname "$0")/.." && pwd)
APP="$OUT/Crayon Cloud.app"
RES="$APP/Contents/Resources"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$RES/crayoncloud-src"
swiftc -O -target "$(uname -m)-apple-macos13.0" -framework AppKit -o "$APP/Contents/MacOS/CrayonCloudMenu" "$HERE/packaging/macos/CrayonCloudMenu/main.swift"

# The package source: pip installs it (with the mlx extra) into ~/Library/Application Support/CrayonCloud/venv on first launch.
cp -R "$HERE/crayoncloud" "$RES/crayoncloud-src/crayoncloud"
cp "$HERE/pyproject.toml" "$HERE/README.md" "$HERE/LICENSE.md" "$HERE/THIRD-PARTY-NOTICES.md" "$RES/crayoncloud-src/"
cp "$HERE/LICENSE.md" "$HERE/THIRD-PARTY-NOTICES.md" "$RES/"
find "$RES/crayoncloud-src" -name "__pycache__" -type d -prune -exec rm -rf {} +

# Icons: the app icon from the 1024 px render, the menu bar icon as a template image.
"$HERE/tools/make-icons.sh" "$RES"
sed "s/__VERSION__/$VERSION/g" "$HERE/packaging/macos/Info.plist" > "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"

codesign --force --sign - "$APP/Contents/MacOS/CrayonCloudMenu"
codesign --force --sign - "$APP"
echo "built $APP"

if [ "$DMG" = "--dmg" ]; then
    STAGE=$(mktemp -d)
    cp -R "$APP" "$STAGE/"
    ln -s /Applications "$STAGE/Applications"
    hdiutil create -volname "Crayon Cloud" -srcfolder "$STAGE" -ov -format UDZO "$OUT/CrayonCloud-v$VERSION-$(uname -m).dmg" >/dev/null
    rm -rf "$STAGE"
    echo "built $OUT/CrayonCloud-v$VERSION-$(uname -m).dmg"
fi
