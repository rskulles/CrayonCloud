#!/usr/bin/env bash
# Build Crayon Cloud.app: a menu bar helper (Swift, compiled here) that sets up a Python environment on first launch
# from the package source shipped inside the bundle, then runs `crayoncloud serve`. The bundle carries its own
# Python (a relocatable CPython from python-build-standalone), so the machine it lands on needs nothing installed.
#
#   tools/make-macos-app.sh <version> <out-dir> [--dmg]
#
# Needs the Xcode Command Line Tools (swiftc, codesign, iconutil) and internet the first time (the Python build is
# cached under ~/.cache/crayoncloud-build). The app is ad-hoc signed, which is enough to run on the machine that
# built it; distributing it to other Macs would need a Developer ID signature and notarization.
set -euo pipefail

VERSION=${1:?version, e.g. 0.1.0}; OUT=${2:?out dir}; DMG=${3:-}
HERE=$(cd "$(dirname "$0")/.." && pwd)
APP="$OUT/Crayon Cloud.app"
RES="$APP/Contents/Resources"

# Pinned python-build-standalone release: bump both together (https://github.com/astral-sh/python-build-standalone/releases).
PBS_TAG=20260901
PBS_PYTHON=3.12.14
case "$(uname -m)" in
    arm64) PBS_ARCH=aarch64 ;;
    x86_64) PBS_ARCH=x86_64 ;;
    *) echo "unsupported architecture $(uname -m)"; exit 1 ;;
esac
PBS_FILE="cpython-$PBS_PYTHON+$PBS_TAG-$PBS_ARCH-apple-darwin-install_only_stripped.tar.gz"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_TAG/$PBS_FILE"
CACHE="${CRAYONCLOUD_BUILD_CACHE:-$HOME/.cache/crayoncloud-build}"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$RES/crayoncloud-src"
swiftc -O -target "$(uname -m)-apple-macos13.0" -framework AppKit -o "$APP/Contents/MacOS/CrayonCloudMenu" "$HERE/packaging/macos/CrayonCloudMenu/main.swift"

# The package source: pip installs it (with the mlx extra) into ~/Library/Application Support/CrayonCloud/venv on first launch.
cp -R "$HERE/crayoncloud" "$RES/crayoncloud-src/crayoncloud"
cp "$HERE/pyproject.toml" "$HERE/README.md" "$HERE/LICENSE.md" "$HERE/THIRD-PARTY-NOTICES.md" "$RES/crayoncloud-src/"
cp "$HERE/LICENSE.md" "$HERE/THIRD-PARTY-NOTICES.md" "$RES/"
find "$RES/crayoncloud-src" -name "__pycache__" -type d -prune -exec rm -rf {} +

# The interpreter: downloaded once, unpacked into Contents/Resources/python (bin/python3 and the standard library).
mkdir -p "$CACHE"
if [ ! -f "$CACHE/$PBS_FILE" ]; then
    echo "downloading $PBS_FILE"
    curl -fsSL -o "$CACHE/$PBS_FILE.part" "$PBS_URL" && mv "$CACHE/$PBS_FILE.part" "$CACHE/$PBS_FILE"
fi
tar -xzf "$CACHE/$PBS_FILE" -C "$RES"          # unpacks a top-level "python" folder
rm -rf "$RES/python/lib/python3.12/test"       # the interpreter's own test suite; a third of the size, never run
find "$RES/python" -name "__pycache__" -type d -prune -exec rm -rf {} +

# Icons: the app icon from the 1024 px render, the menu bar icon as a template image.
"$HERE/tools/make-icons.sh" "$RES"
sed "s/__VERSION__/$VERSION/g" "$HERE/packaging/macos/Info.plist" > "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"

# Every Mach-O in the bundle needs at least an ad-hoc signature on Apple Silicon; the Python tree has hundreds.
find "$RES/python" -type f \( -name "*.dylib" -o -name "*.so" -o -perm -u+x \) -print0 | while IFS= read -r -d '' f; do
    if file -b "$f" | grep -q "Mach-O"; then codesign --force --sign - "$f" 2>/dev/null || true; fi
done
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
