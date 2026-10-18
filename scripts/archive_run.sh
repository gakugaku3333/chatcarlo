#!/bin/bash
# Archive a finished run directory (outside the repo) into a verified .tar.zst, then move the
# original to the Trash. Lossless: extraction restores the same bytes, so recorded SHA256 values
# (manifests, pilot locks, checkpoints) still verify after restoring.
#
#   scripts/archive_run.sh <dir> [<archive_dir>]
#     default archive_dir: ~/.cache/chatcarlo/archive
#   restore:  zstd -dc <archive>.tar.zst | tar -xf - -C <parent of dir>
#
# Steps: (1) SHA256 of every file, (2) tar | zstd -19, (3) extract into a temporary directory and
# compare every hash, (4) only if all match, mv the original to ~/.Trash (never rm).
set -euo pipefail
src=$(cd "$1" && pwd); parent=$(dirname "$src"); name=$(basename "$src")
dest=${2:-$HOME/.cache/chatcarlo/archive}; mkdir -p "$dest"
label=$(echo "${src#$HOME/.cache/chatcarlo/}" | tr '/' '_')
archive="$dest/$label.tar.zst"; sums="$dest/$label.sha256"
[ -e "$archive" ] && { echo "refuse: $archive exists"; exit 1; }
(cd "$parent" && find "$name" -type f -print0 | sort -z | xargs -0 shasum -a 256) > "$sums"
(cd "$parent" && tar -cf - "$name") | zstd -19 -T2 -q -o "$archive"
tmp=$(mktemp -d "$dest/.verify.XXXXXX")
zstd -dc "$archive" | tar -xf - -C "$tmp"
(cd "$tmp" && shasum -a 256 -c --quiet "$sums") || { echo "verify FAILED; original kept"; exit 1; }
n=$(wc -l < "$sums" | tr -d ' ')
mv "$tmp" "$HOME/.Trash/archive_verify_${label}_$(date +%Y%m%d%H%M%S)"
mv "$src" "$HOME/.Trash/${label}_$(date +%Y%m%d%H%M%S)"
echo "archived $n files: $(du -h "$archive" | cut -f1) $archive (original moved to Trash)"
