#!/usr/bin/env bash
# Xuất image, nén, tính MD5 và kích thước .tar chưa nén (Docker guideline mục 11). Cần TEAM.
set -euo pipefail
: "${TEAM:?Đặt TEAM trước}"
IMAGE="zaic2026_datamate_${TEAM}:final"; ARCHIVE="zaic2026_datamate_${TEAM}.tar.gz"
docker image inspect -f '{{.Os}}/{{.Architecture}}' "$IMAGE" | grep -q '^linux/amd64$' || { echo "Image không phải linux/amd64"; exit 1; }
if command -v pigz >/dev/null; then GZ="pigz -p $(nproc)"; GUNZ="pigz -dc"; else GZ="gzip"; GUNZ="gzip -dc"; fi
docker save "$IMAGE" | $GZ > "$ARCHIVE"
echo "== Kích thước .tar CHƯA nén (byte; phải <= 100 GB = 107374182400):"
$GUNZ "$ARCHIVE" | wc -c | tee "${ARCHIVE%.tar.gz}.uncompressed_size_bytes.txt"
md5sum "$ARCHIVE" > "$ARCHIVE.md5"
echo "== MD5:"; cat "$ARCHIVE.md5"
ls -lh "$ARCHIVE"
echo "KHÔNG đổi tên $ARCHIVE sau bước này. Tải lên Google Drive (công khai), rồi dán link + MD5 + kích thước .tar vào form Solution Submission."
